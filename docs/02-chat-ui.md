# 02. 統合チャット UI (`llm-chat`)

## 1. 概要

- ブラウザから 8 モデルを**いつでも切り替えて**チャットできる UI。ポート **5070** (devcontainer で転送済み)。
- `/workspace/LLM/*/model.toml` を持つディレクトリを自動検出する。
  Bonsai 2 と Qwen3.8 Flash Next の `thinking_mode = "reasoning_effort"` は思考オンを medium に設定する。
- 選択されたモデルの `serve.sh` を内部ポート **5071** で 1 つだけ起動し、チャットを中継する。
  Qwen3.8 Flash Next は [Strata](07-strata.md)、その他は llama-server を使用する。
- 会話履歴の保存 (任意)、ウェブ検索 (任意、[03](03-web-search.md))。

```
ブラウザ ──:5070──▶ llm-chat (FastAPI)
                     ├─ ModelManager ── bash serve.sh (PORT=5071) ──▶ llama-server
                     ├─ /api/chat ──(SSE 中継)──▶ :5071/v1/chat/completions
                     ├─ agent.run ── web_search / fetch_page (検索オン時)
                     └─ history/ に JSON + Markdown 保存
```

## 2. プロジェクト作成コマンド

```bash
cd /workspace/LLM/llm-chat
uv init --app --name llm-chat --python 3.12 --no-readme --vcs none .
uv add fastapi "uvicorn[standard]" httpx
uv add ddgs trafilatura          # ウェブ検索用 (03 参照)

uvx ruff check --fix src && uvx ruff format src   # lint / format
uv run llm-chat                                   # 起動 (http://localhost:5070)
```

`pyproject.toml` の `[project.scripts] llm-chat = "llm_chat:main"` によりコマンド化されている。
ディレクトリを移動した場合は `.venv` を作り直す (`rm -rf .venv && uv sync`)。

## 3. ファイル構成

```
llm-chat/src/llm_chat/
├── __init__.py        main(): uvicorn 起動
├── server.py          FastAPI アプリ、ModelManager、履歴 API
├── agent.py           ウェブ検索ツール付きエージェントループ
├── search/            検索プロバイダ抽象化 (03 参照)
└── static/index.html  UI (vanilla JS、marked + DOMPurify を CDN から読込)
```

## 4. 環境変数

| 変数 | 既定 | 内容 |
|---|---|---|
| `LLM_CHAT_HOST` / `LLM_CHAT_PORT` | `0.0.0.0` / `5070` | UI の待受 |
| `LLM_ROOT` | `llm-chat` の親 (`/workspace/LLM`) | モデル検出のルート |
| `LLM_BACKEND_PORT` | `5071` | 内部 llama-server のポート |
| `LLM_HISTORY_DIR` | `llm-chat/history` | 履歴保存先 |

パスは `Path(__file__).resolve().parents[2]` から求めているため、`LLM/` ごと移動しても動く。

## 5. HTTP API

| メソッド | パス | 内容 |
|---|---|---|
| GET | `/` | UI (index.html) |
| GET | `/api/status` | `{active, status, error, loaded_at, models[]}`。status は `stopped`/`loading`/`ready`/`error` |
| POST | `/api/models/{id}/load` | モデル切替を**非同期で**開始 (即座に返る。UI は status をポーリング) |
| POST | `/api/unload` | llama-server を停止し VRAM を解放 |
| GET | `/api/models/{id}/log?n=200` | そのモデルの `server.log` 末尾 |
| POST | `/api/chat` | SSE ストリーム。body: `{model, messages, temperature?, max_tokens?, enable_thinking, web_search}` |
| GET | `/api/history` | 保存済み会話の一覧 |
| GET | `/api/history/{id}` | 会話 1 件 (JSON) |
| PUT | `/api/history/{id}` | 会話を保存 (JSON + Markdown)。id は `^[0-9A-Za-z_-]{1,80}$` のみ許可 |

curl での操作例:

```bash
curl -s localhost:5070/api/status | jq
curl -s -X POST localhost:5070/api/models/qwen3.6-35b-a3b/load
while [[ $(curl -s localhost:5070/api/status | jq -r .status) == loading ]]; do sleep 1; done
curl -sN localhost:5070/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","enable_thinking":false,
       "messages":[{"role":"user","content":"こんにちは"}]}'
curl -s -X POST localhost:5070/api/unload
```

## 6. 主要な関数/メソッド (`server.py`)

### モデル検出

| 関数 | 説明 |
|---|---|
| `discover_models() -> dict[str, ModelInfo]` | `LLM_ROOT/*/model.toml` を走査し、`serve.sh` があるものを `ModelInfo(id, name, description, order, dir)` として `order` 順に返す |
| `ModelInfo.log_path` | `<モデルdir>/server.log` |
| `tail(path, n)` | ログ末尾 n 行 |
| `open_log(path)` | ログを追記モードで開き起動区切り行を書く (async 内でブロッキング open を避けるため `asyncio.to_thread` 経由で呼ぶ) |

### `ModelManager` — llama-server プロセスを 1 つだけ管理

| メソッド | 説明 |
|---|---|
| `snapshot()` | UI 向け状態 dict |
| `request_load(model_id)` | 状態を即 `loading` にして `_load` をバックグラウンドタスクで起動。同じモデルが loading/ready なら何もしない |
| `_load(model_id)` | `asyncio.Lock` 下で旧プロセス停止 → `bash serve.sh` を `HOST=127.0.0.1 PORT=5071` で起動 (`start_new_session=True` でプロセスグループ化、出力は `server.log`) → `_wait_healthy` で待機。途中で別モデルが選ばれたら中断 |
| `_wait_healthy(proc)` | `/health` が 200 になるまで 0.5 秒間隔でポーリング (最大 900 秒)。プロセス終了なら失敗 |
| `_stop_proc()` | `os.killpg(SIGTERM)` → 20 秒待って応答なければ `SIGKILL` |
| `unload()` | 停止して状態を `stopped` に |

アプリ終了時は `lifespan` で `manager.unload()` が呼ばれ、llama-server も確実に止まる。

### チャット中継

- `ChatRequest` (pydantic): `model, messages, temperature, max_tokens, enable_thinking=True, web_search=False`
- `chat(req)`:
  - モデルが `ready` でなければ 409。
  - 通常は `chat_template_kwargs: {"enable_thinking": ...}` を付けて `stream: true` で送る。
    Bonsai 2 では思考オン時に `reasoning_effort=medium`、オフ時に `enable_thinking=false` を送る。
  - `web_search=True` なら `agent.run(BACKEND_URL, payload)` の SSE を返す。
  - それ以外は `relay()` が llama-server の SSE をバイト列のまま中継する (エラーは `data: {"error": ...}` で返す)。

### 会話履歴

| 要素 | 説明 |
|---|---|
| `HistoryMessage` | `role, content, model?, reasoning?, tools?, sources?` |
| `Conversation` | `id, title, created, updated, system, messages[]` |
| `history_path(id, suffix)` | id を正規表現で検証してからパスを作る (パストラバーサル防止、不正なら 400) |
| `to_markdown(conv)` | 閲覧用 Markdown を生成 (思考は `<details>`、ツール実行は引用行、出典はリンク一覧) |
| `list_history()` / `get_history()` / `save_history()` | 一覧・取得・保存。ファイル I/O があるため同期 `def` (FastAPI がスレッドプールで実行) |

保存形式:

```
history/20260924-234410-uxg7.json   # 再読込用 (完全な記録)
history/20260924-234410-uxg7.md     # 閲覧用
```

```bash
# WSL 側からの読み出し例
jq -r '.messages[] | "\(.role): \(.content[:80])"' /workspace/LLM/llm-chat/history/*.json
```

## 7. 主要な関数 (`static/index.html`)

| 関数 | 説明 |
|---|---|
| `refresh()` | `/api/status` を取得しバッジ・ボタン状態を更新。loading 中は 1 秒、それ以外は 5 秒間隔で再実行 |
| `loadModel()` | 選択モデルの `/load` を POST |
| `send()` | 履歴を `{role, content}` だけに整形して `/api/chat` へ送り、SSE を行単位でパース。`reasoning_content` は折りたたみ、`content` は Markdown 描画、`timings` から tok/s 表示、`tool_event` / `sources` を検索表示に反映。`AbortController` で中止可能。完了後 `saveConv()` |
| `newConv()` | 会話 ID (`YYYYMMDD-HHMMSS-xxxx`) を発行し履歴をリセット |
| `saveConv()` | 「履歴を保存」オン時のみ会話全体を PUT。タイトルは最初のユーザー発言の先頭 40 文字 |
| `refreshHistory()` / `openConv(id)` | 過去の会話一覧の取得 / 会話を開いて続きから再開 |
| `renderAssistant(m)` | 保存済みアシスタント発言 (思考・ツール・出典込み) を描画 |
| `upsertTool(el, ev)` | ツール実行状況行 (🔍 検索 / 📄 閲覧) を追加・更新 |
| `renderSources(el, sources, content)` | 本文で引用された `[n]` の出典だけをリンク表示 (引用がなければ参照候補として全件) |
| `storePref(id, key)` | チェックボックス設定を localStorage に保存 (try/catch で保護) |

UI の設定 (履歴保存・ウェブ検索のオン/オフ) はブラウザの localStorage に記憶される。
会話内容は「履歴を保存」オフの間はどこにも書き出されない。
