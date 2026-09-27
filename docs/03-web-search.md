# 03. ウェブ検索 (検索拡張生成)

チャット UI の「🌐 ウェブ検索」をオンにすると、モデルが必要に応じて DuckDuckGo で検索し、
ページ本文を読んで**出典番号付き**で回答する。検索結果を回答生成に使う RAG の一形態
(ローカル文書のベクトル検索 RAG は未実装、[README](README.md#今後の拡張候補) 参照)。

> 検索語と閲覧 URL は外部に送信されるため、既定はオフ。

## 1. 仕組み (ツール呼び出しループ)

```
ユーザーの質問
  → llama-server に tools=[web_search, fetch_page] 付きで送信 (stream)
  → モデルが tool_calls を返す
  → サーバーでツール実行 → 結果を role:"tool" で messages に追加
  → 再送信 … (最大 LLM_SEARCH_MAX_ROUNDS ラウンド、最終ラウンドは tools なし)
  → 出典 [n] 付きの最終回答
```

- モデル側の対応: Qwen3.6 / Gemma 4 / Qwen3.5 はツール呼び出し対応。llama-server は `--jinja` 必須。
- 実測: Qwen3.6-35B-A3B「llama.cpp の最新リリース」→ 検索 2 + 取得 2 で正答 (137 秒)、
  Gemma 4「今日の東京の天気」→ 検索 1 で回答 (42 秒)。

## 2. 依存パッケージ

```bash
cd /workspace/LLM/llm-chat
uv add ddgs trafilatura
```

- `ddgs` — DuckDuckGo 検索 (API キー不要)
- `trafilatura` — HTML から本文抽出

動作確認:

```bash
uv run python -c "from ddgs import DDGS; print(DDGS().text('RTX 5070', region='jp-jp', max_results=3))"
```

## 3. ファイル構成と主要な関数/クラス

```
src/llm_chat/
├── agent.py              エージェントループ
└── search/
    ├── __init__.py       公開 API: SearchProvider, SearchResult, fetch_page, get_provider
    ├── base.py           SearchResult / SearchProvider プロトコル
    ├── providers.py      DuckDuckGoProvider, PROVIDERS 登録表, get_provider()
    └── fetch.py          fetch_page()
```

### `search/base.py`

| 要素 | 説明 |
|---|---|
| `SearchResult(title, url, snippet)` | 検索結果 1 件 (dataclass)。`to_dict()` あり |
| `SearchProvider` (Protocol) | `name: str` と `async search(query, max_results) -> list[SearchResult]` を持つもの |

### `search/providers.py`

| 要素 | 説明 |
|---|---|
| `DuckDuckGoProvider` | `ddgs.DDGS().text(query, region, max_results)` を `asyncio.to_thread` で実行 (ddgs は同期ライブラリ)。地域は `LLM_SEARCH_REGION` (既定 `jp-jp`) |
| `PROVIDERS` | `名前 → ファクトリ` の辞書。**新しい検索エンジンはここに登録する** |
| `get_provider(name=None)` | `LLM_SEARCH_PROVIDER` (既定 `duckduckgo`) のプロバイダを生成。未登録名は `ValueError` |

### `search/fetch.py`

| 関数 | 説明 |
|---|---|
| `fetch_page(url, max_chars)` | http(s) のみ許可。httpx でブラウザ UA を付けて取得 → trafilatura で本文抽出 (スレッド実行) → `max_chars` で切り詰め。本文が取れなければ `ValueError` |

### `agent.py`

| 要素 | 説明 |
|---|---|
| `TOOLS` | OpenAI 形式のツール定義: `web_search(query)`, `fetch_page(url)` |
| `system_prompt()` | 現在日時 (JST) と、検索すべき場面・`[n]` 形式の引用・推測で断定しないことを指示。ユーザーのシステムプロンプトがあれば前置きで結合 |
| `SourceRegistry.number(url, title)` | 1 回答内で URL に通し番号を振る (同じ URL は同じ番号)。検索結果と取得ページで番号を共有 |
| `execute_tool(name, args, sources)` | ツールを実行し `(モデルに返すテキスト, UI 用出典リスト)` を返す。検索結果は `[n] タイトル / URL / 概要` 形式 |
| `run(backend_url, payload)` | エージェントループ本体 (async generator、SSE バイト列を yield) |
| `sse(obj)` | dict を `data: {...}\n\n` に変換 |

`run()` の処理:

1. 各ラウンドで `tools` と `thinking_budget_tokens` を付けて `/v1/chat/completions` をストリーミング。
2. チャンクの `delta.tool_calls` を `index` ごとに連結 (id / name / arguments は分割されて届く)。
3. `content` / `reasoning_content` / `timings` を含むチャンクはそのままクライアントに中継。
4. tool_calls が無ければ `{"sources": [...]}` と `[DONE]` を送って終了。
5. あれば assistant メッセージ (`tool_calls` と `reasoning_content` 付き) を積み、各ツールを実行。
   前後で `{"tool_event": {id, name, args, status: running|done|error, sources?, error?}}` を送る。
   ツールの失敗は `エラー: ...` としてモデルに返し、ループは継続。

## 4. 設定 (環境変数)

| 変数 | 既定 | 内容 |
|---|---|---|
| `LLM_SEARCH_PROVIDER` | `duckduckgo` | 検索プロバイダ |
| `LLM_SEARCH_REGION` | `jp-jp` | DuckDuckGo の地域 |
| `LLM_SEARCH_MAX_RESULTS` | `5` | 1 回の検索件数 |
| `LLM_SEARCH_MAX_ROUNDS` | `6` | ツール呼び出しの最大ラウンド |
| `LLM_FETCH_MAX_CHARS` | `5000` | ページ本文の最大文字数 (ctx 32k を圧迫しないため) |
| `LLM_SEARCH_THINKING_BUDGET` | `2048` | 1 ラウンドの思考トークン上限 |

例: `LLM_SEARCH_MAX_RESULTS=8 uv run llm-chat`

## 5. API プロバイダへの移行

UI やエージェントは `get_provider()` 経由でしか検索しないため、`providers.py` にクラスを追加して登録するだけでよい。

```python
# search/providers.py
import httpx

class TavilyProvider:
    name = "tavily"

    def __init__(self) -> None:
        self.key = os.environ["TAVILY_API_KEY"]

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.key, "query": query, "max_results": max_results},
            )
            r.raise_for_status()
        return [
            SearchResult(title=x["title"], url=x["url"], snippet=x.get("content", ""))
            for x in r.json()["results"]
        ]

PROVIDERS["tavily"] = TavilyProvider
```

```bash
TAVILY_API_KEY=... LLM_SEARCH_PROVIDER=tavily uv run llm-chat
```

(上記は設計例で未検証。実際の API 仕様は各サービスのドキュメントで確認すること)

## 6. 思考ループ問題と対策

テスト中、Qwen3.6 で次の問題が発生した。

| 回 | 現象 | 原因 |
|---|---|---|
| 1 回目 | 検索・取得は成功したが、最終回答前の思考が止まらない (思考 3.8 万文字) | GitHub のリリースページから抽出した本文に日付が無く、整合を取ろうと延々考え続けた |
| 2 回目 | 思考上限 4096 は効いたが、上限まで「b11160 のコミットハッシュを探す。」を反復 | サンプリング設定 (temp 0.6、presence_penalty なし) で反復に陥りやすかった |

対策:

1. **Qwen のサンプリングを公式推奨値に変更** (`serve.sh`): `--temp 1.0 --presence-penalty 1.5`
   (通常チャットにも適用される)
2. **ラウンドごとの思考上限**: リクエストに `thinking_budget_tokens: 2048` を付与 (`agent.py` の `THINKING_BUDGET`)
3. **ラウンド上限**: `MAX_ROUNDS` を超えたら tools を外して回答を強制

対策後は各ラウンドの思考が 200〜450 トークンに収まり、ループは再発していない。

調査に使ったコマンド (SSE を保存してラウンド別に集計):

```bash
curl -sN localhost:5070/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","enable_thinking":true,"web_search":true,
       "messages":[{"role":"user","content":"llama.cppの最新リリースを調べて"}]}' > web.sse
# ツール実行の一覧
grep '^data: {"tool_event' web.sse | sed 's/^data: //' | jq -c '.tool_event | {name,args,status}'
# 最終回答
grep '^data: {' web.sse | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty'
```

> SSE の JSON は `{"tool_event": {` のように**コロンの後に空白がある**ため、
> `grep '"tool_event":{'` では一致しない (調査時にこれで誤判定した)。jq で扱うのが確実。
