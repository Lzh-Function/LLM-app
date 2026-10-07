# 05. 運用・ログ・トラブルシューティング

## 1. プロセス管理

| 用途 | ポート | 起動 | 停止 |
|---|---|---|---|
| チャット UI | 5070 | `cd /workspace/LLM/llm-chat && uv run llm-chat` | Ctrl+C / `pkill -x llm-chat` |
| チャット UI 内部の llama-server | 5071 | UI から自動 | UI の「停止」/ `curl -X POST localhost:5070/api/unload` |
| AivisSpeech / VOICEVOX | 10101 / 50021 | 起動時OFF。UIの「読み上げ」をONにすると起動（導入済み・ローカル接続先の場合） | 「読み上げ」をOFF、またはチャット終了で停止。手動起動したものはそのターミナルでCtrl+C |
| Codex 用 Strata | 8080 | `bash qwen3.8-flash-next/serve-codex.sh` | 起動ターミナルでCtrl+C ([04 参照](04-codex.md#3-gpuとサーバー)) |

バックグラウンドで起動する場合:

```bash
cd /workspace/LLM/llm-chat && (nohup uv run llm-chat > chat.log 2>&1 &)
```

特定ポートの llama-server だけ止める:

```bash
fuser -k 8080/tcp          # ポートを使っているプロセスを kill (最も簡単)

# fuser が無い場合
for p in $(pgrep llama-server); do
  tr '\0' ' ' < /proc/$p/cmdline | grep -q -- "--port 8080" && kill $p
done
```

> **注意: `pkill -f <パターン>` は使わない。** 実行中のシェル自身のコマンドラインにも同じ文字列が含まれるため、
> シェルごと終了してしまう (exit code 144 になった)。プロセス名の完全一致 `pkill -x llm-chat` か、上記の PID 指定を使う。

## 2. ログの場所

| ファイル | 内容 |
|---|---|
| `/workspace/LLM/<モデル>/server.log` | チャット UI から起動した llama-server のログ (起動ごとに追記、ローテーションなし)。UI の「ログ」ボタンでも閲覧可 |
| `/workspace/LLM/qwen3.8-flash-next/server-codex.log` | Codex テスト時に起動したサーバーのログ |
| `/workspace/LLM/llm-chat/chat.log` | nohup 起動時の UI アクセスログ (URL とステータスのみ) |
| `/workspace/LLM/aivisspeech/server.log`、`/workspace/LLM/voicevox/server.log` | 音声エンジン自動起動時のログ。合成するテキストが記録される場合がある |
| `/workspace/LLM/llm-chat/history/` | 「履歴を保存」オン時の会話 (JSON + Markdown) |

- 会話内容は llama-server のログには出ない (既定のログレベル)。
- 「履歴を保存」オフのときは、会話はブラウザのメモリにしか存在しない (再読み込みで消える)。

ログの肥大化が気になったら削除してよい (次回起動時に再作成される):

```bash
: > /workspace/LLM/qwen3.6-35b-a3b/server.log
```

## 3. 確認コマンド集

```bash
nvidia-smi                                            # VRAM 使用量
nvidia-smi --query-gpu=memory.used --format=csv,noheader
free -h                                               # RAM (available を見る)
ss -ltnp | grep -E "5070|5071|8080"                   # 待受ポート
curl -s localhost:5070/api/status | jq                # UI の状態
curl -s localhost:8080/health                         # Codex用Strataの生存確認
/workspace/LLM/llama.cpp/llama-server --list-devices  # GPU 認識
```

## 4. メモリ (WSL2)

`free -h` で `free` が小さく `buff/cache` が大きいのは、GGUF ファイルがページキャッシュに載っているため。
`available` が実際に使える量で、キャッシュは必要に応じて解放される。モデル切替が速いのもこのキャッシュのおかげ。

Windows 側で VmmemWSL のメモリ使用が気になる場合:

```bash
sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'   # その場で解放 (次回のモデル読込が遅くなる)
```

または Windows の `%UserProfile%\.wslconfig`:

```ini
[experimental]
autoMemoryReclaim=gradual
```

(`wsl --shutdown` 後に反映)

## 5. 検閲・外部通信について

- 推論はすべてローカルで完結し、モデレーション API や入出力フィルタは介在しない。
- ただし Qwen / Gemma は学習時に安全性調整を受けているため、内容によってはモデル自身が回答を拒否する。
  拒否を減らしたい場合は abliterated / uncensored 版 GGUF に `serve.sh` のモデルパスを差し替える (性能低下の可能性あり)。
- 外部通信が発生するのは: Hugging Face (ダウンロード時)、cdnjs (UI の marked / DOMPurify をブラウザが取得)、
  ウェブ検索オン時の DuckDuckGo と閲覧先サイト。

## 6. トラブルシューティング

| 症状 | 原因と対処 |
|---|---|
| UI で「起動に失敗しました」 | 「ログ」ボタンで `server.log` を確認。VRAM 不足なら他の GPU プロセス (Codex 用 8080 サーバー等) を停止 |
| モデル切替後に応答しない | `curl localhost:5070/api/status` で status を確認。`loading` が続く場合はログ参照 |
| ウェブ検索で延々と終わらない | 思考ループの可能性。[03 の 6 章](03-web-search.md#6-思考ループ問題と対策) の設定 (`presence_penalty`、`LLM_SEARCH_THINKING_BUDGET`) を確認。UI の「中止」で止められる |
| Codex が極端に遅い | KVと専門家キャッシュの配分、RAM・swapを確認 ([04 の 4 章](04-codex.md#4-コンテキストとメモリ)) |
| コンテキスト変更後にモデルが起動しない | 「メモリを解放中」の完了後、値を下げて再度「切り替え」、または別モデルを選ぶ。画面の「詳細ログ」でOOM・終了コードを確認。停止確認エラーの場合は「停止」で再試行 |
| `LLM/` 移動後に `uv run` が失敗 | `.venv` 内の絶対パスが古い。`rm -rf .venv && uv sync` |
| VS Code (Pylance) が `import httpx` を解決できない | インタプリタが旧 venv を指している。「Python: Select Interpreter」で `/workspace/LLM/llm-chat/.venv` を選択 |
| ページ本文が取れない | ボット対策のあるサイト。`fetch_page` がエラーを返し、モデルは他の情報源で回答を試みる |
