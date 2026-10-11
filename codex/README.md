# Codex with local Flash Next / Qwen3.5-9B / Bonsai 2

```bash
cd /workspace/LLM
uv run codex/setup.py
bash codex/serve.sh flash-next
# 別のターミナルから、作業ディレクトリで:
codex -p flash-next
```

モデルごとにサーバーとCodexの起動名を揃える。

| モデル | サーバー起動 | Codex起動 | API |
| --- | --- | --- | --- |
| Flash Next | `bash codex/serve.sh flash-next` | `codex -p flash-next` | `127.0.0.1:8080/v1` |
| Qwen3.5-9B | `bash codex/serve.sh qwen-9b` | `codex -p qwen-9b` | `127.0.0.1:8083/v1` |
| Bonsai 2 | `bash codex/serve.sh bonsai` | `codex -p bonsai` | `127.0.0.1:8081/v1` |

旧`codex -p qwen`はFlash Nextの互換名として残す。既存のプロジェクト信頼設定と手動追加MCPは、新しい`flash-next`プロファイルへの初回移行時に引き継ぐ。
GPUを使うモデルは一度に一つ。切り替える前にサーバーをCtrl+Cで停止する。

Qwen3.8-Flash-Next SC117 IQ3_XXSをStrataのResponses APIで利用する。
通常の`codex`は従来のGPTを使い、`codex -p flash-next`のときだけQwenの専用設定・カタログを読み込む。
セットアップは以前のグローバルQwen設定をGPTへ戻し、Qwenを別プロファイルとして保存する。
sandboxはworkspace-write、インターネット接続は既定で有効。
Qwenのコンテキストは128K、約98Kで自動要約する。チャット用の32K設定とは別に管理する。

サーバー起動後、`http://127.0.0.1:8080/api-monitor`で入力・思考・応答をリアルタイムに確認できる。
全文をターミナルで追う場合は、別のターミナルで次を実行する。

```bash
bash /workspace/LLM/qwen3.8-flash-next/watch-codex.sh
```

入力・思考・回答・ツール呼び出し・HTTP応答の全文は`qwen3.8-flash-next/codex-traces/`に自動保存する。
`latest/`は最新のサーバー起動分。監視画面は直近100件・各欄262144文字までだが、保存ファイルは省略しない。
ツール実行結果はCodexが次のリクエストに送った範囲で保存する。モデルが出力する思考テキストを記録し、内部テンソルや出力されない計算過程は含まない。

[設定・GPU運用・検証結果](../docs/04-codex.md)を参照。

基本指示はセットアップ時にインストール済みCodexの標準GPT指示を取得する。Qwen用に短縮しない。
Qwenのネット検索は専用の`local_web_search` MCPで行う。セットアップが依存環境を用意し、Codex起動時に自動接続する。APIキーは不要。
`web_search`で検索し、`fetch_page`でページ本文を読む。使い方と出典URLの扱いはMCPのツール説明からモデルに渡される。
`web_search = "live"`は維持するが、現行Strataが除外する標準のホスト型検索とは別の実行経路。

## Bonsai 2 27B Abliterated

同じセットアップで`bonsai`プロファイルも導入する。

```bash
cd /workspace/LLM
uv run codex/setup.py
bash ternary-bonsai-2-27b-abliterated/serve-codex.sh
# 別のターミナルで、作業ディレクトリから:
codex -p bonsai
```

既存のPQ2_0重みをPrism版llama.cppでロードし、Responses変換を介してCodexへ接続する。
標準GPT指示、ファイル編集・シェル実行、検索MCP、思考の表示と全文記録を使用する。コンテキストは98,304トークン、自動要約は73,728トークン。推論レベルの既定は`medium`。
重みとq8_0のKVキャッシュをGPUに置く。起動前にllm-chatをアンロードし、共有GPUロックを予約する。QwenとBonsaiは一台のGPUで切り替えて使う。Ctrl+Cでサーバーとエンジンを停止する。

APIは`127.0.0.1:8081/v1`、バックエンドは`127.0.0.1:8082`。記録は`ternary-bonsai-2-27b-abliterated/codex-traces/`にQwenと同じファイル形式で保存する。

```bash
bash /workspace/LLM/ternary-bonsai-2-27b-abliterated/watch-codex.sh
```

`latest/`は最新の起動分、`live.log`は生成中から追える時系列ログ。`CODEX_TRACE_DIR`で保存先を変更できる。サーバーログは`server-codex.log`、エンジンログは`codex-engine.log`。
QwenのStrata監視画面はQwen専用。Bonsaiの全文確認には保存ファイルと上記コマンドを使う。
設定・メモリ配置・速度の実測は[11-codex-bonsai.md](../docs/11-codex-bonsai.md)を参照。

## Qwen3.5-9B

```bash
bash codex/serve.sh qwen-9b
# 別のターミナルで、作業ディレクトリから:
codex -p qwen-9b
# 全文の生成ログを追う:
bash /workspace/LLM/qwen3.5-9b/watch-codex.sh
```

既存の`Qwen3.5-9B-UD-Q5_K_XL.gguf`と画像エンコーダーを標準llama.cppでロードする。
Bonsaiと共通のResponses変換を使い、標準GPT指示・編集・シェル・検索MCP・思考表示・全文記録に対応する。
コンテキストは131,072トークン、自動要約は98,304トークン。GPU上のq8_0 KVキャッシュを使う。
`medium`はthinking有効、`none`は無効。細かな推論強度は公開しない。
記録は`qwen3.5-9b/codex-traces/`、バックエンドは`127.0.0.1:8084`。
チャット側のコンテキスト設定とは独立している。
実測と選定結果は[12-codex-qwen-9b.md](../docs/12-codex-qwen-9b.md)を参照。
