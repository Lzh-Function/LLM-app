# 12. Codex CLIとQwen3.5-9B

Qwen3.5-9Bは`qwen-9b`、Qwen3.8-Flash-Nextは`flash-next`で指定する。
旧`qwen`はFlash Nextの互換名。通常の`codex`は従来のGPTを使う。

## 起動

```bash
cd /workspace/LLM
uv run codex/setup.py
./llm codex qwen-9b
# 別のターミナルで、作業ディレクトリから:
codex -p qwen-9b
```

Flash Nextへの切り替えは、サーバーをCtrl+Cで停止してから次を実行する。

```bash
./llm codex flash-next
# 別のターミナルで:
codex -p flash-next
```

モデルのディレクトリにある`serve-codex.sh`は`./llm codex`から呼ぶ内部スクリプト。
セットアップはモデルごとの専用カタログとプロファイルを作成し、GPTの既定設定を維持する。
旧`qwen`のプロジェクト信頼設定・手動追加MCPは、`flash-next`への初回移行時に引き継ぐ。

## 9Bの設定

| 項目 | 設定 |
| --- | --- |
| 重み | `qwen3.5-9b/models/Qwen3.5-9B-UD-Q5_K_XL.gguf` |
| エンジン | 標準llama.cpp b11160、CUDA 12.8 |
| 画像エンコーダー | `models/mmproj-F16.gguf`、CPU配置 |
| コンテキスト | 131,072トークン（128K） |
| 自動要約 | 98,304トークン |
| 重み・KV | 全層GPU、K/Vともq8_0 |
| microbatch | 64 |
| Flash Attention | 有効 |
| fit | off、自動的なレイヤー・コンテキスト縮小を使わない |
| 並列スロット | 1 |
| thinking | `medium`で有効、`none`で無効 |
| Responses API | `127.0.0.1:8083/v1` |
| llama.cppバックエンド | `127.0.0.1:8084` |

[公式モデルカード](https://huggingface.co/Qwen/Qwen3.5-9B)のネイティブコンテキストは262,144トークン。
最大長から実測したが、224Kでマシンの操作が著しく重くなったというユーザー報告を受け、128Kを既定にした。
大きい設定のスループットが良くても、このマシンの操作応答を優先する。
チャット用の`model.toml`・`serve.sh`の既定長は独立している。

設定の元は`codex/qwen-9b.config.toml`、`qwen-9b-models.json`、`qwen-9b-runtime.json`。
サーバー・Codex設定・カタログの上限を一致させる。セットアップ後のプロファイルは`~/.codex/qwen-9b.config.toml`。

## ツール・全文記録

基本指示は、セットアップ時にインストール済みCodexの標準GPTモデルから取得した`base_instructions`と`model_messages`をそのまま使う。
編集・シェルは通常のCodexツール、インターネット検索・ページ本文取得は`local_web_search` MCPを使う。
標準のホスト型検索はローカルエンジンに渡さず、MCP経由で実行する。

QwenとBonsaiで`codex/llama_server.py`・`llama_responses.py`を共有する。
llama.cppのChat CompletionsとCodexのResponses API間で、思考・ツール定義・freeform編集・ツール実行結果を変換する。
Qwen3.5はthinkingのON/OFFを公開する。細かなlow/high/xhighの推論強度は公開しない。
過去ターンの思考保持は、このモデルのllama.cppテンプレートが非対応のため適用されない。

起動前にllm-chatをアンロードし、音声・チャット・他のCodexモデルと共有するGPUロックを取得する。
Ctrl+CでAPIとエンジンを停止し、ロックを解放する。外部へ公開せずlocalhostだけで待ち受ける。

```bash
bash /workspace/LLM/qwen3.5-9b/watch-codex.sh
```

全文は`qwen3.5-9b/codex-traces/`にFlash Nextと同じ形式で保存する。
`latest/`が最新の起動分、`live.log`は生成中から追える時系列ログ。
各リクエストの入力、思考、回答、ツール呼び出し、HTTP応答、生成時間を省略せず記録する。
`CODEX_TRACE_DIR`で保存先を変更できる。サーバーログは`server-codex.log`、エンジンログは`codex-engine.log`。
ツール実行結果は、Codexが次のAPIリクエストに送った範囲で記録する。

## コンテキスト比較

2026-10-11、RTX 5070（12,227MiB、driver 591.86）で実モデルをロードして測定した。
同一の約5.4K・約24K入力、temperature=0、thinking無効、出力上限256トークン、プロンプトキャッシュ無効。
出力は213〜256トークンで、表の速度はllama.cppの生成区間の実測値。

| コンテキスト | K / V | 約5.4K入力での生成 | 約24K入力での生成 | GPU使用量ピーク |
| --- | --- | ---: | ---: | ---: |
| 256K | q8 / q8 | 停滞して中止 | 未実施 | 11,911MiB |
| 240K | q8 / q8 | 停滞して中止 | 未実施 | 11,772MiB |
| 224K | q8 / q8 | 66.2 tok/s | 62.4 tok/s | 11,496MiB |
| 192K | q8 / q8 | 68.2 tok/s | 62.6 tok/s | 10,829MiB |
| **128K（採用）** | **q8 / q8** | **68.6 tok/s** | **64.5 tok/s** | **9,491MiB** |
| 64K | q8 / q8 | 66.8 tok/s | 63.0 tok/s | 8,241MiB |
| 256K | q4 / q4 | 62.7 tok/s | 48.0 tok/s | 10,115MiB |
| 256K | q8 / q4 | 70.2 tok/s | 58.8 tok/s | 11,129MiB |

256K・240Kのq8/q8はロード自体は通ったが、最初の生成が停滞したためSIGINTで終了した。
224Kは速度測定が完了したものの、ユーザーが深刻な操作応答の悪化を確認したため採用しない。
128Kから64Kへ下げても速度は改善しなかった。128Kでq8/q8を使う設定を採用する。
この比較はスループットとメモリの測定で、量子化ごとの一般的な回答品質を保証する評価ではない。

機械可読の結果は`codex/qwen-9b-benchmarks.json`。生のログと測定値は`/tmp/qwen9-sweet-spot/`、`/tmp/qwen9-boundary/`、`/tmp/qwen9-mixed/`に保存した。

## 実Codex・長文での確認

128K設定で実Codex CLIを起動し、標準GPT指示を使って次を確認した。

- freeformの`apply_patch`で`squares.py`を作成。
- シェルで実行し、1〜10の二乗和が385であることを確認。
- モデル自身が検索MCPを呼び、Python公式ドキュメントを検索して`fetch_page`で本文を取得。
- 思考、編集・シェル・検索の呼び出し、返されたツール結果、HTTP応答全文をトレースで確認。

作業全体は30.0秒、生成速度は71.2〜72.9 tok/s。
約9.2K〜13.3K入力（標準指示・ツール定義・実行結果を含む）で、9リクエストが成功した。

さらに122,954入力トークンをResponses API経由で送信し、先頭・中央・末尾の3つの異なるキーを正しく取得した。
入力処理は1,694.5 tok/s、生成は48.2 tok/s、入力処理から256トークンの出力まで約78.7秒。
この長文テストは出力上限256のため`incomplete`で終了したが、3キーの取得を検証済み。
128K割当のまま長文入力も通り、GPUピークは9,393MiB、空きRAMは最低39.6GiBだった。

結果とCodex JSONLは`/tmp/codex-qwen-9b/`、既定の全文記録は
`qwen3.5-9b/codex-traces/20261011T131733.542671Z-68911/`。
検証後はAPI・バックエンドの8083/8084を停止し、エンジン終了とGPUロック解放を確認した。

新しい`./llm codex flash-next`と`codex -p flash-next`も実モデルで確認し、検索1回・ページ本文取得2回が成功した。
初回の検索は結果なしで失敗したが、モデル自身が別のクエリで再検索して成功した。
結果は`/tmp/codex-flash-next-renamed/report.json`。
終了後は8080も閉じ、Strataプロセス終了・GPUロック解放を確認した。
セットアップの既存設定保持・移行4件、Responses/HTTPトレース7件、全文トレース6件のテストが通過した。
