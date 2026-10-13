# 11. Codex CLIとTernary Bonsai 2 27B Abliterated

`codex -p bonsai`で、導入済みのHikari07jp版PQ2_0重みを使う。通常の`codex`はGPT、`codex -p qwen`は従来のFlash Nextを使う。

## 起動と停止

```bash
cd /workspace/LLM
uv run codex/setup.py
# ターミナル1
./llm codex bonsai
# ターミナル2: 作業するディレクトリで
codex -p bonsai
# 非インタラクティブ
codex exec -p bonsai "指示"
```

起動ターミナルのCtrl+CでAPIとエンジンを停止し、GPU予約を解放する。起動前にはllm-chatへアンロードを要求し、`irodori-tts/gpu.lock`を予約する。GPU音声や他のCodexモデルが使用中なら起動を拒否する。

Prism版llama.cppの重み・mmprojとラッパーを再利用する。重みの再ダウンロードは不要。APIは`127.0.0.1:8081/v1`、バックエンドは`127.0.0.1:8082`。
`--port`と`--backend-port`で変更できる。APIのポートを変えたら`~/.codex/bonsai.config.toml`の`base_url`も揃える。

## Codexの設定とツール

設定元は`codex/bonsai.config.toml`、モデル情報は`codex/bonsai-models.json`、メモリ配置は`codex/bonsai-runtime.json`。
セットアップがインストール済みCodexの標準GPT指示を取得し、専用カタログへそのまま登録する。既定のGPT、Qwen、追加したMCPや信頼設定は保持する。

PrismのネイティブResponses APIにはCodexのnamespace/customツールへの変換が不足するため、`codex/llama_responses.py`でChat Completionsへ変換する。
StrataのResponses用パーサーとイベント組み立てを再利用し、シェルのfunctionツール、apply_patchのfreeformツール、MCPのnamespace、思考の再送、並列呼び出しの結果順序、画像URLを扱う。バックエンドはPrismで推論する。

推論レベルは`medium`。Bonsaiのテンプレートが受け付ける`none`、`low`、`medium`、`xhigh`をカタログに登録する。[重みの配布元](https://huggingface.co/Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF)も`medium`を推奨している。
`workspace-write`とsandbox内ネット接続を有効にする。Apps・multi-agentはローカルプロファイルで無効。

検索はQwenと同じ`local_web_search` MCPを使う。`web_search`で検索し、`fetch_page`で本文を読む。APIキーや別の検索サーバー起動は不要。
`web_search = "live"`は設定するが、OpenAIのホスト型検索は変換時に除外する。MCP経由のツールをモデルへ渡す。

## コンテキストとメモリ

Codex用コンテキストは98,304トークン（96K）、自動要約は73,728トークン。カタログ・Codex設定・バックエンドの三つを揃え、起動時に実際の`n_ctx`も検査する。
[モデルのネイティブ上限](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)は262,144トークンだが、このGPUでは速度とのバランスを測定して96Kを選んだ。学習済み上限を超えるRoPE拡張は行わない。チャットUIの既定16Kとは別に管理する。

RTX 5070 12GB / RAM約47GiBで、全65層の重み、q8_0のK/Vキャッシュ、再帰状態をGPUへ置く。
`--gpu-layers all --fit off --ubatch-size 64`を指定し、Flash Attentionを有効にする。mmprojはCPU。GPUのKVは3,264MiB、再帰状態は約150MiB、重みは約6,540MiB、計算用領域は約408MiB。
128から64へ物理バッチを縮めることで計算用領域を減らし、96Kでも生成速度を維持できた。
起動時にGPUの空きへコンテキストを自動縮小する設定にはしない。`--context`で小さくする場合は、Codex設定・カタログの上限も変更する。

## 入力・思考・回答の全文記録

Qwenの`TraceStore`とHTTP記録ラッパーを再利用する。保存形式、所有者のみ読み書きできる権限、文字数制限なし、過去ログの保持は同じ。

```bash
bash /workspace/LLM/ternary-bonsai-2-27b-abliterated/watch-codex.sh
```

```text
ternary-bonsai-2-27b-abliterated/codex-traces/
  live.log
  latest/ -> <UTC日時>-<PID>/
  <UTC日時>-<PID>/<リクエストID>/
    request.body
    request.json
    reasoning.txt
    output.txt
    tools.jsonl
    events.jsonl
    response.http
    response.json      # 非ストリーミング・エラー応答
    metadata.json
```

`CODEX_TRACE_DIR`で保存先を指定できる。ツール実行結果はCodexが次のリクエストへ送った範囲で保存する。APIが返す思考テキストを保存し、出力されない内部計算は含まない。
`metadata.json`に使用トークン・バックエンドの`timings`を記録する。ターミナル出力は`server-codex.log`、エンジン出力は`codex-engine.log`。
保存先はGitの対象外。BonsaiにはStrataのブラウザー監視画面を提供しない。記録の閲覧は上記コマンドまたはファイルを使う。

## 速度測定と実モデルでの検証

2026-10-11に、実モデルへ同じ反復英文とPythonの説明を依頼し、入力5,435トークン / 24,335トークン、出力各256トークンで比較した。推論はOFF、temperature=0、入力キャッシュの再利用は0。表の速度は生成だけの値。

| 設定（重み・KVともGPU） | 物理バッチ | 約5K入力時 | 約24K入力時 | GPU使用量の最大値 |
| --- | --- | --- | --- | --- |
| 32K / q8_0 | 128 | 54.46トークン/秒 | 45.72トークン/秒 | 9,353MiB |
| 64K / q8_0 | 128 | 54.55トークン/秒 | 46.69トークン/秒 | 10,744MiB |
| 80K / q8_0 | 128 | 54.21トークン/秒 | 45.67トークン/秒 | 11,338MiB |
| **96K / q8_0（採用）** | **64** | **55.50トークン/秒** | **45.54トークン/秒** | **11,374MiB** |
| 96K / q8_0 | 128 | 46.90トークン/秒 | 40.54トークン/秒 | 11,712MiB |
| 104K / q8_0 | 64 | 44.53トークン/秒 | 38.83トークン/秒 | 11,827MiB |
| 112K / q8_0 | 64 | 15.15トークン/秒 | 未測定 | 11,880MiB |
| 96K / q4_0 | 128 | 45.29トークン/秒 | 26.61トークン/秒 | 10,516MiB |
| 128K / q4_0 | 128 | 45.31トークン/秒 | 26.87トークン/秒 | 11,278MiB |

96Kの採用設定の入力処理は、約5Kで954.3トークン/秒、約24Kで896.4トークン/秒。112Kは生成速度の低下を確認して24K側の処理を中断した。
96KのK=q8_0 / V=q4_0も試したが、約5K入力で生成14.38トークン/秒、入力処理143.4トークン/秒に落ち、採用しなかった。

262Kを確保し、重みをGPU・q8_0 KVをRAMへ置く構成も実際に起動・生成した。入力5,421 / 出力60トークンの別の英文タスクで入力539.1トークン/秒、生成8.74トークン/秒だった。KVと再帰状態がCPUへ移る構成は速度が低いため採用しなかった。
104K以上で実測速度が下がるため96Kを選び、VRAMの余裕を作る目的で64Kへ制限することはしない。

生成時間はバックエンドの`predicted_ms`に基づき、入力処理・検索・ツール実行の時間を含まない。OS全体のGPU使用量を0.5秒間隔で測定した。測定条件・入力・出力の長さで速度は変わり、GPU使用量にもデスクトップ等の使用量を含む。合成入力での小規模測定であり、モデル全般の速度や長文理解の品質を保証するものではない。
比較の記録は`/tmp/bonsai-sweet-spot/`、`/tmp/bonsai-sweet-spot-mixed/`、`/tmp/bonsai-sweet-spot-boundary/`。主要な測定値は`codex/bonsai-benchmarks.json`にも保存する。

採用設定でCodex CLI 0.159.2を実際に動かし、標準GPT指示と`medium`推論で次を確認した。

- apply_patchで`squares.py`を作成し、シェル実行の結果が385であることを確認。
- 検索MCPでPythonの公式ドキュメントを検索し、公式ページ本文を取得。
- ツール結果をモデルへ返し、日本語の最終報告を受信。
- 入力・思考・回答・custom/function/MCP呼び出し・HTTP応答・使用トークン・速度をcodex-tracesへ保存。

Codexの3リクエストの生成速度は49.91〜52.25トークン/秒。作業全体は約48秒で、検索・シェル実行・入力処理を含む。
同じ96K設定のResponses APIへ90,074トークンを実際に入力し、先頭・中間・末尾の指定値をすべて正しく返した。
入力処理は675.6トークン/秒、生成は28.86トークン/秒（256トークン生成）、リクエスト全体は約142.8秒。
90Kを保持すると短い入力より生成速度が下がる。普段のCodexは73,728トークンで自動要約するため、この直接APIの長文試験より早く要約が入る。

検証中のGPU使用量の最大値は11,770MiB、OSの利用可能RAMの最小値は約40.2GiB。記録は`/tmp/codex-bonsai-96k/report.json`、`codex.jsonl`、`answer.txt`、モデルディレクトリの`codex-traces/20261011T125615.799552Z-52980/`に保存した。
終了後は8081・8082ポートの解放、エンジンの残存なし、GPUロックの解放を確認した。停止後のGPU使用量は939MiB。

設定の保持・標準指示・コンテキスト一致のテスト3件、Responses変換・HTTPと全文記録・即時再起動のテスト7件、検索MCPのテスト4件が通過した。

```bash
python3 -m unittest discover -s codex -p test_setup.py -v
strata/.venv/bin/python -m unittest discover -s codex -p test_llama_responses.py -v
codex/search-mcp/.venv/bin/python -m unittest discover -s codex/search-mcp -p test_server.py -v
```
