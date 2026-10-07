# 04. Codex CLIとQwen3.8 Flash Next / Strata

Codexの既定モデルは従来のGPT（この環境では`gpt-6.1-sol`）。
`codex -p qwen`を指定したときだけ、導入済みのSC117版Qwen3.8-Flash-Next GSQ-RCO abliterated IQ3_XXSを利用する。
Strataの`/v1/responses`を使い、ファイル編集・シェル実行・ツール結果を返して続けるエージェント操作に対応する。
Codex CLI 0.159.2、Strataエンジン0.1.40で確認した。モデル重みの導入手順は[07-strata.md](07-strata.md)。

## 1. 設定と起動

この環境の`/home/vscode/.codex/qwen.config.toml`は設定済み。
通常のGPTは`codex`で起動する。Qwenの導入・再設定と利用は次のとおり。

```bash
cd /workspace/LLM
uv run codex/setup.py
# ターミナル1: Codex用のローカルサーバー
bash qwen3.8-flash-next/serve-codex.sh
# ターミナル2: 作業ディレクトリでQwenを指定
cd /path/to/project
codex -p qwen
# 非インタラクティブ
codex exec -p qwen "指示"
```

旧Qwen3.6 / llama.cpp用の`qwen-local.config.toml`は削除した。新しいプロファイル名は`qwen`。
セットアップはQwenのモデル・プロバイダー・専用カタログ・コンテキストを別ファイルに保存する。
以前のグローバルQwen設定が残っていれば、導入前のバックアップからGPT設定を復元し、プロジェクトの信頼設定や追加したMCP等を保持する。
再実行しても既定のGPT設定を変更しない。
以前の設定は`~/.codex/config-backups/flash-next-<日時>/`へ退避する。モデル重みやチャットUIのモデル一覧は削除しない。

設定の元は`codex/qwen.config.toml`、専用モデル情報は`codex/models.json`、基本指示は`codex/instructions.md`。
セットアップは`CODEX_HOME`があればその場所へ、なければ`~/.codex`へ書き込む。
`flash-next-models.json`にはFlash Next一モデルだけを登録する。CodexがOpenAIモデル用の大きなフォールバック定義を使うことを避け、実際のコンテキスト・推論レベル・ツール形式を指定する。

Qwenプロファイルの内容は次の設定。Qwenはログイン・APIキーを要求しない。GPTは従来の認証を使う。

```toml
model = "qwen3.8-flash-next-sc117-abliterated-iq3_xxs"
model_provider = "strata"
model_catalog_json = "/home/vscode/.codex/flash-next-models.json"
model_context_window = 131072
model_auto_compact_token_limit = 98304
model_reasoning_effort = "high"
show_raw_agent_reasoning = true
sandbox_mode = "workspace-write"
web_search = "disabled"

[sandbox_workspace_write]
network_access = true

[model_providers.strata]
name = "Strata (local Flash Next)"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
stream_idle_timeout_ms = 600000
```

カスタムプロバイダー・`responses`・ネットワーク設定は[OpenAI公式設定リファレンス](https://learn.chatgpt.com/docs/config-file/config-reference)に対応する。
別ファイルのプロファイルを`-p`で重ねる方式は[OpenAI公式プロファイル設定](https://learn.chatgpt.com/docs/config-file/config-advanced#profiles)に従う。
Strata側のAPI、namespace/customツール、会話履歴の扱いは[Strata公式資料](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#the-responses-api-and-codex-cli)を参照。

## 2. sandboxとインターネット

`workspace-write`で作業ディレクトリと一時領域への書き込みを許可し、`network_access = true`を既定とした。
ネット接続のための追加フラグは不要。実際のCodexのシェルからHugging FaceへHTTPSで接続し、取得したJSONの内容を確認済み。
承認ポリシーはCodexの通常の`on-request`を使う。ネット接続を有効にするためにsandbox全体を無効化する必要はない。

Qwenプロファイルでは、StrataがOpenAIのホストする`web_search`等のツールを実行しないため、その機能は無効。
`curl`やPythonからのネット取得は利用できる。QwenではAppsとmulti-agentも無効にした。
GPTにはQwen専用のカタログ・コンテキスト・機能制限を適用しない。
MCPやプラグイン、プロジェクトのAGENTS.md・スキルの設定は個別に追加できる。

## 3. GPUとサーバー

`serve-codex.sh`は既定で`127.0.0.1:8080`に起動し、次を行う。

1. ポートが未使用か確認。
2. 起動中のllm-chatへLLMアンロードを要求。llm-chatが停止中ならそのまま進む。
3. `irodori-tts/gpu.lock`を予約し、Strataサーバーへ継承。
4. 既存の重み・packを使い、Codex専用の設定で起動。

GPU音声がONの間は起動を拒否する。Codexサーバー稼働中はllm-chatのLLMやGPU音声の起動もブロックする。
チャット用の`serve.sh`は5071、Codex用は8080。同じGPUに二つのLLMを同時にロードしない。
停止は起動ターミナルでCtrl+C。Strataがエンジンを終了してからGPU予約を解放する。
既定ポートを変える場合は`serve-codex.sh --port <番号>`とCodexの`base_url`の両方を揃える。

ログは`qwen3.8-flash-next/server-codex.log`（サーバー）と`strata/strata-codex-sc117-iq3_xxs.log`（エンジン）。
確認コマンド:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/status
codex --strict-config doctor --summary  # 既定のGPT設定の診断
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

## 4. コンテキストとメモリ

Codex用は128K（131072トークン）、自動要約の閾値は98K（98304トークン）。
llm-chat側もFlash Nextは128Kが既定で、UIから256Kまで変更できる。Codex用とは別に管理する。
専用モデルカタログ・Codex設定・`strata-codex-config.json`の三つを揃えた。
Codex設定の`model_context_window`だけを大きくしても、バックエンドの上限を超えるプロンプトは拒否される。
自動要約の閾値は出力用の余裕を残した値にする。

StrataはGPUの空き領域を専門家キャッシュへ回す。そのためVRAM使用量がカードの上限に近いだけでは、コンテキストを広げられないとは判断できない。
通常はKVキャッシュを増やすと専門家キャッシュが減り、生成速度やRAM使用量に影響する。
エンジン0.1.40の`--kv-grow`は、この環境のRAM節約設定`--resident-experts`と併用できないことを実機で確認した。
Codex用はKVを起動時に確保する。WSLのKV streamingも使わない。

RTX 5070 / RAM約48GBで、128K設定に84031トークンを実際に入力し、末尾の指定文字列を正しく返した。
長文テストは約68.9秒。2秒間隔で取得したメトリクスの最大値は次のとおり。

| 項目 | 実測 |
| --- | --- |
| GPU使用量 | 11968MiB / 12227MiB |
| RAM使用量 | 約40.5GiB / 47.0GiB |
| RAMの残量 | 約6.5GiB |
| GPUの専門家キャッシュ | 32Kで約4.1GiB → 128Kで約2.7GiB |

RAM残量は計測時の全体使用量との差。テスト直後のOSの利用可能メモリも約6.5GiBだった。
GPUの専門家キャッシュが減るため、速度との交換になる。RAMに全専門家を常駐できず、一部はOSのファイルキャッシュに依存する。
128K上限すべてを埋めるテストは未実施。現時点では128Kを採用し、それ以上への拡大は実測して判断する。
長い会話を保持しても、モデルが常に全履歴を正しく利用する保証にはならない。不要なツール出力は短くし、要約と併用する。

## 5. 動作確認

128K設定での速度実測（トークンは文字数とは異なる）:

| テスト | 入力処理 | 生成 |
| --- | --- | --- |
| 起動後のCodex小規模作業、約4～5K入力 | 初回941.7トークン/秒、4.3秒 | 56.5～62.1トークン/秒、各147～187トークン生成 |
| 約84Kの反復英文入力＋英文説明384トークン | 1460.3トークン/秒、57.6秒 | 49.8トークン/秒、7.7秒 |
| 長文処理後のQwenプロファイルでの同じ小規模作業 | 初回128.0トークン/秒、31.7秒 | 37.1～53.5トークン/秒、各155～193トークン生成 |

後者のAPIリクエスト全体は約65.6秒。入力のキャッシュ再利用は0。
長文側は合成した反復入力での計測であり、実際のリポジトリ全体を使った速度評価ではない。
32Kと128Kを同じ入力・出力条件で比較していないため、128Kによる低下率は未測定。
速度は言語・出力内容・MTPの採択率・キャッシュ再利用等で変わる。
長文処理後は小さい入力でも初回処理の待ち時間が大きかった。常に50～60トークン/秒で動くとの評価ではない。

実モデルにCodexから次を指示し、生成物とコマンド実行結果を確認した。

- `fizzbuzz.py`を作成し、Pythonで1～15を実行。
- HTTPSでHugging FaceのJSONを取得。
- PythonでJSONの`id`を照合。
- ツール結果をモデルに返し、最終報告を受け取る。

128K設定で初回プロンプトは約4Kトークン。3リクエスト合計の入力13431、キャッシュ再利用8955、出力493トークン。
Codexの診断はエラー0。StrataのResponses APIテスト21件も通過した。
検証ファイルはこの環境の`/tmp/codex-flash-next-128k-smoke/`、長文テストは`/tmp/strata-codex-context-128k/`に保存した。
GPTへの復帰後も`codex -p qwen`から同じ操作が通った。プロファイルの検証ファイルは`/tmp/codex-qwen-profile-smoke/`。
設定の移行・GPT設定の保持・再実行について`python3 -m unittest discover -s codex -p test_setup.py -v`の3テストが通過した。

この確認は小規模な実行テスト。大きなリポジトリで長時間継続するエージェント作業の品質評価ではない。
