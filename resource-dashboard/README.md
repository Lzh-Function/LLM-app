# LLM Observatory — ローカル・Colabのリソース監視

ローカルのLinux / WSL2と、`colab-qwen` / `colab-glm` で起動したruntimeを、一つのブラウザ画面で監視する。
Python 3.11以上の標準ライブラリだけで起動でき、追加のPython・Node依存のインストールは不要。

```bash
cd /workspace/LLM
./llm-dashboard
```

**[http://localhost:5090](http://localhost:5090)** を開く。Dev ContainerではVS Codeの「ポート」から5090を転送する。
別ディレクトリから `/workspace/LLM/llm-dashboard` と呼んでも使える。

```bash
./llm-dashboard --port 5091       # ポートを変更
./llm-dashboard --interval 5      # 計測間隔を5秒に変更（既定2秒）
```

## 表示できる情報

- CPU名称、使用率、コア・スレッド数、周波数、コアごとの負荷、公開されていれば温度。
- NVIDIA / AMD GPUの名称、使用率、VRAM使用量・空き、温度、消費電力・上限、クロック、ファン。複数GPUは個別表示。
- RAM使用量・利用可能量、コンテナのRAM・CPU割当があればその上限。
- ローカルディスク / runtime SSD / Google Driveの容量、ディスク読み書き速度、ネットワーク送受信速度。
- 稼働モデル名、エンジン・バージョン、量子化、KV形式、MTP、コンテキスト上限・処理中の使用量、モデル配置。
- 生成速度、入力処理速度、前回の速度、完了リクエスト数、累計出力トークン、入力キャッシュ率、処理待ち件数。
- 直近150サンプルのグラフ、表示の一時停止、メトリクスJSONのローカル保存、ローカルAPI接続の追加・削除。

取得できない項目は `—` を表示する。温度・電力・ファン等はVM、WSL、GPUドライバーによって非公開の場合がある。
CPU・ディスクI/O・ネットワーク速度は前回との差分から計算し、最初のサンプルでは未計測。
CPU使用率はOSから見える論理CPU全体が分母。コンテナ割当は別途表示する。
RAM使用量は `MemTotal - MemAvailable`。再利用・回収可能なファイルキャッシュを全量使用中のRAMとして扱わない。
WSL2 / Containerでは、そのLinux環境から見えるCPU・RAM・ディスクの値を表示する。Windows全体のセンサー監視ではない。
Drive容量はDriveFSの報告値を表示し、runtime SSD容量と区別する。

## 接続方法

左側の監視対象で「ローカル」「Colab · Qwen」「Colab · GLM」を切り替える。未起動の対象は未接続として残り、自動で再確認する。

**ローカル**：チャット用の5071、Strata Codex用の8080、`codex/*-runtime.json` の `backend_port` を自動確認する。
通常の起動方法を変えずに使える。別ポートのサーバーは「接続管理 → API接続を追加」でURLを登録する。
追加したAPIのハードウェア情報は、このダッシュボードを実行しているローカルマシンのもの。
APIは `http://127.0.0.1:PORT` / `http://localhost:PORT` / `http://[::1]:PORT` を指定できる。

**Colab**：いつもの `./colab-qwen codex` / `./colab-glm codex` / `start` を起動しておけば、既存SSH ControlMasterソケットとAPI転送ポートを自動検出する。
監視用に新しいColab sessionを作らず、認証・Driveマウント・モデルの再起動も行わない。
SSHを開いているこのワークスペースと同じマシン・ユーザーからダッシュボードを起動する。
接続が切れたら最終取得時刻と未接続状態を明示し、古い値をライブの値として扱わない。

**Strata / Maya**：JSONの `/metrics` と対応していれば `/v1/status` から取得する。生成の直近速度と、前回・累計平均を区別する。
**llama.cpp / Prism fork**：[公式のPrometheus metrics](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#get-metrics-prometheus-compatible-metrics-exporter)から累計平均の生成・入力処理速度を表示する。瞬間速度として表示しない。
このリポジトリのllama.cpp起動スクリプトとColab GLMのllama.cpp起動では `--metrics` を有効にした。既に起動中のサーバーへこの変更を適用するには次回起動が必要。
他の起動方法を使う場合も `--metrics` を付ける。メトリクスを公開しないOpenAI互換サーバーでは、ハードウェアや公開済みモデル情報だけを表示する。

## 保存と既存の会話保護

ダッシュボードはローカルで動作し、静的ファイルやフォントを外部CDNから取得しない。
Colab側はSSHから標準ライブラリの収集コードを `python3 -B -c` でメモリ上だけで実行し、監視ファイルやログをruntime SSD・Driveへ作成しない。
既存の推論サーバーの書き込み制限を変更しない。プロンプト、応答、思考本文、会話ログや `/api/requests` は収集しない。
推論APIのレスポンスから必要な数値・モデル情報だけを選び、ブラウザへ渡す。監視目的の推論リクエストは送らない。

時系列データはダッシュボードのローカルプロセスのRAMにだけ保持し、終了時に消える。
追加した接続名・API URLだけをローカルの `resource-dashboard/.runtime/targets.json` に保存する（git対象外）。
「スナップショット」はブラウザからローカルPCへメトリクスJSONを保存する。
表示の一時停止中もバックエンドの監視は続く。ダッシュボードをCtrl+Cで終了しても、推論サーバーやColab runtimeは停止しない。
標準では `127.0.0.1` にだけ待ち受ける。

## 確認

```bash
python3 -B -m unittest discover -s resource-dashboard -p 'test_*.py'
node --check resource-dashboard/static/app.js
```

ローカルRTX 5070と、稼働中のColab A100・Qwen無検閲版から実データを取得して確認した。
デスクトップ・390px幅の表示、モデル一覧、接続の追加・削除、更新の停止・再開、表示期間・グラフ切り替え、未接続表示をブラウザで確認した。
Colab GLMは未稼働のため、監視の実機確認は未実施。既存SSHの検出とAPIのメトリクス対応は同じ方式で実装している。
