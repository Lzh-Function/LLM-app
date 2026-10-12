# Colab A100でStrata IQ3_SをCodexから使う

Qwen3.8-Flash-Next GSQ-RCO **IQ3_S**を、ColabのA100 high-memで実行する。
起動時に**通常版**または**無検閲版（abliterated）**を選べる。どちらもStrataが動かせる最大の量子化tierはIQ3_Sで、取得手順・SHA-256検証・pack生成・Codex接続は共通。
モデル・MTP・生成済みpackはGDriveの `MyDrive/Strata-data/` に保存する。
GGUFをローカルPCやColabの `/content` へコピーしない。ソース、Python環境、ビルドの作業ファイルだけをColabの一時ディスクに置く。
SSHでAPIをローカルの `127.0.0.1:8085` へ転送する。Strata側は専用の `127.0.0.1:18080` のみで待ち受ける。
ColabのJupyterが使う8080番との競合を避け、ビルド・取得より前にポートの空きを確認する。

## 通常版と無検閲版の選択

既定は通常版。無検閲版は `--variant uncensored` を付ける。

```bash
./colab-qwen codex                      # 通常版（既定）
./colab-qwen codex --variant uncensored # 無検閲版
./colab-qwen start --variant uncensored
```

無検閲版は [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) のIQ3_Sを使う。
2026-10-12にHugging FaceのAPIで確認したダウンロード数では、無検閲・abliterated系のGGUFの中で最も多く（1,018,399 DL、likes 217、gated設定なし）、
同じGSQ-RCOをabliterateした派生（orcarouter 629,886 DLだがgatedでログイン必須、huihui-ai 339,162 DL、mradermacher 148,339 DL）より多い。
SC117は通常版と同じ `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` をbaseとし、48レイヤー分の144テンソルだけをabliterated版へ差し替えた配布で、量子化の再学習はしていない。
同梱の `strata/rt/` は、通常版のMTP実行用ファイルとして既に使っているものと同じrevision・SHA-256で共有する。つまり**MTPのdraft headは両変種で同一**（abliterateされていない）で、差し替わるのは主モデルの重み側。

無検閲版は配布元のREADMEどおり、**公式GSQ-RCOのpackを流用せず、そのGGUFからpackを作り直す**。
`tools/iq_pack.py` は `--gguf` と同じフォルダ内の `-0000N-of-0000M.gguf` を兄弟shardとして探し、`split.count` / `split.no` の一致を検査するため、
2つの変種のshardを同じフォルダに置いてはいけない（shard 2はバイト一致でも別ファイルとして持つ）。
そのため保存先は `models/IQ3_S-abliterated/` と `packs/iq3_s-abliterated/` に分けている。
**両変種をGDriveに置くとGGUFだけで約167GB**（通常版83.62GB + 無検閲版83.86GB）。空き容量の検査は選択した変種の未取得分だけを見る。

## 初回に必要な操作

`colab-cli` のログイン済みGoogleアカウントのGDriveに、**90GB程度以上の空き**を用意する。
GGUFの2分割だけで通常版83,617,662,656 bytes（約83.62GB）、無検閲版83,863,586,496 bytes（約83.86GB）。MTPは約0.82GB、さらにdense packと小さなビルド成果物が加わる。
Driveマウント時に表示されるブラウザ認証では、**表示された要求権限をすべて選択して許可する**。
この環境では、一部の権限だけを許可すると認証情報の転送後に `mount failed` となり、全選択でマウントが成功した。
許可後はターミナルに戻ってEnterを押す。Hugging Faceのログインは不要。

自分のターミナルで、次を実行する。

```bash
cd /workspace/LLM
./colab-qwen codex
```

この1コマンドがruntime作成→GDriveマウント・認証→初回ビルド→ダウンロード・SHA-256検証→pack作成→Strata起動→Codex起動を行う。
`--variant` を付けなければ通常版で、変種ごとに初回ダウンロードとpack生成が行われる。既に取得済みの変種は再取得しない。
初回準備の待機上限は3時間、runtime全体の上限は4時間。必要なら `./colab-qwen codex --startup-minutes 300 --max-hours 6` で、準備待ち5時間・runtime全体6時間に変更できる。
**Codexを終了するとSSH接続とモデルを終了し、GDriveの同期完了を待ってから `colab stop -s strata-qwen` を実行する。**
同期待ちは既定で最大10分。初回の約84GBの書き込みに長めの時間を確保するなら `./colab-qwen codex --flush-minutes 30` を使う。
同期完了を確認できない場合も時間上限でruntimeをstopし、未同期データは次回再取得が必要になり得ることを表示して失敗終了する。
Codexは起動したときの作業ディレクトリを使う。別プロジェクトの場合は絶対パスで呼ぶ。

```bash
cd /path/to/project
/workspace/LLM/colab-qwen codex
# 非対話タスクも実行可能。GDrive認証が必要なときは人のターミナルで実行する。
/workspace/LLM/colab-qwen codex -- exec 'READMEを読んで構成を説明して'
```

初回準備は時間がかかる。クラウド同期済みの `.gguf.part` はDriveに残り、次回は続きを取得する。
SHA-256が一致したファイルだけを正式名に変更する。ただし、Colabでの書き込み・検証完了とGDriveのクラウド同期完了は別。
GDrive画面に小さなJSONだけ先に見える場合もあり、`.verified.json` だけでは本体のクラウド保存を保証しない。
2回目以降はGGUF本体の存在・サイズ・更新時刻・検証マーカーが一致すれば再取得・全量ハッシュを省く。
全量再検証が必要なときは `--reverify` を付ける。

## 起動・終了を分ける

```bash
# ターミナル1: 準備完了後も、このコマンドを起動したまま使う。
./colab-qwen start

# ターミナル2: 上で起動したStrataへCodexを接続する。
./colab-qwen client
# または現在のCodex/スクリプトからAPIを呼び出す。
curl http://127.0.0.1:8085/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-iq3_s","input":"日本語で一文だけ挨拶して","max_output_tokens":64,"reasoning":{"effort":"none"}}'

# セッション終了。Strata、トンネル、Colab runtimeを終了する。
./colab-qwen stop
# 緊急時は直接:
colab stop -s strata-qwen
```

無検閲版で起動したruntimeへ同じcurlを使う場合は、`model` を `qwen3.8-flash-next-iq3_s-uncensored` に変える。
`./colab-qwen client` は起動中の変種をAPI応答から判定して、対応するCodexモデルを接続する。
`start`のターミナルでCtrl+Cを押しても、GDrive同期待ちを経てruntimeをstopする。
`client`の終了だけでは共有runtimeを止めないため、最後に `stop` する。
`codex`モードではCodexが終了したときに自動でstopする。
起動失敗、SSH切断、SIGTERM/SIGHUP、時間上限でも、マウント済みなら同期を試みてから停止する。
別ターミナルの `stop` も起動元の同期・停止処理が終わるまで待つ。
既存の `strata-qwen` があると、新規起動は拒否する。必要なら先に `stop` する。

```bash
./colab-qwen doctor   # ローカル依存とCodexカタログの確認。runtimeを作成しない。
./colab-qwen status   # runtimeとAPIの状態。
./colab-qwen doctor --variant uncensored  # 無検閲版の取得サイズとカタログを確認。
./colab-qwen check-drive  # GDriveだけを確認し、成功・失敗どちらでもruntimeをstop。
./colab-qwen start --max-hours 2  # 初回準備を含む2時間の上限。
./colab-qwen codex --context 65536 --max-hours 3
```

既定は準備待ち最大180分、準備・利用の最大4時間。終了処理の同期待ちはこの上限とは別枠で最大10分（`--flush-minutes`）。同期待ち中もruntimeは稼働する。
長時間の初回ダウンロードなら `--startup-minutes 300 --max-hours 6` など、必要な上限を明示する。
時間上限で中断したファイルもクラウド同期済みなら次回再開できる。長い待機期間は自分で `stop` する。
直接の `colab stop`、ローカルマシンの電源断、SIGKILLでは同期処理を実行できず、未同期データを失う場合がある。
停止したVMの一時キャッシュからは回収できない。緊急時の直接の `colab stop` はruntime解放を優先する。

## 設定と保存先

- runtime起動は指定どおり `colab new -s strata-qwen --gpu A100 --high-mem`。
- Strata 0.1.43、ソースcommit `7695859aa13afb248bdb4d36f9b2e95707db079d` を固定。A100用のsm_80バイナリをColabのCUDA Toolkitでビルドする。
- [ISTA-DASLabのIQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S)をrevision・サイズ・SHA-256で固定。
- 無検閲版は[SC117のIQ3_S](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/eec4e10a2c29b440a2ae85591fa11569ce752963/IQ3_S)を同じ手順で固定（shard 1が55,063,448,064 bytes、shard 2が28,800,138,432 bytes）。
- MTPの実行用ファイルは[SC117の配布済みrt](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/eec4e10a2c29b440a2ae85591fa11569ce752963/strata/rt)を使う。既存ローカル構成で元のQwen MTPと一致を確認したファイルで、主モデルにはSC117の改変済み重みを使わない。
- 変種ごとの `model` 名・pack先・ファイル一覧は `manifest.json` の `variants` に置き、`default_variant` が既定になる。Codexのモデル名は通常版 `qwen3.8-flash-next-iq3_s`、無検閲版 `qwen3.8-flash-next-iq3_s-uncensored`。
- 128K context、int8 KV、MTP、CJK draft vocabulary。テキスト入力のみ。画像用の追加モデルは取得しない。
- `--resident-experts`でexpertをGPU/RAMに置き、PLE表も `--ple-io ram`でRAMへ読む。GDriveへのランダム参照を避けるため、A100と96GiB以上のRAMを検査する。
- [公式Codex provider設定](https://learn.chatgpt.com/docs/config-file/config-reference)に従い、Responses APIを使用。専用モデルカタログとCLIオーバーライドを使い、通常のCodex設定は書き換えない。基本指示はインストール済みCodexの標準指示を取得する。
- 既存の検索MCP用Python環境があれば接続する。推論とツール実行の実機確認は下記のとおり未完了。
- Drive内: `models/IQ3_S/`、`models/IQ3_S-abliterated/`、`packs/iq3_s/`、`packs/iq3_s-abliterated/`、`mtp/rt/`、`artifacts/`、`logs/engine.log`。
- ローカル: `codex/colab-qwen/.runtime/` にSSH鍵、接続ログ、専用カタログ。git対象外、ディレクトリ権限700。
- Colab内: `/content/strata-qwen/bootstrap.log` にビルド・取得・サーバーのログ。準備失敗時は末尾をローカルの `.runtime/bootstrap-error.log` に保存する。
- 終了時の同期確認: `.runtime/drive-sync-status.json`。`confirmed: true` ならColabの `drive.flush_and_unmount()` の正常終了を確認した。

保存先を変える場合は `--data-dir /content/drive/MyDrive/任意のフォルダ`。
MyDriveの外や未マウントのディレクトリは拒否し、一時SSDへのモデル保存に切り替えない。

## runtime作成時に `503 Service Unavailable` が出る場合

`colab new` の `/tun/m/assign` に対する応答なので、GDrive認証より前のruntime割当段階で失敗している。
A100 high-memの空き状況やColab側のサービスの問題が考えられるが、この応答だけでは詳細原因や復旧時間は分からない。
時間を置いて同じ起動コマンドを再実行する。自動連続再試行や別のGPUへの変更は行わない。
スクリプトは割当失敗を簡潔に表示し、`.runtime/allocation-error.json` に段階・要求した構成・分類を保存する。
失敗時も停止処理を通す。`colab sessions` はサーバー側の割当確認に使える。

## GGUFがGDriveの画面に見えない場合

保存先は `MyDrive/Strata-data/models/IQ3_S/`（無検閲版は `models/IQ3_S-abliterated/`）。ColabのDriveFSは書き込みをキャッシュするため、
runtime内にGGUFと検証JSONがあっても、クラウドへの送信がまだ終わっていない場合がある。
新しい終了処理では、モデルの書き込みを止めてから `drive.flush_and_unmount()` で同期を待ち、成功・失敗を明示する。
待機上限は `--flush-minutes`。上限や通信エラーで完了を確認できない場合でも、runtimeの停止処理を通す。
本体が無い検証JSONだけの状態なら、次回起動は本体の欠落を検出して再ダウンロードする。JSONだけで取得済みとは判断しない。
取得前にDriveの報告する空き容量と未取得サイズを比較し、容量不足ならビルド・ダウンロードへ進まない。

## 認証後に `ValueError: mount failed` が出る場合

`Credentials propagated` は認証情報の転送完了を示す。DriveFSの起動・接続成功とは別なので、その後にマウントが失敗する場合がある。
マウント失敗だけでは権限不足か、DriveFS側の障害かは断定できない。
まず同意画面で要求された権限をすべて選択したか確認する。この環境では、それによりCLI経由のマウントが成功した。
CLI経由の認証が繰り返し失敗する場合、まず自分のターミナルで次を実行する。

```bash
./colab-qwen check-drive --drive-auth browser
```

表示されたURLを開くと、CLIで作成した既存のruntimeに接続するColab画面を開ける。
新しいruntimeは作成せず、表示された次のコードをブラウザのセルで実行してDriveアクセスを許可する。

```python
from google.colab import drive
drive.mount('/content/drive')
```

マウント成功後、ターミナルに戻ってEnterを押す。ブラウザ操作待ちは最大5分、マウント確認は最大20秒。
指定のA100 runtimeを作成してマウントだけを確認し、モデル取得やビルドはせず、最後にruntimeをstopする。
成功したら `./colab-qwen codex --drive-auth browser` で、同じ認証方法を使って本起動できる。
`check-drive` が停止したruntimeを本起動で引き継ぐことはできず、本起動時には新しいruntimeで再度マウントする。

既定のCLI認証は `./colab-qwen check-drive`。インストール済みCLIの認証処理を使い、補助スクリプトで `exec` と同じカーネルに接続する。
失敗時の診断は、マウントと同じセルの中で取得し、停止前に `.runtime/drive-mount-error.json` へ保存する。
このファイルにはマウント状態、DriveFSログの有無・サイズ、認証・権限・接続・クラッシュなど既知のエラーの検出数を記録し、ログ原文や認証情報は保存しない。
CLIのマウント失敗後は別の `exec` で診断を取り直さない。CLI全体にも5分の時間制限を設け、応答しない場合はプロセスを終了してruntime停止へ進む。
セル自体が完了しない場合は診断を保存できないが、runtimeは停止処理を通る。

インストール済み `colab-cli` は、Colab側のPython例外を表示しても終了コードが0になる場合がある。
このフレームワークは各Python実行の完了マーカーと実際のマウント状態を確認し、失敗したらuploadへ進まずruntimeをstopする。

## この環境での確認状況（2026-10-12）

指定コマンドでA100 high-mem runtimeの割当、RAM約167GiB、`nvcc`の存在を確認。
GDriveは未マウントだったため認証・モデル取得は進めず、runtimeをstopし、サーバー上に残っていないことを確認した。
`doctor`でCodexの標準指示を含む専用カタログ生成を確認。
再開ダウンロード、SHA-256不一致、変更済みキャッシュ、GDrive保存制約、失敗時の終了処理、停止のサーバー確認、CLIの終了コード0でもリモート例外を検出すること、プロセスの時間制限、同一セル内の失敗診断、ブラウザ認証の分岐、割当503時の停止、ポート競合、空き容量、同期完了・失敗と停止の順序など48件のテストが通過。
変種選択では、既定が通常版であること、未知の変種名をビルド・取得前に拒否すること、両変種のshardが別フォルダ・別packに分かれていること、`engine_config` が選択変種のshard・pack・モデル名を使うこと、`--variant` がColab側の起動コマンドへ渡ること、healthが両変種のモデル名を受け付け未知のモデル名は弾くことを確認した。
さらに完成した `start` コマンドで、A100作成→GDrive未認証検出→自動stop→停止のサーバー確認まで実機で通った。
配布元のモデルAPIに対して、manifestの固定revision・サイズ・LFS SHA-256も照合した。無検閲版のIQ3_S 2分割（55,063,448,064 / 28,800,138,432 bytes）と共有するMTP 4点も、SC117のrevision `eec4e10a2c29b440a2ae85591fa11569ce752963` で同じ照合を通した。
無検閲版の実機起動（Colab上のダウンロード・pack生成・推論）はまだ未実施。
ユーザーの `check-drive` 実行で、同意画面の要求権限をすべて許可した後のCLI経由マウント、MyDriveの存在、runtimeの自動停止まで成功した（2026-10-12 10:08 UTCのログ）。
ユーザーの本起動ログでColab上のCUDAビルド、GGUF/MTPの取得・検証、pack生成まで完了した。その後8080番の競合で起動が失敗してruntimeが停止し、GDrive画面ではJSONのみ残っているとの報告があった。
8080番を専用18080番へ変更し、終了時の同期待ちを追加した。
**モデル本体のクラウド同期、修正後のIQ3_S推論・Codexの実ツール呼び出しは実機確認が必要。**

```bash
python3 codex/colab-qwen/test_framework.py
```
