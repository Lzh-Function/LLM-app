# Colab A100でStrata IQ3_SをCodexから使う

Qwen3.8-Flash-Next GSQ-RCO **IQ3_S**を、ColabのA100 high-memで実行する。
起動時に**通常版**または**無検閲版（abliterated）**を選べる。どちらもStrataが動かせる最大の量子化tierはIQ3_Sで、取得手順・SHA-256検証・pack生成・Codex接続は共通。
GGUFのshardであるモデル本体は、Hugging FaceからColabの一時SSD `/content/strata-qwen/model-data/` へ直接ダウンロードし、GDriveには置かない。
MTP・生成済みpack・ビルド済みengineはGDriveの `MyDrive/Strata-data/` に保存し、起動時にSSDへコピーする。Strataは常にSSD側から読み出す。ローカルPCへモデルはコピーしない。
SSHでAPIをローカルの `127.0.0.1:8085` へ転送する。Strata側は専用の `127.0.0.1:18080` のみで待ち受ける。
ColabのJupyterが使う8080番との競合を避け、ビルド・取得より前にポートの空きを確認する。
起動・停止・診断はリポジトリ直下の `./llm colab qwen ...` を使う。

## 通常版と無検閲版の選択

既定は通常版。無検閲版は `--variant uncensored` を付ける。

```bash
./llm colab qwen codex                      # 通常版（既定）
./llm colab qwen codex --variant uncensored # 無検閲版
./llm colab qwen start --variant uncensored
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
GGUFはGDriveに保存しない。GDriveに残るのはMTP・pack・engineだけで、空き容量の検査もDrive側に残すファイルの未取得分だけを見る。

## 初回に必要な操作

GDriveには **数GBの空き**があれば足りる（MTP約0.82GB、dense pack約2GB、engineビルド成果物）。
GGUFの2分割は通常版83,617,662,656 bytes（約83.62GB）、無検閲版83,863,586,496 bytes（約83.86GB）で、こちらはGDriveではなくHugging Faceから一時SSDへ直接入る。SSDはColabの約253GBを使い、本体・pack・DriveFSキャッシュが同じディスクを共有するため、起動前に空き容量を検査する。
Driveマウント時に表示されるブラウザ認証では、**表示された要求権限をすべて選択して許可する**。
この環境では、一部の権限だけを許可すると認証情報の転送後に `mount failed` となり、全選択でマウントが成功した。
許可後はターミナルに戻ってEnterを押す。Hugging Faceのトークンは任意（[Hugging Faceトークン](#hugging-faceトークン任意)参照）。無くても無認証でダウンロードできる。

自分のターミナルで、次を実行する。

```bash
cd /workspace/LLM
./llm colab qwen codex
```

この1コマンドがruntime作成→GDriveマウント・認証→初回ビルド→必要なダウンロード→SSDへのコピー・必要なSHA-256検証→packのコピーまたは作成→Strata起動→Codex起動を行う。
`--variant` を付けなければ通常版で、変種ごとに初回ダウンロードとpack生成が行われる。既に取得済みの変種は再取得しない。
初回準備の待機上限は3時間、runtime全体の上限は4時間。必要なら `./llm colab qwen codex --startup-minutes 300 --max-hours 6` で、準備待ち5時間・runtime全体6時間に変更できる。
Strata内部のREADY待ちも `--startup-minutes` と同じ秒数を起動前の環境変数 `STRATA_ENGINE_READY_S` へ渡す。サーバー既定の900秒で、Driveからの読込やGPUキャッシュ充填途中に終了しないようにする。外側の準備待ち・runtime時間上限も引き続き適用する。
**Codexを終了するとSSH接続とモデルを終了し、GDriveの同期完了を待ってから `colab stop -s strata-qwen` を実行する。**
同期待ちはColab標準と同じ最大24時間。同期が終われば直ちに停止する。同期失敗時はruntimeを残し、`./llm colab qwen stop` で再試行する。
同期完了を確認できない場合も時間上限でruntimeをstopし、未同期データは次回再取得が必要になり得ることを表示して失敗終了する。
Codexは起動したときの作業ディレクトリを使う。別プロジェクトの場合は絶対パスで呼ぶ。

```bash
cd /path/to/project
/workspace/LLM/llm colab qwen codex
# 非対話タスクも実行可能。GDrive認証が必要なときは人のターミナルで実行する。
/workspace/LLM/llm colab qwen codex -- exec 'READMEを読んで構成を説明して'
```

モデル本体のダウンロードは開始・20秒ごとの転送GBと速度・完了を表示し、SHA-256は転送中に計算する（検証だけの再読込は発生しない）。
SHA-256が一致したファイルだけを正式名に変更し、`.verified.json` にサイズとSHA-256を記録する。一時ファイル `.part` はSSDに残るため、同じruntime内なら続きから再開できる。
runtimeが変わるとSSDごと消えるので、本体は再びHugging Faceからダウンロードする。同じruntimeでサーバーを再起動すると `Verified runtime SSD copy (no transfer)` と表示され、本体の転送はゼロ。
SHA-256が不一致のときは `.part` を捨てて先頭からやり直す（壊れた続きを拾い続けない）。
packが未作成ならSSD上のGGUFからSSD上に生成し、完成品をDriveに保存する。既存packはDriveからSSDへコピーする（約2GB）。

### 実測した転送速度（2026-10-13、A100 high-mem）

| 経路 | 実測 |
|---|---|
| Hugging Face 単一ストリーム | 154.7 MB/s |
| Hugging Face 4並列 Range | 294.2 MB/s |
| DriveFS 読み（キャッシュ済み） | 73.0 MB/s |
| 一時SSD 書き込み / 読み込み | 279 MB/s / 5.4 GB/s |

単一ストリームのままでもDrive経由のコピーより2.1倍速い（84.44GBで約19分→約9分）。4並列は294 MB/sまで出るが、それはSSDの書き込み上限で、しかもSHA-256を転送中に刻めなくなるため再読込（84GBで約4分）で相殺される。1本でSHA-256を転送中に計算する現設計が実質的な最速なので、並列化は入れていない。

Driveに残すMTPとpackのキャッシュでは `--skip-cache-mtime` が既定で有効。Drive同期後の更新日時の差を無視して検証済みキャッシュを使うため、指定は不要。
本体のサイズと `.verified.json` のサイズ・SHA-256がmanifestと一致すれば、更新日時を比較せず全量ハッシュを省く。ログは `Verified cache (mtime check skipped)`。
モデル本体は常に転送中にSHA-256を検証するため、更新日時の判定は使わない。既存のSSDコピーで全量再検証が必要なのは `--reverify` を付けたときだけで、ログは `Runtime SSD copy needs a SHA-256 recheck`。

```bash
./llm colab qwen codex --variant uncensored
./llm colab qwen start
./llm colab qwen start --reverify # 全量SHA検証
```

## Hugging Faceトークン（任意）

5リポジトリはすべて `gated: false` / `private: false` なので、トークンがなくても動く。あるのは429（レート制限）の低減と、将来ライセンス同意を要求するリポジトリが増えたときの保険で、速度は上がらない。

```bash
mkdir -p ~/.config/hf
printf '%s\n' 'hf_...' > ~/.config/hf/token
chmod 600 ~/.config/hf/token
```

read-only（fine-grainedなら "Read" 権限のみ、リポジトリスコープ不要）で十分。書き込みは使わない。

`start` がそのファイルを `colab upload` で `/content/strata-qwen/.hf_token` に送る。`colab exec --env` は使わない。実体が `os.environ['KEY'] = '値'` をセルの先頭に足すだけで、値が `~/.config/colab-cli/colab.log` と `~/.config/colab-cli/history/*.jsonl` に平文で残る（sentinelで実測: log 2件 / history 1件）。upload は contents APIなのでファイル本体はこれらのログに出ない（実測: 0件）。

トークンはランタイムの一時SSDにだけ置き、GDriveには置かない。Driveは同じGoogleアカウントの他のデバイスから読めるうえ、runtimeを越えて残る。`stop` 時に削除し、runtime停止でも一緒に消える。値はログにもコマンドラインにも出さない。Hugging Faceが401/403を返した場合は無認証で自動リトライし、429は `Retry-After` を見て待つ。


## 起動・終了を分ける

```bash
# ターミナル1: 準備完了後も、このコマンドを起動したまま使う。
./llm colab qwen start

# ターミナル2: 上で起動したStrataへCodexを接続する。
./llm colab qwen client
# または現在のCodex/スクリプトからAPIを呼び出す。
curl http://127.0.0.1:8085/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-iq3_s","input":"日本語で一文だけ挨拶して","max_output_tokens":64,"reasoning":{"effort":"none"}}'

# セッション終了。Strata、トンネル、Colab runtimeを終了する。
./llm colab qwen stop
# 緊急時は直接:
colab stop -s strata-qwen
```

無検閲版で起動したruntimeへ同じcurlを使う場合は、`model` を `qwen3.8-flash-next-iq3_s-uncensored` に変える。
`./llm colab qwen client` は起動中の変種をAPI応答から判定して、対応するCodexモデルを接続する。
`start`のターミナルでCtrl+Cを押しても、GDrive同期待ちを経てruntimeをstopする。
`client`の終了だけでは共有runtimeを止めないため、最後に `stop` する。
`codex`モードではCodexが終了したときに自動でstopする。
起動失敗、SSH切断、SIGTERM/SIGHUP、時間上限でも、マウント済みなら同期を確認してから停止する。同期に失敗した場合は未同期データを保護するためruntimeを残し、`stop` で再試行できる。
別ターミナルの `stop` も起動元の同期・停止処理が終わるまで待つ。
既存の `strata-qwen` があると、新規起動は拒否する。必要なら先に `stop` する。

```bash
./llm colab qwen doctor   # ローカル依存とCodexカタログの確認。runtimeを作成しない。
./llm colab qwen status   # runtimeとAPIの状態。
./llm colab qwen doctor --variant uncensored  # 無検閲版の取得サイズとカタログを確認。
./llm colab qwen check-drive  # GDriveだけを確認し、成功・失敗どちらでもruntimeをstop。
./llm colab qwen start --max-hours 2  # 初回準備を含む2時間の上限。
./llm colab qwen codex --context 65536 --max-hours 3
```

既定は準備待ち最大180分、準備・利用の最大4時間。終了処理の同期待ちはこの上限とは別枠で、Colabの `flush_and_unmount()` と同じ最大24時間（`--flush-minutes`）。24時間必ず待つわけではなく、同期が終われば直ちに停止する。同期待ち中はA100の利用時間を消費する。上限やエラーで同期を確認できなければruntimeを残す。
長時間の初回ダウンロードなら `--startup-minutes 300 --max-hours 6` など、必要な上限を明示する。
時間上限で中断したファイルもクラウド同期済みなら次回再開できる。長い待機期間は自分で `stop` する。
直接の `colab stop`、ローカルマシンの電源断、SIGKILLでは同期処理を実行できず、未同期データを失う場合がある。
停止したVMの一時キャッシュからは回収できない。緊急時の直接の `colab stop` はruntime解放を優先する。


同期待ちの失敗でruntimeが残った場合は `./llm colab qwen stop` で再試行する。
DriveFSが完了前にマウントを外して終了した場合も、既存の認証・キャッシュで再マウントして同期を再試行する。マウントが外れただけでは同期済みと判定しない。
同期確認を省いて解放する必要がある場合だけ `./llm colab qwen stop --force` を使う（未同期データを失う可能性がある）。

IQ packの再利用は `.complete.json` のモデル識別子と `dense.bin`・`index.txt`・`native_experts.txt`・tokenizerで確認する。別のMTP pack用の `dense.txt` は要求しない。
起動待ち中はbootstrapとRAM上の推論診断から追加行だけをSSH経由でローカルへ表示する。以前の末尾を繰り返し表示せず、RAMの出力をColabのbootstrap.logへコピーしない。

Drive認証後のruntime再接続・作業ディレクトリ準備は、1回最大120秒、ローカルCLIのタイムアウト時に最大3回まで接続を作り直して再試行する。
ここでColab CLI自身の接続初期化が詰まる場合もある。再試行は再実行しても結果が変わらない準備セルに限定し、モデル起動やDrive同期のセルは自動で再実行しない。
セル実行の失敗段階・待機上限・試行回数・エラー種別はローカルの `.runtime/setup-error.json` に保存する。認証情報や推論出力はこのJSONに含めない。

## 設定と保存先

- runtime起動は指定どおり `colab new -s strata-qwen --gpu A100 --high-mem`。
- Strata 0.1.43、ソースcommit `7695859aa13afb248bdb4d36f9b2e95707db079d` を固定。A100用のsm_80バイナリをColabのCUDA Toolkitでビルドする。
- [ISTA-DASLabのIQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S)をrevision・サイズ・SHA-256で固定。
- 無検閲版は[SC117のIQ3_S](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/eec4e10a2c29b440a2ae85591fa11569ce752963/IQ3_S)を同じ手順で固定（shard 1が55,063,448,064 bytes、shard 2が28,800,138,432 bytes）。
- MTPの実行用ファイルは[SC117の配布済みrt](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/eec4e10a2c29b440a2ae85591fa11569ce752963/strata/rt)を使う。既存ローカル構成で元のQwen MTPと一致を確認したファイルで、主モデルにはSC117の改変済み重みを使わない。
- 変種ごとの `model` 名・pack先・ファイル一覧は `manifest.json` の `variants` に置き、`default_variant` が既定になる。Codexのモデル名は通常版 `qwen3.8-flash-next-iq3_s`、無検閲版 `qwen3.8-flash-next-iq3_s-uncensored`。
- 次回の新規起動から既定は256K（262,144トークン）context。稼働中のStrataには反映されない。`--context 131072` で従来の128Kにも戻せる。int8 KV、MTP、CJK draft vocabulary。テキスト入力のみ。画像用の追加モデルは取得しない。
- モデル・MTP・pack・tokenizerの参照先はすべてruntimeのSSD。`--resident-experts`でexpertをGPU/RAMに置き、PLE表も `--ple-io ram`でRAMへ読むため、A100と96GiB以上のRAMを検査する。
- [公式Codex provider設定](https://learn.chatgpt.com/docs/config-file/config-reference)に従い、Responses APIを使用。専用モデルカタログとCLIオーバーライドを使い、通常のCodex設定は書き換えない。基本指示はインストール済みCodexの標準指示を取得する。
- 既存の検索MCP用Python環境があれば接続する。推論とツール実行の実機確認は下記のとおり未完了。
- Drive内: `models/IQ3_S/`、`models/IQ3_S-abliterated/`、`packs/iq3_s/`、`packs/iq3_s-abliterated/`、`mtp/rt/`、`artifacts/`。推論エンジンのログは `/dev/shm/colab-privacy-*/inference.log`（RAM）へ保存する。
- runtime SSD内: `/content/strata-qwen/model-data/` 配下に選択した変種の `models/`、`packs/` と共有の `mtp/rt/`。runtime停止時に消える。推論中は読み取り専用。
- ローカル: `codex/colab-qwen/.runtime/` にSSH鍵、接続ログ、専用カタログ。git対象外、ディレクトリ権限700。
- Colab内: `/content/strata-qwen/bootstrap.log` にビルド・取得・保護ランチャーの起動結果だけを保存する。推論中の標準出力・標準エラーはRAMに保存する。準備・起動失敗時はRAMのエンジン診断もローカルの `.runtime/bootstrap-error.log` に回収する。
- 終了時の同期確認: `.runtime/drive-sync-status.json`。`confirmed: true` ならColabの `drive.flush_and_unmount()` の正常終了を確認した。

Driveの永続キャッシュ先を変える場合は `--data-dir /content/drive/MyDrive/任意のフォルダ`。
この指定にはMyDriveの外や未マウントのディレクトリを許可しない。推論用コピー先は上記のruntime SSDに固定し、容量不足時にDriveからの直接推論へ切り替えない。

## 会話内容をColabのディスクへ保存しないための制限

`start` / `codex` は共通の `../colab_privacy.py` をruntimeへアップロードし、モデル準備前に保護機能を確認する。
GPU runtimeでは保護下でCUDAの初期化も事前検査し、重み取得前にGPUとの互換性を確認する。
準備が終わったら、推論サーバーと子エンジンを **Landlock ABI 3以上の書き込み制限**の中で起動する。
Landlockが使えないkernelでは、Colab内に必要に応じて `bubblewrap` を導入し、推論専用の読み取り専用マウント名前空間へ切り替える。
この方式もモデル準備前に実際の書き込み拒否を検査する。必要なNVIDIAノードだけを持つ `/dev` を使い、SSH/APIのためネットワークは共有する。
Colabは新しい `/proc` のマウントも拒否するため、元の `/proc` を使う。CUDAが更新する自分の `/proc/<PID>/task/<TID>/comm`（RAM上のスレッド名）だけを追加で許可する。ディスクへ通じるmagic linkなどはネイティブの書き込み制限で拒否する。CUDAとpsutilのPID参照を保つため、PID名前空間と起動元のプロセスグループは維持する。
`/content`、GDrive、通常の `/tmp`、ホームなどへのファイル作成・追記・切り詰め・移動・削除を拒否する。
Landlockで許可する書き込み先は、tmpfsであることを確認した `/dev/shm`、`/dev/null` / `/dev/zero` / NVIDIAのデバイスと、上記のprocメタデータ。
bubblewrapでは `/tmp` と `/dev` もRAM上に作り、通常のディスクとGDriveのマウントが読み取り専用であることを内側のランチャーで確認する。
モデルやpack、エンジンの取得・生成・保存は準備段階で行い、推論中のSSD/GDriveからのモデル読み出し・読み取り専用mmapは維持する。
将来、モデル重みを複製する追加のディスクキャッシュを使う場合も、会話受付前の準備段階で作成する。モデルフォルダ全体を推論中に書き込み可能にする例外は設けない。

エンジンログ・標準出力・標準エラーはtmpfsと確認した `/dev/shm/colab-privacy-*/inference.log` に保存し、起動元から継承したディスクのログFDは閉じる。起動失敗時はSSHでRAMのログをローカルの `.runtime/bootstrap-error.log` に回収する。
APIモニター、slot保存、デバッグ・トレース・dump設定を無効化し、Pythonのbytecode保存とCUDAのディスクキャッシュも止める。
一時ファイルとXDGキャッシュの保存先は `/dev/shm/colab-privacy-*`。RAM上に残る一時ファイルには会話内容が入り得るが、ディスクへは保存しない。
OSのswapは無効化して確認する（SSDからモデル重みを読み出すオフロードとは別）。core dumpはhard limitを0にする。
Colabのパイプ式core handlerは管理側の読み取り専用設定なので変更しない。代わりに、事前に `cc` でビルドした小さな固定コードの `privacy-core-*.so` をPythonと子エンジンに継承する。
このライブラリは各execの直後に `PR_SET_DUMPABLE=0` を設定し、外部handlerへ会話メモリを渡さない。ライブラリ自体は会話を含まないコード成果物としてColabのbootstrapフォルダへ保存する。
推論中は `open` / `openat` / `fopen` / `truncate` と64bit版でも、RAM・許可したデバイス以外への書き込みを拒否する。元の `/proc` のmagic link経由のディスク書き込みも通常のファイルAPIでは拒否する。
この追加ライブラリは信頼する推論エンジンのログ・ダンプ抑制用であり、任意の未信頼コードが直接syscallを呼ぶ場合までの完全な隔離を保証するものではない。
事前に実際のexecで適用を検査する。動的ELF64ではないエンジンや、ライブラリを適用できないruntimeでは起動を拒否する。
推論プロセスのLinux capabilitiesを落とし、対応kernelではLandlockの拒否ログも抑制する。

保護を適用できないruntimeでは、制限を弱めて起動せず失敗して終了処理へ進む。
`bootstrap.log` に `COLAB_PRIVACY_ENFORCED` が出てからAPIが利用可能になる。推論の診断ログはRAMにあり、起動失敗時にローカルへ回収できる。
変更は次回の `start` / `codex` から適用され、すでに起動中のサーバーへ `client` で接続するだけでは適用されない。
過去のログは自動削除しない。Googleの管理基盤や他の独立したプロセスまで含めた保存ゼロの保証ではない。

ローカルの検証は `python3 codex/test_colab_privacy.py`。実kernelでの書き込み拒否、子プロセスへの継承、ログFDの閉鎖とモデルの読み取り専用mmapを検証する。
モデル全体の保護付きAPI推論は未検証。2026-10-13のQwen A100では、保護下のCUDA初期化、PCIe計測、322MiBのembeddingのRAM固定、native projection、PLEのRAM読込、MTPロードまで進んだ。GPUキャッシュ充填中にStrata内部の900秒タイムアウトで終了したため、READY待ちを上記の起動待ち上限と合わせた。この待機時間修正後のA100実機起動は未確認。GLMモデル本体でのGPU推論は未検証。
2026-10-13の実際のColab A100起動では、Landlock syscallが `Function not implemented` を返して停止したため、上記の読み取り専用マウント方式を追加した。
同日のColab CPU runtime（kernel `6.6.122+`）では最終版の `COLAB_PRIVACY_PREFLIGHT_OK` と `COLAB_PRIVACY_LIVE_CHECK_OK` を確認済み。
モデル相当ファイルの読み取り専用mmap、ディスクへの作成・上書きの拒否、RAM一時ファイルの読み書き、継承したディスクログFDの閉鎖、子exec後のcapabilitiesゼロを確認し、検証runtimeを停止した。GDriveとGPUモデル本体はこの検証で使っていない。
固定revisionのStrataを使うローカルCPUモックでは、実Landlock・capabilities制限・ログFD閉鎖を適用してResponses APIの往復を確認した。
ローカル環境はswap有効かつ `/dev/shm` が読み取り専用のため、このモック確認だけはホストの事前検査と一時領域を差し替えた。本番ランチャーに迂回オプションはない。

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
./llm colab qwen check-drive --drive-auth browser
```

表示されたURLを開くと、CLIで作成した既存のruntimeに接続するColab画面を開ける。
新しいruntimeは作成せず、表示された次のコードをブラウザのセルで実行してDriveアクセスを許可する。

```python
from google.colab import drive
drive.mount('/content/drive')
```

マウント成功後、ターミナルに戻ってEnterを押す。ブラウザ操作待ちは最大5分、マウント確認は最大20秒。
指定のA100 runtimeを作成してマウントだけを確認し、モデル取得やビルドはせず、最後にruntimeをstopする。
成功したら `./llm colab qwen codex --drive-auth browser` で、同じ認証方法を使って本起動できる。
`check-drive` が停止したruntimeを本起動で引き継ぐことはできず、本起動時には新しいruntimeで再度マウントする。

既定のCLI認証は `./llm colab qwen check-drive`。インストール済みCLIの認証処理を使い、補助スクリプトで `exec` と同じカーネルに接続する。
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
再開ダウンロード、SHA-256不一致、変更済みキャッシュ、GDrive保存制約、失敗時の終了処理、停止のサーバー確認、CLIの終了コード0でもリモート例外を検出すること、プロセスの時間制限、同一セル内の失敗診断、ブラウザ認証の分岐、割当503時の停止、ポート競合、空き容量、同期完了・失敗と停止の順序などに加え、READY待ち時間の伝達、RAM診断の転送、差分ログ表示、IQ packの再利用などのテストが通過。更新日時比較の省略、検証記録の不一致、サイズ不一致、`--reverify` の優先も検証する。
SSD配置についても、コピー中のSHA検証でDriveを一度だけ読むこと、破損・サイズ違いのコピーを採用しないこと、容量不足時に取得前に停止すること、packの完成マーカーを最後にコピーすること、ログをコピー対象に含めないこと、両変種のモデル参照先がSSDであることをローカルテストで確認した。SSD配置後のA100実機での起動・速度比較は未実施。
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
