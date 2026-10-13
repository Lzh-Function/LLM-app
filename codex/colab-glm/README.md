# Colab A100でGLM-5.3をCodexから使う

GLM-5.3／GLM-5.3-Flashを、ColabのA100 high-memで実行する。
起動時に**3変種**から選び、それぞれエンジンが異なる（`flash`はProject Maya、他2つはllama.cpp）。
重み・Mayaのpack・ビルド済みエンジンはGDriveの `MyDrive/GLM53-data/` に保存する（Qwenの `MyDrive/Strata-data/` とは別フォルダ）。
GGUFはローカルPCにはコピーしない。推論時にはColabの `/content/glm53-a100/model-data/` へコピーを試し、容量・I/Oの問題があればDriveから直接読む。ソース、Python環境、ビルドの作業ファイルもColabの一時ディスクに置く。
SSHでAPIをローカルの `127.0.0.1:8086` へ転送する。サーバー側は専用の `127.0.0.1:18081` のみで待ち受ける。
ColabのJupyterが使う8080番との競合を避け、ビルド・取得より前にポートの空きを確認する。
選定根拠と容量・速度の予算は [docs/14-colab-glm53-runtime-plan.md](../../docs/14-colab-glm53-runtime-plan.md)。
起動・停止・診断はリポジトリ直下の `./llm colab glm ...` を使う。

## 変種の選択

既定は `flash`。

```bash
./llm colab glm codex                              # flash（既定・Maya）
./llm colab glm codex --variant flash-uncensored   # 無検閲Flash（llama.cpp）
./llm colab glm start --variant full --context 32768
```

| variant | 配布 | 量子化 | 重み | エンジン | context |
| --- | --- | --- | ---: | --- | ---: |
| `flash` | [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) `Maya-L` | gate/up IQ3_S・down IQ4_XS・一部Q5_K | 156.32GB / 145.58GiB（4分割） | Project Maya 1.1.1 | 32768 / 65536 / **131072** |
| `flash-uncensored` | [BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF) `IQ3_XXS` | IQ3_XXS | 134.31GB / 125.08GiB（3分割） | llama.cpp | 32768 / 65536 / **131072** |
| `full` | [autotrust/GLM-5.3-GGUF-DGX-Spark](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark) `SLIM-E192` | routed experts IQ2_XXS、256→192 expert | 160.76GB / 149.72GiB（4分割） | llama.cpp | **32768 / 65536** |

`--context` の選択肢は変種ごとに `manifest.json` が持つ。既定は上の太字。
Codexのモデル名は `glm-5.3-flash-maya-l` / `glm-5.3-flash-uncensored-iq3_xxs` / `glm-5.3-slim-e192`。
**同時起動はできない。** 重みの合計がRAM+VRAM（246.44GiB）を超えるため、1runtimeで1変種だけ。
切り替えは `stop` → 別の `--variant` で `start`。既にダウンロード済みの変種の再取得はない。
`client` は起動中の変種をAPI応答から判定して接続するため、`client --variant` は起動中の変種と違うと失敗する。

## 初回に必要な操作

`colab-cli` のログイン済みGoogleアカウントのGDriveに、選んだモデルの重みとpack分の空きを用意する。
3変種すべてならGGUFだけで約451.39GBなので、**460GB程度以上のクラウド側の空き**を目安にする。
空きはDrive APIの `storageQuota` で確認する。ColabのDriveFSマウントが返す空き容量は一時ディスク／キャッシュの値なので、GDrive全体の空きとして扱わない。
Driveマウント時のブラウザ認証では**表示された要求権限をすべて選択して許可**する。許可後はターミナルに戻ってEnterを押す。Hugging Faceのログインは不要。

```bash
cd /workspace/LLM
./llm colab glm codex
```

この1コマンドが runtime作成→GDriveマウント・認証→GGUF取得・SHA-256検証→ソース取得・**エンジンビルド（sm_80）**→（Mayaは）pack生成→runtime SSDへのコピーを試行→サーバー起動→Codex起動を行う。
`flash` の初回は Maya のエンジンコンパイルと pack 生成（全テンソルのインデックスと小さなfloat重み・tokenizerで1〜2GB。expertはGGUFに残る）が加わり、`flash-uncensored`／`full` は llama-server のビルドになる。
初回準備の待機上限は既定240分、runtime全体の上限は6時間。必要なら `--startup-minutes 300 --max-hours 8` で変える。
**Codexを終了するとSSH接続とモデルを終了し、GDriveの同期完了を待ってから `colab stop -s glm53-a100` を実行する。**
同期待ちはColabの `flush_and_unmount()` と同じ最大24時間（`--flush-minutes` で変更可）。同期が終われば直ちに停止する。上限やエラーで同期を確認できなければruntimeを残す。同期待ち中はA100の利用時間を消費する。

## 2回目以降はDriveのキャッシュを使う

初回に作ったものは、変種・エンジンごとにDriveへ保存し、次回以降は再取得・再ビルドしない。

```
MyDrive/GLM53-data/
├── models/Maya-L/            Maya-L 4分割と、Mayaが生成する pack/（Maya自身の配置）
├── models/bolding-iq3_xxs/   IQ3_XXS 3分割
├── models/slim-e192/         SLIM-E192 4分割
└── engines/
    ├── llama-<sig>/llama-server
    └── maya-<sig>/{strata, MAYA-BUILD.json}
```

- GGUFはサイズ・`.verified.json` が一致すれば再取得も全量ハッシュも省く。更新日時の比較は既定で省略する。再検証は `--reverify`。
- 検証記録の不一致や `--reverify` で再検証するときは、ダウンロードはし直さず、検証後に記録を更新する。検証開始・20秒ごとの読取GBと速度・完了を表示する。

`--skip-cache-mtime` は既定で有効。Drive同期後の更新日時の差を無視して検証済みキャッシュを使うため、指定は不要。
本体のサイズと `.verified.json` のサイズ・SHA-256がmanifestと一致すれば、更新日時を比較せず全量ハッシュを省く。ログは `Verified cache (mtime check skipped)`。
初回ダウンロードは従来どおりSHA-256を検証する。検証記録が無い・壊れている・記録のSHA-256/サイズが異なる場合は全量検証し、本体のサイズが異なる場合は拒否する。
`--reverify` を指定すれば全量検証を優先する。既定では同じサイズの内容変更は検出しないため、内容を検査するときは `--reverify` を使う。従来の更新日時比較を使う場合は `--no-skip-cache-mtime` を付ける。

```bash
./llm colab glm codex
./llm colab glm start
./llm colab glm prepare
./llm colab glm start --reverify # 全量SHA検証
```

- `--repack` は無い。Mayaは `models/Maya-L/pack/native_experts.txt` が同じGGUFを指し、`index.txt`・`dense.bin`・tokenizerが揃っていればpackを作り直さない。
- `<sig>` は「エンジン種別 + 固定commit + nvccのパスとバージョン + libc + CPU flags」のSHA-256先頭20桁。これらの条件が変わると再ビルドになる。
- 復元した Maya の `build/strata` と `MAYA-BUILD.json` は、Maya自身のstamp検査を通るため、一致すればコンパイルを飛ばす。
- ソース（llama.cpp、project-maya、Mayaが使うllama.cpp）と `.venv` はColabの一時ディスクに置く。runtimeごとに作り直す。

## 推論時のruntime SSDコピー

`start` / `codex` は選択した変種のGGUFをDriveから `/content/glm53-a100/model-data/` へコピーし、Mayaの場合は同じモデルフォルダのpackもコピーする。コピー中にGGUFのSHA-256を照合し、全ファイルが揃ったときだけSSDのパスでエンジンを起動する。`prepare` はDriveへの永続保存とエンジン準備だけを行い、SSDコピーはしない。

DriveFSの読取キャッシュは同じruntime SSDを使うが、容量の使い方は起動ごとに変わる。そのため事前の空き容量だけでコピーを諦めず、5GBの空きを残しながら実際にコピーを試す。容量不足やコピー中のI/Oエラーでは途中のSSDコピーを削除し、検証済みDriveファイルを直接読む設定で起動する。Drive側ファイルのサイズ変更やコピー中のSHA-256不一致は破損の可能性があるため、Drive読込へ切り替えず起動を停止する。

ログの `Runtime SSD model copy complete` はSSD読込、`using Google Drive model files` はDrive読込を表す。SSDコピーはruntime停止時に消えるため、次回起動でも再度コピーを試す。Drive上のモデルとエンジンキャッシュは引き続き永続化する。

## 準備だけまとめて行う

```bash
./llm colab glm prepare                  # 3変種すべて: 重み取得 + 2エンジンビルド + Mayaのpack
./llm colab glm prepare --variant full   # 1変種だけ。この場合のみ --context を指定できる
```

サーバーもCodexも起動しない。runtime作成→GDriveマウント→変種ごとの空き容量確認→GGUF取得・SHA-256検証→エンジンビルド→（`flash` は）pack生成までを1runtimeで済ませ、完了後にGDriveを同期して停止する。
contextは各変種の既定値（`flash`／`flash-uncensored` は131072、`full` は65536）を使い、`--context` は `--variant` を1つに絞った時だけ受け付ける。
待機上限は既定480分、runtime全体は10時間、GDrive同期待ちは最大24時間（`--startup-minutes`・`--max-hours`・`--flush-minutes` で変更可）。
完了は `bootstrap.log` の `GLM_PREPARE_COMPLETE ["flash", "flash-uncensored", "full"]` で分かる。以後の `start`／`codex` はキャッシュを使うだけになり、初回の待ち時間がサーバーを待つ時間から消える。
準備プロセスはこの完了行を出して正常終了する。完了行がある正常終了を `GLM_BOOTSTRAP_FAILED` と表示しない。クラウドへの保存確認は、その後の「GDriveの同期完了を確認しました。」まで待つ。

ログが更新されず、進んでいるか分からない場合は、別ターミナルで確認する。

```bash
./llm colab glm progress
```

既存のSSH接続で準備プロセスと子プロセス、GGUFの読取／書込位置、直近のログを表示する。runtimeの作成や停止は行わない。
`state=D` はI/O待ち、`R` は実行中、`S` は待機。読み取り位置の増加はSHA-256検証でも起こるので、ダウンロードと区別してログを見る。
準備待ちのログが変わらないときも、30秒ごとにこの確認コマンドを案内する。
`prepare`／`start` のログは前回の読取位置から追加された行だけを表示する。以前の末尾8行を繰り返す表示は並列実行を意味しない。
既に実行中のPythonプロセスにはローカルファイルの編集だけでは修正が反映されない。

## 起動・終了を分ける

```bash
# ターミナル1: 準備完了後も、このコマンドを起動したまま使う。
./llm colab glm start --variant full

# ターミナル2: 上で起動したサーバーへCodexを接続する。
./llm colab glm client
# または現在のCodex/スクリプトからAPIを呼び出す。
curl http://127.0.0.1:8086/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-slim-e192","input":"日本語で一文だけ挨拶して","max_output_tokens":64,"reasoning":{"effort":"none"}}'

# セッション終了。サーバー、トンネル、Colab runtimeを終了する。
./llm colab glm stop
# 緊急時は直接:
colab stop -s glm53-a100
```

`/v1/responses` はMaya・llama.cppの両方が本体で提供する。変換層は置いていない。
`start`のターミナルでCtrl+Cを押しても、GDrive同期待ちを経てruntimeをstopする。
`client`の終了だけでは共有runtimeを止めないため、最後に `stop` する。`codex`モードではCodex終了時に自動でstopする。
起動失敗、SSH切断、SIGTERM/SIGHUP、時間上限でも、マウント済みなら同期を確認してから停止する。同期に失敗した場合は未同期データを保護するためruntimeを残し、`stop` で再試行できる。
既存の `glm53-a100` があると新規起動は拒否する。

```bash
./llm colab glm doctor                    # ローカル依存とCodexカタログの確認。runtimeを作成しない。
./llm colab glm doctor --variant full     # 変種ごとの取得サイズとカタログを確認。
./llm colab glm status                    # runtimeとAPIの状態。
./llm colab glm progress                  # 準備中の読取位置・子プロセスを確認。新規runtimeなし。
./llm colab glm check-drive               # GDriveだけを確認し、成功・失敗どちらでもruntimeをstop。
./llm colab glm start --max-hours 2       # 初回準備を含む2時間の上限。
./llm colab glm codex --context 32768 --max-hours 3
```


同期待ちの失敗でruntimeが残った場合は `./llm colab glm stop` で再試行する。
DriveFSが完了前にマウントを外して終了した場合も、既存の認証・キャッシュで再マウントして同期を再試行する。マウントが外れただけでは同期済みと判定しない。
同期確認を省いて解放する必要がある場合だけ `./llm colab glm stop --force` を使う（未同期データを失う可能性がある）。

Drive認証後のruntime再接続・作業ディレクトリ準備は、1回最大120秒、ローカルCLIのタイムアウト時に最大3回まで接続を作り直して再試行する。
ここでColab CLI自身の接続初期化が詰まる場合もある。再試行は再実行しても結果が変わらない準備セルに限定し、モデル起動やDrive同期のセルは自動で再実行しない。
セル実行の失敗段階・待機上限・試行回数・エラー種別はローカルの `.runtime/setup-error.json` に保存する。認証情報や推論出力はこのJSONに含めない。

## 設定と保存先

- runtime起動は `colab new -s glm53-a100 --gpu A100 --high-mem`。A100（sm_80）とRAM 150GiB以上を検査し、違えばビルド・取得前に失敗する。
- Project Maya v1.1.1 commit `eb9b7ec88d6f7802f15b218b551550f749c7a94a`、llama.cpp commit `efe3626b119ff81033e309c3ba07a770bef2c546`、Mayaが使うllama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` を固定。
- llama-serverのCMake設定は `LLAMA_BUILD_COMMON=ON`・`LLAMA_BUILD_TOOLS=ON`・`LLAMA_BUILD_SERVER=ON`。固定したllama.cppでは `tools/server` がこの条件の下にあるため、TOOLSをOFFにすると `llama-server` の対象が生成されない。テスト・examples・unified app・Web UIはOFFで、ビルド対象は `llama-server` に絞る。
- GGUFはrevision・サイズ・SHA-256で固定。分割GGUFは最初のshardを `-m` に渡し、兄弟shardは同じフォルダから読む。
- `flash` は `STRATA_GLM_RAM_RESIDENT=1 STRATA_GLM_KV_INT8=1 STRATA_GLM_RAM_HEADROOM_GB=16`、draftingはモデル自身のMTP block、`--no-vision` で画像エンコーダー（mmproj）の取得とビルドを省略。
- llama.cppの変種は `-ngl 99 -cmoe --moe-cache-mib <32768|16384> -fa on -np 1 --jinja --temp 0.7 --min-p 0.05`。重みがVRAM 79.39GiBを超えるため、expertはRAMに置いてGPUキャッシュで補う。
- `full`（SLIM-E192）はMTPテンソルを持たないため、speculative decodeは使わない。
- 変種ごとの `model` 名・エンジン・context選択肢・ファイル一覧は `manifest.json` の `variants` に置き、`default_variant` が既定になる。
- [公式Codex provider設定](https://learn.chatgpt.com/docs/config-file/config-reference)に従いResponses APIを使用。専用モデルカタログとCLIオーバーライドを使い、通常のCodex設定は書き換えない。
- ローカル: `codex/colab-glm/.runtime/` にSSH鍵、接続ログ、専用カタログ、`drive-sync-status.json`。git対象外、権限700。
- Colab内: `/content/glm53-a100/bootstrap.log` にビルド・取得・保護ランチャーの起動結果だけを保存する。推論中の標準出力・標準エラーはRAMに保存する。準備・起動失敗時はRAMのエンジン診断もローカルの `.runtime/bootstrap-error.log` に回収する。

永続保存先を変える場合は `--data-dir /content/drive/MyDrive/任意のフォルダ`。MyDriveの外や未マウントのディレクトリは拒否する。推論用の一時コピー先は常にruntime SSDの `/content/glm53-a100/model-data/`。

## 会話内容をColabのディスクへ保存しないための制限

全3変種の `start` / `codex` は共通の `../colab_privacy.py` をruntimeへアップロードし、モデル準備前に保護機能を確認する。
GPU runtimeでは保護下でCUDAの初期化も事前検査し、重み取得前にGPUとの互換性を確認する。
モデル取得・pack生成・エンジンビルドは従来どおりディスクへ保存する。`prepare` は会話サーバーを起動しないため、書き込み制限の対象外。
推論は **Landlock ABI 3以上の書き込み制限**の中で実行し、子プロセスにも制限を継承する。
Landlockが使えないkernelでは、Colab内に必要に応じて `bubblewrap` を導入し、推論専用の読み取り専用マウント名前空間へ切り替える。
この方式もモデル準備前に実際の書き込み拒否を検査する。必要なNVIDIAノードだけを持つ `/dev` を使い、SSH/APIのためネットワークは共有する。
Colabは新しい `/proc` のマウントも拒否するため、元の `/proc` を使う。CUDAが更新する自分の `/proc/<PID>/task/<TID>/comm`（RAM上のスレッド名）だけを追加で許可する。ディスクへ通じるmagic linkなどはネイティブの書き込み制限で拒否する。CUDAとpsutilのPID参照を保つため、PID名前空間と起動元のプロセスグループは維持する。
`/content`、GDrive、通常の `/tmp`、ホームなどへのファイル作成・追記・切り詰め・移動・削除を拒否する。
書き込み先はtmpfsと確認した `/dev/shm`、`/dev/null` / `/dev/zero` / NVIDIAのデバイスと、上記のprocメタデータに限定する。
bubblewrapでは `/tmp` と `/dev` もRAM上に作り、通常のディスクとGDriveのマウントが読み取り専用であることを内側のランチャーで確認する。
SSD/GDriveからのモデル重みの読み出し・読み取り専用mmapと、RAM/VRAM間のexpertキャッシュは維持する。
モデル重みのruntime SSDコピーも会話受付前の準備段階で作成する。推論中にモデルフォルダへの書き込みを広く許可する例外は設けない。

Mayaは `--no-start` で準備を済ませた設定から `serve/server.py` を直接起動する。起動時の再ビルド・設定touch・更新再起動を介さない。
MayaのエンジンログはRAMの `/dev/shm/colab-privacy-*/inference.log`、APIモニターとslot保存は無効。llama.cppも `--log-disable` を付ける。
全変種で標準出力・標準エラーをRAMへ保存し、継承したディスクのログFDを閉じ、デバッグ・トレース・dump設定を取り除く。起動失敗時はSSHでRAMの診断をローカルの `.runtime/bootstrap-error.log` に回収する。
Pythonのbytecode保存とCUDAのディスクキャッシュを止め、一時ファイルとXDGキャッシュは `/dev/shm/colab-privacy-*` に置く。
このRAM上の一時領域には会話内容が入り得る。OSのswapは無効化して確認し、RAMがディスクへ退避されないようにする（モデル重みのSSDオフロードとは別）。
core dumpのhard limitを0にする。Colabのパイプ式core handlerは管理側の読み取り専用設定なので変更しない。
代わりに、事前に `cc` でビルドした固定コードの `privacy-core-*.so` をPythonと子エンジンに継承し、各execの直後に `PR_SET_DUMPABLE=0` を設定する。
ライブラリは会話を含まないコード成果物としてColabのbootstrapフォルダへ保存する。実際のexecで適用を事前検査し、動的ELF64ではないエンジンや適用できないruntimeでは起動を拒否する。
推論中は `open` / `openat` / `fopen` / `truncate` と64bit版でも、RAM・許可したデバイス以外への書き込みを拒否する。元の `/proc` のmagic link経由のディスク書き込みも通常のファイルAPIでは拒否する。
この追加ライブラリは信頼する推論エンジンのログ・ダンプ抑制用であり、任意の未信頼コードが直接syscallを呼ぶ場合までの完全な隔離を保証するものではない。
Linux capabilitiesを落とし、対応kernelではLandlockの拒否ログも抑制する。

保護を適用できないruntimeでは、制限を弱めて起動せず失敗して終了処理へ進む。
`bootstrap.log` に `COLAB_PRIVACY_ENFORCED` が出てからAPIが利用可能になる。推論の診断ログはRAMにあり、起動失敗時にローカルへ回収できる。
変更は次回の `start` / `codex` から適用する。既存サーバーへ `client` で接続するだけでは適用されず、過去のログも自動削除しない。
Googleの管理基盤や独立した別プロセスまで含めた保存ゼロの保証ではない。

ローカルの検証は `python3 codex/test_colab_privacy.py`。実kernelでの書き込み拒否、子プロセスへの継承、ログFDの閉鎖とモデルの読み取り専用mmapを検証する。
モデル全体の保護付きAPI推論は未検証。2026-10-13のQwen A100では、保護下のCUDA初期化、PCIe計測、322MiBのembeddingのRAM固定とnative packの読込まで確認した（3分の起動試験ではAPI準備完了に到達せず終了）。GLMモデル本体でのGPU推論は未検証。
2026-10-13のQwenのColab A100起動では、Landlock syscallが `Function not implemented` を返して停止したため、GLMにも共通の読み取り専用マウント方式を追加した。
同日のColab CPU runtime（kernel `6.6.122+`）では最終版の `COLAB_PRIVACY_PREFLIGHT_OK` と `COLAB_PRIVACY_LIVE_CHECK_OK` を確認済み。
モデル相当ファイルの読み取り専用mmap、ディスクへの作成・上書きの拒否、RAM一時ファイルの読み書き、継承したディスクログFDの閉鎖、子exec後のcapabilitiesゼロを確認し、検証runtimeを停止した。GDriveとGPUモデル本体はこの検証で使っていない。
固定revisionのMayaを使うローカルCPUモックでは、実Landlock・capabilities制限・ログFD閉鎖を適用してChat Completions APIの往復を確認した。
ローカル環境はswap有効かつ `/dev/shm` が読み取り専用のため、このモック確認だけはホストの事前検査と一時領域を差し替えた。本番ランチャーに迂回オプションはない。
この確認で、固定revisionのMayaサーバーは `/v1/responses` に404を返す既存の互換性問題も判明した。今回の保存抑制とは別に、MayaへCodexを接続するにはResponses変換などの対応が必要。

## 実装の確認状況

`python3 codex/colab-glm/test_framework.py` で94件が通過。
再開ダウンロード、SHA-256不一致、変更済みキャッシュ、Drive保存制約、空き容量、ポート競合、A100/RAMの検査、
変種とcontextの選択（未知の値をビルド・取得前に拒否）、エンジンキャッシュキーの分離、
llama-serverのsm_80ビルドとDriveキャッシュの復元・保存、Mayaのengine restore/saveとconfigの `model_name` 上書き、
両エンジンの起動コマンド（shard・`--alias`・port・context・`--config`）、healthの `/health` + `/props` 判定（ロード中の `max_context 0` を未準備扱い）、
トンネルのポート、`--variant` のColab側起動コマンドへの伝達、同期→停止の順序と失敗時の停止、
`prepare` が全変種の取得と両エンジン生成を exec なしで終えること（単一変種指定では他変種に触れないこと、
port forward とCodex検査を飛ばすこと）までを検証した。
差分ログの重複防止、途中のUTF-8行、ログ切り詰め、正常終了と実際の失敗の区別も検証した。
実行中のColabでMaya-Lの4分割が保存済みであること、同期後のmtime変更により全量SHA-256再確認が発生すること、読取位置の増加（約99MB/s）を確認した。
Drive APIの実空きが約5.12TBだった一方、DriveFSの `disk_usage` はColabの約253GBの一時ディスクを示していた。容量誤判定をDrive APIへ修正し、同じruntimeの準備プロセスへ反映した。
MayaのCUDAビルド・pack作成はユーザーのログで確認済み。無検閲Flashの3分割も取得・SHA-256検証が完了した後、TOOLS=OFFによるllama-serverの対象欠落でビルドが失敗した。
COMMON・TOOLS・SERVERをONへ修正し、固定ソースの実際のCMake設定で「OFF時は対象なし、ON時は対象あり」をローカルで確認した（CUDAコンパイルはこの検査には含まない）。
2026-10-12のユーザーのColab実行ログで、全3変種の `GLM_PREPARE_COMPLETE` と、`full` でのビルド済みllama-serverキャッシュ再利用を確認した。
完了後の正常終了を監視側が `GLM_BOOTSTRAP_FAILED` と表示していた問題は修正済み。
**推論とCodexの実ツール呼び出しは未確認。prepare完了後も、クラウドへの保存はGDrive同期完了のログで確認する。**
