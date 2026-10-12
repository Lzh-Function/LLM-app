# Colab A100でGLM-5.3をCodexから使う

GLM-5.3／GLM-5.3-Flashを、ColabのA100 high-memで実行する。
起動時に**3変種**から選び、それぞれエンジンが異なる（`flash`はProject Maya、他2つはllama.cpp）。
重み・Mayaのpack・ビルド済みエンジンはGDriveの `MyDrive/GLM53-data/` に保存する（Qwenの `MyDrive/Strata-data/` とは別フォルダ）。
GGUFをローカルPCやColabの `/content` へコピーしない。ソース、Python環境、ビルドの作業ファイルだけをColabの一時ディスクに置く。
SSHでAPIをローカルの `127.0.0.1:8086` へ転送する。サーバー側は専用の `127.0.0.1:18081` のみで待ち受ける。
ColabのJupyterが使う8080番との競合を避け、ビルド・取得より前にポートの空きを確認する。
選定根拠と容量・速度の予算は [docs/14-colab-glm53-runtime-plan.md](../../docs/14-colab-glm53-runtime-plan.md)。

## 変種の選択

既定は `flash`。

```bash
./colab-glm codex                              # flash（既定・Maya）
./colab-glm codex --variant flash-uncensored   # 無検閲Flash（llama.cpp）
./colab-glm start --variant full --context 32768
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
./colab-glm codex
```

この1コマンドが runtime作成→GDriveマウント・認証→GGUF取得・SHA-256検証→ソース取得・**エンジンビルド（sm_80）**→（Mayaは）pack生成→サーバー起動→Codex起動を行う。
`flash` の初回は Maya のエンジンコンパイルと pack 生成（全テンソルのインデックスと小さなfloat重み・tokenizerで1〜2GB。expertはGGUFに残る）が加わり、`flash-uncensored`／`full` は llama-server のビルドになる。
初回準備の待機上限は既定240分、runtime全体の上限は6時間。必要なら `--startup-minutes 300 --max-hours 8` で変える。
**Codexを終了するとSSH接続とモデルを終了し、GDriveの同期完了を待ってから `colab stop -s glm53-a100` を実行する。**
同期待ちは既定で最大10分。初回の約150GBの書き込みに長めの時間を確保するなら `--flush-minutes 30`。

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

- GGUFはサイズ・更新時刻・`.verified.json` が一致すれば再取得も全量ハッシュも省く。再検証は `--reverify`。
- Driveの初回同期で更新時刻が変わる場合は、次回にSHA-256を全量再確認する。ダウンロードはし直さず、検証後に記録を更新する。検証開始・20秒ごとの読取GBと速度・完了を表示する。
- `--repack` は無い。Mayaは `models/Maya-L/pack/native_experts.txt` が同じGGUFを指し、`index.txt`・`dense.bin`・tokenizerが揃っていればpackを作り直さない。
- `<sig>` は「エンジン種別 + 固定commit + nvccのパスとバージョン + libc + CPU flags」のSHA-256先頭20桁。これらの条件が変わると再ビルドになる。
- 復元した Maya の `build/strata` と `MAYA-BUILD.json` は、Maya自身のstamp検査を通るため、一致すればコンパイルを飛ばす。
- ソース（llama.cpp、project-maya、Mayaが使うllama.cpp）と `.venv` はColabの一時ディスクに置く。runtimeごとに作り直す。

## 準備だけまとめて行う

```bash
./colab-glm prepare                  # 3変種すべて: 重み取得 + 2エンジンビルド + Mayaのpack
./colab-glm prepare --variant full   # 1変種だけ。この場合のみ --context を指定できる
```

サーバーもCodexも起動しない。runtime作成→GDriveマウント→変種ごとの空き容量確認→GGUF取得・SHA-256検証→エンジンビルド→（`flash` は）pack生成までを1runtimeで済ませ、完了後にGDriveを同期して停止する。
contextは各変種の既定値（`flash`／`flash-uncensored` は131072、`full` は65536）を使い、`--context` は `--variant` を1つに絞った時だけ受け付ける。
待機上限は既定480分、runtime全体は10時間、GDrive同期待ちは30分（`prepare` 専用の既定値。`--startup-minutes`・`--max-hours`・`--flush-minutes` で変更可）。
完了は `bootstrap.log` の `GLM_PREPARE_COMPLETE ["flash", "flash-uncensored", "full"]` で分かる。以後の `start`／`codex` はキャッシュを使うだけになり、初回の待ち時間がサーバーを待つ時間から消える。
準備プロセスはこの完了行を出して正常終了する。完了行がある正常終了を `GLM_BOOTSTRAP_FAILED` と表示しない。クラウドへの保存確認は、その後の「GDriveの同期完了を確認しました。」まで待つ。

ログが更新されず、進んでいるか分からない場合は、別ターミナルで確認する。

```bash
./colab-glm progress
```

既存のSSH接続で準備プロセスと子プロセス、GGUFの読取／書込位置、直近のログを表示する。runtimeの作成や停止は行わない。
`state=D` はI/O待ち、`R` は実行中、`S` は待機。読み取り位置の増加はSHA-256検証でも起こるので、ダウンロードと区別してログを見る。
準備待ちのログが変わらないときも、30秒ごとにこの確認コマンドを案内する。
`prepare`／`start` のログは前回の読取位置から追加された行だけを表示する。以前の末尾8行を繰り返す表示は並列実行を意味しない。
既に実行中のPythonプロセスにはローカルファイルの編集だけでは修正が反映されない。

## 起動・終了を分ける

```bash
# ターミナル1: 準備完了後も、このコマンドを起動したまま使う。
./colab-glm start --variant full

# ターミナル2: 上で起動したサーバーへCodexを接続する。
./colab-glm client
# または現在のCodex/スクリプトからAPIを呼び出す。
curl http://127.0.0.1:8086/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-slim-e192","input":"日本語で一文だけ挨拶して","max_output_tokens":64,"reasoning":{"effort":"none"}}'

# セッション終了。サーバー、トンネル、Colab runtimeを終了する。
./colab-glm stop
# 緊急時は直接:
colab stop -s glm53-a100
```

`/v1/responses` はMaya・llama.cppの両方が本体で提供する。変換層は置いていない。
`start`のターミナルでCtrl+Cを押しても、GDrive同期待ちを経てruntimeをstopする。
`client`の終了だけでは共有runtimeを止めないため、最後に `stop` する。`codex`モードではCodex終了時に自動でstopする。
起動失敗、SSH切断、SIGTERM/SIGHUP、時間上限でも、マウント済みなら同期を試みてから停止する。
既存の `glm53-a100` があると新規起動は拒否する。

```bash
./colab-glm doctor                    # ローカル依存とCodexカタログの確認。runtimeを作成しない。
./colab-glm doctor --variant full     # 変種ごとの取得サイズとカタログを確認。
./colab-glm status                    # runtimeとAPIの状態。
./colab-glm progress                  # 準備中の読取位置・子プロセスを確認。新規runtimeなし。
./colab-glm check-drive               # GDriveだけを確認し、成功・失敗どちらでもruntimeをstop。
./colab-glm start --max-hours 2       # 初回準備を含む2時間の上限。
./colab-glm codex --context 32768 --max-hours 3
```

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
- Colab内: `/content/glm53-a100/bootstrap.log` にビルド・取得・サーバーのログ。準備失敗時は末尾をローカルの `.runtime/bootstrap-error.log` に保存する。

保存先を変える場合は `--data-dir /content/drive/MyDrive/任意のフォルダ`。MyDriveの外や未マウントのディレクトリは拒否し、一時SSDへのモデル保存に切り替えない。

## 実装の確認状況

`python3 codex/colab-glm/test_framework.py` で69件が通過。
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
