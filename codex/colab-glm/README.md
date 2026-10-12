# Colab A100でGLM-5.3をCodexから使う

GLM-5.3／GLM-5.3-Flashを、ColabのA100 high-memで実行する。
起動時に**3変種**から選び、それぞれエンジンが異なる（`flash`はProject Maya、他2つはllama.cpp）。
重み・Mayaのpack・ビルド済みエンジンはGDriveの `MyDrive/Strata-data/glm53/` に保存する。
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

`colab-cli` のログイン済みGoogleアカウントのGDriveに、**1変種あたり150GB程度以上の空き**を用意する（Qwenのruntimeと合わせてもDriveの空きは制約にならない）。
Driveマウント時のブラウザ認証では**表示された要求権限をすべて選択して許可**する。許可後はターミナルに戻ってEnterを押す。Hugging Faceのログインは不要。

```bash
cd /workspace/LLM
./colab-glm codex
```

この1コマンドが runtime作成→GDriveマウント・認証→ソース取得→**エンジンビルド（sm_80）**→GGUFダウンロード・SHA-256検証→（Mayaは）pack生成→サーバー起動→Codex起動を行う。
`flash` の初回は Maya のエンジンコンパイルと pack 生成（全テンソルのインデックスと小さなfloat重み・tokenizerで1〜2GB。expertはGGUFに残る）が加わり、`flash-uncensored`／`full` は llama-server のビルドになる。
初回準備の待機上限は既定240分、runtime全体の上限は6時間。必要なら `--startup-minutes 300 --max-hours 8` で変える。
**Codexを終了するとSSH接続とモデルを終了し、GDriveの同期完了を待ってから `colab stop -s strata-glm` を実行する。**
同期待ちは既定で最大10分。初回の約150GBの書き込みに長めの時間を確保するなら `--flush-minutes 30`。

## 2回目以降はDriveのキャッシュを使う

初回に作ったものは、変種・エンジンごとにDriveへ保存し、次回以降は再取得・再ビルドしない。

```
MyDrive/Strata-data/glm53/
├── models/Maya-L/            Maya-L 4分割と、Mayaが生成する pack/（Maya自身の配置）
├── models/bolding-iq3_xxs/   IQ3_XXS 3分割
├── models/slim-e192/         SLIM-E192 4分割
└── engines/
    ├── llama-<sig>/llama-server
    └── maya-<sig>/{strata, MAYA-BUILD.json}
```

- GGUFはサイズ・更新時刻・`.verified.json` が一致すれば再取得も全量ハッシュも省く。再検証は `--reverify`。
- `--repack` は無い。Mayaは `models/Maya-L/pack/native_experts.txt` が同じGGUFを指し、`index.txt`・`dense.bin`・tokenizerが揃っていればpackを作り直さない。
- `<sig>` は「エンジン種別 + 固定commit + nvccのパスとバージョン + libc + CPU flags」のSHA-256先頭20桁。ColabのCUDAツールキットが変わった時だけ再ビルドになる。
- 復元した Maya の `build/strata` と `MAYA-BUILD.json` は、Maya自身のstamp検査を通るため、一致すればコンパイルを飛ばす。
- ソース（llama.cpp、project-maya、Mayaが使うllama.cpp）と `.venv` はColabの一時ディスクに置く。runtimeごとに作り直す。

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
colab stop -s strata-glm
```

`/v1/responses` はMaya・llama.cppの両方が本体で提供する。変換層は置いていない。
`start`のターミナルでCtrl+Cを押しても、GDrive同期待ちを経てruntimeをstopする。
`client`の終了だけでは共有runtimeを止めないため、最後に `stop` する。`codex`モードではCodex終了時に自動でstopする。
起動失敗、SSH切断、SIGTERM/SIGHUP、時間上限でも、マウント済みなら同期を試みてから停止する。
既存の `strata-glm` があると新規起動は拒否する。

```bash
./colab-glm doctor                    # ローカル依存とCodexカタログの確認。runtimeを作成しない。
./colab-glm doctor --variant full     # 変種ごとの取得サイズとカタログを確認。
./colab-glm status                    # runtimeとAPIの状態。
./colab-glm check-drive               # GDriveだけを確認し、成功・失敗どちらでもruntimeをstop。
./colab-glm start --max-hours 2       # 初回準備を含む2時間の上限。
./colab-glm codex --context 32768 --max-hours 3
```

## 設定と保存先

- runtime起動は `colab new -s strata-glm --gpu A100 --high-mem`。A100（sm_80）とRAM 150GiB以上を検査し、違えばビルド・取得前に失敗する。
- Project Maya v1.1.1 commit `eb9b7ec88d6f7802f15b218b551550f749c7a94a`、llama.cpp commit `efe3626b119ff81033e309c3ba07a770bef2c546`、Mayaが使うllama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` を固定。
- GGUFはrevision・サイズ・SHA-256で固定。分割GGUFは最初のshardを `-m` に渡し、兄弟shardは同じフォルダから読む。
- `flash` は `STRATA_GLM_RAM_RESIDENT=1 STRATA_GLM_KV_INT8=1 STRATA_GLM_RAM_HEADROOM_GB=16`、draftingはモデル自身のMTP block、`--no-vision` で画像エンコーダー（mmproj）の取得とビルドを省略。
- llama.cppの変種は `-ngl 99 -cmoe --moe-cache-mib <32768|16384> -fa on -np 1 --jinja --temp 0.7 --min-p 0.05`。重みがVRAM 79.39GiBを超えるため、expertはRAMに置いてGPUキャッシュで補う。
- `full`（SLIM-E192）はMTPテンソルを持たないため、speculative decodeは使わない。
- 変種ごとの `model` 名・エンジン・context選択肢・ファイル一覧は `manifest.json` の `variants` に置き、`default_variant` が既定になる。
- [公式Codex provider設定](https://learn.chatgpt.com/docs/config-file/config-reference)に従いResponses APIを使用。専用モデルカタログとCLIオーバーライドを使い、通常のCodex設定は書き換えない。
- ローカル: `codex/colab-glm/.runtime/` にSSH鍵、接続ログ、専用カタログ、`drive-sync-status.json`。git対象外、権限700。
- Colab内: `/content/strata-glm/bootstrap.log` にビルド・取得・サーバーのログ。準備失敗時は末尾をローカルの `.runtime/bootstrap-error.log` に保存する。

保存先を変える場合は `--data-dir /content/drive/MyDrive/任意のフォルダ`。MyDriveの外や未マウントのディレクトリは拒否し、一時SSDへのモデル保存に切り替えない。

## 実装の確認状況

`python3 codex/colab-glm/test_framework.py` で53件が通過。
再開ダウンロード、SHA-256不一致、変更済みキャッシュ、Drive保存制約、空き容量、ポート競合、A100/RAMの検査、
変種とcontextの選択（未知の値をビルド・取得前に拒否）、エンジンキャッシュキーの分離、
llama-serverのsm_80ビルドとDriveキャッシュの復元・保存、Mayaのengine restore/saveとconfigの `model_name` 上書き、
両エンジンの起動コマンド（shard・`--alias`・port・context・`--config`）、healthの `/health` + `/props` 判定（ロード中の `max_context 0` を未準備扱い）、
トンネルのポート、`--variant` のColab側起動コマンドへの伝達、同期→停止の順序と失敗時の停止までを検証した。
**Colab上での実ダウンロード、エンジンビルド、推論とCodexの実ツール呼び出しは未実施。**
