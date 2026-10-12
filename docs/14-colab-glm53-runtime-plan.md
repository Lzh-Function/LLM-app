**GLM-5.3／FlashをColab A100で使う実行計画 — 2026-10-12**

[13-colab-glm53-research.md](13-colab-glm53-research.md) の調査を受けて、このruntimeで実際に起動する4変種（通常版Flash・無検閲Flash・通常版GLM-5.3・無検閲GLM-5.3）の配布、エンジン、コンテキスト長、容量予算を決めた文書。ここでは**まだ実装していない**。決めた内容と、Colab側で先に測る必要がある項目を分けて書く。

容量・速度の判定は公開ファイルサイズと配布者の測定からの推定で、このA100でのロード・推論実測は未実施。実測値は「Colab runtimeで測る項目」にまとめた。

## 結論（4変種）

| variant | 配布（固定revision） | 量子化 | 重みサイズ | HF downloads | エンジン | context |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `flash` | [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da) `Maya-L` | gate/up IQ3_S・down IQ4_XS・一部Q5_Kの混合 | **156.32GB / 145.58GiB** | 71,734 | Project Maya | 131072 |
| `flash-uncensored` | [BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF/tree/8a31ca66485373f9d5c002f5400b2aecaff84dfe) `IQ3_XXS` | IQ3_XXS（3分割＋mmproj 1.13GB） | **134.31GB / 125.08GiB** | **484,227** | llama.cpp（補助候補としてMaya `--gguf-dir`） | 131072 |
| `full` | [autotrust/GLM-5.3-GGUF-DGX-Spark](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark/tree/dc02b40177c64563867260cd349b2fbffb38c102) `SLIM-E192` | routed experts IQ2_XXS（2.06bpw）・他Q8_0、256→192 expert | **160.76GB / 149.72GiB** | 8,963 | llama.cpp | 65536 |
| `full-uncensored` | **未確定**（下記「通常版の無検閲版が埋まらない」） | — | — | — | — | — |

downloads は2026-10-12にHF APIで取得した値。無検閲FlashをBoldingBuildsにした根拠は、確認したFlash無検閲GGUFの中でdownload数が2桁大きく、配布者が空応答・coherenceのgateを通した現行版だから。通常版FlashをMaya-Lにしたのは、Mayaの公開量子化で最大かつ作者評価がFP8の99.2%（5タスク各400問のzero-shot選択式）で、RAM resident運用の想定がこのruntimeと一致するため。

`full` を autotrust の SLIM-E192 にしたのは容量とエージェント用途の実測値のため。expertを25%構造的に削った派生で、作者のFP8段階のA/Bでは HumanEval 95.1→95.1、BFCL live 69.6→69.4、BFCL multi-turn 73.0→72.5 を保ち、C-Eval 92.0→84.7、GPQA-Diamond 86.9→83.3 を落としている。中国語知識と一部知識系を代償に、ツール使用とコードを保つ設計で、Codex運用の方向に合う。未改変の重みにこだわるなら [unsloth UD-IQ1_S](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771)（216.71GB / 201.83GiB）が代替だが、下の表で余裕が44.61GiBしか残らず、IQ1_Sの品質保持は72.6%。

## 容量予算

RAMは `MemTotal 175,166,472 KiB` = **167.05GiB = 179.37GB**、VRAMは80GB = **79.39GiB**。合計 **246.44GiB**。これはColabのこのruntimeで記録された値で、今回のマシンではなくColab側の値として使っている。

| variant | 重み | KV cache | 重み以外の残り（KV控除前） |
| --- | ---: | ---: | ---: |
| `flash` @131072 | 145.58GiB | 1.78GiB（fp16）／1.13GiB（`STRATA_GLM_KV_INT8=1`） | 100.86GiB |
| `flash-uncensored` @131072 | 125.08GiB | 同上（glm5-next） | 121.36GiB |
| `full` @65536 | 149.72GiB | 4.8GiB | 96.72GiB |
| `full` @131072 | 149.72GiB | 9.6GiB | 96.72GiB |
| `full`（UD-IQ1_S代替）@65536 | 201.83GiB | 4.8GiB | 44.61GiB |

KV cache の係数は、FlashがMayaの報告で128K状態が1.78GB（fp16）・1.13GB（int8）、GLM-5.3が配布者READMEの「16Kトークンあたり約1.2GiB」（圧縮MLA/DSA cache、約75KiB/token）。CUDA context、compute buffer、prefill buffer、OS、SSHトンネルは上の「残り」から払う。

**同時起動はできない。** `flash`（145.58GiB）と `flash-uncensored`（125.08GiB）の合計は270.66GiBで、246.44GiBを超える。4変種は**1つずつ起動**し、切り替えは停止→起動で、重みはDriveに既にある前提。Qwenのruntimeと同時に走らせない。MayaのRAM tierはピン留めされたメモリで、空きRAMから自動計算されるため、他プロセスに食られると tier が小さくなる。

## 速度の見通し

エージェント運用で効くのは decode 速度と prefill 速度で、両方とも「1トークンあたりの重み読み出し量」と、それがVRAMから来るかRAM／PCIe越しかに決まる。

| | active params | 1トークンの重み読み出し | 出典 |
| --- | ---: | ---: | --- |
| GLM-5.3-Flash | 18B | 約7.2〜7.7GB | 量子化bit数からの計算（IQ3級） |
| GLM-5.3（SLIM-E192） | 40B | 約25GB | 配布者README（decodeはmemory-bandwidth bound） |

通常版はFlashの約3.4倍の重みトラフィックで、A100のHBM2e（80GBで約2TB/s）に全部載らない分はPCIe越しになる。Colab A100のPCIe世代はこのruntimeで未測定なので、ここが読めない最大の要因。

公開されている近縁の測定値:

| 構成 | 結果 |
| --- | --- |
| RTX 4090 24GB + 192GB RAM、Maya-L、128K | decode 16.4 tok/s、prefill 1371 tok/s |
| RTX 5090 32GB + 96GB RAM、Maya-M | decode 30+ tok/s、prefill 約2000 tok/s |
| B200 180GB にSLIM-E192を常駐 | decode 42 tok/s（単一）、pp512/pp2048 730〜760 tok/s |

VRAM 80GBは4090の24GBより大きく、MayaはVRAMに載らないexpertをRAM tierにピン留めして使うため、Maya-Lは16.4 tok/sより速い方向。ただしRAM帯域・PCIe世代・Colabの割当が不明なので、**このA100の予測値は出さない**。まず実測する。

MTP（推論加速）の有無は配布で違う。GGUFヘッダとテンソル名を実ファイルから確認した。

| 配布 | arch | block_count | MTP（`nextn`） |
| --- | --- | ---: | --- |
| BoldingBuilds IQ3_XXS | `glm5-next` | 46 | あり（`blk.45.nextn.*`） |
| unsloth GLM-5.3 UD-IQ1_S | `glm-dsa` | 79 | あり（`blk.78.nextn.*`） |
| autotrust SLIM-E192 | `glm-dsa` | 78 | **なし**（blk.0〜77のみ） |

SLIM-E192はMTPブロックごと落ちている。llama.cpp側でspeculative decodeの加速が効かない分、実測でFlashとの差が開く可能性がある。llama.cppはFlashのMTPを [PR #29928](https://github.com/ggml-org/llama.cpp/pull/29928)（2026-10-07 merge）、`glm-dsa` 系を [#19460](https://github.com/ggml-org/llama.cpp/pull/19460)／[#24770](https://github.com/ggml-org/llama.cpp/pull/24770)／[#25980](https://github.com/ggml-org/llama.cpp/pull/25980)／[#26474](https://github.com/ggml-org/llama.cpp/pull/26474) でサポートしている。

## コンテキスト長のスイートスポット

Flashは **131072**。KV cacheが1.13〜1.78GiBしか食わず、Mayaの選択肢（`CONTEXTS = [8192, 32768, 65536, 131072]`）の上限がここ。モデル仕様は1,048,576だが、1Mをこのruntimeで快適に常駐できるとは扱わない。Codexのツール結果が積み上がるので、65536では途中で切り詰めやすい。

通常版は **65536** を既定にする。KV 4.8GiBで、prefill bufferとcompute bufferの余裕を残せる。131072（KV 9.6GiB）も容量的には狙えるが、まず65536で速度・安定を確認してから上げる。

## 通常版GLM-5.3の無検閲版が埋まらない

確認した中で最も小さい通常版無検閲GGUFでも [softwareweaver Q2_K 315.90GB / 294.20GiB](https://huggingface.co/softwareweaver/GLM-5.3-Uncensored-GGUF)、[Infatoshi EXL3 3.0bpw 292.65GB / 272.55GiB](https://huggingface.co/Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw) で、246.44GiBに常駐しない。REAP等のexpert pruningを適用した通常版GGUFも探したが見つからない（Flashの [REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) と、通常版はMLX配布のみ）。HFのdownload数で上位を取れる配布に限定すると、`full-uncensored` はこのruntimeでは成立しない。

選択肢は3つ。

1. **`flash-uncensored` を無検閲版の答えにする。** 容量・実績・MTPのすべてで最も素直。無検閲の通常版を使う理由が「744Bの知識量」なら、次項とのトレードオフになる。
2. **LoRAを `full` に適用する。** [ChuckMcSneed/GLM-5.3-UNCENSORED-LORA-GGUF](https://huggingface.co/ChuckMcSneed/GLM-5.3-UNCENSORED-LORA-GGUF)（1.34GB、274 downloads）は dealignai の GLM-5.3-UNCENSORED-FP8 から抽出したLoRAで、作者自身が "It's not very good." と書いている。baseは JANGQ-AI/GLM-5.3-FP8 で、IQ2_XXS＋expert pruning済みのSLIM-E192に当てた品質は未検証。llama.cppのLoRAが `glm-dsa` のMoEで効くかも未確認。実験枠としては置けるが、既定にはしない。
3. **240GB以下の通常版無検閲配布が出るまで待つ。** HFのdownload数条件を満たす配布が現れた時点でmanifestに追加する形にしておけば、実装は変わらない。

実装は `full-uncensored` を variant として予約だけしておき、未選択のうちに起動しようとしたら明示的にエラーにする運用を想定する。

## エンジンと起動設定

### Project Maya（`flash`）

[mw00/project-maya](https://github.com/mw00/project-maya) v1.1.1、commit `eb9b7ec88d6f7802f15b218b551550f749c7a94a`。A100のsm_80対応、OpenAI／Anthropic APIを `:8080`、tool callのstream対応。

```bash
STRATA_GLM_RAM_RESIDENT=1 STRATA_GLM_KV_INT8=1 STRATA_GLM_RAM_HEADROOM_GB=16 \
  python3 maya.py --yes --model Maya-L --context 131072 --no-vision \
  --models-dir /content/drive/MyDrive/Strata-data/glm53/models --port 18080 --host 127.0.0.1
```

- `STRATA_GLM_RAM_RESIDENT=1` はVRAMに載らない全expertをRAM tierに置き、収まらなければ起動に失敗する設定。推論中にディスク／Driveへexpertを読みに行かない構成にできる。[SETTINGS.md](https://github.com/mw00/project-maya/blob/eb9b7ec88d6f7802f15b218b551550f749c7a94a/docs/SETTINGS.md)
- `STRATA_GLM_RAM_HEADROOM_GB` の既定は6GB。ColabはOS・SSHトンネル・CUDA host側が同じRAMを共有するので16GBへ上げる。RAM tierはピン留めでスワップできない。
- `STRATA_GLM_PROMOTE_MIN=6` は RAM_RESIDENT と組み合わせた設定で、1回のrouteでしか使わないexpertをVRAMへ昇格させる churn を止めた例（4090 D + 128GBでdecode 1.9倍）がある。A100では未検証なので、既定で入れず比較して決める。
- drafting は既定が `--draft mtp`。`dflash2` は A100以上で単一GPU +19% の報告があるが **CC BY-NC-ND** なので、配布物としてDriveに置く場合はライセンスを確認する。
- `--no-vision` で画像エンコーダー（mmproj 1.14GB）とビルドを省略。

### llama.cpp（`flash-uncensored`・`full`）

`glm5-next` は [PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773)（2026-09-30 merge）で本体に入り、`glm-dsa` も本体側。分割GGUFは最初のshardを渡す。

```bash
llama-server -m models/bolding-iq3_xxs/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 -fa on -c 131072 -np 1 --host 127.0.0.1 --port 18080
```

`full` は `-c 65536`。MTPテンソルを含む配布ではspeculative decodeが使えるが、フラグ名は導入するllama.cppのビルドで確認する（#29928以降で変更されている）。配布者推奨のサンプリングは `--temp 0.7 --min-p 0.05`。2bit級ではgreedyで長い連鎖ループに入ることがあるため。

`flash-uncensored` をMayaで動かす経路（`--gguf-dir`）は「動くがMayaは計測していない」と作者が明記しているexperimental扱い。まずllama.cppで測り、遅い場合にMaya側を試す。

## Colab runtimeで測る項目

**このマシンで測ったRAM／VRAMはColabの値にならない。** 以下はColabのruntime内で実行し、結果を `docs/` に保存する。

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
awk '/MemTotal|MemAvailable|SwapTotal/ {print}' /proc/meminfo
lspci -vv 2>/dev/null | grep -i -A2 'LnkCap\|LnkSta' | head -20   # PCIe世代・幅
df -h / /content /tmp; lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
free -g; nproc; lscpu | grep -E 'Model name|NUMA|Socket'
```

Driveの実測（初回ダウンロードとロード時間に直結する）:

```bash
# 1) HFからのダウンロード帯域（1GBだけ）
curl -L -o /tmp/hf.bin --max-filesize 1073741824 \
  https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF/resolve/main/IQ2_XXS/GLM-5.3-Flash-Uncensored-IQ2_XXS-00001-of-00003.gguf
# 2) Drive上の既存GGUFの順次読み
dd if="/content/drive/MyDrive/Strata-data/models/IQ3_S/..." of=/dev/null bs=16M count=256 iflag=direct 2>&1 | tail -1
```

判定基準:

- Driveの順次読みが100MB/s未満だと、Maya-L 145.58GiBのロードだけで25分以上かかる。その場合は一時ディスクへコピーする価値があるか `df -h` の空きを見る（145GiB載らないならDrive直読みを受け入れる）。
- `lspci` のLnkStaがGen3 x16（約12〜13GB/s）なら、RAM tier依存のdecodeは遅くなる方向。Gen4 x16なら4090系の公開値に近い挙動を期待できる。
- `MemAvailable` が167GiBより大きく減っているruntimeなら、起動前に他セッションを止める。

## 実装方針（`colab-glm`）

[codex/colab-qwen](../codex/colab-qwen) の作りを踏襲する。`manifest.json` に `variants` と `default_variant` を置き、`--variant` で起動時に選び、未取得分だけを初回にDriveへダウンロードしてSHA-256で検証する。保存先は `/content/drive/MyDrive/Strata-data/glm53/` に変種別フォルダを分ける。

```
MyDrive/Strata-data/glm53/
├── models/maya-l/            Maya-L 4分割（145.58GiB）
├── models/bolding-iq3_xxs/   IQ3_XXS 3分割（125.08GiB）
├── models/slim-e192/         SLIM-E192 4分割（149.72GiB）
├── packs/                    Mayaのpack（GGUFから生成する派生データ）
└── engines/                  ビルド済みエンジン（commit名でキャッシュ）
```

Qwenのruntimeと同じく、保存先はDrive内のみ（SSDフォールバックなし）、空き容量検査は選択変種の未取得分だけを見る、ポート競合はビルド・ダウンロード前に失敗、失敗時はruntime stopまで行う、を維持する。エンジンがMayaとllama.cppで2系統になる分、`engines/` のキャッシュキーにエンジン名とcommitを入れる。

**全変種をDriveに置くとGGUFだけで約420GiB**（145.58 + 125.08 + 149.72）。Driveの空きを先に確認する。Qwen両変種（約167GB）とGLM 3変種を同時に保持すると約600GB級になる。

## 決まっていないこと

1. `full-uncensored` を上記3案のどれにするか。
2. `full` を SLIM-E192（pruned・MTPなし）と UD-IQ1_S（未改変・MTPあり・余裕44.61GiB）のどちらにするか。実測速度が出るまで決めきれない。
3. Mayaの `STRATA_GLM_PROMOTE_MIN` などのチューニングを既定に入れるか。A100での測定がない。
4. Colabの一時ディスク容量とDrive読み帯域。上のprobeで確定する。
