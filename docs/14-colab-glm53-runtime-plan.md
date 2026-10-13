**GLM-5.3／FlashをColab A100で使う実行計画 — 2026-10-12**

[13-colab-glm53-research.md](13-colab-glm53-research.md) の調査を受けて、このruntimeで実際に起動する3変種（通常版Flash・無検閲Flash・通常版GLM-5.3）の配布、エンジン、コンテキスト長、容量予算を決めた文書。決めた内容と、Colab側で先に測る必要がある項目を分けて書く。

実装は [codex/colab-glm](../codex/colab-glm)（起動方法と運用は [README](../codex/colab-glm/README.md)）。この文書は選定の根拠と容量・速度の予算のまま残し、実装と食い違った箇所（Driveの配置、ポート、packの置き場所）は下の「実装」で直している。

容量・速度の判定は公開ファイルサイズと配布者の測定からの推定で、このA100でのロード・推論実測は未実施。実測値は「Colab runtimeで測る項目」にまとめた。

## 結論（3変種）

| variant | 配布（固定revision） | 量子化 | 重みサイズ | HF downloads | エンジン | context |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `flash` | [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da) `Maya-L` | gate/up IQ3_S・down IQ4_XS・一部Q5_Kの混合 | **156.32GB / 145.58GiB** | 71,734 | Project Maya | 131072 |
| `flash-uncensored` | [BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF/tree/8a31ca66485373f9d5c002f5400b2aecaff84dfe) `IQ3_XXS` | IQ3_XXS（3分割＋mmproj 1.13GB） | **134.31GB / 125.08GiB** | **484,227** | llama.cpp（補助候補としてMaya `--gguf-dir`） | 131072 |
| `full` | [autotrust/GLM-5.3-GGUF-DGX-Spark](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark/tree/dc02b40177c64563867260cd349b2fbffb38c102) `SLIM-E192` | routed experts IQ2_XXS（2.06bpw）・他Q8_0、256→192 expert | **160.76GB / 149.72GiB** | 8,963 | llama.cpp | 65536 |

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

**同時起動はできない。** `flash`（145.58GiB）と `flash-uncensored`（125.08GiB）の合計は270.66GiBで、246.44GiBを超える。3変種は**1つずつ起動**し、切り替えは停止→起動で、切り替え先の重みはそのruntimeのSSDに無いためHFからダウンロードし直す。Qwenのruntimeと同時に走らせない。MayaのRAM tierはピン留めされたメモリで、空きRAMから自動計算されるため、他プロセスに食られると tier が小さくなる。

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

## 通常版GLM-5.3の無検閲版は作らない

確認した中で最も小さい通常版無検閲GGUFでも [softwareweaver Q2_K 315.90GB / 294.20GiB](https://huggingface.co/softwareweaver/GLM-5.3-Uncensored-GGUF)、[Infatoshi EXL3 3.0bpw 292.65GB / 272.55GiB](https://huggingface.co/Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw) で、246.44GiBに常駐しない。REAP等のexpert pruningを適用した通常版GGUFも探したが見つからない（Flashの [REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) と、通常版はMLX配布のみ）。HFのdownload数で上位を取れる配布に限定すると、`full-uncensored` はこのruntimeでは成立しない。

無検閲が必要なら `flash-uncensored` を使う。`full-uncensored` は作らない。240GB以下でdownload数上位を取れる通常版無検閲配布が出た時点で、manifestに variant を1行足せば追従できる。

## エンジンと起動設定

### Project Maya（`flash`）

[mw00/project-maya](https://github.com/mw00/project-maya) v1.1.1、commit `eb9b7ec88d6f7802f15b218b551550f749c7a94a`。A100のsm_80対応、OpenAI／Anthropic APIを既定の `:8080`（このruntimeでは `18081`）、tool callのstream対応。

```bash
STRATA_GLM_RAM_RESIDENT=1 STRATA_GLM_KV_INT8=1 STRATA_GLM_RAM_HEADROOM_GB=16 \
  python3 maya.py --yes --model Maya-L --context 131072 --no-vision \
  --models-dir /content/drive/MyDrive/GLM53-data/models --port 18081 --host 127.0.0.1
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
  -ngl 99 -cmoe --moe-cache-mib 32768 -fa on -c 131072 -np 1 \
  --host 127.0.0.1 --port 18081 --alias glm-5.3-flash-uncensored-iq3_xxs --jinja
```

Codexが使う `/v1/responses` は、llama.cppのserver本体とMayaの `serve/server.py` の両方が既に実装している。変換層は要らない。
起動完了の判定は、llama.cppが `/health` に status しか返さず model と `n_ctx` を `/props` に置くため、`/health` と `/props` の2つを見る。Mayaは `/health` にも model と max_context を持つ。

`full` は `-c 65536`。MTPテンソルを含む配布ではspeculative decodeが使えるが、フラグ名は導入するllama.cppのビルドで確認する（#29928以降で変更されている）。配布者推奨のサンプリングは `--temp 0.7 --min-p 0.05`。2bit級ではgreedyで長い連鎖ループに入ることがあるため。

**`-ngl 99` だけでは起動しない。** 両変種の重み（125.08GiB／149.72GiB）はVRAM 79.39GiBを超える。`-cmoe` は MoE expert 重みをCPU RAMに置くフラグで、`glm5-next` も `glm-dsa` もテンソル名が標準の `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` なのでそのまま効く。dense・attention・共有expertはVRAMに残る。`--moe-cache-mib` はCPU側expertのGPUキャッシュ（既定0）で、RAM帯域とPCIe往復を削るための値。`-ncmoe N`（先頭N層のexpertだけCPUへ）と合わせて、値はロード時のVRAM実測で決める。まず `--moe-cache-mib 0` で入り、空きVRAMを見て上げる。

`flash-uncensored` をMayaで動かす経路（`--gguf-dir`）は「動くがMayaは計測していない」と作者が明記しているexperimental扱い。まずllama.cppで測り、遅い場合にMaya側を試す。

### autotrust配布の「DGX Spark向け」表記について

リポジトリ名・ファイル名（`GLM-5.3-Q2-DGX-Spark-0000N-of-00004.gguf`）・READMEの肩書き（"packaged for DGX Spark owners"）にSparkが出る。ただし配布物はハードウェア非依存のGGUFで、Spark向けなのはREADMEのビルド手順と2台構成のレシピだけ。実ファイルをHTTP Rangeで直接読んでヘッダを確認した（revision `dc02b40`）。

| 項目 | 実測値 |
| --- | --- |
| `general.architecture` | `glm-dsa` |
| `general.basename` / `general.name` | `GLM-5.3-SLIM-E192` |
| `block_count` / `expert_count` / `leading_dense_block_count` | 78 / 192 / 3 |
| `context_length` | 1048576 |
| テンソル型（shard 1） | Q8_0 ×262、F32 ×210、IQ2_XXS ×60 |
| テンソル型（shard 4） | Q8_0 ×136、F32 ×109、IQ2_XXS ×37 |

テンソル型は全て標準ggml型で、独自フォーマットは無い。README自身の "runs on upstream llama.cpp"、"Standard llama.cpp glm-dsa layout (same tensor names, MLA `attn_k_b`/`attn_v_b` split and metadata as llama.cpp's own GLM 5.2/5.3 conversions)" と一致する。

Spark固有の要素と、A100での扱い:

- `-DCMAKE_CUDA_ARCHITECTURES=121a-real` はGB10のarchフラグ。A100では `sm_80` でビルドする。
- 2台Sparkの `rpc-server` / `--rpc` / `--tensor-split 1,1` は128GB×2に載せるための構成で、A100＋RAM167GiBでは不要。
- llama.cpp側にSpark専用パスはあるが、A100ではコンパイル時に消えるだけ。[mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu) のL2プリフェッチは `#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK`（`common.cuh` で 1210）で囲まれ、コメントにも "only enabled on DGX Spark… On the higher-bandwidth parts the kernel has little exposed latency left to hide and the extra requests cost more than they save"。`cc == GGML_CUDA_CC_DGX_SPARK` のne11上限チューニングも、他archは汎用デフォルトに落ちる。
- IQ2_XXSはMMVQで全arch対応（`get_vec_dot_q_cuda()` に `case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;`）。capabilityゲート（`fast_bf16_hardware_available`、`bf16_mma_hardware_available`、`ampere_mma_available`、`cp_async_available`）は `GGML_CUDA_CC_AMPERE = 800` 以上で成立するのでA100は通る。
- `glm-dsa` はupstream本体にマージ済み（[#19460](https://github.com/ggml-org/llama.cpp/pull/19460) アーキ追加、[#24770](https://github.com/ggml-org/llama.cpp/pull/24770) indexerテンソル任意化、[#25980](https://github.com/ggml-org/llama.cpp/pull/25980) NextN/MTP spec-decode、[#26474](https://github.com/ggml-org/llama.cpp/pull/26474) indexer cache）。`src/models/glm-dsa.cpp` は標準の `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` を使うので、`-cmoe` / `-ncmoe` / `--moe-cache-mib` がそのまま効く。
- A100でGLM-5系をGGUF＋llama.cppで動かす前例として [jimmylegendary/glm5-llamacpp-a100](https://github.com/jimmylegendary/glm5-llamacpp-a100) がある（8×A100＋1TB RAM前提、"upstream vLLM/SGLang sparse-MLA support is unreliable" な環境向け。方向性としては参考になるが、この構成とは桁が違う）。

残るリスクは2つ。作者の実測は 2×DGX Spark / 1×DGX Spark（paging、低一桁t/s） / B200・GB200 / Mac Studio 256GB の4構成で、**A100やHopperのデータは無い**。そしてREADMEの「180GB GPUに常駐」はA100 80GBでは成立しないため、expertをRAMに置いてPCIe越しに読む構成になる。

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

## 実装（`colab-glm`）

[codex/colab-qwen](../codex/colab-qwen) の作りを踏襲する。`manifest.json` に `variants` と `default_variant` を置き、`--variant` で起動時に選び、未取得分だけを初回にダウンロードしてSHA-256で検証する。ファイルごとの `fetch` が取得先を決め、GGUFのshard（`huggingface`）はHFからruntime SSDへ直接、それ以外の細かいパーツ（`drive`）はQwenの `MyDrive/Strata-data/` とは別の `/content/drive/MyDrive/GLM53-data/` に置く。
エンジンはMayaとllama.cppの2系統で、ビルド済み成果物もDriveにキャッシュする。

```
MyDrive/GLM53-data/
├── models/Maya-L/pack/       Mayaが生成したpack（1〜2GB）
└── engines/
    ├── llama-<sig>/llama-server
    ├── maya-<sig>/{strata, MAYA-BUILD.json}
    └── maya-packs.json       packの保存先索引
```

`models/Maya-L/` はMaya自身の配置（`--models-dir` の下にモデル名のフォルダ）で、**packはGGUFと同じフォルダに `pack/` として生成される**。Qwenのような `packs/` フォルダは作らない。packは全テンソルのインデックスと小さなfloat重み・tokenizer（合計1〜2GB）で、expertはGGUFに残る。Mayaは `pack/native_experts.txt` が同じGGUFを指し、`PACK_FILES` が揃っていれば作り直さない。
`<sig>` はエンジン種別・固定commit・nvccのパスとバージョン・libc・CPU flagsのSHA-256先頭20桁。これらの条件が変わると再ビルドになる。Maya側は `build/strata` と `MAYA-BUILD.json` をそのまま復元し、Maya自身のstamp検査（source hashとarchs）を通す。
llama.cppは `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80` で `llama-server` だけを作り、Mayaは `maya.py --yes --no-start` でセットアップした後に `--config` で起動する2段構えにする（`--no-start` だけだとdashboard起動を飛ばせないため）。
永続保存先はDrive内のみ、ポート競合はビルド・ダウンロード前に失敗、失敗時はruntime stopまで行う、を維持する。2026-10-13の変更で、`start` / `codex` はGGUFをHFからruntime SSDへ直接ダウンロードするようになった。DriveからSSDへコピーする経路と、容量不足時にDriveの重みを直接読むフォールバックは廃止している（DriveFSの実測73MB/sに対しHF単一ストリームは154.7MB/s、4並列Rangeは294.2MB/s）。詳細は [colab-glm README](../codex/colab-glm/README.md#モデル本体のhugging-face直接ダウンロード)。
`prepare` はサーバーを起動せず、`--variant` を省略すると3変種分の取得と2エンジンのビルド・Mayaのpack生成を1runtimeで済ませ、エンジンとpackをDriveへキャッシュする。変種ごとにSSD上の本体を解放するため、420GiBを1枚のSSDに載せる必要は無い。
待機上限は `prepare` だけ480分、runtime 10時間、GDrive同期待ち30分の既定値。contextは各変種の既定値を使い、`--context` は単一変種の指定時に限る。

**GGUFはDriveに置かない**ため、Drive側の必要空き量はエンジン成果物とpackだけで済む。代わりにruntime SSD（約253GB）が本体・pack・DriveFSキャッシュを共有するので、取得前に `check_runtime_space` で容量を検査する。

容量の確認にはDrive APIの `about.storageQuota` を使う。2026-10-12の稼働runtimeでは、
DriveFSマウントの `disk_usage` はColabの一時ディスクと同じ約253GBの総量を返したため、
これをクラウド側の残容量としてモデル全体のサイズと比較すると、容量不足を誤判定した。
実際のクラウド空きは約5.12TBだった。検証JSONに記録したmtimeも初回のDrive同期で変わり、
次回にGGUFの全量SHA-256再確認が走った（約99MB/s）。モデル再取得とは別工程なので進捗を表示する。

## 決まっていないこと

1. ~~`full` を SLIM-E192（pruned・MTPなし）と UD-IQ1_S（未改変・MTPあり・余裕44.61GiB）のどちらにするか。~~ **SLIM-E192に決めた。** UD-IQ1_Sは余裕44.61GiBでprefill bufferとcompute bufferが置けず、IQ1_Sの品質保持は72.6%。A100の実測が出たら、速度が低一桁ならUD-IQ1_S側を `--gguf-dir` 抜きで別variantとして足す余地は残す。
2. Mayaの `STRATA_GLM_PROMOTE_MIN` などのチューニングを既定に入れるか。A100での測定がない。
3. Colabの一時ディスク容量とDrive読み帯域。上のprobeで確定する。
