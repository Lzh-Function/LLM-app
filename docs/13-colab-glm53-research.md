**GLM-5.3／Flashの通常版・Uncensored／Abliterated版をColab A100で動かす調査 — 2026-10-12**

**Uncensored／Abliteratedを優先する場合の候補を追加して結論を更新した。** 公開利用指標を重視する第一候補はOrcaRouter系の **BoldingBuilds IQ3_XXS（134.31GB）＋llama.cpp**、量子化精度を上げる候補は **Huihui UD-Q4_K_XL（199.71GB）＋llama.cpp**。Mayaの導入・運用を優先するなら **Maya-M-Derisked（116.00GB）** が選べるが、公開直後の実験版で利用実績・評価は小さい。

前の **Maya-L（156.32GB）** は通常重みの量子化で、無検閲版ではない。また、通常版GLM-5.3の **UD-IQ2_M（238.58GB）** という容量上限を、そのまま人気Uncensored配布の上限と読んではいけない。今回確認した通常版Uncensoredの代表的なGGUFは最小でも315.90GBあり、このruntimeのRAM／VRAMへ常駐できない。可否は公開実装と実ファイルサイズからの推定で、Colabでのロード・推論実測ではない。

モデルのファイルサイズは配布元Hugging Face APIの各分割を合算して確認した。固定revision、bytes、SHA-256と換算値は [glm53-model-sizes.json](glm53-model-sizes.json) に保存した。モデル本体の取得、runtime作成、既存Qwen構成の変更は行っていない。

追加した派生モデル20リポジトリのdownload数、likes、gated状態、親モデル、固定revision、分割サイズは [glm53-uncensored-models.json](glm53-uncensored-models.json) に保存した。調査中にもdownload数が更新されるため、以下は2026-10-12に各APIで取得した値である。

**「シェア率」の扱いと利用規模**

実際の稼働台数や利用者数を公開データだけで確定できないので、市場シェアの割合は計算しない。HFの月間download数を露出・利用規模の代理指標にし、likesと配布時期も併記する。同じ重みの親・再量子化・ミラーは別の利用者集団とは限らず、合算しない。GGUFは全GGUFファイルへのアクセスを数え、分割や形式の違いで数え方も変わるため、FP8とGGUFを厳密に順位比較しない。[HFの集計仕様](https://huggingface.co/docs/hub/models-download-stats)

| 配布・系統 | 月間downloads | likes | このruntimeでの位置付け |
| --- | ---: | ---: | --- |
| [BoldingBuilds / OrcaRouter Flash GGUF](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF) | **484,227** | 23 | 確認したFlash GGUFでdownload指標が大きい。134.31GB以下の量子化があり第一候補 |
| [OrcaRouter Flash FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) | **221,675** | 289 | 同系統の親モデル。328.34GBで常駐不可。gated |
| [dealignai Flash UNCENSORED FP8](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8) | **40,047** | 151 | 別の大きな派生系統。元FP8をこのruntimeへ常駐する構成にはしない |
| [Huihui Flash abliterated GGUF](https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF) | **27,094** | **229** | 多段階の量子化。4bit級・高精度寄りの候補 |
| [AliceThirty Flash UNCENSORED GGUF](https://huggingface.co/AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF) | 22,918 | 18 | dealignai由来。metadataが旧 `glm5next` のためエンジン互換性の追加確認が必要 |
| [OrcaRouter Flash GGUF](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-GGUF) | 18,266 | 122 | 同じOrcaRouter系。Q4_K_M 192.97GB。gated |
| [DogContext Flash Q2 DS4](https://huggingface.co/DogContext/GLM-5.3-Flash-Uncensored-Q2-ds4) | 16,970 | 14 | 96.51GBだが配布者はDS4／Metal専用と明記。今回のA100導入候補にはしない |
| [GCSA-AiLab Flash Uncensored RCO-GSQ](https://huggingface.co/GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF) | 8,152 | 7 | LoRA v2をmergeした137.07GBの混合量子化。追加比較候補 |
| [softwareweaver 通常版Uncensored GGUF](https://huggingface.co/softwareweaver/GLM-5.3-Uncensored-GGUF) | 2,830 | 13 | 最小Q2_Kが315.90GB。常駐不可 |
| [Infatoshi 通常版Uncensored EXL3 3.0bpw](https://huggingface.co/Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw) | 2,656 | **325** | 292.65GB。常駐不可。作者は8×A100 40GBで検証 |
| [dealignai 通常版UNCENSORED FP8](https://huggingface.co/dealignai/GLM-5.3-UNCENSORED-FP8) | 2,578 | 137 | 上の通常版GGUF／EXL3の親系統。FP8は容量不足 |
| [audnai penclaw通常版abliterated](https://huggingface.co/audnai/penclaw-GLM-5.3-abliterated) | 1,952 | **388** | likesが大きいが、今回の常駐予算に合う公開低bit配布を確認できず。manual gateあり |
| [Maya-M-Derisked](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF) | **737** | 2 | 約116GB、Maya公式セットアップの選択肢。人気より運用適合性で選ぶ候補 |
| [alesha-pro Maya-M abliteration vector版](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-Maya-M-GGUF) | 678 | 1 | 約116GB。未改変のMaya-M＋実行時vector＋エンジンpatchという別方式 |

系統の根拠は各配布者のmodel cardとHF APIのbase_model。一般Uncensoredとは別枠で、[dealignai GLM-5.3-CYBERSECURITY-FP8](https://huggingface.co/dealignai/GLM-5.3-CYBERSECURITY-FP8)は100,586 downloads／660 likesがある。ただし専門用途の配布で、FP8だけで約755.63GBあり今回の容量範囲外。

**実行する量子化・変更内容・互換性をまとめた候補**

| 優先条件 | 選ぶ配布・量子化 | 実サイズ | エンジンと根拠 |
| --- | --- | ---: | --- |
| 人気と公開検証を重視 | **BoldingBuilds / OrcaRouter IQ3_XXS** | **134.31GB** | current llama.cpp。公開4段階では最大サイズ。配布者の小規模coherence gateを通過 |
| 人気系統で少し軽く | BoldingBuilds / OrcaRouter IQ2_M | 121.62GB | llama.cpp。2026-10-08追加、coherence gate通過 |
| 4bit級の精度を優先 | **Huihui UD-Q4_K_XL** | **199.71GB** | llama.cpp。容量上は余裕を残す常駐候補 |
| 高精度化の容量上限を試す | Huihui UD-Q5_K_XL | 240.31GB | llama.cpp。短context・単一リクエストの実験候補 |
| 4bit級を少し軽く | Huihui UD-IQ4_XS | 156.82GB | llama.cpp。Maya-Lとは重みもレシピも違う |
| Mayaで簡便に運用 | **Maya-M-Derisked** | **116.00GB** | Mayaに `--model Maya-M-Derisked` の選択肢あり。resident構成を狙える |
| merged LoRAを比較 | GCSA-AiLab RCO-GSQ Q4 | 137.07GB | llama.cpp。Q4という名前だが実際は複数dtypeの約3.5bit級。adapterは再適用しない |
| OrcaRouterのEXL3を比較 | MikeRoz 3.05／4.05bpw | 126.11／165.07GB | ExLlamaV3＋CPU offload候補。download数865／871で主力GGUFより小さい |

サイズ・固定revisionは追加JSONで追跡できる。[BoldingBuilds](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF)、[Huihui](https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF)、[Maya-M-Derisked](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF)、[GCSA技術説明](https://huggingface.co/GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF/blob/main/docs/technical-en.md)、[MikeRoz](https://huggingface.co/MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3)。容量判定は前回と同じRAM／VRAM予算からの推定で、ロード・速度の実測は未実施。

BoldingBuildsはOrcaRouterによる重み編集の再量子化。現行配布はattention・shared expert等を高精度に保った再作成版で、過去の低bit版には空応答・coherenceの問題があった。最初に試すなら現行IQ3_XXSまたはIQ2_Mを選び、古い同名ファイルを流用しない。coherence gateは小規模なコード・数学・指示追従テストであり、独立したCodex実運用評価ではない。また、空の最終回答を「拒否しなかった」と数える評価の問題を配布者自身が報告している。Uncensoredの見出しや低いrefusal数だけを品質保証にしない。[配布者の検証・revision履歴](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF)

HuihuiはUnsloth GGUF由来で、作者は0-basedの15〜35層を編集し、他の層とexpertは未編集と明記。OrcaRouter系と同じ編集範囲ではない。「Uncensored」のラベルから全モード・全入力で拒否しないとは判断しない。[編集範囲](https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF)

Maya-M-DeriskedはBlackfrost_AIのDWMによる直接の重み編集。Maya-Mのdtype・形状・分割・総サイズを保ち、MTPは未改変。配布者は128KのロードとAPI smoke testを確認しているが、包括的なclean-versus-edited品質評価は未完了。Mayaセットアップへの組込みは確認したが、このColabでの動作は未実測。[配布仕様](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF)、[Mayaの選択肢](https://github.com/mw00/project-maya/blob/eb9b7ec88d6f7802f15b218b551550f749c7a94a/maya.py)

alesha-pro版は未改変のMaya-Mに2.2MBの実行時vectorを加えるもので、patch済みMayaが必要。作者の測定はMaya-M・4×RTX3090で、通常のMaya-Lをそのまま無検閲版として扱える根拠にはしない。Maya-Lを無検閲優先候補へ置き換えるには、別の派生配布やpatch・評価の確認が必要。[配布者の方式と測定](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-Maya-M-GGUF)

GCSAはLoRA v2 merge済みの完全モデル。参考評価はNVFP4モデル＋adapterで行われ、公開GGUFの直接評価ではない。容量的に合っても、その評価を137GB GGUFの実測結果として扱わない。[技術説明](https://huggingface.co/GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF/blob/main/docs/technical-en.md)

OrcaRouterの親FP8と公式GGUFはAPI上 `gated=auto`、匿名でのREADME取得も401だった。これらを直接取得する場合はHFログインと配布ページのアクセス承諾が必要。今回のBoldingBuilds、Huihui、Maya-M-Deriskedのmetadata・model cardは匿名で取得できた。調査のためにgateを回避したり、認証情報を追加したりはしていない。

**通常版GLM-5.3のUncensoredは、今回の人気配布では常駐が難しい**

softwareweaverの最小Q2_Kは **315.90GB / 294.20GiB**、InfatoshiのEXL3 3.0bpwは **292.65GB / 272.55GiB**。前者のファイル名は `Q2_K-BF16-MTP` で、一律2bitのサイズと推定してはいけない。drowzeysの2.75bpw版もmetadata合計約276.21GBで予算を上回る。既存の人気配布を選ぶ条件では、通常版Uncensoredを約239GBで起動できるという結論にはならない。[softwareweaver](https://huggingface.co/softwareweaver/GLM-5.3-Uncensored-GGUF)、[Infatoshi](https://huggingface.co/Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw)

通常重みのUD-IQ2_Mを使う容量評価は後段に比較用として残す。無検閲の重みをそのサイズへ自力量子化する作業やadapterの追加は別の検証であり、今回確認した完成済み人気配布とは区別する。通常版のLoRA抽出版には配布者自身が品質を低く評価しているものもあり、軽い追加ファイルだけで同等の完成版になるとは扱わない。

**容量の前提**

前のQwen準備で確認したMemTotalは175,166,472 KiB。これは **167.05GiB = 179.37GB** であり、167GBと同じではない。VRAMは今回指定された80GBを確保できる前提。前回の実機確認にはVRAM容量の記録がないので、次回起動時に `nvidia-smi` で確認する。

167GBと80GBを十進数で足すと247GB、実測RAMを換算して80GBを加えると約259GB。ただしOS、CUDA、KV cache、KDA state、prefill buffers、MTP、ロード時の一時領域が必要で、全量を重みに使えない。RAMとVRAMに同じexpertを複製する実装やCPU用に重みを展開する実装では、ファイルサイズより多くの容量を消費する。ここでの「常駐」は全重みをRAM／VRAMに収め、推論中のモデル重みのディスク読み出しを避ける方式を指す。

**Project Mayaの通常版・容量比較**

[Project Maya](https://github.com/mw00/project-maya)はStrataを基にGLM-5.3-Flash向けに作られたエンジン・サーバー・量子化ツール。今回確認したソースはv1.1.1、commit `eb9b7ec88d6f7802f15b218b551550f749c7a94a`。A100のsm_80に対応し、GPU expert cache、RAM tier、ディスクtierを使う。通常版GLM-5.3向けではない。

| Mayaのモデル | 重みサイズ | このruntimeでの容量評価 |
| --- | ---: | --- |
| Maya-S24 | 94.70GB | 十分収まる。小VRAM向けなので80GB GPUでは優先度を下げる |
| Maya-S | 96.47GB | 速度重視の比較候補 |
| Maya-M | 116.00GB | 速度と精度の比較候補。RAM／VRAM常駐が十分狙える |
| **Maya-L** | **156.32GB / 145.58GiB** | **Mayaの公開量子化では最大。今回の品質優先候補** |

サイズは[配布元](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da)の分割合計。Maya-Lはgate/upをIQ3_S、downをIQ4_XS、一部をQ5_Kにし、attention等にも別の精度を使う混合量子化。名前だけから「一律3bit」とは解釈しない。[量子化レシピと公開評価](https://github.com/mw00/project-maya/blob/eb9b7ec88d6f7802f15b218b551550f749c7a94a/bench/results/MAYA-L.md)

Mayaには `--glm-ram-resident` または `STRATA_GLM_RAM_RESIDENT=1` がある。GPU外の全expertをRAMへ置き、収まらなければ起動時に失敗させる。[設定説明](https://github.com/mw00/project-maya/blob/eb9b7ec88d6f7802f15b218b551550f749c7a94a/docs/SETTINGS.md)と[実装](https://github.com/mw00/project-maya/blob/eb9b7ec88d6f7802f15b218b551550f749c7a94a/src/core/glm_fast_path.cu)を確認し、warm-up後に全expertの配置を検査する処理も確認した。`STRATA_GLM_WARM=0`にはしない。

最初の測定は1リクエスト・32Kまたは128K context・テキストのみ・resident有効が適切。resident確認ログとdisk readカウンターを見て、prefill／decodeの両方でDriveからのexpert読み出しが残らないかを確認する。GDriveのマウント環境での起動は未検証。ソースには直接I/Oを使えないファイルシステムでbuffered readへ戻る処理があるが、これだけでDrive上の全操作の互換性は保証できない。

Maya-Lの作者評価は5つのzero-shot選択式タスク各400問で、FP8モデルの平均スコアの99.2%を維持。これはCodexでの長時間コーディングやツール使用の性能保証とは異なる。同じ公開評価のMaya-Mは97.9%。Maya-LとMは同じheld-out文章で比較されているが、他社量子化の評価と直接順位付けしない。[Maya-L評価](https://github.com/mw00/project-maya/blob/eb9b7ec88d6f7802f15b218b551550f749c7a94a/bench/results/MAYA-L.md)

速度の参考値として作者が集めた報告には、RTX 5090 32GB＋96GB RAMでMaya-Mが30+ tok/s、RTX 4090 24GB＋192GB RAMでMaya-Lが16.4 tok/s（128K設定）がある。ただしCPU、PCIe、RAM帯域、実際の入力長が異なり、このA100 runtimeの予測値としては使えない。MayaがCPUとPCIeの処理分担を校正するのもこのため。[公開測定一覧](https://github.com/mw00/project-maya/blob/eb9b7ec88d6f7802f15b218b551550f749c7a94a/README.md)

**一般GGUFで狙える上限**

| モデル | 量子化 | 重みサイズ | 常駐構成の評価（推定） |
| --- | --- | ---: | --- |
| GLM-5.3-Flash | UD-IQ3_XXS | 120.37GB | 容量に余裕あり |
| GLM-5.3-Flash | UD-IQ4_XS | 156.82GB | 有力。Maya-Lとは別レシピ |
| GLM-5.3-Flash | **UD-Q4_K_XL** | **199.71GB** | **容量に余裕を残す高精度候補** |
| GLM-5.3-Flash | UD-Q5_K_XL | 240.31GB | 短context・1リクエストで検証する上限候補 |
| GLM-5.3-Flash | UD-Q6_K_XL | 291.83GB | RAM＋VRAM常駐の範囲外 |
| GLM-5.3 | UD-IQ1_S | 216.72GB | 常駐候補。低bit量子化の品質を評価する必要あり |
| GLM-5.3 | UD-IQ1_M | 228.49GB | 実測RAM容量なら検討可能 |
| GLM-5.3 | **UD-IQ2_M** | **238.58GB** | **通常版の常駐上限候補。余裕は小さい** |
| GLM-5.3 | UD-Q2_K_XL | 253.88GB | バッファ等の余裕がほぼなく、常駐用途では非推奨 |
| GLM-5.3 | UD-IQ3_XXS | 281.69GB | RAM＋VRAM常駐の範囲外 |

[Flash配布元](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/a38483c8cd5df544f53d70fb281afe97369d5ab6)、[通常版配布元](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771)からサイズを算出。判定はOS等の余裕を含めたこちらの推定で、作者の実測ではない。公開[通常版手順](https://unsloth.ai/docs/models/glm-5.3)も2bit級に合計約245GBを案内しているが、CPU／GPUの配置や作業領域次第で変わる。

最大候補のFlash Q5と通常版IQ2_Mは、重みをGPUへ約65-75GB配置すると残り約164-175GBがホスト側になる。実測179.37GB RAMから見ても余裕は小さく、KV cacheとprefill作業領域の予算を厳密に確認する必要がある。初回は32K contextと小さめのmicrobatchで試す。1M context対応というモデル仕様から、その長さで快適に常駐できるとは推定しない。

今回確認したFlashのGGUF metadataは `glm5-next`。過去の `glm5next` と取り違えない。現在のllama.cppは[Flash対応PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773)を2026-09-30に、[MTP対応PR #29928](https://github.com/ggml-org/llama.cpp/pull/29928)を2026-10-07にmerge済み。古いforkを前提にした投稿をそのまま手順に使わない。

**調べたエンジンの比較**

| エンジン | このA100 runtimeでの位置付け | 公開情報から分かる制約 |
| --- | --- | --- |
| **Project Maya** | Flashの運用候補 | 通常重みはMaya-Lまで公開。Uncensored優先ならMaya-M-Derisked。RAM resident設定あり。A100対応。通常版GLM-5.3は対象外。他のGGUFを `--gguf-dir` で使う経路はexperimental |
| **llama.cpp** | Flash Q4／Q5、通常版IQ1／IQ2の比較基準 | GPUに載らない層／expertをCPUへ配置。CUDA sm_80対応。実際の速度はCPU・RAM帯域にも依存 |
| **ExLlamaV3 / TabbyAPI** | FlashのEXL3を比較する候補 | Flash対応、CPU MoE offloadあり。一般GGUFは使わない。worker arena／一時領域を含む実RAM使用量を測る必要あり |
| **Phobos** | Strata派生の別候補だが初手には選ばない | 専用IQ4 179.10GB、公開最小RAMは192GB。今回の179.37GB RAMは要件未達で、80GB VRAMで削減できるか未検証。Unsloth GGUFの単純流用不可 |
| **glm-flash-lite** | 低RAM＋高速NVMe構成の研究として参考 | 125GB EXL3なのに公開all-RAMレシピは約238GB free RAMを要求。NVMe版は55GiBで動くが、O_DIRECTとlocal expert store、強いCPU／NVMeを前提とする |
| **Helios** | 今回は優先しない | 2×RTX 3090向けにtrunkとexpert poolを分けた実装。単体A100へそのまま適用できる実測なし |
| **KTransformers** | 公開Flashレシピは容量不足 | native FP8が約306GiB、system RAMを350GB以上要求。量子化GGUFのサイズとこのFP8手順を混同しない |
| **vLLM / SGLang** | 公開高速レシピは今回の第一候補ではない | Flashの公開CUDAレシピはHopper／Blackwellが中心。NVFP4はBlackwellを要求し、A100へ同じコマンドを適用できない |
| **strata-glm** | 比較研究として参考、導入対象にしない | 作者がarchived experimentと明記。RTX 5090＋NVFP4＋2 NVMeの実装で、token ID入出力のみ。公開サーバーなし |

各行の根拠: [Maya](https://github.com/mw00/project-maya)、[llama.cpp](https://github.com/ggml-org/llama.cpp)、[ExLlamaV3のoffload仕様](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md)、[Phobos](https://github.com/mussonking/Phobos)、[glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite)、[Helios](https://github.com/neron82/helios-glm53flash)、[KT Flash手順](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md)、[vLLM Flash手順](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)、[SGLang Flash手順](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)、[strata-glm](https://github.com/sergqwer/strata-glm)。Hugging Faceが自動生成する各アプリ向けの起動例は、実装の対応確認の根拠にしていない。

**VRAM 80GBだけで完結させる場合**

今回確認した非pruned版の最小Flash GGUFでも87-96GB程度あり、上記候補はいずれもRAMとの併用が必要。Maya-LをVRAMだけへ載せる構成ではない。

別枠として、50%のexpertを削除した[REAP50](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)にはIQ3_M 72.13GB、Q3_K_M 78.78GBがあり、前者は短contextでGPU完結の容量候補となる。後者は80GB GPUでは作業領域が厳しい。これは単なる量子化ではなく321Bから約165Bへexpertを削った派生モデルで、元のFlashと同一能力とは扱わない。今回はRAMが多いので、最初からこの変更を加える理由は小さい。

**GDriveとColabの一時ディスク**

Maya-L／Mはresident構成が容量的に狙えるため、永続保存はDrive、起動時にRAM／VRAMへ読み込み、推論中はexpertをDriveから読まない方針をまず検証できる。ただしMayaのGDrive運用は実機未確認で、ロード速度も別に測る。画像やセッションcacheは別のI/Oになるため、expert常駐がすべてのDriveアクセスゼロを意味するわけではない。

Flash Q6や通常版3bit以上を無理に動かすなら、一部expertをディスクから読む必要がある。この方式は容量の上限を広げるが、Driveをexpertのランダム読み出し元にする構成は採用しない。Colabの一時ディスクの空き容量と実I/Oを先に確認する。[Colab公式FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)もDrive I/Oの削減を勧めている。一時ディスクの性能・容量はhigh-mem指定だけでは確定せず、公開NVMe／RAID0環境の速度を再現できるとは推定しない。

無検閲優先で次に実機検証するなら、人気・公開検証を重視してBoldingBuilds / OrcaRouter IQ3_XXS、精度寄りでHuihui UD-Q4_K_XL、Mayaの運用を重視してMaya-M-Deriskedを比較する。最初は32Kまたは128K・単一リクエスト。Mayaではresidentを有効にする。通常重みとの比較用にMaya-Lも残す。通常版GLM-5.3の人気Uncensored配布は容量不足なので、今回のruntimeではFlashを優先する。起動から終了・失敗時までruntime stopを含める。今回の作業は調査・資料更新までで、既存Qwenフレームワークには組み込んでいない。
