# データ容量とモデル整理案

調査日: 2026-10-07 15:09 UTC。対象は `/workspace/LLM` 以下。下記の容量表・整理案は削除前の調査記録。ユーザー指定に基づく削除実施結果は末尾に追記した。

## 計測方法と総量

`du -x -B1 -d 1 .` による割り当て済み容量は **235,068,506,112 bytes = 235.07 GB = 218.92 GiB**。隠しディレクトリを含み、別ファイルシステムへは進まず、シンボリックリンクの参照先は追わない。通常サイズの合計は `du -x --apparent-size -B1 -s .` で **234,773,251,200 bytes = 234.77 GB**。以下の容量は GB（10億 bytes）で統一する。

`df -B1 .` の空きは約516.39 GB。同じファイルシステムにある、このディレクトリ以外の使用量は本調査の対象外。

## LLM

「本体」は主要GGUFのファイルサイズ。「保存領域」は画像エンコーダー・変換データ・設定・ログ等を含む割り当て済み容量。速度は既存のローカル検証記録で、今回新たな推論ベンチマークは行っていない。

| モデル | 本体 GB | 保存領域 GB | 既存の生成速度 | 整理案 |
|---|---:|---:|---|---|
| Qwen3.8-Flash-Next SC117 IQ3_XXS | 76.14 | 85.53 | 128K Codexで約37～62 tok/s | 主力として残す |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 22.36 | 23.26 | 約51 tok/s | Flash Nextと役割が重なる。省RAM・起動約10秒のMoE予備機が必要なら残す |
| Gemma 4 26B-A4B UD-Q4_K_XL | 17.01 | 18.20 | 約45 tok/s | Qwenとは別系列の比較・予備機として残す |
| Qwen3.8-27B UD-Q4_K_M | 16.46 | 17.39 | 約6.8 tok/s | 削除優先。現在のGPUではCPUオフロードが重い |
| Ternary Bonsai 2 27B PQ2_0 | 7.21 | 7.84 | この環境で比較速度未測定 | 改変版とどちらか一つにする |
| Ternary Bonsai 2 27B Abliterated PQ2_0 | 7.21 | 7.21 | この環境で比較速度未測定 | 制約の少ない軽量27Bが欲しいならこちらを残す |
| Qwen3.5-9B UD-Q5_K_XL | 6.74 | 7.66 | 約74 tok/s | 起動約4秒の軽量・高速枠として残す |
| Dolphin 3.0 Llama 3.1 8B Q6_K | 6.60 | 6.60 | 約80 tok/s | 軽量枠は9B、制約の少ない枠はBonsai/Flashと重なるため削除候補 |

LLM保存領域合計は約173.68 GB。Flash Nextのラッパーディレクトリ自体は約0.00011 GBで、重みは `Strata-data/` にある。この表のFlash保存領域にはStrataエンジンの0.79 GBは含めていない。

他モデルの速度は主に32Kコンテキスト・思考OFFの記録。Flashの128K記録とは入力・出力・キャッシュ条件が異なるため、公平な速度ランキングではない。実測の出典は [LLM導入・性能](01-setup-models.md)、[Strata](07-strata.md)、[Codex](04-codex.md)。

[Prism公式カード](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)によればBonsai 2はQwen3.8-27B派生。Qwen3.8-27B、Bonsai通常版、Bonsai改変版を全て残す意義は、比較用途を除くと薄い。ただし量子化・追加学習・abliterationで挙動は変わり、回答品質が同等とは断定しない。改変版はv0.1プレビューで、通常版との人による品質比較は未実施。

モデルの大きさや公式ベンチマークだけで、この環境の量子化版・改変版の日本語品質やコーディング能力の優劣は決められない。上記の整理案は、主に実用速度・資源負担・用途の重複を根拠とする。

## 音声・文字起こしモデル

| モデル・データ | 容量 GB | 現在の役割と整理案 |
|---|---:|---|
| Irodori v4 Large、実行用BF16重み | 6.58 | GPU声作成・作品合成。残す |
| Irodori v4 Large、変換元FP32重み | 13.15 | BF16と二重保存。通常実行には不要で整理候補 |
| Irodori v4.1 Small-MF重み | 3.09 | llm-chatのCPU読み上げ。残す |
| Irodori v4.1 Small RF重み | 3.06 | 初期比較用。現行の声作成はLargeなので整理候補 |
| Irodori共有codec重み | 0.43 | MF・Largeとも使用。残す |
| Qwen3-TTS 1.7B VoiceDesign一式 | 4.52 | 説明から声を作成。残す |
| Qwen3-TTS 1.7B Base一式 | 4.54 | 参照音声から作品を合成。残す |
| faster-whisper medium本体 | 1.53 | 現行の文字起こし既定。残す |
| faster-whisper small本体 | 0.48 | 速度比較・軽量代替用。容量が小さいため削除優先度は低い |
| AivisSpeech まお | 0.258 | 好みの声かどうかで判断 |
| AivisSpeech コハク | 0.255 | 好みの声かどうかで判断 |
| AivisSpeech共有日本語BERTキャッシュ | 0.654 | AivisSpeech使用時に必要 |
| VOICEVOXモデル群（26ファイル） | 1.62 | キャラクター固有の声のため、汎用TTSで完全に置き換えられるとは限らない |

Irodori全体は **34.82 GB**（モデル領域26.36 GB、実行環境等8.42 GB、保存音声・評価等）。Qwen-TTS全体は **17.19 GB**（モデル9.06 GB、実行環境等8.12 GB）。IrodoriとQwenのCUDA環境だけで約16.52 GBあり、両者間でハードリンク共有はされていない。IrodoriはPython 3.10、QwenはPython 3.12であり、環境統合には依存関係と実動作の再検証が必要。使い続けるエンジンの `.venv` を削除して節約する案は採らない。

[Qwen公式](https://github.com/QwenLM/Qwen3-TTS)でもVoiceDesignとBaseは別役割で、声作成→参照から合成の両方を行う構成では両方必要。両モデルが持つspeech tokenizerの重み各0.682 GBは、保存済みSHA256が一致している。将来の取得・更新処理も対応させて共通化すれば約0.68 GB削減できるが、モデルを一つ丸ごと消すこととは異なる。

Irodori Large RF40は既存検証でRTF約0.85～0.98、Qwen Baseは約1.35、Irodori MF CPUは約1.09～1.36。文章・声・条件が異なるので、速度も音質も直接の優劣判定には使わない。**IrodoriとQwenの音質は、同じ原稿で試聴してから片方を外す判断がよい。** 会話用CPU MFと作品用GPUモデルは別用途として扱う。詳細は [Irodori](08-irodori-tts.md) と [Qwen-TTS](09-qwen-tts.md)。

## 共有エンジン・アプリ

| 保存領域 | GB | 内訳・用途 |
|---|---:|---|
| `aivisspeech/` | 2.48 | エンジン、2話者、共有BERT等 |
| `voicevox/` | 2.22 | エンジンとモデル群 |
| `llm-chat/` | 2.58 | ASRキャッシュ2.02 GB、Python環境0.55 GB、履歴・ライブラリ等 |
| `llama.cpp/` | 1.09 | Qwen/Gemma/Dolphinの共有エンジン |
| `strata/` | 0.79 | Flash用エンジン・ソース・環境・バックアップ等 |
| `llama-prism/` | 0.21 | Bonsai用エンジン |
| `voice-synthesize/` | 0.0072 | 独立サービス・作品保存等 |

`Strata-data/` 85.53 GBの内訳は、本体GGUF 76.14 GB、画像エンコーダー0.91 GB、dense pack 1.54 GB、MTP一式6.93 GB、tokenizer等。

MTPは実行用 `mtp/rt/` が0.824 GB、取得元BF16 `mtp/tensors/` が5.214 GB、中間 `mtp/mtp-q2_0.gguf` が0.889 GB。現在のllm-chat/Codex設定は `--mtp .../mtp/rt` を使うので、元データ・中間GGUFの **約6.10 GB** は追加の整理候補。ただし再変換時には再取得が必要で、元テンソルがなければStrataセットアップの元重み検証も省略される。実行用 `rt/`、Flash本体2分割、dense pack、画像エンコーダーは保持する。

## 推奨する順番と回収見込み

自分ならLLMは **Flash Next + Gemma 4 + Qwen3.5-9B + Bonsaiどちらか一つ** にする。主力、別系列、軽量高速、軽量27Bという役割を残す。Bonsai改変版を残す場合の見積もりは以下。

| 段階 | 整理対象 | 追加回収 GB | 累計回収 GB | 残量 GB |
|---|---|---:|---:|---:|
| 1 | Qwen3.8-27B領域、Dolphin領域、Bonsai通常版の本体のみ | 31.19 | 31.19 | 203.87 |
| 2 | Qwen3.6-35B-A3B領域 | 23.26 | 54.45 | 180.61 |
| 3 | Irodori Large FP32本体、比較用Small RF領域 | 16.22 | 70.68 | 164.39 |
| 任意の追加 | MTP取得元テンソル・中間GGUF | 6.10 | 76.78 | 158.29 |

Qwen3.6はFlash Nextの長い起動時間やRAM消費を避けたい時の予備として価値があるため、段階2は実際に使わないことを確認して決める。最初の段階だけでも約31.2 GB、実用構成を絞り込む段階3までなら全体の約30%を減らせる。

削除するならファイルの依存関係も同時に整理する必要がある。

- **Bonsai通常版のディレクトリ全体を消さない。** 改変版の `serve.sh` / `model.toml` は通常版領域の約0.63 GBのmmprojを参照する。本体GGUFだけを外すか、mmprojを共通領域へ移して参照を更新する。
- **Irodori Largeのtokenizer・BF16重み・共有codecを保持する。** 変換元FP32だけを外す。通常BF16起動のコードは元FP32を読まないが、`setup.sh` は次回実行時にFP32を再取得する。Small RFも現行プロファイルでは使わないが、setupは再取得するため、継続して減らすなら取得方針も変更する。
- ASRのsnapshotは実体blobへのシンボリックリンク。smallを外す場合、snapshotだけの削除ではほとんど容量を回収できない。medium等から参照されないblobのみ対象にする。
- 保存された参照音声、作品、会話履歴は再取得できるモデルと別扱いにする。現在は小容量で、削減対象に含めない。

## 最上位ディレクトリの実測値

以下は調査時の割り当て済みbytes。測定後にこの文書を追加した分や、動作中サービスの追記分は含まない。

```text
85525639168  Strata-data
34822713344  irodori-tts
23259942912  qwen3.6-35b-a3b
18204106752  gemma4-26b-a4b
17392152576  qwen3.8-27b
17187954688  qwen-tts
 7835475968  ternary-bonsai-2-27b
 7661907968  qwen3.5-9b
 7206236160  ternary-bonsai-2-27b-abliterated
 6596091904  dolphin3.0-llama3.1-8b
 2578137088  llm-chat
 2480517120  aivisspeech
 2220359680  voicevox
 1086427136  llama.cpp
  794718208  strata
  207069184  llama-prism
    7163904  voice-synthesize
    1490944  .git
     184320  docs
     106496  qwen3.8-flash-next
      49152  codex
      40960  .ruff_cache
235068506112 total（ルート直下ファイル・ディレクトリ自体も含む）
```

## 削除実施結果

2026-10-07、ユーザー指定に基づいて以下を削除した。

- Qwen3.6-35B-A3B、Qwen3.8-27B、Bonsai通常版、Dolphinの重み・起動スクリプト・UI設定。
- Irodori比較用Small RFのモデル領域とLarge変換元FP32重み。

削除処理の直前・直後の実測は **235,068,551,168 → 164,390,113,280 bytes**。約 **70.68 GB（65.82 GiB）** を回収し、残量は約 **164.39 GB（153.10 GiB）**。後からの文書追記やログ更新で多少変動する。

保持したLLMはFlash Next、Gemma 4、Qwen3.5-9B、Bonsai Abliteratedの4つ。通常版のBonsai画像エンコーダーを改変版の`models/`へ移し、起動・UI設定・取得スクリプトの参照先も更新した。取得スクリプトは削除済みLLMの画像エンコーダーを再取得しない。

Irodori会話用Small-MF、Large BF16、Large tokenizer、共有codec、CUDA実行環境、保存した参照音声・作品・会話履歴、Qwen-TTS、ASR、AivisSpeech、VOICEVOX、Strataの中間データは保持した。Irodoriのsetupは比較用RFを取得せず、検証済みBF16を再利用して変換元FP32の再取得を避ける。初回・再変換時もFP32は変換後に削除し、元のSHA256を`source_weights`と`derived.source_sha256`に保持する。`--keep-large-fp32`指定時のみFP32を保持する。

削除後にllm-chatを再起動し、実際のAPIで4モデルのみ表示され、全ての画像エンコーダー参照が有効なことを確認した。実モデルでもBonsai改変版の16K起動、llm-chatのIrodori CPU Small-MFと独立サービスのGPU Large BF16のロード・ウォームアップ音声生成が成功。確認後はモデル・音声ランタイムを停止状態へ戻した。Irodori取得処理も実行し、元FP32・比較用RFが復活しないことを確認した。既存Pythonテスト81件と取得・再変換に関する追加テスト5件が成功。
