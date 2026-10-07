# 08. Irodori-TTSの導入検討

調査日: 2026-10-07。前半は導入前の公開モデルカード・実装の調査。後半の「実装と使い方」に現在の構成を記載する。ユーザー指定により声作成・独立合成はLarge GPU、会話はSmall-MF CPUとした。前半のSmall候補作成案は変更前の調査記録。音質の人による評価とStrata同時実行は未測定。

同日の容量整理で、比較用Small RFとLargeの変換元FP32重みを削除した。会話用Small-MF、作品用Large BF16、Largeのtokenizer、共有codec、保存音声・作品は保持。setup既定はMF・Large・codecのみ取得し、変換後のFP32はSHA256を残して削除する。既存BF16のrevision・変換情報・SHA256を検証できる場合はFP32を再取得しない。明示的なFP32使用が必要な場合だけ`bash irodori-tts/setup.sh --keep-large-fp32`を使う。

llm-chatの会話読み上げには **Irodori-TTS-v4.1-Small-MF、既定4ステップ** を最初に評価する。音質・読みの比較基準は通常のv4.1-Smallの40ステップ。Largeは声の指定・再現を優先する単独合成の比較候補とする。実行方式は、機能が揃う公式Pythonサーバーを基本にし、CPUの速度・RAMが不足する場合にC++実装とGPU用メモリ予約を比較する。

追加要件: **captionから声の候補を生成し、選んだWAVを以後の参照にする。複数の声をライブラリとして保持する。** この用途では、声の候補作成を通常Small RF40、日常の読み上げをMF4で行う構成を推す。Large INT8は、必要なら候補作成時にStrataを止めて単独で比較する。いずれも提案段階で、生成音声を別モデルで参照した場合の声の再現度は試聴して確認する。

| モデル | 規模 | 通常の生成設定 | 配布チェックポイント | 用途の判断 |
|---|---|---|---|---|
| v4.1-Small-MF | 約0.8B | MeanFlow、4ステップ | FP32、約3.09 GB | 会話向けの第一候補 |
| v4.1-Small | 約0.77B | RF、40ステップ | FP32、約3.06 GB | 読み・音質の比較基準 |
| v4-Large | 約3.29B | RF、40ステップで公式評価 | FP32、約13.2 GB | 声の指定・再現を重視する比較候補 |

容量はダウンロードする重みファイルの大きさ。実行時のRAM・VRAM使用量ではない。公式Python経路は別途Semantic-DACVAE-Japanese-32dimの重み約430 MB、tokenizer、Python/PyTorch等の依存が必要。BF16実行にしても、配布元のFP32ファイルが半分の容量でダウンロードされるわけではない。
容量の根拠: [Smallのファイル](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small/tree/main)、[MFのファイル](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF/tree/main)、[Largeのファイル](https://huggingface.co/Aratako/Irodori-TTS-v4-Large/tree/main)、[codecのファイル](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim/tree/main)。

MFは単に通常版のステップ数を減らしたモデルではない。少ない反復用に蒸留され、CFGも学習に取り込まれている。推論時のCFGやSwayの調整は適用されない。通常RFを4ステップにした場合より、読みと話者再現の公式評価が良い。ただし40ステップRFと同等の音質と断定できず、人による大規模な音質評価もない。
[MFモデルカード](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF)。

同じJKYB-Parakeetの難読漢字評価では、読みの正解率はSmall RF40が93.42%、MF4が92.76%、Large RF40が92.80%。Largeは声の記述への追従と話者類似度を改善しているが、読みについてSmallを上回るわけではない。会話読み上げでLargeを最優先にする根拠は弱い。
[MFの比較](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF)、[Largeの比較](https://huggingface.co/Aratako/Irodori-TTS-v4-Large)。

公式の通常Small量子化版はINT8 weight-only/dynamic（各872 MiB）、INT4 weight-only（813 MiB）、FP8 weight-only（873 MiB）、FP8 dynamic（906 MiB）。INT4でINT8から削減できるのは59 MiBなので、容量面での利益は小さい。速度と音質は別途比較する。公式Largeはそれぞれ3,662 / 3,662 / 2,818 / 3,665 / 3,659 MiB。
これらはtorchaoの形式で、GGUFではない。CUDAで検証され、CPU実行は未検証。codecや実行用バッファも別途必要。通常Smallの量子化ファイルをMFとして使うことはできない。
[Small量子化](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-Quantized)、[Large量子化](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized)。

今の環境はRTX 5070 12 GB、Core Ultra 7 270K Plus、Windowsの物理RAM64 GB、WSL上限48 GB。SC117 IQ3_XXSの既存実測ではGPU使用量は約11,900 MiB（全体約12,227 MiB）、RAMはWSL全体で最大42.83 GiB。TTS用のGPU余地はほぼない。GPUで動かす場合、先にStrataのexpert cacheを小さくするメモリ配分が必要になる。
GPU cacheを減らすとCPU側で扱うexpertが増えるため、生成速度とRAM/SSD参照の両方に影響し得る。単にWSLの上限を上げてもVRAM不足は解決しない。既存測定の根拠は [07-strata.md](07-strata.md)、[06-voice-chat.md](06-voice-chat.md)。

Strataには`vram_elastic`と`POST /v1/vram`があるが、今の`--resident-experts`方式では利用できない。ローカルの`strata/src/program/generate.cpp`も、resident complementが有効な場合のVRAM変更を拒否する。現在の方式を維持したまま音声ON/OFFとGPU cacheの縮小・拡大を連動させることはできない。
GPUを併用するなら、起動時にTTS分を予約しておく案が基本になる。音声OFFでTTSのGPUメモリを解放しても、Strataのcacheはそのまま。モデルを再ロードする手順をユーザー操作なしに毎回挟む設計は避ける。
[Strataの詳細](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)。

| 実行方式 | 利点 | 判断が必要な点 |
|---|---|---|
| 公式Irodori-TTS-Server、MF、CPU | VoiceDesign・参照音声・caption・絵文字を使える。StrataのGPU配分を維持できる | このCPUでの速度とロード時ピークRAMは未測定。Strataの15 CPU workersとの競合も測る |
| speech.cpp、MF、F16、CPU | PyTorch不要。CPU版バイナリがあり、軽量な候補 | caption/VoiceDesignは未実装。参照音声が必要。速度は実測が必要 |
| 公式サーバー、MF BF16、GPU、codec CPU | 機能を保ちつつDiTをGPUで実行 | Strata用VRAMを減らす必要。ロード時を含めた実測で予約量を決める |
| speech.cpp、MF、F16/Q8、GPU | 低遅延の公開測定がある | captionを使えない。Strata用VRAMの調整は同じく必要 |

公式サーバーは`/v1/audio/speech`と音声一覧APIを提供し、モデルとcodecのデバイスを別々に指定できる。サーバー既定モデルは古いv4-Smallなので、v4.1-Small-MFを明示する必要がある。通常の応答は完成音声で、SSEモードも文のチャンク単位の配信。MFの既定4ステップを使い、RF用の40ステップ設定を引き継がない。
[公式サーバー](https://github.com/Aratako/Irodori-TTS-Server)、[設定例](https://github.com/Aratako/Irodori-TTS-Server/blob/main/.env.example)。

speech.cpp開発者の測定では、MF F16・4ステップ、RTX 2080/Vulkanで最初の音声の中央値0.13秒、RTF0.10、GPUメモリ増加約2.1 GB。Q8では約1.5 GB。20文、起動・準備完了後の測定であり、この5070・CPU・Strata同時実行の性能ではない。RTFは生成時間÷音声時間で、小さいほど速い。
同実装にはモデル・codec込みF16約1.9 GB、Q8変換約1.2 GBの形式がある。旧形式のモデルと別codecを要求するリリースと、新形式の一体ファイルを要求するリリースがあるため、実装と重みのrevisionを揃える。speech.cpp、audio.cpp、llama.cpp、StrataのGGUFは互換とは扱わない。MFの新配布先はREADMEに記載されているが、今回ブラウザーではそのカードを取得できなかったため、導入時には実ファイルの存在とSHAを改めて確認する。
[speech.cpp実装者の説明・測定](https://github.com/nyosegawa/speech.cpp)、[通常Smallの現行GGUFカード](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF)。

llm-chatへの変更は、既存の音声ON/OFF管理と合成・再生キューを活用し、Irodori用のAPI変換を追加する形が良い。現在の`/audio_query`→`/synthesis`と数値style IDはAivis/VOICEVOX向けなので、そのままURLだけを差し替えることはできない。

- Irodori選択時はそのエンジンだけを起動する。Aivis/VOICEVOXを同時に常駐させない。
- 起動時OFF、ONでロードと短いウォームアップ、OFFでプロセス終了・RAM/VRAM解放。`/health`だけでは公式サーバーのモデルロード完了を保証しない。
- 一文単位で合成し、生成済み音声を再生中に次の文を作る。5～10秒程度の音声を初期の分割目標にして、途切れと最初の待ち時間を測る。
- 一つの参照音声を各文で共用する。公式経路は参照latentの事前作成、speech.cppはvoiceファイルで参照の再エンコードを避ける。
- captionが使える公式経路では、LLMの感情選択を固定プリセットのcaptionに変換する。従来の数値style IDを流用しない。
- 「再生停止」と「音声OFF」は別に扱う。再生停止で毎回モデルを再ロードする構成にしない。

参照なしVoiceDesignを文ごとに独立実行すると声が揺れる可能性がある。声の指定を重視する場合も、固定した参照を使い、captionで話し方を調整する方針が良い。モデルカードは30秒程度以上のきれいな参照を推奨している。公式の音声サンプルを参考にし、選ぶ録音の利用条件も確認する。
[Smallの参照条件](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)、[公式サーバーのチャンク・参照仕様](https://github.com/Aratako/Irodori-TTS-Server)。

導入判断の実測は以下を目安にする。数字は提案する採用基準で、公表された実測値ではない。

1. 短文・長文・固有名詞・数字・英単語を含む日本語20～50文を、同じ参照でMF4とRF40から合成する。人が音質・読み間違い・語尾の余分な発話を確認する。
2. 初回起動と定常時を分け、最初の音声までの時間、RTF、RAM/VRAMの最大値を記録する。目標は定常時RTF0.7以下、最初の音声1～2秒以内。
3. Strataの回答生成とTTSを同時に実行し、LLMのtok/s・応答開始時間・swap増加を比較する。CPU版の実用性はここで判断する。
4. CPUでは待ち時間が大きければ、MFのGPU実行を単独で測り、ピークVRAMに余裕を足してStrataの予約量を決める。最初の実験範囲としてTTS分3～4 GiBを検討するが、固定の必要量とは扱わない。
5. LargeはSmallとの差を試聴してから検討する。先にSmall-MFと良い参照を使う方が、12 GB GPU・48 GB WSLでは導入負担が小さいと判断する。

声のライブラリ作成は次の流れを想定する。

1. 声の高さ・響き・息の量・年齢感・話し方をcaptionに書き、同じ試聴文でseedを変えて候補を生成する。初期設定は一度に1候補、通常Small RF40。複数候補の同時生成はVRAMが増えるので逐次実行する。
2. 候補ごとに試聴し、選んだWAVそのものを保存する。captionとseedだけの保存では別文章でも同じ声になることを保証できない。モデルrevision、seed、生成設定、試聴文、caption、WAVのSHAも保存して再評価できるようにする。
3. 選んだ声に永続IDと表示名を付け、以後はそのIDの参照を指定してMFで合成する。TTSモデルは声の数だけ必要になるわけではなく、同じモデルで各参照を使い分ける。
4. 声の確認は、元の試聴文の再生だけでなく、別の短文・長文・感情表現をMFで合成して行う。合成した参照からの声の再現度・雑音・発声の癖の継承は未検証。
5. 同じ声の補助参照を複数持つ場合は、同一話者のクリップとして明示的にまとめる。別々のseedで得た異なる声の候補を、一つの話者の参照に混ぜない。

公式サーバーは`voice: "none"`と`irodori.caption`で参照なしVoiceDesignを行え、保存したWAVを別のリクエストの`voice`として登録できる。例として`voice_a.wav`、`voice_b.wav`は別のvoice IDになる。同じ話者の複数参照は`voices.json`の`ref_wavs`でまとめられる。この対応はサーバーにあるが、候補の一覧・試聴・お気に入り・登録画面はllm-chat側に追加する。
[公式VoiceDesign・音声登録](https://github.com/Aratako/Irodori-TTS-Server)。

初期の参照用試聴文は、極端な感情・笑い・音楽・長い無音を避け、自然な話し方で複数の音を含む文章にする。短いWAVでもまず試せるが、モデルカードの参照推奨は30秒程度以上のきれいな音声。固定秒数を無理に指定して引き伸ばすより、必要に応じて同じ声の複数クリップを使う案を評価する。読み上げ時のcaptionは、参照と矛盾する声質変更を避け、主に話し方や感情に使う。
[Smallの参照条件](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)、[候補生成の設定](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md)。

この要件ではcaption未実装のspeech.cppを声の候補作成の主経路にしない。公式経路で候補を作った後、固定参照の読み上げだけspeech.cppへ渡す構成は比較できる。ただし新しいGGUFと実装の対応、参照の再エンコード、元のWAVとの声の一致を確認する。

ライブラリのWAV自体は小さい。48 kHz・16 bit・mono・30秒なら音声データ約2.88 MB、20声でも約57.6 MB。モデルやPyTorchの容量と分けて管理する。一つの実行プロセスでRFとMFを同時に常駐させず、声作成と会話読み上げのモード切り替えでロードするモデルを選ぶ案が良い。

Small・MFのモデルはMIT、LargeはT5Gemma由来のGemma条件。いずれもモデルカード記載の利用条件を確認する。Aivis/VOICEVOXとの音質比較は未実施で、Irodoriの方が必ず自然だとは断定しない。
[Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)、[MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF)、[Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large)。

## 実装と使い方（2026-10-07）

ユーザー指定により、**声作成と独立した高品質合成はLLMをアンロードしてGPUのLargeを使用する**構成に変更した。会話中の読み上げはSmall-MFをCPUで動かす。公式Pythonサーバーは専用uv環境で動かし、llm-chat自身へPyTorchを追加しない。

| モード | モデル | デバイス・精度 | 既定ステップ | LLMの扱い |
|---|---|---|---|---|
| 会話 `chat` | v4.1-Small-MF、約0.8B | CPU FP32、codec CPU FP32 | MF4 | ロード状態を維持 |
| 声作成 `design` | v4-Large、約3.29B | GPU BF16、codec CPU FP32 | RF40 | 起動前にアンロード |
| 独立合成 `synthesize` | 同じv4-Large | GPU BF16、codec CPU FP32 | RF40、4～120で調整可能 | 起動前にアンロード |

LargeはINT8/INT4を使わず、全パラメータを保持する。配布FP32重みからCPU上でBF16チェックポイントを作成し、元ファイル・変換ファイル双方のSHA256を記録する。公式ローダーはGPU転送後にdtypeを変更するため、FP32の約13.2GBを直接指定すると12GB GPUではロード時に収まらない。BF16ファイルを事前作成してこの一時的な消費を避ける。tokenizerは元チェックポイントのものを使う。

`design`と`synthesize`は同じモデル・codecなので切り替えても再ロードしない。会話との切り替えでは旧プロセスを停止して別モデルをロードする。声の数だけモデルを常駐させる必要はない。

### 導入・起動

```bash
cd /workspace/LLM
bash irodori-tts/setup.sh       # CUDA 12.8版、Large/MF/codec、BF16変換後FP32は削除
bash voice-synthesize/serve.sh # http://localhost:5080
```

llm-chatを使う場合は別ターミナルで:

```bash
cd /workspace/LLM/llm-chat
uv run llm-chat                # http://localhost:5070
```

この環境にはCUDA版と全重みを導入済み。setup既定は`cu128`。再導入では`install.json`に記録したrevisionを維持し、更新時だけ`bash irodori-tts/setup.sh --update-models`を使う。重み・専用環境・ライブラリ・合成結果・ログはGit管理外。Largeにはモデルカード記載のGemma条件が適用される。

**独立サービスの使い方**:

1. `http://localhost:5080`で「GPUモデルを起動」を押す。llm-chatが動いていれば`POST /api/unload`でLLMを停止し、Largeのロードと短文ウォームアップを待つ。llm-chat自体の起動は不要。
2. 「声を作る」に声の説明・試聴文・seedを入力して候補生成。試聴して「声として登録」を押す。手持ちのWAVも登録できる。
3. 「音声作品を制作」で作品の声の説明・原稿・作品名を入力して合成する。声ライブラリの登録は不要。「登録した声を使う」も選べ、参照WAVを作品側へコピーする。最大20000文字を文単位に分割し、同じ参照latentを使って結合する。句読点がない長い文も160文字以下に分け、1回のGPU推論への巨大な入力を避ける。
4. 完成WAVを再生・ダウンロード。原稿TXT・生成設定・作品専用の声・完成音声は専用の`productions/`へ保存する。「作品ZIPを保存」で一式を取得できる。「この作品をもとに制作」では保存した声のまま原稿・設定を変えて新しい作品を作る。生成条件、参照SHA、モデルrevision、合成時間、RTFもJSONで保存する。「詳細設定」ではステップ数・CFG・seedを調整できる。ステップ増加が常に音質向上になるとは限らないので試聴して選ぶ。
5. 「音声OFF・GPU解放」で管理プロセスを終了し、GPUを解放する。LLMは自動で再ロードしない。

**llm-chatの使い方**:

1. 「声ライブラリ」→「声作成モードを起動」。LLMがロード中でも待機タスクをキャンセルしてプロセスを停止し、Large GPUへ切り替える。
2. 声候補を生成・試聴・登録。複数候補は逐次生成し、指定seedを1ずつ増やす。元WAV・ID・条件を保持して名前やお気に入りを変更できる。
3. 「会話モードで試す」でLargeを停止し、CPU Small-MF4へ切り替える。別文章を試聴し、「会話で選ぶ」で話者にする。
4. LLMをロードして会話。声色の自動選択は固定captionの感情プリセットに変換する。合成と再生は別キューで進め、完成音声を最大2件先行して保持する。
5. 「音声停止」は再生だけ停止。「読み上げOFF」「音声OFF」は音声プロセスを停止する。CPU会話モードのON/OFFはLLMの状態を変えない。

会話用の声ライブラリは両UIから使える。音声作品の制作はこのライブラリと独立しており、作品専用の声をライブラリへ登録しない。登録済みの声を選んだ場合もWAVを作品内へコピーするので、元ライブラリを削除・改名しても作品や再制作に影響しない。llm-chat向けの参照音声作りは別画面にまとめる。別々のseedの候補を同一話者の参照へ混ぜない。補助参照は登録済みの声へ、同じ話者であることを明示して追加する。参照WAVは1ファイル32MiB以下、0.2～120秒、同一声の参照合計120秒以下。30秒程度以上のきれいな参照を推奨するが、短い候補も利用できる。

### GPU排他制御

llm-chatとvoice-synthesizeは`irodori-tts/gpu.lock`のOSロックを共有する。GPU音声がONの間、LLMのロード要求は409で止める。別の音声サービスがGPUを使用中なら、そのUIでOFFにしてから切り替える。LLMの通常ロード・モデル切り替えも同じロックを取得する。子プロセスへロックFDを継承し、管理アプリが終了しても子モデルが生きている間は予約が残る。

独立サービスはローカルのllm-chatへアンロード要求を送り、停止を確認してから予約を取る。接続先が起動していなければそのまま進む。接続先がエラーを返したり停止を確認できなかったりする場合はGPUモデルを起動しない。llm-chatが別ポートなら`LLM_CHAT_URL`を合わせる。

この排他制御の対象は両アプリが管理するプロセス。手動起動のStrataなどは対象外なので、先に停止しておく。管理外のローカルGPU Irodoriがすでにポートを使っている場合はエラーにする。CPUの既存エンジンは再利用してもOFFで停止しない。

### 保存場所と設定

```text
irodori-tts/
  runtime/                          公式サーバー + 専用 .venv
  models/{chat,large,codec}/         revisionを固定した重み・tokenizer
  models/large/bf16/                変換済み全パラメータBF16重み
  install.json                      revision、SHA256、変換情報、依存
  voices/{chat,large}/.latents/      再作成可能な参照WAV・latentキャッシュ
  gpu.lock                          両アプリのGPU排他ロック
  server.log
llm-chat/voice-library/<ID>/
  reference.wav                     登録した元WAV
  reference-<UUID>.wav               同じ話者の補助参照（任意）
  metadata.json                     永続ID、名前、SHA256、caption、seed、条件
voice-synthesize/productions/<ID>/
  audio.wav                         完成した音声作品
  manuscript.txt                    原稿（UTF-8）
  settings.json                     原稿・声の選び方・生成設定
  metadata.json                     作品名、参照情報、時間・RTF・SHA256
  voice/<voice-ID>/                  作品専用のWAV・声の条件
  product.zip                       完成音声・原稿・設定・作品専用の声の一式
voice-synthesize/outputs/<ID>/       旧形式の合成結果（読み取り互換）
```

バックアップ対象は声ライブラリと必要な作品ディレクトリ全体（または作品ZIP）。作品は元ライブラリを参照せず保存したWAVを使う。制作失敗時の一時領域は削除し、完成後にだけ作品ディレクトリを原子的に公開する。旧`outputs/`は移動せず一覧・取得・削除に対応する。`voices/`はモデル停止中に削除して再作成できる。`install.json`も保存すると同じrevisionで再導入できる。両アプリのライブラリ変更はファイルロックと原子的なメタデータ置換で保護する。

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `IRODORI_BACKEND` | `cu128` | setup時のPyTorch配布元。`cpu` / `rocm`も指定可能 |
| `IRODORI_CHAT_MODEL_DEVICE` / `_PRECISION` | `cpu` / `fp32` | 会話モデル |
| `IRODORI_CHAT_CODEC_DEVICE` / `_PRECISION` | `cpu` / `fp32` | 会話codec |
| `IRODORI_QUALITY_MODEL_DEVICE` / `_PRECISION` | `cuda` / `bf16` | 声作成・独立合成モデル |
| `IRODORI_QUALITY_CODEC_DEVICE` / `_PRECISION` | `cpu` / `fp32` | GPU余裕を確保するcodec |
| `IRODORI_CPU_THREADS` | `4` | OMP/MKLスレッド |
| `LLM_IRODORI_URL` | `http://127.0.0.1:8088` | llm-chatの音声バックエンド |
| `VOICE_SYNTHESIZE_TTS_URL` | `http://127.0.0.1:8089` | 独立サービスの音声バックエンド |
| `LLM_CHAT_URL` | `http://127.0.0.1:5070` | 独立サービスからのLLMアンロード先（ローカルHTTP限定） |
| `VOICE_SYNTHESIZE_HOST` / `_PORT` | `127.0.0.1` / `5080` | 独立UIの待受け。CLI `--host` / `--port`でも変更可能 |
| `LLM_IRODORI_MODEL_NAME` | `irodori-tts` | OpenAI互換APIのmodel名 |
| `LLM_IRODORI_API_KEY` | 未設定 | 管理音声サーバー・クライアントのBearer認証 |
| `LLM_VOICE_LIBRARY_DIR` | `llm-chat/voice-library` | 両UI共通の声保存先 |
| `VOICE_SYNTHESIZE_PRODUCTION_DIR` | `voice-synthesize/productions` | 作品専用の保存領域 |
| `VOICE_SYNTHESIZE_OUTPUT_DIR` | 未設定 | 保存先の旧互換設定。PRODUCTION_DIRを優先 |

設定例は`irodori-tts/.env.example`。コピーした`.env`を`set -a; source irodori-tts/.env; set +a`で読み込んでからアプリを起動する。自動では読み込まない。旧`IRODORI_MODEL_DEVICE/PRECISION`・`IRODORI_CODEC_DEVICE/PRECISION`は会話側の互換設定として残る。GPU会話へ変更する場合もLLMアンロードとGPU排他が適用される。12GB GPUではLarge FP32は収まらないので既定BF16を使う。

### APIと検証

| 操作 | API |
|---|---|
| llm-chat音声ON/OFF | `PUT /api/voice/runtime` (`enabled`, `engine: "irodori"`, `mode: "chat" / "design"`) |
| 独立サービス音声ON/OFF | 同じAPI (`enabled`, `mode: "design" / "synthesize"`) |
| 準備状況・GPU予約・プロファイル | `GET /api/voice/runtime` |
| ライブラリ一覧 | `GET /api/voice/library` |
| Large声候補生成 | `POST /api/voice/library/candidates` (`caption`, `text`, `seed`, `count`, `name`, RF設定) |
| 登録・名前・お気に入り | `PATCH /api/voice/library/<ID>` (`register: true`, `name`, `favorite`) |
| WAV取り込み・補助参照 | `POST /api/voice/library/import` / `<ID>/references` (multipart) |
| 元WAV・条件 | `GET /api/voice/library/<ID>/audio` / `metadata` |
| 別文章で比較 | `POST /api/voice/library/<ID>/preview` (`text`, `preset`, `mode`, `seed`) |
| 会話読み上げ（llm-chat） | `POST /api/voice/synthesize` (`engine: "irodori"`, `voice_id`, `preset`, `text`) |
| 独立長文合成 | `POST /api/synthesis` (`text`, `name`, `voice_caption`, `preset`, `caption`, `seed`, RF設定)。任意で`voice_id`または`source_product_id`を指定 |
| 作品一覧・保存 | `GET /api/synthesis`, `GET /api/synthesis/<ID>/audio` / `manuscript` / `settings` / `metadata` / `archive` |
| 制作画面の選択肢 | `GET /api/synthesis/options`（声ライブラリを読まず取得可能） |
| 削除 | `DELETE /api/voice/library/<ID>` / `DELETE /api/synthesis/<ID>` |

RF設定は`num_steps`（既定40、4～120）、`cfg_scale_text` / `cfg_scale_caption`（既定3）、`cfg_scale_speaker`（既定5）。CFGは0～10。独立サービスはLLM会話・履歴のAPIやModelManagerを持たない。`voice_id`も`source_product_id`も指定しなければ声の説明から作品専用の声を用意する。長文で話者が変わらないよう、そのWAVを全チャンクの固定参照に使う。これは作品内にだけ保存し、声候補や会話話者へは追加しない。

```bash
cd /workspace/LLM/llm-chat
uv run python -m unittest discover -s tests -q
node tests/test_voice_queue.cjs
# 一時ライブラリでLarge GPU候補→登録→CPU MF会話→latent再利用→OFFまで確認
uv run python tests/smoke_irodori.py --output ../irodori-tts/evaluation/smoke-new
# 同じ参照でCPU MF4とGPU Large RF40を比較（Large起動でLLMをアンロード）
uv run python ../irodori-tts/benchmark.py --voice <ID>
# LLMとの同時実行はCPU会話だけを指定する
uv run python ../irodori-tts/benchmark.py --voice <ID> --modes chat --concurrent-chat
```

GPU排他・LLMロード中の停止・失敗時の解放・独立合成・長文分割を含む55件のPythonテスト、ブラウザーでの両UI、合成/再生キューを確認した。導入済み実モデルでもLarge候補作成→CPU MF会話への切り替え、独立サービスのRF40/RF80合成、WAV・条件保存、OFFによる解放を確認済み。作品制作の追加確認では、登録声なしで約29秒の原稿音声を生成し、会話ライブラリが空のまま作品一式を保存・ZIP取得できた。同じ作品の声で別原稿の制作も確認した。追加確認データは`irodori-tts/evaluation/production-separated-20261007/`に保存した。

評価ツールは既定20文、`--texts`・`--count`・`--modes`・`--preset`・`--seed`・`--output`を指定できる。準備時間、リクエスト時間、RTF、SHA256、TTSのRSS/HWM、WSLメモリ・swap、GPU全体使用量を保存する。GPU全体値は表示・他プロセスを含み、サンプル間のピークは取り逃がし得る。終了時は元の音声設定へ戻すが、アンロードしたLLMは自動復帰させない。`human_review`は人が試聴して記入する欄で、音質を自動判定しない。

### Large GPU構成の実測

RTX 5070、モデルCUDA BF16、codec CPU FP32、CPU 4スレッド、LLM停止状態で機能確認した。

| 項目 | 実測 |
|---|---|
| Large起動～ウォームアップ | 約19.8秒 |
| Large RF40声候補（24.48秒の音声） | 合成20.73秒、RTF0.85 |
| 固定参照RF40長文（27.36秒の音声） | 合成26.70秒、RTF0.98 |
| 固定参照RF80短文（6.36秒の音声） | 合成8.85秒、RTF1.39 |
| GPU全体使用量の最大サンプル | 9,256MiB（約9.04GiB） |
| 声作成→独立合成の切り替え | 同じPID、再ロードなし |
| OFF後 | 子プロセス終了、GPU予約解除を確認 |

音声・生成条件・GPUサンプルは`irodori-tts/evaluation/large-gpu-synthesis/`へ保存した。RF40とRF80は異なる文章なのでステップ数の音質・速度比較には使えない。単独GPUでの機能確認であり、最大長入力や全CFG設定のVRAM上限を保証する測定ではない。人による音質・読み・声の一致の評価は未実施。

### 初期Small CPU構成の参考値

CPU / FP32、codec CPU / FP32、OMP/MKL 4スレッド、Strata停止状態で実行。これは少数文での機能確認で、採用基準を満たす性能評価ではない。

| 項目 | 実測 |
|---|---|
| RFの起動～ウォームアップ完了 | 約18.7秒 |
| MFの起動～ウォームアップ完了 | 約9.4秒 |
| RF40候補（12.36秒の音声） | 合成42.16秒、RTF3.41 |
| MF4別文章1（5.96秒の音声） | 合成6.50秒、RTF1.09 |
| MF4別文章2（3.32秒の音声） | 合成4.51秒、RTF1.36 |
| MFプロセスのVmHWM | 5,376,056 KiB（約5.13 GiB） |
| MFプロセスの最後のRSS | 4,843,836 KiB（約4.62 GiB） |

MFの合成時間は参照準備後のspeech APIの時間。初回参照エンコード・アップロードとブラウザー再生開始の時間は別途加わる。ロード時RSSの時系列は別の`benchmark.py`で測る。実測データと試聴WAVはこの環境の`irodori-tts/evaluation/smoke-20261007/`に保存してある。**CPU既定設定はRTF0.7・最初の音声1～2秒という提案目標に未達**。音質・声の再現度の人による確認、20～50文評価、Strata併用、スレッド調整・GPU予約量の検証はこれから行う。

### Qwen3-TTSも選択可能

独立`voice-synthesize`にQwen3-TTS 1.7B VoiceDesign / Baseを追加した。画面上部のエンジン選択から切り替える。GPU排他と作品の保存領域は共通で、声作成・合成用のモデルは一つずつロードする。詳細は[Qwen3-TTSの導入・運用](09-qwen-tts.md)を参照。llm-chatの会話用CPU Irodori構成は引き続き利用できる。
