# Voice Synthesize

LLMと共存させず、Irodori-TTS-v4-Large（約3.29B）またはQwen3-TTS 1.7Bの全重みをGPU BF16で使う、声作成・長文合成の独立サービス。
原稿と声の説明だけで音声作品を制作できる。作品はllm-chatの声ライブラリとは別の保存領域へ、完成WAV・原稿・設定・作品専用の声をまとめて保存する。

```bash
cd /workspace/LLM
bash irodori-tts/setup.sh
bash qwen-tts/setup.sh  # Qwenを使う場合
bash voice-synthesize/serve.sh
```

`http://localhost:5080`の「音声作品を制作」で声の説明と原稿を入力して合成する。
声ライブラリへの登録は不要。原稿はUTF-8のTXTからも取り込める。
手持ちの音声で作品を制作する場合は「声の選び方」で「参照WAVをアップロード」を選び、WAVと原稿を指定して合成する。
複数のWAVを選ぶと、それぞれの話者ラベルが必須になる。例えば一つ目を`speaker A`、二つ目を`speaker B`として、原稿を次のように入力する。

```text
[speaker A] こんにちは。今日は何をしましょうか。
[speaker B] 公園へ散歩に行きたいです。
[speaker A] いいですね。一緒に行きましょう。
```

ラベルは任意の名前でよく、角括弧内と完全に一致させる。同じ話者の再登場も可能。タグは読み上げず、各区間を対応する参照で独立して推論する。長い区間は最大160文字で分割し、すべての生成音声の接続箇所に1秒の無音を挿入する。WAVが一つならラベルは任意で、普通の原稿をそのまま使える。
Irodori・Qwenの両方に対応。Qwenの参照文章はファイルごとに入力できる。参照アップロードの合計は256 MiBまで、各話者の参照は先頭120秒・保存後32 MiB以下。別話者の参照を結合して一つの声にはしない。
作品ZIPには全話者の参照WAVと条件、タグ付き原稿、区間WAV、タイムライン、完成WAVを保存する。「この作品をもとに制作」で全話者を引き継ぎ、元作品を削除しても再制作できる。作品カードで保存した話者の参照を試聴し、区間WAVも取得できる。
Irodoriは音声のSHA256ごとに参照潜在を準備して`ref_latents`として再利用し、再登場や別作品へのコピーで再エンコードしない。Qwenは各参照のクローンプロンプトを再利用する。[公式パラメーター説明](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md)でも繰り返し推論には`--ref-latent`が高速な経路とされている。Speaker Inversionは話者埋め込みを別途学習する方式で、任意のアップロードから自動学習すると初回の時間が増えるため、このアップロード制作フローでは参照潜在キャッシュを使う。
参照WAVは作品専用の声として保存し、作品ZIPと「この作品をもとに制作」にも引き継ぐ。120秒を超えるWAVは先頭120秒を使い、カットした秒数を結果に表示する。
最大20000文字。完成WAVは4時間・1 GiBまで保存できる。Irodoriは既定RF40、QwenはVoiceDesignで作った声をBaseでクローンする。詳細設定は各エンジンのサンプラーとseedに対応する。
完成WAV・原稿TXT・生成条件を個別に保存でき、「作品ZIPを保存」で一式をダウンロードできる。
「この作品をもとに制作」では、作品に保存した声を使い、原稿・設定を変えて新しい作品を制作する。
保存済み作品と参照ボイスの一覧では「名前を変更」を開き、新しい名前を入力して「保存」を押す。名前は1～80文字。合成後でも変更でき、作品の設定とZIPにも反映する。音声モデルの起動は不要。

「登録した声を使う」も選択可能。使ったWAVを作品内へコピーするため、元ライブラリの削除・変更後も作品の保存・再制作に影響しない。
llm-chat用の参照音声作りは別の「llm-chat用の参照音声」画面で行う。
手持ちのPCM WAVは256 MiBまで取り込める（保存後は32 MiB以下）。120秒を超えるWAVは先頭120秒を保存する。
補助参照は既存の参照と合計120秒に収まる長さまで先頭から保存し、カットした秒数を画面に表示する。
カットした録音の読み上げ文章は元の文章として保存し、Qwenのクローンには自動で使わない。使う場合は保存された音声に合う文章を合成画面で入力する。
声作成と合成の切り替えで同じLargeモデルを再ロードしない。

GPU音声ONの前にローカルllm-chat（既定5070）へLLMアンロードを要求する。
llm-chatが停止していても利用可能。GPUを使う間はLLMの再ロードをブロックする。
別の音声サービスから切り替えるときは先にそのGPU音声をOFFにする。
終了・OFFでGPUを解放し、LLMは自動で再ロードしない。

UIは5080、Irodoriバックエンドは8089、Qwenは8090。`serve.sh --host 127.0.0.1 --port 5080`で待受けを変更可能。
`LLM_CHAT_URL`、`VOICE_SYNTHESIZE_TTS_URL`、`VOICE_SYNTHESIZE_QWEN_URL`で接続先を変更できる。
会話用の声は`llm-chat/voice-library/`、作品は`voice-synthesize/productions/<ID>/`へ保存する。

```text
productions/<ID>/
  audio.wav           完成音声
  manuscript.txt      原稿（UTF-8）
  settings.json       声の選び方、seed、ステップ数、CFG等
  metadata.json       作品名、モデル、時間・RTF、参照SHA256
  voice/<voice-ID>/   作品専用のWAVと条件（会話ライブラリへの登録なし）
  segments/0001.wav   複数話者作品の生成区間（metadata.jsonに話者・原稿・時刻・条件）
  product.zip         原稿・設定・完成音声・作品専用の声の一式
```

保存先は`VOICE_SYNTHESIZE_PRODUCTION_DIR`で変更できる。
旧`VOICE_SYNTHESIZE_OUTPUT_DIR`は互換設定として利用可能。旧`outputs/`のWAV・条件も一覧・ダウンロード・削除できる。
作品を削除してもllm-chat用の声は削除しない。制作に失敗した未完成の作品は一覧へ出さない。

モデル条件、設定、API、GPU排他制御、検証方法は[運用手順](../docs/08-irodori-tts.md#実装と使い方2026-10-07)を参照。

Qwenの参照文章、話し方の指定、生成設定と実測は[Qwen3-TTSの運用手順](../docs/09-qwen-tts.md)を参照。

複数参照のAPIは`POST /api/synthesis/upload`。multipartの`files`をファイル数だけ繰り返し、`settings`のJSON文字列に同じ順序の`speakers`を指定する。一つだけの場合は従来の`file`も使用できる。

```json
{
  "engine": "irodori",
  "name": "二人の会話",
  "text": "[speaker A]こんにちは。[speaker B]おはよう。",
  "speakers": [
    {"label": "speaker A", "reference_text": "一つ目のWAVの正確な文章（任意）"},
    {"label": "speaker B", "reference_text": "二つ目のWAVの正確な文章（任意）"}
  ]
}
```

複数話者では、全体の`reference_text`は指定しない。ラベル未入力・重複・未知のタグ・空区間・タグ前の文章は422。失敗した作品は公開しない。話者の参照は`GET /api/synthesis/<ID>/speakers/<0始まりの番号>/audio`、区間音声は`GET /api/synthesis/<ID>/segments/<0始まりの番号>/audio`で取得できる。

検証は`llm-chat/tests/test_dialogue.py`と`test_synthesis_upload.cjs`。実際のIrodori/Qwenクライアントを通し、モデル応答のみをモックにして話者切替・キャッシュ・1秒のPCM無音・原稿保持・ZIP・再制作・失敗時の片付けを確認する。実モデルでの複数話者音質と総生成時間の比較は未測定。
