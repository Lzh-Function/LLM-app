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
最大20000文字。完成WAVは4時間・1 GiBまで保存できる。Irodoriは既定RF40、QwenはVoiceDesignで作った声をBaseでクローンする。詳細設定は各エンジンのサンプラーとseedに対応する。
完成WAV・原稿TXT・生成条件を個別に保存でき、「作品ZIPを保存」で一式をダウンロードできる。
「この作品をもとに制作」では、作品に保存した声を使い、原稿・設定を変えて新しい作品を制作する。

「登録した声を使う」も選択可能。使ったWAVを作品内へコピーするため、元ライブラリの削除・変更後も作品の保存・再制作に影響しない。
llm-chat用の参照音声作りは別の「llm-chat用の参照音声」画面で行う。
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
  product.zip         原稿・設定・完成音声・作品専用の声の一式
```

保存先は`VOICE_SYNTHESIZE_PRODUCTION_DIR`で変更できる。
旧`VOICE_SYNTHESIZE_OUTPUT_DIR`は互換設定として利用可能。旧`outputs/`のWAV・条件も一覧・ダウンロード・削除できる。
作品を削除してもllm-chat用の声は削除しない。制作に失敗した未完成の作品は一覧へ出さない。

モデル条件、設定、API、GPU排他制御、検証方法は[運用手順](../docs/08-irodori-tts.md#実装と使い方2026-10-07)を参照。

Qwenの参照文章、話し方の指定、生成設定と実測は[Qwen3-TTSの運用手順](../docs/09-qwen-tts.md)を参照。
