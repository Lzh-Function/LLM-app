# Local Qwen3-TTS

公式Qwen3-TTS 1.7B VoiceDesign / BaseをCUDA BF16で使う独立バックエンド。
LLMアンロード・GPU排他・モデル切り替え・長文作品保存は`voice-synthesize`から利用する。

```bash
cd /workspace/LLM
bash qwen-tts/setup.sh
bash voice-synthesize/serve.sh
```

`http://localhost:5080`で「Qwen3-TTS 1.7B」を選び、声の説明と原稿を入力して制作する。
作品専用の参照音声・文章も保存するため、声ライブラリへ登録せずに同じ声を再利用できる。
GPU上には一モデルだけを載せる。終了時は画面の「音声OFF・GPU解放」。

[導入・モデル・API・設定・実測](../docs/09-qwen-tts.md)を参照。
