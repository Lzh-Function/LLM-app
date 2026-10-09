# Local Qwen3-TTS

公式Qwen3-TTS 1.7B VoiceDesign / BaseをCUDA BF16で使う独立バックエンド。
LLMアンロード・GPU排他・モデル切り替え・長文作品保存は`voice-synthesize`から利用する。

```bash
cd /workspace/LLM
bash qwen-tts/setup.sh
bash voice-synthesize/serve.sh
```

`http://localhost:5080`で「Qwen3-TTS 1.7B」を選び、声の説明と原稿を入力して制作する。
手持ちの音声を使う場合は「参照WAVをアップロード」を選ぶ。120秒を超えるWAVは先頭120秒を自動で使う（アップロード最大256 MiB、作品に保存する参照は最大32 MiB）。
Qwenバックエンドの`/local/voices/prepare`へ直接送る場合も、最大256 MiBを受け付けて先頭120秒を使う。
120秒はこのアプリの上限であり、公式SDKが定めた秒数上限ではない。切り取り時は参照全文との不一致を避けるため、文章を使わず話者埋め込みでクローンする。
作品専用の参照音声・文章も保存するため、声ライブラリへ登録せずに同じ声を再利用できる。
GPU上には一モデルだけを載せる。終了時は画面の「音声OFF・GPU解放」。

[導入・モデル・API・設定・実測](../docs/09-qwen-tts.md)を参照。
