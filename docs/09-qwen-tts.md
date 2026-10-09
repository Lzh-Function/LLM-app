# Qwen3-TTSの導入と音声作品制作

2026-10-07、`voice-synthesize`にQwen3-TTSを追加した。画面上部でIrodori v4 LargeとQwen3-TTS 1.7Bを選択できる。
モデルは公式の1.7B全重みをCUDA BF16で使い、量子化しない。声作成と作品合成はLLMをアンロードしてから実行する。

## 起動と制作

```bash
cd /workspace/LLM
# 初回導入（この環境では導入済み）
bash qwen-tts/setup.sh
# Irodoriも使う場合
bash irodori-tts/setup.sh
# 制作画面
bash voice-synthesize/serve.sh
```

`http://localhost:5080`で音声エンジンを「Qwen3-TTS 1.7B」にする。
「作品専用の声を指定」で声の説明・原稿・作品名を入力し、「作品を合成して保存」を押す。
必要なモデルは自動起動する。作品は`voice-synthesize/productions/<ID>/`に保存し、WAV、UTF-8原稿、設定、条件、参照WAV・その文章をZIPで取得できる。
原稿は最大20000文字。完成WAVの受け入れ上限は4時間・1 GiB。句読点を優先して最大160文字ずつに分け、全チャンクで同じ声のクローンプロンプトを使い、チャンク間に120msの無音を挟む。

「この作品をもとに制作」では、保存済みの声と設定を使って別原稿を制作できる。
「登録した声を使う」では、既存のIrodori/Qwen生成声や取り込みWAVも選べる。参照音声を作品内へコピーするため、元ライブラリを削除しても作品の再制作は可能。
「参照WAVをアップロード」では、ライブラリへ登録せずに手持ちのWAVを作品合成に使える。アップロードは最大256 MiB、120秒超は先頭120秒を自動で使う。切り取り後の作品用参照は最大32 MiB。
複数ファイルを選ぶ場合は各ファイルに異なる話者ラベルを付け、原稿を`[speaker A]文章[speaker B]文章`のように区切る。各話者の参照と文章から個別のクローンプロンプトを作り、再登場では再利用する。各区間を独立して合成し、長い区間も最大160文字ずつに分割して、すべての生成区間の間に1秒の無音を入れる。アップロード256 MiBはファイル合計、参照120秒・32 MiBは各話者ごとの上限。
作品専用の声をllm-chatの声ライブラリへ自動登録することはない。

「llm-chat用の参照音声」画面でもQwenの声候補を作り、試聴して登録できる。
Qwenの試聴文は160文字以下、候補数は1～8。同じ声の識別が崩れないよう声作成は一回の生成に収める。
登録済みWAVはllm-chatの既存Irodori CPU合成の参照にも使える。llm-chatの会話中にQwenをGPUで常駐させる構成は追加していない。

## モデルとGPUの使い方

| 用途 | 公式モデル | 実行 |
|---|---|---|
| 声の説明から参照WAVを作成 | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | CUDA BF16、PyTorch SDPA |
| 参照WAVから長文を合成 | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | CUDA BF16、PyTorch SDPA |

これは公式の[Voice Design → Clone](https://github.com/QwenLM/Qwen3-TTS#voice-design-then-clone)の流れに沿った構成。
各モデルはspeech tokenizerも含む。GPU上にはモデルとcodecを一組だけ載せ、VoiceDesignからBaseへの切り替え時は古いモデルを解放する。
バックエンドのプロセスを維持して内部のモデルだけを切り替え、声のクローンプロンプトはCPUテンソルでキャッシュする。
IrodoriとQwenの切り替えでは、元バックエンドを停止してから次を起動する。

GPU起動前にローカルllm-chat（既定5070）へ`POST /api/unload`を送り、GPU音声中のLLMロードは409でブロックする。
共有の`irodori-tts/gpu.lock`を子プロセスへ継承するため、親アプリが異常終了しても動作中のバックエンドとの排他は継続する。
作業後は「音声OFF・GPU解放」。LLMを自動で再ロードしない。他アプリのGPU音声がONなら、そのアプリでOFFにしてから切り替える。

FlashAttentionは導入せず、PyTorch SDPAを明示している。この環境ではCUDAコンパイラに依存せず実行できる。
公式SDKの音声生成は完成WAVを返す。この実装は生成完了後の保存・再生に対応し、ストリーミング配信は実装していない。

## 参照音声の文章と話し方

[Baseの公式モデルカード](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)に沿って、正確な参照文章があれば音声コード・文章・話者埋め込みを使うICL方式でクローンする。
生成した参照は読み上げ文章を自動保存する。取り込むWAVは「読み上げ内容」を入力でき、作品の詳細設定でも参照文章を上書きできる。
作品での上書きは作品内に保存し、共有ライブラリへ反映しない。

参照文章がなければ`x_vector_only_mode=True`で話者埋め込みのみを使う。条件の`metrics.clone_mode`に`icl`または`speaker_embedding_only`を記録する。
Qwenでは先頭の参照WAV一つを使う。Irodori用の補助参照を追加していてもQwenは結合しない。
参照としては短く明瞭な音声を用意すると扱いやすい。作品専用に生成する既定の参照文章は約50文字。

### 参照音声の長さと自動切り取り

公式READMEは[3秒の音声によるクローン](https://github.com/QwenLM/Qwen3-TTS#released-models-description-and-download)を紹介している。3秒は最大長ではない。
導入済みの公式SDKと[現在の公式実装](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)の`create_voice_clone_prompt`には、参照を一定秒数で拒否・切り取りする処理や`--max-ref-seconds`設定はない。受け取った音声を音声トークナイザーと話者埋め込み抽出へ渡す。無制限に動作する保証はなく、長い参照ではメモリ使用量が増え、ICLでは参照音声コード・文章もモデルへの入力になる。

この環境では作品制作とQwenバックエンドを先頭120秒までに揃えている。`POST /local/voices/prepare`も120秒超の音声を自動で切り取り、最大256 MiBまで受け付ける。120秒以下の参照はそのまま使う。
切り取った場合は、参照全文が切り取り音声に一致しなくなるため`transcript`をクローンには使わず、話者埋め込み方式にする。120秒以下の音声に正確な文章を添えればICL方式を使える。
バックエンドの準備結果に`source_reference_seconds`、`reference_seconds`、`reference_trimmed`、`transcript_ignored_due_to_trim`を返す。`reference_sha256`はアップロード元音声のSHA256。

Baseに自由な話し方の指示を与える機能はないため、感情プリセット・追加captionはQwen選択中は無効にしている。
話し方はVoiceDesignの声の説明で指定し、その参照から引き継ぐ。APIもQwenで非neutralプリセットや追加captionを指定した場合は422を返す。

## 生成設定とAPI

| Qwen設定 | 既定 | 範囲 |
|---|---|---|
| `temperature` | 0.9 | 0より大きく2以下 |
| `top_p` | 1.0 | 0より大きく1以下 |
| `top_k` | 50 | 1～1000 |
| `repetition_penalty` | 1.05 | 1～2 |
| `max_new_tokens` | 2048 | 256～4096、チャンク単位 |
| `seed` | ランダム | 0～4294967295 |

サンプラーはtalker/subtalkerとも有効。温度・top-p・top-kは両方に指定する。
Irodori用のRFステップ・CFGはQwen選択時には使わない。seedはチャンクごとに`seed + index`を32bitに丸めて使う。
生成条件・モデルrevision・重みSHA256・参照SHA256・合成時間・RTF・チャンク数を作品に保存する。

既存のAPIに`engine: "qwen"`を追加した。省略時は互換のためIrodori。

```bash
# 準備完了はGET /api/voice/runtimeのreadyで確認
curl -sS -X PUT http://localhost:5080/api/voice/runtime \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"engine":"qwen","mode":"synthesize"}'

curl -sS -X POST http://localhost:5080/api/synthesis \
  -H 'Content-Type: application/json' \
  -d '{"engine":"qwen","text":"今日の出来事を、静かな声でお届けします。","voice_caption":"落ち着いた女性の声。丁寧な日本語のナレーション。","name":"ナレーション","seed":42}'

curl -sS -X PUT http://localhost:5080/api/voice/runtime \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false,"engine":"qwen"}'
```

候補生成は`mode: "design"`をONにして`POST /api/voice/library/candidates`へ`engine: "qwen"`、`caption`、`text`を指定する。
作品へWAVを直接アップロードするAPIは`POST /api/synthesis/upload`。multipartの`file`にWAV、`settings`に`{"engine":"qwen","text":"原稿","reference_text":"参照の読み上げ内容（任意）"}`というJSON文字列を指定する。`voice_id`・`source_product_id`との同時指定はできない。
取り込みWAVのmultipartフィールドに任意の`transcript`を追加できる。
既存の保存作品はIrodoriとして扱い、設定を再利用する時は作品のエンジンを選び直す。

## インストールと保存先

独立した`qwen-tts/.venv`を使い、llm-chatのPython依存へTorch/Transformersを追加しない。
Python 3.12、Torch/Torchaudio 2.10.0+cu128、Transformers 4.57.3、qwen-tts 0.1.1。
依存は`uv.lock`、公式ソースは`022e286b98fbec7e1e916cb940cdf532cd9f488e`で固定。
初回のモデルrevisionもsetupで固定し、取得した重みSHA256は`qwen-tts/install.json`に記録し、通常のsetup再実行ではそのrevisionを維持する。
明示的にモデルを更新する場合は`bash qwen-tts/setup.sh --update-models`。
初回導入後の`serve.sh`はHugging Face/Transformersをオフライン設定で動かす。

```text
qwen-tts/
  setup.sh / serve.sh / service.py / chunking.py
  pyproject.toml / uv.lock
  runtime/                  固定revisionの公式SDK
  models/design/ / base/    モデルとspeech tokenizer
  install.json              revision・SHA256・依存バージョン
  voices/.prompts/           再作成可能なCPUクローンプロンプト
  server.log
  evaluation/               この環境の実機検証データ
```

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `VOICE_SYNTHESIZE_QWEN_URL` | `http://127.0.0.1:8090` | Qwenバックエンド |
| `LLM_QWEN_TTS_API_KEY` | 未設定 | QwenバックエンドとクライアントのBearer認証 |
| `LLM_CHAT_URL` | `http://127.0.0.1:5070` | LLMアンロード先 |
| `VOICE_SYNTHESIZE_PRODUCTION_DIR` | `voice-synthesize/productions` | 共通の作品領域 |
| `LLM_VOICE_LIBRARY_DIR` | `llm-chat/voice-library` | 共通の会話用声ライブラリ |

通常は`voice-synthesize`から起動し、GPU排他とLLMアンロードを使う。
`bash qwen-tts/serve.sh --port 8090`でもバックエンド単体を起動できるが、単体CLI自体はLLMアンロード・GPU予約を行わない。
管理外のローカルQwenが既に起動していれば、制作画面はGPU所有権を確認できないため起動を拒否する。

## 検証結果

RTX 5070 12GB、CUDA BF16、SDPA、LLM停止状態で確認した。

| 内容 | 実測 |
|---|---|
| Baseバックエンド起動・全モデルロード | 約4.13秒（ページキャッシュあり） |
| 声の説明→専用参照→214文字の原稿合成→作品保存 | 約71.45秒 |
| 原稿の合成だけ（2チャンク） | 約52.43秒、完成音声38.76秒、RTF1.35 |
| 保存した同じ声で別原稿を合成 | 約14.99秒、完成音声11.12秒、RTF1.35 |
| 作品ZIP | 完成WAV・原稿・設定・条件・参照WAV・参照文章を確認 |
| VoiceDesign声候補 | 6.32秒の音声、候補APIで生成・保存を確認 |
| 文章のない参照 | 話者埋め込み方式で5.52秒の音声を生成 |
| OFF | バックエンド終了・GPU予約解除、GPU使用量約1.5GiBまで低下 |

検証データは`qwen-tts/evaluation/20261007/`。CPUのWhisper mediumで完成音声を文字起こしし、214文字の原稿と照合した。句読点を除く読み上げ文章の一致を確認した（音質の採点とは別）。実測は機能確認であり、Irodoriとの同一原稿・同一声による音質比較ではない。
最大20000文字や全生成設定の速度・VRAMを保証する測定ではなく、人による音質の採点も実施していない。

Pythonの62テストでGPU排他、切り替え、失敗時の解放、原稿保持、参照文章、エンジン混同の拒否、元作品削除後の声再利用を検証した。
ブラウザーでエンジン選択、Qwen設定、参照文章、候補生成、作品の再制作、ZIP、Irodoriへの画面切り替え、モバイルの配置を確認した。

```bash
uv run --project llm-chat python -m unittest discover -s llm-chat/tests -q
node llm-chat/tests/test_voice_queue.cjs
node llm-chat/tests/test_synthesis_upload.cjs
qwen-tts/.venv/bin/python -m unittest discover -s qwen-tts/tests -q
```

参照アップロードはQwenの作品制作APIとバックエンド準備処理で検証している。バックエンドのテストではモデル推論をモックに置き換え、120秒境界、先頭音声の保持、32 MiBを超える元音声、切り取り時の文章除外、キャッシュ再利用、不正音声の拒否を確認する。実モデルでの120秒参照による音質・VRAM測定は未実施。

2026-10-07に原稿上限を20000文字へ拡張。UI・TXT取り込み・制作API・両バックエンドを揃え、20000文字の原稿保持とZIP保存、20001文字の拒否を検証した。Qwenの実バックエンドAPIでは推論をスタブに置き換え、125チャンク全てに同じ声のプロンプトを渡すことを確認した。20000文字全体の実モデルでの合成は未実施。
