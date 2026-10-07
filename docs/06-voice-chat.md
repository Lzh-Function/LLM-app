# 06. 音声チャット（録音ボタン方式）

## 構成

- `llm-chat` (5070) がブラウザーの録音ファイルを受け、Silero VAD と faster-whisper (`medium`, CPU INT8) で文字起こしする。発話が認識でき、モデルが送信可能なら自動送信する。無音・認識失敗時は送らず、モデルが応答中または未起動なら結果を入力欄に残す。
- 既存の `/api/chat` が LLM の回答をストリームで返す。表示する本文だけを句読点で区切り、`llm-chat` から選択した音声エンジンに送る。思考内容と検索イベントは読み上げない。
- AivisSpeech Engine 1.2.0 と VOICEVOX Engine 0.25.2 の Linux x64 版を同じ Dev Container で CPU 実行する。別の Docker コンテナは使わない。
- WAV はブラウザーで再生する。録音や合成音声は会話履歴に保存しない。文字起こししたテキストは、履歴保存をオンにした場合だけ通常の発言として保存する。

## 起動

現在の作業領域には [公式 Linux x64 リリース 1.2.0](https://github.com/Aivis-Project/AivisSpeech-Engine/releases/tag/1.2.0) と、下記 2 モデルを導入済み。別環境で再現する場合は、公式リリースの `AivisSpeech-Engine-Linux-x64-1.2.0.7z.001` を `aivisspeech/` に取得して `uvx --from py7zr py7zr x <取得したファイル> /workspace/LLM/aivisspeech` で展開する。モデルは表の AivisHub 詳細ページから AIVMX を取得し、`aivisspeech/data/AivisSpeech-Engine/Models/` に配置する。

```bash
cd /workspace/LLM/llm-chat
uv run llm-chat                     # http://localhost:5070
```

`uv run llm-chat` 起動時の読み上げはOFF。UIの「読み上げ」をONにした時だけ、導入済みのAivisSpeech・VOICEVOXをバックグラウンドで起動する。
OFFにすると再生・一覧の定期確認を止め、チャット自身が起動した両エンジンと子プロセスを終了してRAMを解放する。起動途中でもOFFにでき、再度ONにすると起動し直す。
既定ポートは127.0.0.1:10101と127.0.0.1:50021、CPU実行。Aivis/VOICEVOXとIrodori CPU会話モードの切り替えではllama.cpp・Strataを再ロードしない。IrodoriのLarge GPU声作成・独立合成はLLMをアンロードする（[手順](08-irodori-tts.md#実装と使い方2026-10-07)）。チャットは音声の起動完了を待たずに利用できる。
Ctrl+Cで終了すると、チャット自身が起動した音声エンジンと子プロセスも停止する。
すでにポートが使用されている場合は新たに起動せず、既存サービスをOFF時・終了時にも停止しない。
外部URLやパス付きプロキシURLを指定している場合、またはエンジンが未導入の場合は、自動起動の対象から外れる。
管理処理は既存の `serve.sh` を実行する。エンジン本体とAIVMXの導入は引き続き別途行う。
エンジン初回起動時のBERTデータ取得や、初回文字起こし時のSTTモデル取得は下記の従来動作を引き継ぐ。

ON後の起動中は `/api/voice/voices` が一時的に503になることがある。画面はON中だけ10秒間隔で再確認し、起動後に自動復帰する。OFF中のAPIは空の一覧を返す。
片方が先に起動した場合はその話者を利用でき、もう片方の起動が終わるまで一覧を再確認する。
「音声再確認」でも取得し直せる。ターミナルには `starting speech engine` / `speech engine ready` と、失敗した場合のログ場所を表示する。
エンジンのログは `aivisspeech/server.log` と `voicevox/server.log` に追記する。

手動で管理する場合は、各エンジンを別ターミナルで先に起動し、UIの「読み上げ」をONにする。

```bash
cd /workspace/LLM/aivisspeech && bash serve.sh
cd /workspace/LLM/voicevox && bash serve.sh
cd /workspace/LLM/llm-chat && uv run llm-chat
```

音声一覧は両エンジンへ並行して問い合わせ、各エンジンへの問い合わせ全体を5秒で打ち切る。
ブラウザー側も8秒で確認を打ち切り、選択欄に接続失敗・タイムアウトを表示する。

録音はブラウザーのマイク権限が必要で、`localhost:5070` から開く。録音ボタンをもう一度押すと終了し、最大 30 秒で自動終了する。発話を認識したら自動送信する。「読み上げ」をオンにすると、選んだ声で応答を再生する。送信中の「中止」は再生中の音声も止める。

「声色を自動選択」がオンのときは、選択欄で**話者**を指定する。LLM は回答の雰囲気に合う声色を、その話者に存在するスタイルから 1 つ選ぶ。選択された声色は回答カード上部に表示し、履歴にはエンジン名・スタイル ID・名前を回答本文とは別に保存する。自動選択をオフにすると手動の声色選択欄が現れ、指定した声色をそのまま使う。

## 音声モデルとデータ

| モデル | 配布元 | ライセンス | SHA-256 |
|---|---|---|---|
| まお 1.2.0 | https://hub.aivis-project.com/aivm-models/a59cb814-0083-4369-8542-f51a29e72af7 | ACML 1.0 | `f87ccea2e8e2de0e0bfe52e803945af903b4086bf25621a015111628f00e4119` |
| コハク 1.1.0 | https://hub.aivis-project.com/aivm-models/22e8ed77-94fe-4ef2-871f-a86f94e9a579 | ACML 1.0 | `3f5c08b52bb8a64efd361268580c81510f96c927cd6905aa7dbae6851333270a` |

エンジン本体は `/workspace/LLM/aivisspeech/Linux-x64/`、モデルは `/workspace/LLM/aivisspeech/data/AivisSpeech-Engine/Models/` に置く。エンジンの初回起動で必要な BERT データも `data/` に保存される。STT モデルは初回の文字起こし時に `/workspace/LLM/llm-chat/voice-models/` へ取得する。これらの大容量ファイルはソース管理に入れない。

試聴用の短文は `/workspace/LLM/aivisspeech/mao-test.wav` と `kohaku-test.wav` に生成済み。ブラウザーの「読み上げ」と音声選択欄からも切り替えられる。

VOICEVOX は [公式 Engine 0.25.2 Linux CPU x64 リリース](https://github.com/VOICEVOX/voicevox_engine/releases/tag/0.25.2) を `voicevox/linux-cpu-x64/` に展開済み。取得した `voicevox_engine-linux-cpu-x64-0.25.2.7z.001` の SHA-256 は `bab016a966131b89bad398e7b898f8c742617dc80a7da49bf494cd07133dbc7a`。同版の `/speakers` から、次の5人だけを画面に表示する。

| VOICEVOX 話者 | スタイル |
|---|---|
| ずんだもん | ノーマル、あまあま、ツンツン、セクシー、ささやき、ヒソヒソ、ヘロヘロ、なみだめ |
| 冥鳴ひまり | ノーマル |
| 中国うさぎ | ノーマル、おどろき、こわがり、へろへろ |
| 東北ずん子 | ノーマル |
| 東北きりたん | ノーマル |

VOICEVOX の合成モデルは公式 Engine 配布物に含まれる。`voicevox/serve.sh` で直接起動し、追加の Docker コンテナは不要。音声エンジン間のスタイル ID 衝突を避けるため、内部では `エンジン:話者名` とエンジン別のスタイル ID を使う。

5人の「ノーマル」で合成した試聴用 WAV を `voicevox/<話者名>-test.wav` に保存済み。

## 初回の動作確認

2026-10-07の初回の起動管理検証では、両エンジンが停止した状態から起動し、6.27秒で7話者の一覧取得が成功。その後、起動時OFF・UIのON/OFFで起動停止する方式へ変更した。
まおの初回合成は2.91秒（WAV 281,872 bytes）、ずんだもんは0.90秒（155,692 bytes）。短文「こんにちは。音声チャットのテストです。」を使用。
ブラウザーで7話者、読み上げの有効化、話者・声色の選択、再確認後の選択維持を確認した。
終了後は自動起動した両エンジンのポートが閉じたことも確認。
既存サービスの再利用、起動途中の停止、終了済みlauncherが残したworkerの回収、ON/OFF API、LLM状態維持などを含む最新のテスト25件が成功。

起動時OFFへの変更後は、SC117 IQ3_XXSを先にロード（55.37秒でready）し、実ブラウザーから音声ON→OFF→再ON→OFFを確認した。
両回のONで7話者を取得し、まお・ずんだもんのWAV合成が成功。OFF中は11秒以上待っても話者一覧への定期問い合わせが発生せず、両エンジンの停止も確認した。
起動途中のOFF、ページ再読み込み時の状態復元も成功。音声切り替え前後でLLMのロード時刻は同一で、OFF後にもStrataが「こんにちは！」と応答した。
検証終了後、UI・LLM・両音声エンジンの検証用ポートがすべて閉じた。結果は `docs/tmp_voice-toggle-results.json` に保存。

同日の同時利用検証では、両音声エンジンで合成してからSC117 IQ3_XXSをロードし、44.31秒でready。
Strataの日本語応答「こんにちは。」をまおで合成し、WAV 105,748 bytesを取得した。
WSL上限48GBで起動・回答・合成は成功したが、音声エンジンが加わると全エキスパートを常駐する余裕は不足した。
Strataの安全処理が、profileで優先する約33.74GiBをRAMに保持し、残りの一部をGGUFからファイル参照するモードへ切り替えた。
応答後の監視APIではWSL全体の使用RAM40.2GiB、GPU11,933MiB。ファイル参照はOSのページキャッシュを使う場合もある。
音声なしで測った生成速度を音声併用時の速度として扱わない。この検証は短い挨拶の生成・合成であり、長文やSTTとの同時実行の性能は未測定。

2026-09-26 に Dev Container 内で AivisSpeech Engine 1.2.0 と `qwen3.5-9b` を起動して測定した。短文「こんにちは。音声チャットのテストです。」を使い、モデル読込後の 1 回の計測値。利用者のマイク入力や長文での性能を示す値ではない。

| 操作 | 所要時間 | 結果 |
|---|---:|---|
| まお・ノーマルの合成 | 0.97 秒 | WAV 283,812 バイト |
| コハク・ノーマルの合成 | 1.35 秒 | WAV 295,374 バイト |
| STT medium / CPU INT8 | 1.77 秒 | 「こんにちは 音声チャットのテストです」 |
| STT small / CPU INT8 | 0.72 秒 | 「こんにちは、音声チャットのテストです。」 |

同じ WAV では `small` の方が速く、文字起こしも成立した。既定値は引き継ぎ案どおり `medium` とし、実マイクの発話で比較してから変更する。

## 設定

| 変数 | 既定値 | 内容 |
|---|---|---|
| `VV_CPU_NUM_THREADS` | `8` | AivisSpeech の CPU スレッド数 |
| `LLM_AIVIS_URL` | `http://127.0.0.1:10101` | `llm-chat` から見た AivisSpeech の接続先 |
| `LLM_VOICEVOX_URL` | `http://127.0.0.1:50021` | `llm-chat` から見た VOICEVOX の接続先 |
| `LLM_STT_MODEL` | `medium` | faster-whisper モデル名。比較時は `small` に変更 |
| `LLM_STT_THREADS` | `8` | STT の CPU スレッド数 |
| `LLM_STT_DIR` | `llm-chat/voice-models` | STT モデル保存先 |

## API

- `GET /api/voice/runtime`: `{enabled, starting, stopping}` を返す。`GET /api/status` の `voice` にも同じ状態を含む。
- `PUT /api/voice/runtime`: `{enabled: true}` で起動、`{enabled: false}` で停止。連続切り替えは直列処理し、ONの重複で二重起動しない。
- `GET /api/voice/voices`: ON中は、まお・コハクと指定の VOICEVOX 5話者のスタイル一覧を取得。OFF中は `[]`。`key` は `エンジン:話者名`。
- `POST /api/voice/transcribe`: `multipart/form-data` の `file`。最大 12 MiB、録音時間 0.2～30 秒。`{text, duration, speech_detected}` を返す。
- `POST /api/voice/synthesize`: `{text, style_id, engine}` を受け、指定エンジンの WAV を返す。テキストは最大 200 文字。旧リクエストは `engine=aivis` として扱う。OFF中は409。
- `POST /api/chat`: 読み上げの自動選択時だけ `voice_speaker` に `エンジン:話者名` を指定する。サーバーはその話者の声色候補をシステム指示に追加し、モデルの `[[VOICE_STYLE:...]]` を SSE の `{voice_style:{id,name}}` イベントへ分離する。回答本文、画面表示、TTS にタグは渡さない。タグがない・候補外の場合は選択欄の声色を使う。

## 現段階の制約

- 発話終了の自動判定と、再生中の発話による割り込みは次段階。現在は録音ボタンで区切る。
- STT の `small` と `medium` の品質・遅延、および TTS のスレッド数は、実マイク入力と llama-server の同時稼働条件で比較する必要がある。
- 読み上げは完成した WAV を順番に再生する。ブラウザー側の自動再生制限で再生できない場合は、画面下部の状態表示を確認する。
