# llm-chat — 統合ローカル LLM チャット

`/workspace/LLM/*/model.toml` を持つディレクトリを LLM として自動検出し、
ブラウザから選んだモデルの `serve.sh` (llama-server または Strata) を起動してチャットする。
VRAM 12GB では 1 モデルずつしか載らないため、切り替え時は前のモデルを停止してから起動する。

## 起動

```bash
cd /workspace/LLM/llm-chat
uv run llm-chat          # http://localhost:5070
```

- 画面上部でモデルを選び「切り替え」→ バッジが `ready` になったら送信可能
- 会話の途中でモデルを切り替えても履歴はそのまま引き継がれる
- 「コンテキスト」でトークン数を指定して「切り替え」を押すと適用。同じモデルでも値を変えれば再起動する
- 「思考モード」オフで `enable_thinking=false` をテンプレートに渡す。
  Bonsai 2 と Qwen3.8 Flash Next ではオン時に `reasoning_effort=medium` を渡す
- 温度欄が空ならモデルごとの既定値 (`serve.sh` の `--temp`)
- 「ログ」で選択中モデルの `server.log` を表示
- 「履歴を保存」オンで、応答ごとに会話を `history/<日時>-<id>.json` (再読み込み用) と
  `.md` (閲覧用) に保存する。オフの間は何も書き出さない (設定はブラウザに記憶)。
  「過去の会話…」から保存済みの会話を開いて続きを話せる

## コンテキスト長

画面上部の数値欄は512トークンからモデルの上限まで指定できる。
候補には128K・192K・256Kも含め、各モデルの上限を超える候補は表示しない。
1K = 1024トークンで、文字数ではない。入力・システム指示・会話履歴・出力を合わせた長さ。
値はモデルごとにブラウザーへ記憶する。変更は再起動後に適用し、状態バッジには実際に起動した長さを表示する。
履歴は再起動で削除しない。

| モデル | 既定 | 上限 |
| --- | --- | --- |
| Qwen3.8 Flash Next / Strata | 128K | 256K |
| Qwen3.5-9B / Qwen3.6-35B-A3B / Qwen3.8-27B / Gemma 4 | 32K | 256K |
| Bonsai 2 / Bonsai 2 Abliterated | 16K | 256K |
| Dolphin 3.0 Llama 3.1 8B | 32K | 128K |

上限は導入済みGGUFの`<architecture>.context_length`から確認して`model.toml`へ記載した。
モデルの対応上限であって、このPCで起動・生成できることを保証する値ではない。
長いコンテキストはKVキャッシュ等のメモリを増やすため、速度低下だけでなく起動失敗・OOMも起こり得る。
起動に失敗した場合はエラーとログを確認し、値を下げて再度「切り替え」を押す。
メモリ割り当て失敗は、モデル名・指定トークン数・再試行の案内とともに表示する。
強制終了は「メモリ不足の可能性」として扱い、原因を断定しない。現在の起動分のログは「詳細ログ」で確認できる。
後処理中は「メモリを解放中」と表示し、親サーバーと同じプロセスグループのエンジン・画像処理を停止する。
通常停止が20秒で完了しなければ強制終了し、終了確認にも期限を設ける。
停止を確認してからGPU予約を解放し、コンテキスト変更・別モデルへの再試行を有効にする。
停止を確認できない例外は`cleanup_failed`として表示し、モデルの重複起動を防ぐため予約を保持する。
「停止」から再試行できる。
KVの配置・形式・アーキテクチャで増加量は異なる。
[llama.cppの設定資料](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)も参照。

APIは`POST /api/models/<id>/load`に`{"context_length": 131072}`を送る。
本文を省略した従来の呼び出しは、そのプロセス内で最後に選んだ値、未選択ならモデルの既定値を使う。
上限超過・512未満・整数以外は422で拒否し、起動中モデルには触れない。

llama.cppには`CTX_SIZE`経由で`--ctx-size`を渡し、統合UIでは1スロットとして使用する。
Strataは`chat_server.py`がチャット専用の実行設定に`--max-context`を設定する。
導入元の設定とCodexの128K設定は書き換えない。単体起動でも`CTX_SIZE=196608 bash qwen3.8-flash-next/serve.sh`で変更できる。

検証ではFlash Nextを統合APIから128Kで起動し、同じモデルを192Kへ変更してチャット応答を確認した。
llama.cpp側もDolphinの8K起動・16Kへの再起動とチャット応答を確認した。
256KはUI/APIの許可範囲と設定への反映をテスト済みだが、実機での起動・256K入力は未検証。
コンテキスト設定が大きくても、テストで上限いっぱいの履歴を埋めたわけではない。
失敗処理は実際のプロセスを使ってOOMログ・SIGKILL・起動タイムアウトを模擬し、
終了した親が残した子プロセスの停止・GPU予約の解放・短いコンテキストと別モデルへの復帰を確認した。
ブラウザーでもエラー表示・詳細ログ・後処理完了後の再試行を確認した。実メモリを枯渇させるテストは行っていない。

## 画像添付

「画像添付」から PNG・JPEG・WebP を選ぶか、入力欄へ画像を貼り付ける。
1回の発言に最大4枚を添付でき、送信前に削除できる。サムネイルを押すと拡大表示する。
画像だけの送信も可能。元画像は1枚20MiB・2000万画素以下で、ブラウザー内で長辺1600px以下の
JPEGに変換する（透明部分は白背景、アニメーションは静止画になる）。

対応モデルは Qwen3.5-9B、Qwen3.6-35B-A3B、Qwen3.8-27B、Gemma 4 26B-A4B、
Ternary Bonsai 2 27B とその Abliterated 版、Qwen3.8 Flash Next (Strata)。Dolphin はテキストのみ。
画像付きの会話を Dolphin に切り替えた場合は送信できないため、画像対応モデルへ戻すか新しい会話を開始する。
画像は会話の続きにも渡され、「履歴を保存」がオンの場合だけ JSON・Markdown に画像データも保存する。
過去の会話を開くと画像も復元される。音声読み上げ・ウェブ検索も併用できる。

画像用モデルを未取得の環境では、先に以下を実行する。

```bash
bash /workspace/LLM/download-mmproj.sh
```

Qwen3.8 Flash Next の画像用ファイルは専用のセットアップで取得する。
導入と実測条件は [Strata の運用手順](../docs/07-strata.md)を参照。

各 `model.toml` の `mmproj` に画像用ファイルのパスを指定し、`serve.sh` から読み込む。
Bonsai の2モデルは親モデルの Q8_0 projector を共有する。画像エンコーダーは CPU で処理し、
12GB VRAM でテキストモデルの配置を維持する。画像付きの最初の応答はテキストのみの場合より時間がかかる。
単体起動では `MMPROJ` 環境変数でパスを変更できる。

## 音声チャット

ブラウザーの録音ボタンで発話を文字起こしし、認識できたら自動送信する。
「読み上げ」をオンにすると、応答本文を AivisSpeech Engine の「まお」「コハク」、または VOICEVOX Engine の「ずんだもん」「冥鳴ひまり」「中国うさぎ」「東北ずん子」「東北きりたん」で再生する。
「声色を自動選択」がオンなら、話者は選択欄で指定し、声色は LLM が回答ごとに選ぶ。
`uv run llm-chat` 起動時の読み上げはOFF。UIの「読み上げ」をONにすると導入済みの両エンジンをCPUで起動し、OFFにすると停止してメモリを解放する。LLMのロード状態は維持する。
すでに手動起動済みのサービスは再利用し、OFF時にも停止しない。チャット終了時にも、自身が起動した音声エンジンを停止する。
起動手順、モデル、設定は
[音声チャットの運用手順](../docs/06-voice-chat.md)を参照。
ON中に一覧の取得に失敗した場合や、一部のエンジンがまだ起動中の場合は、10秒間隔で再確認する。OFF中は確認しない。
エンジンを後から起動してもページの再読み込みなしで利用できる。「音声再確認」で手動確認も可能。

### Irodoriと声ライブラリ

Irodoriも音声エンジンとして選択できる。「声ライブラリ」から説明文とseedで声の候補を作り、
試聴・お気に入り・名前変更・登録を行う。手持ちのWAVも登録可能。
候補生成はLarge GPU BF16 / RF40、会話はCPU Small-MF4。声作成の起動時にLLMをアンロードする。選んだWAVを永続IDで参照し、参照latentを再利用する。
同じ話者の補助参照を追加でき、LLMの声色選択は固定の感情captionに変換する。
合成済み音声を再生しながら次の文を作り、「音声停止」は再生だけ、「読み上げOFF」は管理プロセスを終了する。
Irodori選択時はAivisSpeech/VOICEVOXを新たに起動しない。

```bash
cd /workspace/LLM
bash irodori-tts/setup.sh   # 専用CUDA環境、Large・MF・比較用RF・codecを導入
cd llm-chat
uv run llm-chat
```

「声ライブラリ」→「声作成モードを起動」→候補生成・登録→「会話モードで試す」→話者選択で利用する。
保存先は`llm-chat/voice-library/`。起動時はOFFで、ON後にモデルのロードと短文ウォームアップを行う。
CPU単独の初回動作確認ではMF4のRTF約1.09～1.36で、会話向けの目標にはまだ届いていない。
高品質合成専用の独立UIは`bash voice-synthesize/serve.sh`（リポジトリ直下で実行）、`http://localhost:5080`。
原稿と声の説明からIrodori LargeまたはQwen3-TTS 1.7BをGPUで使って音声作品を制作し、`voice-synthesize/productions/`へWAV・原稿・設定・作品専用の声を保存する。
会話用の声登録は不要。登録済みの声を使う場合も作品内へコピーして保存し、ZIPで一式を取得できる。GPU音声がONの間はLLMロードを409で止める。
Qwenの導入・声作成・長文制作は[Qwen3-TTSの運用手順](../docs/09-qwen-tts.md)を参照。
設定・実測・MF/Large比較・Strata同時実行の測定は[Irodoriの運用手順](../docs/08-irodori-tts.md#実装と使い方2026-10-07)を参照。

## ウェブ検索

「🌐 ウェブ検索」をオンにすると、モデルにツール `web_search` / `fetch_page` を渡す。
モデルが必要と判断したときだけ検索し、サーバー側で実行した結果を渡して出典番号 [n] 付きで回答させる
(`src/llm_chat/agent.py`)。検索語と閲覧 URL は外部に送信されるので既定はオフ。

| 環境変数 | 既定 | 内容 |
|---|---|---|
| `LLM_SEARCH_PROVIDER` | `duckduckgo` | 検索プロバイダ |
| `LLM_SEARCH_REGION` | `jp-jp` | DuckDuckGo の地域 |
| `LLM_SEARCH_MAX_RESULTS` | `5` | 1 回の検索で返す件数 |
| `LLM_SEARCH_MAX_ROUNDS` | `6` | ツール呼び出しの最大ラウンド数 (超えたらツールなしで回答) |
| `LLM_FETCH_MAX_CHARS` | `5000` | ページ本文の最大文字数 |
| `LLM_SEARCH_THINKING_BUDGET` | `2048` | 1 ラウンドあたりの思考トークン上限 (思考ループ防止) |

### 検索 API への移行

`src/llm_chat/search/providers.py` に `SearchProvider` (`name` と
`async search(query, max_results) -> list[SearchResult]`) を実装したクラスを追加し、
`PROVIDERS` に登録して `LLM_SEARCH_PROVIDER=<name>` で起動するだけ。UI やエージェント側の変更は不要。

## 構成

```
/workspace/LLM/
├── llama.cpp/            共有ランタイム (prebuilt CUDA 12.8, bin/ + llama-server ラッパー)
├── llama-prism/          Bonsai 2 専用の Prism ML 版 llama.cpp
├── strata/              Strata 本体と専用 Python 環境
├── Strata-data/         Flash Next の GGUF、変換済みデータ、MTP
├── qwen3.8-flash-next/   Strata 用の serve.sh / setup.sh / model.toml
├── ternary-bonsai-2-27b-abliterated/ Hikari07jp v0.1 PQ2_0 + mmproj
├── gemma4-26b-a4b/       models/*.gguf, serve.sh, model.toml, server.log
├── qwen3.5-9b/           同上
└── llm-chat/             このアプリ (FastAPI, ポート 5070 / 内部 llama-server は 5071)
    └── history/          保存された会話履歴
```

## モデルの追加

新しいディレクトリに `models/` へ GGUF を置き、`serve.sh` (PORT/HOST 環境変数を受け取る)
と `model.toml` (`id`, `name`, `description`, `order`) を作ればアプリ再起動で一覧に出る。
コンテキストを変更できるよう、`default_context_length`と`max_context_length`も指定し、
`serve.sh`で`CTX_SIZE`環境変数を受け取る。省略時は安全側に両方32768として扱う。
Bonsai 2 のように思考オンで `reasoning_effort=medium` を使うモデルには
`thinking_mode = "reasoning_effort"` を追加する。
画像対応モデルには `mmproj = "models/mmproj-F16.gguf"` のように画像用ファイルの相対パスを追加し、
`serve.sh` に `--mmproj` を設定する。

各 `serve.sh` は単体でも実行でき、`http://127.0.0.1:5071/v1` を OpenAI 互換 API として使える。
