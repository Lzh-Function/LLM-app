# llm-chat — 統合ローカル LLM チャット

`/workspace/LLM/*/model.toml` を持つディレクトリを LLM として自動検出し、
ブラウザから選んだモデルの `serve.sh` (llama-server) を起動してチャットする。
VRAM 12GB では 1 モデルずつしか載らないため、切り替え時は前のモデルを停止してから起動する。

## 起動

```bash
cd /workspace/LLM/llm-chat
uv run llm-chat          # http://localhost:5070
```

- 画面上部でモデルを選び「切り替え」→ バッジが `ready` になったら送信可能
- 会話の途中でモデルを切り替えても履歴はそのまま引き継がれる
- 「思考モード」オフで `enable_thinking=false` をテンプレートに渡す。
  Bonsai 2 ではオン時に `reasoning_effort=medium` を渡す
- 温度欄が空ならモデルごとの既定値 (`serve.sh` の `--temp`)
- 「ログ」で選択中モデルの `server.log` を表示
- 「履歴を保存」オンで、応答ごとに会話を `history/<日時>-<id>.json` (再読み込み用) と
  `.md` (閲覧用) に保存する。オフの間は何も書き出さない (設定はブラウザに記憶)。
  「過去の会話…」から保存済みの会話を開いて続きを話せる

## 画像添付

「画像添付」から PNG・JPEG・WebP を選ぶか、入力欄へ画像を貼り付ける。
1回の発言に最大4枚を添付でき、送信前に削除できる。サムネイルを押すと拡大表示する。
画像だけの送信も可能。元画像は1枚20MiB・2000万画素以下で、ブラウザー内で長辺1600px以下の
JPEGに変換する（透明部分は白背景、アニメーションは静止画になる）。

対応モデルは Qwen3.5-9B、Qwen3.6-35B-A3B、Qwen3.8-27B、Gemma 4 26B-A4B、
Ternary Bonsai 2 27B とその Abliterated 版。Dolphin はテキストのみ。
画像付きの会話を Dolphin に切り替えた場合は送信できないため、画像対応モデルへ戻すか新しい会話を開始する。
画像は会話の続きにも渡され、「履歴を保存」がオンの場合だけ JSON・Markdown に画像データも保存する。
過去の会話を開くと画像も復元される。音声読み上げ・ウェブ検索も併用できる。

画像用モデルを未取得の環境では、先に以下を実行する。

```bash
bash /workspace/LLM/download-mmproj.sh
```

各 `model.toml` の `mmproj` に画像用ファイルのパスを指定し、`serve.sh` から読み込む。
Bonsai の2モデルは親モデルの Q8_0 projector を共有する。画像エンコーダーは CPU で処理し、
12GB VRAM でテキストモデルの配置を維持する。画像付きの最初の応答はテキストのみの場合より時間がかかる。
単体起動では `MMPROJ` 環境変数でパスを変更できる。

## 音声チャット

ブラウザーの録音ボタンで発話を文字起こしし、認識できたら自動送信する。
「読み上げ」をオンにすると、応答本文を AivisSpeech Engine の「まお」「コハク」、または VOICEVOX Engine の「ずんだもん」「冥鳴ひまり」「中国うさぎ」「東北ずん子」「東北きりたん」で再生する。
「声色を自動選択」がオンなら、話者は選択欄で指定し、声色は LLM が回答ごとに選ぶ。
エンジンは同じ Dev Container 内で直接起動する。起動手順、モデル、設定は
[音声チャットの運用手順](../docs/06-voice-chat.md)を参照。

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
├── ternary-bonsai-2-27b/             Bonsai 2 PQ2_0
├── ternary-bonsai-2-27b-abliterated/ Hikari07jp v0.1 PQ2_0
├── qwen3.8-27b/          Qwen3.8-27B UD-Q4_K_M
├── qwen3.6-35b-a3b/      models/*.gguf, serve.sh, model.toml, server.log
├── gemma4-26b-a4b/       同上
├── qwen3.5-9b/           同上
├── dolphin3.0-llama3.1-8b/ Dolphin 3.0 Llama 3.1 8B Q6_K
└── llm-chat/             このアプリ (FastAPI, ポート 5070 / 内部 llama-server は 5071)
    └── history/          保存された会話履歴
```

## モデルの追加

新しいディレクトリに `models/` へ GGUF を置き、`serve.sh` (PORT/HOST 環境変数を受け取る)
と `model.toml` (`id`, `name`, `description`, `order`) を作ればアプリ再起動で一覧に出る。
Bonsai 2 のように思考オンで `reasoning_effort=medium` を使うモデルには
`thinking_mode = "reasoning_effort"` を追加する。
画像対応モデルには `mmproj = "models/mmproj-F16.gguf"` のように画像用ファイルの相対パスを追加し、
`serve.sh` に `--mmproj` を設定する。

各 `serve.sh` は単体でも実行でき、`http://127.0.0.1:5071/v1` を OpenAI 互換 API として使える。
