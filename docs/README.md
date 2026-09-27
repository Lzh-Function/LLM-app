# ローカル LLM 環境ドキュメント

WSL2 + Dev Container 上で、llama.cpp を使ってローカル LLM を動かし、
統合チャット UI・ウェブ検索・Codex 連携まで行うための構築記録。

## 環境

| 項目 | 値 |
|---|---|
| GPU | GeForce RTX 5070 (VRAM 12GB、うち Windows 表示で約 1.4〜1.6GB 使用) |
| RAM | 48GB (WSL 内から 47GiB) |
| CPU | Core Ultra 7 270K Plus (24 スレッド) |
| OS | Ubuntu 24.04 (Dev Container on WSL2) |
| ドライバ | 591.86 / CUDA 13.1 |
| ツール | uv 0.12.17, Node 24, Rust, Codex CLI 0.156.1 |

## ディレクトリ構成

```
/workspace/LLM/
├── docs/                 このドキュメント
├── llama.cpp/            共有推論エンジン (prebuilt CUDA 12.8, b11160)
│   ├── bin/              llama-server 本体 + libcudart/libcublas
│   ├── llama-server      LD_LIBRARY_PATH を設定するラッパー
│   └── VERSION
├── llama-prism/          Bonsai 2 専用の Prism ML fork (CUDA 12.8, b10735)
├── ternary-bonsai-2-27b/             PQ2_0 (7.21GB)
├── ternary-bonsai-2-27b-abliterated/ Hikari07jp v0.1 PQ2_0 (7.21GB)
├── qwen3.8-27b/          Dense 27B UD-Q4_K_M (16.5GB)
├── qwen3.6-35b-a3b/      MoE 35B/A3B  UD-Q4_K_XL (21GB)
├── gemma4-26b-a4b/       MoE 26B/A4B  UD-Q4_K_XL (16GB)
├── qwen3.5-9b/           Dense 9B     UD-Q5_K_XL (6.3GB)
├── dolphin3.0-llama3.1-8b/ Dense 8B   Q6_K (6.6GB)
│   ├── models/*.gguf     モデル本体
│   ├── serve.sh          llama-server 起動スクリプト (単体でも使える)
│   ├── model.toml        チャット UI 用メタデータ
│   └── server.log        起動ログ (追記)
└── llm-chat/             統合チャット UI (uv + FastAPI, :5070)
    ├── src/llm_chat/     サーバー / エージェント / 検索 / 静的 UI
    └── history/          保存された会話履歴 (JSON + Markdown)
```

## クイックスタート

```bash
# チャット UI (ブラウザで http://localhost:5070)
cd /workspace/LLM/llm-chat && uv run llm-chat

# Codex からローカル LLM を使う
CTX_SIZE=65536 PORT=8080 /workspace/LLM/qwen3.6-35b-a3b/serve.sh   # ターミナル 1
codex -p qwen-local                                                # ターミナル 2
```

> VRAM 12GB では **同時に 1 モデルしか載らない**。チャット UI と Codex 用サーバーは同時に使わないこと。

Bonsai 2 の 2 モデルは Prism ML fork を使い、既定コンテキストは VRAM を考慮して 16K。
Hikari07jp 版は v0.1 プレビュー。思考切替は両モデルだけ `reasoning_effort` に合わせている。

## ドキュメント一覧

1. [01-setup-models.md](01-setup-models.md) — llama.cpp 導入、Hugging Face からのモデル取得、`serve.sh`、性能実測
2. [02-chat-ui.md](02-chat-ui.md) — 統合チャット UI の構成、API、主要な関数/メソッド、会話履歴
3. [03-web-search.md](03-web-search.md) — ウェブ検索 (検索拡張生成) の仕組み、プロバイダ抽象化、思考ループ対策
4. [04-codex.md](04-codex.md) — Codex CLI との連携、コンテキスト長と速度
5. [05-operations.md](05-operations.md) — 運用・ログ・メモリ・トラブルシューティング
6. [06-voice-chat.md](06-voice-chat.md) — Dev Container 内の AivisSpeech、録音、文字起こし、読み上げ

## 今後の拡張候補

- ローカル文書 RAG (埋め込みモデル Qwen3-Embedding-0.6B を CPU で動かし sqlite-vec/LanceDB に格納、`search_docs` ツールとして追加)
- 検索 API (Tavily / Brave) プロバイダの追加 → [03-web-search.md](03-web-search.md#5-api-プロバイダへの移行)
- Codex 用サーバーの `-ub 2048` によるプロンプト処理高速化の検証
