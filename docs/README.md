# ローカル LLM 環境ドキュメント

WSL2 + Dev Container 上で、llama.cpp と Strata を使ってローカル LLM を動かし、
統合チャット UI・ウェブ検索・Codex 連携まで行うための構築記録。

## 環境

| 項目 | 値 |
|---|---|
| GPU | GeForce RTX 5070 (VRAM 12GB、うち Windows 表示で約 1.4〜1.6GB 使用) |
| RAM | Windows物理64GB、WSL上限48GB（WSL内から約47GiB） |
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
├── strata/              Qwen3.8 Flash Next 用 Strata (CUDA 13.0)
├── Strata-data/         Flash Next SC117 IQ3_XXS、共用MTP、画像エンコーダー
├── qwen3.8-flash-next/   統合 UI から Strata を起動するラッパー
├── ternary-bonsai-2-27b-abliterated/ Hikari07jp v0.1 PQ2_0 (7.21GB) + mmproj
├── gemma4-26b-a4b/       MoE 26B/A4B  UD-Q4_K_XL (17.01GB)
├── qwen3.5-9b/           Dense 9B     UD-Q5_K_XL (6.74GB)
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
bash /workspace/LLM/qwen3.8-flash-next/serve-codex.sh  # ターミナル 1
codex -p qwen                                     # ターミナル 2（通常のcodexはGPT）
```

> VRAM 12GB では **同時に 1 モデルしか載らない**。チャット UI と Codex 用サーバーは同時に使わないこと。

Bonsai 2 Abliteratedは Prism ML fork を使い、既定コンテキストは VRAM を考慮して 16K。
Hikari07jp 版は v0.1 プレビュー。Bonsai 2 と Flash Next の思考切替は `reasoning_effort` に合わせている。

2026-10-07にQwen3.6-35B-A3B、Qwen3.8-27B、Bonsai通常版、Dolphinを削除し、現在のLLMは上記4モデル。Bonsaiの画像エンコーダーは改変版の`models/`に移した。音声はIrodori会話用Small-MF・作品用Large BF16・共有codecとQwen-TTSを保持し、比較用Small RFとLarge変換元FP32だけを整理した。[削除結果と容量](10-storage-audit.md#削除実施結果)

## ドキュメント一覧

1. [01-setup-models.md](01-setup-models.md) — llama.cpp 導入、Hugging Face からのモデル取得、`serve.sh`、性能実測
2. [02-chat-ui.md](02-chat-ui.md) — 統合チャット UI の構成、API、主要な関数/メソッド、会話履歴
3. [03-web-search.md](03-web-search.md) — ウェブ検索 (検索拡張生成) の仕組み、プロバイダ抽象化、思考ループ対策
4. [04-codex.md](04-codex.md) — Strata / Flash Nextを使うCodex CLI、sandboxネット接続、コンテキスト
5. [05-operations.md](05-operations.md) — 運用・ログ・メモリ・トラブルシューティング
6. [06-voice-chat.md](06-voice-chat.md) — Dev Container 内の AivisSpeech、録音、文字起こし、読み上げ
7. [07-strata.md](07-strata.md) — Qwen3.8 Flash Next SC117 IQ3_XXS、Strataの導入と実測、起動手順
8. [08-irodori-tts.md](08-irodori-tts.md) — Irodori-TTSの比較、声ライブラリとllm-chatへの実装、独立voice-synthesize、GPU/CPU運用・実測・評価ツール

9. [09-qwen-tts.md](09-qwen-tts.md) — Qwen3-TTS 1.7Bの導入、声作成・長文クローン、GPU排他と独立作品保存
10. [10-storage-audit.md](10-storage-audit.md) — 保存容量の実測、各モデルの役割、断捨離候補と回収容量

## 今後の拡張候補

- ローカル文書 RAG (埋め込みモデル Qwen3-Embedding-0.6B を CPU で動かし sqlite-vec/LanceDB に格納、`search_docs` ツールとして追加)
- 検索 API (Tavily / Brave) プロバイダの追加 → [03-web-search.md](03-web-search.md#5-api-プロバイダへの移行)
- Codex 用サーバーの `-ub 2048` によるプロンプト処理高速化の検証
