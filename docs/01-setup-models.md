# 01. llama.cpp とモデルのセットアップ

## 1. モデル選定の考え方

- VRAM 12GB に全部載る Dense モデルは 14B 程度が上限。
- **MoE モデル**は推論時に一部のパラメータしか使わないため、エキスパート層を CPU 側 RAM に逃がし、
  アテンション・共有層・KV キャッシュを GPU に置けば 30B 級でも実用速度で動く。
- 採用モデル:

| ディレクトリ | モデル | 種類 | 量子化 | サイズ | 配置 |
|---|---|---|---|---|---|
| `qwen3.6-35b-a3b` | Qwen3.6-35B-A3B | MoE 35B / active 3B | UD-Q4_K_XL | 21GB | GPU + CPU (`--fit`) |
| `gemma4-26b-a4b` | Gemma 4 26B-A4B-it | MoE 26B / active 4B | UD-Q4_K_XL | 16GB | GPU + CPU (`--fit`) |
| `qwen3.5-9b` | Qwen3.5-9B | Dense 9B | UD-Q5_K_XL | 6.3GB | 全層 GPU |
| `dolphin3.0-llama3.1-8b` | Dolphin 3.0 Llama 3.1 8B | Dense 8B | Q6_K | 6.6GB | 全層 GPU |
| `qwen3.8-27b` | Qwen3.8-27B | Dense 27B | UD-Q4_K_M | 16.5GB | GPU + CPU (`--fit`) |
| `ternary-bonsai-2-27b` | Ternary Bonsai 2 27B | Dense 27B | PQ2_0 | 7.21GB | Prism ML fork / GPU |
| `ternary-bonsai-2-27b-abliterated` | Hikari07jp v0.1 | Dense 27B | PQ2_0 | 7.21GB | Prism ML fork / GPU |
| `qwen3.8-flash-next` | Qwen3.8 Flash Next | MoE 125B / active 6B | IQ2_XS | 約68GB | Strata / GPU + RAM + SSD |

Flash Next は専用エンジン Strata を使うため、[07-strata.md](07-strata.md) の手順を参照。

## 2. llama.cpp (推論エンジン) の導入

llama.cpp 導入時はコンテナに CUDA Toolkit (nvcc) が無かったため、**公式の prebuilt CUDA バイナリ**を使用。
後から Strata 用に CUDA Toolkit 13.0 を導入したが、既存の llama.cpp は同梱ランタイムで動く。
ドライバが CUDA 13.1 なので、互換性の確実な **CUDA 12.8 版**を選んだ (RTX 50 系 = sm_120 対応)。

```bash
# 最新リリースと成果物の確認
curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" \
  | jq -r '.[] | .tag_name, (.assets[].name)'

# 本体 + CUDA ランタイム (libcudart/libcublas) を取得して展開
mkdir -p /workspace/LLM/llama.cpp && cd /workspace/LLM/llama.cpp
B=b11160
curl -sL -o llama.tgz  https://github.com/ggml-org/llama.cpp/releases/download/$B/llama-$B-bin-ubuntu-cuda-12.8-x64.tar.gz
curl -sL -o cudart.tgz https://github.com/ggml-org/llama.cpp/releases/download/$B/cudart-llama-$B-bin-ubuntu-cuda-12.8-x64.tar.gz
tar xzf llama.tgz && tar xzf cudart.tgz
mv llama-$B bin && mv cudart-llama-$B-bin-ubuntu-cuda-12.8-x64/* bin/
rmdir cudart-llama-$B-bin-ubuntu-cuda-12.8-x64 && rm llama.tgz cudart.tgz
echo $B > VERSION

# GPU 認識確認
LD_LIBRARY_PATH=bin bin/llama-server --list-devices
#   CUDA0: NVIDIA GeForce RTX 5070 (12226 MiB, 11017 MiB free)
```

### ラッパー `llama.cpp/llama-server`

同梱ライブラリを解決するため `LD_LIBRARY_PATH` を設定してから本体を `exec` する。

```bash
#!/usr/bin/env bash
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin"
export LD_LIBRARY_PATH="${BIN_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${BIN_DIR}/llama-server" "$@"
```

### アップデート手順

上記の `B=` を新しいタグにして同じ手順で `bin/` を差し替えるだけ (モデル・スクリプトは変更不要)。

### Bonsai 2 専用の Prism ML fork

PQ2_0 は独自カーネルと Hadamard 変換を使うため、通常の `llama.cpp` では実行できない。
既存の推論エンジンを残したまま、Prism ML の CUDA 12.8 バイナリ
`prism-b10735-842b188` を `llama-prism/bin/` に配置した。`llama-prism/llama-server` は
同ディレクトリのライブラリと既存の CUDA 12.8 ランタイムを読み込む。
各 Bonsai モデルの `serve.sh` のみがこのエンジンを使う。

モデル本体は以下をそれぞれの `models/` に取得した。

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir /workspace/LLM/ternary-bonsai-2-27b/models
hf download Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF \
  Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf \
  --local-dir /workspace/LLM/ternary-bonsai-2-27b-abliterated/models
```

12GB VRAM では 32K コンテキストが収まらないという配布元の既知の問題があるため、
両モデルの `CTX_SIZE` 既定値は 16K にした。`--fit` が有効なので空き VRAM が減れば
設定が自動調整される。Hikari07jp 版は `reasoning_effort=medium` を既定にし、
チャット UI の思考オフは `enable_thinking=false` に変換する。

Qwen3.8-27B は Unsloth の UD-Q4_K_M を取得し、既存の llama.cpp で起動する。
重みが VRAM 12GB を超えるので `--fit` によって一部を CPU に配置する。

```bash
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf \
  --local-dir /workspace/LLM/qwen3.8-27b/models
```

Dolphin 3.0 は[公式の Llama 3.1 8B GGUF](https://huggingface.co/dphn/Dolphin3.0-Llama3.1-8B-GGUF)から Q6_K を取得する。
ChatML 形式のため `--jinja` で内蔵テンプレートを使う。配布元はシステムプロンプトの設定を推奨しており、
チャット UI の「システム」欄に例えば `You are Dolphin, a helpful AI assistant.` を指定できる。

```bash
hf download dphn/Dolphin3.0-Llama3.1-8B-GGUF Dolphin3.0-Llama3.1-8B-Q6_K.gguf \
  --local-dir /workspace/LLM/dolphin3.0-llama3.1-8b/models
```

## 3. Hugging Face からのモデルダウンロード

Python は uv 経由で使う方針のため、`huggingface_hub` の `hf` CLI を **`uvx` で一時実行**する (インストール不要)。

```bash
# 利用可能な GGUF とサイズの確認 (API)
curl -s "https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main?recursive=true" \
  | jq -r '.[] | select(.type=="file") | "\(.size/1e9|.*100|floor/100)GB \(.path)"' | grep gguf

# ダウンロード (各モデルディレクトリの models/ に保存)
cd /workspace/LLM/qwen3.6-35b-a3b
uvx --from huggingface_hub hf download unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --local-dir models

cd /workspace/LLM/gemma4-26b-a4b
uvx --from huggingface_hub hf download unsloth/gemma-4-26B-A4B-it-GGUF \
    gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf --local-dir models

cd /workspace/LLM/qwen3.5-9b
uvx --from huggingface_hub hf download unsloth/Qwen3.5-9B-GGUF \
    Qwen3.5-9B-UD-Q5_K_XL.gguf --local-dir models

# 完了後、中間キャッシュは削除してよい
rm -rf /workspace/LLM/*/models/.cache
```

メモ:
- 長時間かかるので `nohup ... > download.log 2>&1 &` でバックグラウンド実行した。
- hf-xet は領域を先に確保して書き込むため、途中の `du` は進捗の目安にならない。
- 未認証でも取得可能 (`HF_TOKEN` を設定するとレート制限が緩和される)。

## 4. 起動スクリプト `serve.sh`

### 画像入力用ファイル

Qwen・Gemma・Bonsai の6モデルの画像用ファイルは、クローン後に以下で取得する（合計約4.6GB）。

```bash
bash /workspace/LLM/download-mmproj.sh
```

Qwen と Gemma はそれぞれの Unsloth 配布リポジトリの `mmproj-F16.gguf` を使用する。
Bonsai は Prism ML 配布の `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` を使用し、
Abliterated 版も同じファイルを共有する。ダウンロードは `.part` に保存し、完了後に正式名へ変更する。
各 `serve.sh` は `--mmproj` でこのファイルを読み込み、未取得なら取得方法を表示して終了する。
`--no-mmproj-offload` で画像エンコーダーを CPU に配置し、VRAM の追加消費を抑える。

2026-09-29 に6モデルすべてを既定の起動設定で起動し、統合チャット API 経由で画像を送信した。
全モデル（Hikari07jp の Abliterated 版を含む）が、テスト画像の「赤い円と青い四角」を正しく回答した。
添付の使い方は [llm-chat/README.md](../llm-chat/README.md#画像添付) を参照。

各モデルディレクトリに置く。**単体で OpenAI 互換 API サーバーとして使える**し、チャット UI からも呼ばれる。
環境変数 `HOST` / `PORT` / `CTX_SIZE` で上書きでき、追加引数は `"$@"` でそのまま渡る。

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ../llama.cpp/llama-server \
    --model models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --alias qwen3.6-35b-a3b \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-5071}" \
    --ctx-size "${CTX_SIZE:-32768}" \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --fit on --fit-target 768 \
    --threads 8 \
    --jinja --reasoning-format deepseek \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 1.5 \
    "$@"
```

### 主なオプション

| オプション | 意味 |
|---|---|
| `--fit on --fit-target 768` | VRAM に収まるよう未指定パラメータを自動調整。MoE では**エキスパート重みを自動で CPU に逃がす** (`--n-cpu-moe` の手動調整が不要)。768MiB の余白を残す |
| `--n-gpu-layers all` + `--fit off` | 9B / Dolphin 8B の全層を GPU に固定 |
| `--flash-attn on` | Flash Attention |
| `--cache-type-k/v q8_0` | KV キャッシュを 8bit 量子化して VRAM 節約 |
| `--jinja` | GGUF 内蔵のチャットテンプレートを使う (ツール呼び出しに必須) |
| `--reasoning-format deepseek` | 思考を `reasoning_content` に分離して返す |
| `--presence-penalty 1.5` | (Qwen) 思考の反復ループ防止。Qwen 公式の思考モード推奨値 |
| `--alias` | API 上のモデル名 |

### モデル別サンプリング

| モデル | temp | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| Qwen3.6-35B-A3B / Qwen3.5-9B | 1.0 | 0.95 | 20 | 1.5 |
| Gemma 4 26B-A4B | 1.0 | 0.95 | 64 | – |

> 当初 Qwen は temp 0.6 / presence_penalty なしにしていたが、ウェブ検索時に思考が反復ループしたため公式推奨値に変更した
> ([03-web-search.md](03-web-search.md#6-思考ループ問題と対策) 参照)。

### 単体での利用例

```bash
PORT=8080 /workspace/LLM/qwen3.6-35b-a3b/serve.sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"こんにちは"}],
       "chat_template_kwargs":{"enable_thinking":false}}' | jq -r '.choices[0].message.content'
```

llama-server が提供する主なエンドポイント: `/health`, `/v1/chat/completions`, `/v1/responses` (Codex 用), `/v1/messages` (Anthropic 形式)。

## 5. 性能実測 (ctx 32k, 思考オフ)

| モデル | 起動時間 | 生成速度 | VRAM |
|---|---|---|---|
| Qwen3.6-35B-A3B | 約 10 秒 | 約 51 tok/s | 約 11.5GB |
| Gemma 4 26B-A4B | 約 15 秒 | 約 45 tok/s | 約 11.6GB |
| Qwen3.5-9B | 約 4 秒 | 約 74 tok/s | 約 8.5GB |
| Qwen3.8-27B UD-Q4_K_M | 約 17 秒 | 約 6.8 tok/s | 約 10.5GB |
| Dolphin 3.0 Llama 3.1 8B Q6_K | 約 3 秒 | 約 80 tok/s | 約 9.4GiB |

Qwen3.8-27B は 32K コンテキスト・思考オフで、整数列挙の 256 トークン生成を
2 回測定して 6.84 / 6.83 tok/s。8K コンテキストでは 7.27 tok/s だったため、
既定値は他モデルと同じ 32K とした。VRAM 使用量は `nvidia-smi` の GPU 全体値で、
32K 時は 10,482 MiB 使用 / 1,462 MiB 空き。重みが 16.5GB の Dense モデルで、
一部レイヤーの CPU オフロードにより、既存の MoE モデルより生成が遅い。
上の既存モデルの速度値は別の入力で測ったため、厳密な同条件比較ではない。

Dolphin は 32K コンテキストで全層 GPU に載せ、84 トークンの日本語応答で
80.29 tok/s を記録した。起動中の GPU 全体使用量は 9,442 MiB。
統合チャット UI 経由の通常応答も確認済み。現行 GGUF の ChatML テンプレートでは
OpenAI 互換 API の `tools` を渡してもツール定義がプロンプトに反映されず、
強制ツール呼び出しでも通常の文章応答になった。Dolphin での UI のウェブ検索は
機能が確認できていないため、通常チャット用途として扱う。

`--load-mode none` (mmap 無効) も試したが、Gemma が 45 → 31 tok/s に低下したため**既定 (mmap) のまま**にしている。

## 6. モデルの追加方法

1. `/workspace/LLM/<新ディレクトリ>/models/` に GGUF をダウンロード
2. 既存の `serve.sh` をコピーしてモデルパス・alias・サンプリングを変更 (`HOST`/`PORT` 環境変数を受けること)
3. `model.toml` を作成:
   ```toml
   id = "my-model"
   name = "My Model"
   description = "説明"
   order = 4
   ```
4. チャット UI を再起動すると一覧に出る
