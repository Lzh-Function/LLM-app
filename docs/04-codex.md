# 04. Codex CLI との連携

llama-server は OpenAI **Responses API** (`/v1/responses`) を提供しているため、
Codex CLI の独自プロバイダとして接続できる。ログイン・API キー・トークン課金は不要。

- Codex CLI: 0.156.1 (2026-02 に `wire_api = "chat"` は廃止 → `"responses"` を使う)
- 推奨モデル: **Qwen3.6-35B-A3B** (エージェント型コーディング向け。9B は長い手順で破綻しやすい)

## 1. プロファイル設定

Codex 0.156 の `-p <name>` は **`~/.codex/<name>.config.toml` をベース設定の上に重ねる**仕様。
ベースの `config.toml` には手を入れず、ローカル用プロファイルを別ファイルにしている。

`~/.codex/qwen-local.config.toml`:

```toml
# codex -p qwen-local : ローカル llama.cpp の Qwen3.6-35B-A3B を使う
# 事前に: CTX_SIZE=65536 PORT=8080 /workspace/LLM/qwen3.6-35b-a3b/serve.sh
model = "qwen3.6-35b-a3b"
model_provider = "llamacpp"
model_context_window = 65536

# サンドボックス内のコマンドにネットワークを許可 (hf download / curl で HF API 等を使うため)
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = true

[model_providers.llamacpp]
name = "llama.cpp (local)"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
stream_idle_timeout_ms = 10000000
```

- `env_key` を書かないので API キーは要求されない。
- `workspace-write` は既定でネットワーク遮断だが、`network_access = true` で**既定で許可**している。
  書き込み可能なのは作業ディレクトリ (`-C` で指定) と `/tmp` のみ。
- `~/.codex` は Docker named volume (`devcontainer-codex`) なのでコンテナ再作成後も残る。

## 2. 起動手順

```bash
# ターミナル 1: Codex 用サーバー (チャット UI とは別ポート 8080、ctx 64k)
#   ※ 先にチャット UI のモデルは「停止」しておく (VRAM は 1 モデル分しかない)
CTX_SIZE=65536 PORT=8080 /workspace/LLM/qwen3.6-35b-a3b/serve.sh

# ターミナル 2: インタラクティブ
cd /path/to/project
codex -p qwen-local

# 非インタラクティブ (1 回実行して終了)
codex exec -p qwen-local --skip-git-repo-check -s workspace-write "指示"
```

| 起動方法 | モード |
|---|---|
| `codex -p qwen-local` | インタラクティブ (TUI) |
| `codex exec -p qwen-local "..."` | 非インタラクティブ |

接続確認: 起動時のヘッダーに `model: qwen3.6-35b-a3b` / `provider: llamacpp` が表示される。

`-p` を使わない場合の代替: `codex -c model_provider=llamacpp -c model=qwen3.6-35b-a3b`
(この場合 provider 定義はベースの `config.toml` に必要)。

### Codex 用サーバーの停止

使い終わったら止めて VRAM を解放する (チャット UI を使う前にも必要)。

```bash
# フォアグラウンドで起動した場合: そのターミナルで Ctrl+C
#   (serve.sh は exec で llama-server に置き換わるので Ctrl+C で本体まで止まる)

# バックグラウンド (nohup 等) で起動した場合: ポート 8080 のプロセスだけを止める
fuser -k 8080/tcp

# fuser が無い場合の代替: コマンドラインに --port 8080 を含む llama-server だけを kill
for p in $(pgrep -x llama-server); do
  tr '\0' ' ' < /proc/$p/cmdline | grep -q -- "--port 8080" && kill $p
done

# 停止確認
curl -s localhost:8080/health || echo "stopped"
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 約 1.4GB に戻れば解放済み
```

> - `pkill llama-server` は**使わない**。チャット UI の内部サーバー (5071) まで止まる。
> - `pkill -f 8080` のような `-f` 指定も**使わない**。実行中のシェル自身にも一致して終了する
>   ([05 の 1 章](05-operations.md#1-プロセス管理))。

## 3. 動作確認に使ったコマンド

```bash
# Responses API を直接叩く
curl -s localhost:8080/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","input":"1+1は？数字だけ答えて"}' | jq '.output'

# Codex 非インタラクティブテスト
mkdir -p /tmp/codex-test
codex exec -p qwen-local --skip-git-repo-check --ephemeral -s workspace-write -C /tmp/codex-test \
  "fizzbuzz.py を作成して 1〜15 の FizzBuzz を出力するようにし、python3 で実行して結果を確認して。" < /dev/null

# サーバー側のリクエストごとの速度
grep -E "prompt eval time|       eval time" /workspace/LLM/qwen3.6-35b-a3b/server-codex.log
```

## 4. コンテキスト長と速度 (重要)

Codex はシステムプロンプト + ツール定義だけで約 7,000 トークン使うため、チャット用の 32k より広げる必要がある。
ただし広げるほど KV キャッシュが VRAM を占め、`--fit` が MoE 層を CPU に追い出すので遅くなる。

| ctx | 生成速度 | 初回プロンプト (6,889 tok) | FizzBuzz タスク全体 |
|---|---|---|---|
| 32k (チャット) | 約 51 tok/s | – | – (Codex には狭い) |
| **64k (採用)** | **約 25 tok/s** | 32 秒 (217 tok/s) | **56 秒** |
| 128k | 3.25 tok/s | 58 秒 (119 tok/s) | 4 分以上 |

- 2 回目以降のリクエストはプロンプトキャッシュ (前回との共通部分の再利用) で 1〜3 秒。
- 64k を超える長い作業では Codex の自動 compaction (履歴要約) に頼る。

## 5. モデルのダウンロード (ネットワーク利用)

Codex 組み込みの `web_search` は使えないが、シェル経由の `curl` / `uvx ... hf download` は使える。
ファイル一覧は Hugging Face API で取得できるため、ウェブ検索は不要。

```bash
cd /workspace/LLM && codex -p qwen-local
# 例: 「HF API で unsloth/Qwen3.8-27B-GGUF のファイル一覧を調べ、UD-Q4_K_XL を
#      qwen3.8-27b/models に nohup でバックグラウンドダウンロードして」
```

- 実測: 「HF API で unsloth/Qwen3.5-9B-GGUF のファイル一覧を表示し README.md をダウンロード」を
  `curl` 2 回で正常に完了 (sandbox 表示: `workspace-write ... (network access enabled)`)。
- 数十 GB のダウンロードはコマンドのタイムアウトを避けるため `nohup ... &` を指示する。
- モデルは学習時点以降の新しいリポジトリ名を知らないので、API での検索 (`/api/models?search=...&author=unsloth`) を指示すると確実。

## 6. 既知の警告・制限

| 表示 | 意味 |
|---|---|
| `Model metadata for 'qwen3.6-35b-a3b' not found` | Codex がモデル情報を持っていない。フォールバック値で動作し、今回の範囲では問題なし |
| `Codex could not find bubblewrap on PATH` | サンドボックス用。同梱版で代替される |
| サーバーログ `unsupported Responses tool type 'web_search' skipped` | Codex 組み込みの web_search は llama.cpp 非対応のため使えない |

## 7. Claude Code について (検討のみ)

llama-server は Anthropic 形式 (`/v1/messages`) も提供しており、
`ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=dummy ANTHROPIC_MODEL=qwen3.6-35b-a3b claude`
で技術的には接続可能。ただし公式サポート外・システムプロンプトが大きく 64k を圧迫・
自動要約が 200k 前提、といった理由から **Codex を採用**した (未検証)。

## 8. 今後の改善候補

- `-ub 2048` (ubatch 拡大) で CPU オフロード時の初回プロンプト処理を高速化できる可能性 (未計測)
