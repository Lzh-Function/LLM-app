# 07. Qwen3.8 Flash Next と Strata

Qwen3.8 Flash Next の IQ2_XS 版を [Strata](https://github.com/Niko1221/Strata) で動かす。
MoE のエキスパートは RAM に保持し、一部を GPU にキャッシュする。
大きな n-gram テーブルは SSD 上の GGUF を参照するので、12GB VRAM でも実行できる。
RAM と SSD も使う構成であり、モデル全体が VRAM に収まるわけではない。

## 導入した構成

- Strata 0.1.29、コミット `d6708a4`。Linux バイナリがリリースに無いためソースからビルド。
- [ISTA-DASLab の IQ2_XS](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。エキスパートを削減した Coder 版ではなく、元モデル。
- RTX 5070 12GB、Core Ultra 7 270K Plus、WSL から見える RAM 約47GiB。
- コンテキスト 32,768、8-bit KV、MTP speculative decoding、CJK を含む draft vocabulary。
- 画像エンコーダーは CPU。低 RAM モードと実験的な control vector は無効。
- 公式セットアップが CUDA Toolkit 13.0 を apt で Dev Container 全体へ導入。Python の `.venv` 外に配置し、既存モデルは従来の CUDA 12.8 ランタイムを使用する。

配布元の [RAM・SSD 条件と実測](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md) を確認したうえで、
ホスト SSD の空き198GBを基準に IQ2_XS を選んだ。WSL の `df` はホストの空き容量の代わりに使わない。
通常モードでは GGUF のエキスパートを別ファイルに複製しないため、低 RAM モードの約36GBの追加コピーを避けられる。

## 容量と動作・速度の実測 (2026-09-30)

`du -s -B1` による実使用量:

| 対象 | GB (10億 bytes) | GiB |
|---|---:|---:|
| モデル・画像・変換済み pack・MTP (`Strata-data`) | 77.4 | 72.1 |
| Strata ソース・ビルド・専用 Python 環境 | 0.65 | 0.61 |
| CUDA Toolkit (`/usr/local/cuda-13.0`) | 5.18 | 4.82 |
| NVIDIA 開発ツール (`/opt/nvidia`) | 2.39 | 2.23 |
| 上記合計 | 85.6 | 79.8 |

その他のシステム依存パッケージとファイルシステムの増分もあるため、導入予算は約90GBで見る。
申告された空き198GBには収まる。差し引き約108GBを残す見込みだが、ホストの実残量はWindows側で確認する。

統合 UI の ModelManager から起動し、`/api/chat` → Strata の SSE を実際に中継して測定。
起動から `ready` までは30.04秒。速度測定は思考オフ、温度0.2、生成上限256トークン、MTP有効。

| 入力 | 入力トークン | 生成トークン | 生成速度 | 最初の応答まで | 応答全体 |
|---|---:|---:|---:|---:|---:|
| 日本語の説明文 | 51 | 256 | 56.1 tok/s | 0.60秒 | 5.14秒 |
| Python の BFS コード生成 | 48 | 256 | 62.5 tok/s | 0.71秒 | 4.78秒 |
| 約1Kトークンの英文を踏まえた説明 | 998 | 256 | 66.2 tok/s | 1.54秒 | 5.38秒 |

約1K入力のプロンプト処理は666.6 tok/s。速度はサーバーが返した `timings` のエンジン計測値。
最初の応答と全体時間はクライアントで測定し、HTTP中継も含む。
これらは短い入力での単発実測。生成は256トークンで打ち切り、32K全体を使う長文の速度は未測定。
上流のWindows実測とはプロンプト・RAM・実行環境が異なる。

同時に以下も確認した。

- 思考オンで `17×23 = 391` を回答し、思考と本文を別々にUIへ中継。
- PNGの図形を「左：赤い円／右：青い四角」と正しく回答。画像付きの全体時間1.34秒。
- headless Chromiumで8モデルの選択肢、画像添付ボタン、実際の日本語応答、送信後の操作復帰を確認。
- 速度が `null` の短い応答もUIが完了するよう修正し、ブラウザで確認。
- 既存の画像・音声関連テスト15件が成功。
- 検証終了で推論エンジン・画像エンコーダー・検証用UIがすべて終了し、ポートとGPUメモリを解放。

測定中のWSL全体の使用RAMは約39.4〜39.9GiB (モデル以外のプロセスも含む)。
GPUメモリは監視APIの報告で最大11,844MiB、GPUキャッシュは3,898エキスパート。
検証後はWSL使用RAMが約3.1GiBへ戻った。
エンジンの既定値を使用し、CPU worker数などの追加キャリブレーションは行っていない。

## 起動

統合チャット UI を起動し、モデル一覧で **Qwen3.8 Flash Next 125B (Strata)** を選んで「切り替え」を押す。
テキスト・思考モード・画像添付を既存 UI から利用できる。

```bash
cd /workspace/LLM/llm-chat
uv run llm-chat
# ブラウザで http://localhost:5070 を開く
```

単体の OpenAI 互換 Chat Completions API は以下で起動する。ブラウザは自動で開かない。

```bash
PORT=8080 bash /workspace/LLM/qwen3.8-flash-next/serve.sh
# http://127.0.0.1:8080/v1/chat/completions
```

`/v1/responses` は実装されていないため、Responses API を使うクライアントとはそのまま接続できない。
統合 UI は `/v1/chat/completions` と SSE を使う。
モデルの切り替え・UI 終了時は Strata と子プロセスも停止する。

## 配置・再セットアップ

| パス | 内容 |
|---|---|
| `LLM/strata/` | 上流リポジトリ、専用 `.venv`、ビルド済みエンジン |
| `LLM/strata/strata-iq2_xs.json` | コンテキスト、KV、エンジンとモデルのパス |
| `LLM/Strata-data/models/IQ2_XS/` | 約68GBの GGUF 2分割 |
| `LLM/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 画像エンコーダー |
| `LLM/Strata-data/packs/iq2_xs/` | dense weights、tokenizer、GGUF のエキスパート索引 |
| `LLM/Strata-data/mtp/` | MTP の取得・変換済みデータ |
| `LLM/qwen3.8-flash-next/server.log` | 統合 UI 経由の起動ログ |
| `LLM/strata/strata-iq2_xs.log` | エンジンのログ |

初回導入や途中のダウンロードを再開する場合:

```bash
bash /workspace/LLM/qwen3.8-flash-next/setup.sh
```

完了済みファイルを再利用し、`.part` の取得は途中から再開する。
上流が生成した `run-iq2_xs.sh` はブラウザを開くため、ここでは上記 `serve.sh` を使う。
`strata/` と `Strata-data/` は Git 管理対象から除外している。
上流更新時はサーバーを停止してから `strata/` を更新し、再セットアップする。
このラッパーは既存チェックアウトを自動更新しない。

RAM が約48GBなので、他の大きなアプリや LLM を同時に常駐させると余裕が少ない。
この WSL 環境では、まず検証済みの32K設定を使う。
64K以上では WSL の pinned memory 制限により、配布元セットアップが KV streaming を無効にする。
