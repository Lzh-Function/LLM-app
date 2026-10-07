# 07. Qwen3.8 Flash Next と Strata

Qwen3.8 Flash Next の SC117 Abliterated IQ3_XXS 版を [Strata](https://github.com/Niko1221/Strata) で動かす。
低RAM常駐モードで、GPUに置くエキスパートを除いた分を RAM に保持する。
大きな n-gram テーブルは SSD 上の GGUF を参照するので、12GB VRAM でも実行できる。
RAM と SSD も使う構成であり、モデル全体が VRAM に収まるわけではない。

## 導入した構成

- Strata 0.1.40.1、コミット `82f46a8`、CUDA エンジン 0.1.40。Linux バイナリがリリースに無いためソースからビルド。2026-10-07 に 0.1.29 (`d6708a4`) から更新。
- [SC117 の IQ3_XXS](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/eec4e10a2c29b440a2ae85591fa11569ce752963/IQ3_XXS)。リビジョン `eec4e10a2c29b440a2ae85591fa11569ce752963` に固定。2026-10-07 に ISTA-DASLab IQ2_XS から置換。
- RTX 5070 12GB、Core Ultra 7 270K Plus、Windows物理RAM 64GB、WSL上限48GB（見える RAM 約47GiB）。WSLの上限は増量していない。
- ベース設定・従来の速度測定はコンテキスト32,768。現在の統合チャットは128Kが既定で、UIから256Kまで指定できる。
- 8-bit KV、MTP speculative decoding、CJK を含む draft vocabulary。
- CPU expert pool は旧IQ2_XSの実測で採用した15 workersを引き継ぐ。WSLでP/Eコア情報が取得できないため、校正情報にも保存。IQ3_XXSのworkers比較は未実施。
- 画像エンコーダーは CPU。`--resident-experts` を有効化し、実験的な control vector は無効。
- 起動時のresident安全余裕は `STRATA_RESIDENT_HEADROOM_GIB=6`。既定4GiBの検証より常駐量を減らすために設定。
- 公式セットアップが CUDA Toolkit 13.0 を apt で Dev Container 全体へ導入。Python の `.venv` 外に配置し、既存モデルは従来の CUDA 12.8 ランタイムを使用する。

ホストSSDの空き容量はWSLの `df` から判断しない。今回もユーザーの指示により空き容量の確認は行っていない。
SC117専用のnative packを作り直し、エキスパートをGGUFから直接参照する。
上流セットアップの低RAMモードが作る約43GBの `experts.bin` は作らず、モデルを重複保存しない。

## SC117への置換と共用ファイル (2026-10-07)

旧IQ2_XSの第1分割、専用dense packと索引、旧ダウンロード完了マーカーを削除（合計40,764,086,799 bytes）。
旧モデルはディスクに残していないため、以前の設定・エンジンのバックアップだけではIQ2_XSへ戻せない。
削除対象一覧は `strata/backups/2026-10-07-sc117/removed-iq2-files.json` に保存。

SC117 IQ3_XXSのGGUFは合計76,142,283,328 bytes（約76.14GB）。
第2分割28,800,138,432 bytesは旧モデルのファイルと配布元のSHA-256が一致したため、SC117のファイル名へ移動して再利用。
新規取得は第1分割47,342,144,896 bytesのみ。両分割のSHA-256を検証し、`.part` から正式名へ変更する。
MTPの取得・変換済みデータも共用する。MTPは元のQwenのdraft headで、Abliterated化された重みではない。
手元の `mtp-q2_0.gguf`、実行用 `dense.bin`・`experts.bin`・`draft_vocab.bin` はSC117配布版とSHA-256が一致。
比較結果は `strata/backups/2026-10-07-sc117/mtp-comparison.json` に保存。

画像モデルは名前だけで判断せず、SC117配布版と比較した。
手元907,543,008 bytes、SC117版907,542,944 bytesとファイルサイズは64 bytes異なるが、
334テンソルの名前・型・形状・相対オフセット、および907,523,008 bytesの全重み領域のSHA-256が一致。
重み領域のSHA-256は `8623953074e256b4b213a6237903a6ca0dd5993eb6c5f11995741bb4cb76e36b`。
差分は `general.name`、`general.quantized_by`、`general.repo_url`、`general.tags` のメタデータのみなので、既存mmprojを再利用した。
比較結果は `strata/backups/2026-10-07-sc117/mmproj-comparison.json` に保存。

tokenizerもSC117のGGUFから書き出した結果と旧版を比較し、一致すれば共用する。
native packのdense weightsと索引はSC117から作り直す。
配布者の[移行手順](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/blob/eec4e10a2c29b440a2ae85591fa11569ce752963/strata/README.md)のとおり、旧packをSC117に流用しない。

tokenizerの5ファイルは書き出し結果とバイト単位で一致したため、旧版を `Strata-data/shared/qwen3.8-flash-next-tokenizer/` へ移動して共用。
新packの `tokenizer/` はこのディレクトリへのシンボリックリンク。
旧モデルからのデータファイルの純増は約8.12GB（第1分割の差と小さな索引差）。Windows側SSDの実残量は測定していない。

## SC117 IQ3_XXSの実測 (2026-10-07、48GB設定)

上記設定で統合UIのModelManagerから起動し、58.03秒でready。
エンジンログでresidentモードを確認。RAMに37.11GiBのエキスパートを保持し、GPU cacheは2,571エキスパート。
`experts.bin` の追加コピーも、SSD参照のみのmmapモードへのフォールバックも発生していない。

思考オフ、温度0.2、生成上限256トークン、MTP有効の単発測定:

| 入力 | 入力トークン | 生成トークン | 生成速度 | 最初の応答まで | 応答全体 |
|---|---:|---:|---:|---:|---:|
| 日本語の説明文 | 51 | 256 | 54.0 tok/s | 0.73秒 | 5.45秒 |
| PythonのBFS | 48 | 256 | 59.1 tok/s | 0.76秒 | 5.07秒 |
| 約1Kトークンの英文を踏まえた説明 | 998 | 256 | 54.8 tok/s | 1.51秒 | 6.17秒 |

約1K入力の処理は698.6 tok/s。画像の形と色（左：赤い円、右：青い四角）も正答し、画像リクエストの全体時間は3.55秒。
思考オンの `17×23=391`、Responses API、既存UIのweb_searchツール定義による呼出し・検証用結果の受け渡しも確認。
ツール検証の検索結果は固定データであり、外部検索の実行速度・結果品質の測定ではない。
headless ChromiumでSC117の表示名、画像添付可能状態、日本語の実応答、送信後の操作復帰も確認。
既存UIの画像・音声関連テスト15件も成功。

WSL全体のメモリを1秒ごとに計測し、`MemTotal - MemAvailable` の最大は42.83GiB、最小availableは4.21GiB。
画像リクエスト後の監視APIは使用RAM42.7GiB、GPUは各リクエスト後の監視で最大11,899MiB。
swap使用量は開始時1.59GiB、起動中の最大3.18GiBと約1.58GiB増えた。48GBでも起動・推論は通ったが、余裕が大きい構成ではない。
residentの安全余裕6GiBは起動時の割当判定に使う値で、実行中ずっと6GiBの空きを保証するものではない。
エンジンの検証中ログでは、RAMとVRAMのエキスパート交換後もSSDからのexpert blob読み込みは0。
PLE参照や、RAMに退避できなかったprefill用GPU領域ではSSD参照があり得るため、全処理でSSDを使わないという意味ではない。

既定4GiBの最初の検証では、resident量38.64GiB、WSLメモリの最大44.03GiB、swapは1.43→最大3.84GiB。
6GiBに調整後、上記機能検証を再実行して全て成功した。
初回の生成速度は35.7〜39.8 tok/sで、再検証は54.0〜59.1 tok/s。キャッシュ・同時実行負荷なども異なるため、差を設定変更だけの効果とは断定しない。
32Kコンテキスト設定での短い入力の測定であり、32K全体を埋める入力や複数アプリとの同時実行は未測定。

検証結果・メモリ時系列・スクリプトは `strata/backups/2026-10-07-sc117/`、最初の設定での結果はその `headroom4/` に保存。
検証終了後はUI・Strata・画像エンコーダーを停止し、メモリを解放した。

上記の速度・メモリ測定では音声エンジンを起動していない。
同日の音声併用検証では、AivisSpeech・VOICEVOXで合成してからStrataを起動し、回答の読み上げまで成功。
ただし、音声エンジンのメモリが加わるため、Strataは安全処理で約33.74GiBのexpertをRAMに保持し、残りの一部をGGUFのファイル参照へ切り替えた。
同時利用の詳細は [音声チャットの実測](06-voice-chat.md#初回の動作確認) を参照。
音声は起動時OFFで、UIの「読み上げ」をONにすると起動、OFFにすると停止する。LLMは再ロードしない。
Strataのexpert配置はモデルのロード時に決まるため、音声OFFでRAMを解放しても、配置が自動で拡大するわけではない。

## この環境に関係する更新 (2026-10-07)

| 更新 | この構成への影響 |
|---|---|
| [0.1.30](https://github.com/Niko1221/Strata/releases/tag/v0.1.30)、[0.1.38](https://github.com/Niko1221/Strata/releases/tag/v0.1.38)、[0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39) の入力処理・MTP・生成カーネルの改善 | RTX 5070 の通常チャットにも適用。上流の速度測定は主に Q2_0 / IQ3_XXS なので、IQ2_XS の結果は下の実測を参照。長文の改善幅は空きVRAMとpackによる。 |
| [0.1.31](https://github.com/Niko1221/Strata/releases/tag/v0.1.31) の CJK tokenizer 修正 | 日本語などが改行なしで長く続く入力のtokenizeが高速化。日本語用 MTP vocabulary (`cjk`) を明示して維持。 |
| [0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39)、[0.1.40](https://github.com/Niko1221/Strata/releases/tag/v0.1.40) の CPU worker 推奨・P/E コア判定修正 | Core Ultra 7 270K Plus は 8P + 16E、推奨15 workers。ただしこのWSLのsysfsには判定情報が無く、再セットアップだけでは既定23 workers。実測で15を選び、再セットアップでも保持されるよう校正情報に保存。 |
| [0.1.40](https://github.com/Niko1221/Strata/releases/tag/v0.1.40) の Linux read-ahead / メモリ読み込み改善 | ソースをビルドし直すと適用。起動は今回28.01秒、最終設定では32.82秒。0.1.29時の30.04秒とはファイルキャッシュなどの条件が異なるため、cold startの改善率は未測定。 |
| [0.1.37](https://github.com/Niko1221/Strata/releases/tag/v0.1.37)〜[0.1.40.1](https://github.com/Niko1221/Strata/releases/tag/v0.1.40.1) の停止・再起動・NaN修正 | フリーズやキャンセル後の復帰を改善。再起動中の待機リクエストが永久に待つ問題も修正。コードや思考内に引用されたtool callの誤認も修正され、UIのウェブ検索にも関係する。 |
| [0.1.39 の Responses API](https://github.com/Niko1221/Strata/releases/tag/v0.1.39) | Strataから `/v1/responses` を直接利用可能。stateless方式で、`previous_response_id`・hosted tools・reasoning summaryには非対応。統合UIは引き続きChat Completionsを利用。 |
| [0.1.38](https://github.com/Niko1221/Strata/releases/tag/v0.1.38)、[0.1.40](https://github.com/Niko1221/Strata/releases/tag/v0.1.40) の HTTP / 画像入力チェック | Host・Originの検証とネットワーク上の画像パスの制限が追加。既存UIのlocalhost中継とdata URL画像は動作確認済み。 |

並列生成は12GBカードでexpert cacheを圧迫するため1リクエストずつを維持。
AMD・Strix Halo・複数GPUの改善は現在の単体RTX 5070には該当しない。
64K以上のWSLでKV streamingが使えない制約も残るため、32K / int8 KVを維持する。
低RAMモードのGGUF直接参照は、上記のSC117への置換で利用する。以下の更新実測は置換前のIQ2_XSの記録。

更新後、統合UI経由の日本語・思考・画像入力、ブラウザ操作、Responses APIの実応答を確認。
Strataの設定・CPU判定・tool call・再起動関連テスト123件と、既存UIの画像・音声関連テスト15件が成功。
モデル・pack・MTPは既存ファイルを再利用し、再ダウンロードは無し。

同じプロンプト、思考オフ、温度0.2、生成256トークンの単発比較:

| 入力 | 0.1.29 生成 tok/s | 更新後23 workers | 最終15 workers | 最終設定の最初の応答まで |
|---|---:|---:|---:|---:|
| 日本語の説明文 | 56.1 | 70.7 | 77.7 | 0.57秒 |
| Python BFS | 62.5 | 83.6 | 86.8 | 0.57秒 |
| 約1Kトークンの英文を踏まえた説明 | 66.2 | 74.5 | 85.7 | 1.45秒 |

約1K入力の処理は旧版666.6、更新後23 workersで744.6、最終15 workersで730.9 tok/s。
画像の形と色は両設定で正答。画像の全体時間は23 workersで1.23秒、15 workersで3.99秒であり、画像処理の高速化は確認できていない。
計測中のWSL使用RAMは約36.0〜39.5GiB、GPUは最大11,944MiB。
旧測定は2026-09-30、今回は2026-10-07で、同時実行アプリ・キャッシュ・生成内容が異なり得る。
速度差はこの入力での参考値であり、32K長文や全入力での改善を保証しない。

CPU workersは同じエンジンで23→15→23の順に比較。
上流の校正用3プロンプト、温度0、生成128トークンで、各起動後に1巡warm-up、2巡測定した。
各巡の3入力の中央値を取り、さらに2巡の中央値を取ると、23 workersは68.9 / 82.1 tok/s、15 workersは85.7 tok/s。
23 workersの測定間にも変動があり、GPU cache量も違ったため、改善率を固定値として扱わない。
15 workersを採用し、`strata-iq2_xs.json` の `--pool-workers 15` とStrataのハードウェア別校正情報へ保存した。
他のCPUやコンテキスト・画像設定を変えた場合は、校正情報のハードウェアキーが変わるので再測定する。
最終15 workersでも、日本語・思考・画像・Responses API・ブラウザUIの検証を再実行して成功。
検証終了時はエンジン・画像エンコーダー・検証用UIを停止し、WSLのavailable RAMは約45GiBに戻った。
検証データと再実行用スクリプトは `strata/backups/2026-10-07-update-checks/` に保存した。

## 初回導入時の容量と動作・速度の実測 (2026-09-30、0.1.29)

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

統合チャット UI を起動し、モデル一覧で **Qwen3.8 Flash Next 125B SC117 (Strata)** を選んで「切り替え」を押す。
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
# 長さを変更する場合:
CTX_SIZE=196608 PORT=8080 bash /workspace/LLM/qwen3.8-flash-next/serve.sh
```

0.1.39から `/v1/responses` も利用できる（stateless方式、上の対応範囲を参照）。
統合 UI は `/v1/chat/completions` と SSE を使う。
モデルの切り替え・UI 終了時は Strata と子プロセスも停止する。
統合UIの「コンテキスト」変更も、同じモデルを停止して再起動する。
詳細と全モデルの上限は[llm-chatのコンテキスト設定](../llm-chat/README.md#コンテキスト長)を参照。

## 配置・再セットアップ

| パス | 内容 |
|---|---|
| `LLM/strata/` | 上流リポジトリ、専用 `.venv`、ビルド済みエンジン |
| `LLM/strata/strata-sc117-iq3_xxs.json` | コンテキスト、KV、エンジンとモデルのパス |
| `LLM/strata/strata-chat-sc117-iq3_xxs.json` | 起動時に生成するチャット用設定。UIで選んだコンテキストを反映 |
| `LLM/qwen3.8-flash-next/chat_server.py` | `CTX_SIZE`をチャット用Strata設定へ反映するランチャー |
| `LLM/qwen3.8-flash-next/strata-config.json` | 上記設定の再生成用テンプレート（`@ROOT@` は導入先へ置換） |
| `LLM/Strata-data/models/sc117_iq3_xxs/` | 約76.14GBの GGUF 2分割、SHA-256と配布元記録 |
| `LLM/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 画像エンコーダー |
| `LLM/Strata-data/packs/sc117_iq3_xxs/` | SC117専用dense weights、tokenizer、GGUFのエキスパート索引。`experts.bin` の複製なし |
| `LLM/Strata-data/mtp/` | MTP の取得・変換済みデータ |
| `LLM/qwen3.8-flash-next/server.log` | 統合 UI 経由の起動ログ |
| `LLM/strata/strata-sc117-iq3_xxs.log` | エンジンのログ |

既存Strata環境でSC117を再取得・再設定、または途中のダウンロードを再開する場合:

```bash
bash /workspace/LLM/qwen3.8-flash-next/setup.sh
```

完了済みファイルを再利用し、`.part` の取得は途中から再開する。
`setup.sh` は `install_sc117.py` を使い、上流のモデル選択セットアップは呼び出さない。
既存のStrataエンジン・専用venv・MTP・画像モデルが必要。新規のStrata本体導入は上流の手順に従う。
SC117のGGUFとpackだけを取得・生成し、追加の巨大なモデルキャッシュや `experts.bin` を作らない。
統合UIからは上記 `serve.sh` を使う。ブラウザは自動で開かない。
`strata/` と `Strata-data/` は親リポジトリの Git 管理対象から除外している。
モデルの配布リビジョンは固定し、Strataの既存チェックアウト・エンジンは自動更新しない。
サーバーを停止し、Strataソースの変更を `git -C /workspace/LLM/strata status --short` で確認してから更新する。
Strata本体を更新する際はその版の変更内容を確認し、更新後のソースからエンジンを再ビルドして検証する。
SC117用 `setup.sh` はエンジンを再ビルドしないため、`git pull` とこのスクリプトだけでは本体の更新は完了しない。

```bash
bash /workspace/LLM/qwen3.8-flash-next/setup.sh
```

0.1.40.1では上流履歴の整理で通常のpullが一度失敗する変更があった。
今回は旧HEADを `backup/pre-0.1.40.1-20261007` ブランチに残して最新タグへ切り替え済み。
旧エンジン・設定・起動スクリプトは `strata/backups/2026-10-07-v0.1.29/` に保存。
これらはIQ2_XS時代の記録。現在のSC117設定を0.1.29で使う手順は検証していない。
IQ2_XSへ戻す場合は専用の第1分割とpackを再取得・再生成し、ラッパーも戻す必要がある。

RAM が約48GBなので、他の大きなアプリや LLM を同時に常駐させると余裕が少ない。
統合チャットのFlash Nextは128Kが既定で、モデル上限の256KまでUIから選択できる。
大きくした場合、起動できるかはRAM・VRAMの余裕に依存する。Codex用は別の128K設定で、約84Kの入力とツール操作を実機で確認した。
メモリ実測・起動手順は[04-codex.md](04-codex.md)を参照。
64K以上では WSL の pinned memory 制限により、配布元セットアップが KV streaming を無効にする。
