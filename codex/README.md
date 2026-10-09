# Codex with local Flash Next

```bash
cd /workspace/LLM
uv run codex/setup.py
bash qwen3.8-flash-next/serve-codex.sh
# 別のターミナルから、作業ディレクトリで:
codex -p qwen
```

Qwen3.8-Flash-Next SC117 IQ3_XXSをStrataのResponses APIで利用する。
通常の`codex`は従来のGPTを使い、`codex -p qwen`のときだけQwenの専用設定・カタログを読み込む。
セットアップは以前のグローバルQwen設定をGPTへ戻し、Qwenを別プロファイルとして保存する。
sandboxはworkspace-write、インターネット接続は既定で有効。
Qwenのコンテキストは128K、約98Kで自動要約する。チャット用の32K設定とは別に管理する。

サーバー起動後、`http://127.0.0.1:8080/api-monitor`で入力・思考・応答をリアルタイムに確認できる。
全文をターミナルで追う場合は、別のターミナルで次を実行する。

```bash
bash /workspace/LLM/qwen3.8-flash-next/watch-codex.sh
```

入力・思考・回答・ツール呼び出し・HTTP応答の全文は`qwen3.8-flash-next/codex-traces/`に自動保存する。
`latest/`は最新のサーバー起動分。監視画面は直近100件・各欄262144文字までだが、保存ファイルは省略しない。
ツール実行結果はCodexが次のリクエストに送った範囲で保存する。モデルが出力する思考テキストを記録し、内部テンソルや出力されない計算過程は含まない。

[設定・GPU運用・検証結果](../docs/04-codex.md)を参照。
