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

[設定・GPU運用・検証結果](../docs/04-codex.md)を参照。
