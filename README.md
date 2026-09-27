# ローカル LLM 環境

構築方法と運用方法は [docs/README.md](docs/README.md) を参照。

## GitHub で管理する場合

このディレクトリをリポジトリのルートにする。`serve.sh`、`model.toml`、
`llm-chat` のソース、`uv.lock`、手順書を管理し、ダウンロードしたモデル、
推論エンジンのバイナリ、会話履歴、録音、ログ、認証情報は `.gitignore` で除外する。
private リポジトリでも認証情報や会話履歴をコミットしないこと。

最初にこのディレクトリで `git init` し、初回のコミット前に
`git status --short` で追加対象を確認する。
除外理由は `git check-ignore -v <パス>` で確認できる。
新しい種類のモデルや生成物を追加した場合は、コミット前に `.gitignore` を更新する。

GitHub CLI を使う場合は、追加対象を確認してコミットした後、
`gh repo create <owner>/<repo> --private --source=. --remote=origin --push`
で private リポジトリを作成できる。作成後は GitHub 上でも公開範囲を確認する。

クローン後は [モデルの取得手順](docs/01-setup-models.md) と
[音声機能の導入手順](docs/06-voice-chat.md) に従い、除外されたファイルを各環境で取得する。
`llama.cpp/bin/` と `llama-prism/bin/` の実行ファイルも取得し直す必要がある。
`.gitignore` は既に追跡済みのファイルには効かないため、該当する場合は
`git rm --cached <パス>` で追跡対象から外す。過去のコミットに含まれた秘密情報は
履歴からの除去と認証情報の更新が必要になる。
