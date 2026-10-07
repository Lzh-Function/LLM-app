"""Start Codex's local Strata endpoint with the same GPU exclusion as voice/chat."""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def unload_chat():
    request = urllib.request.Request("http://127.0.0.1:5070/api/unload", method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=60) as response:
            state = json.load(response)
        if state.get("active") is not None or state.get("status") != "stopped":
            raise SystemExit("llm-chatのLLMアンロードを確認できません")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, OSError) and exc.reason.errno == errno.ECONNREFUSED:
            return
        raise SystemExit(f"llm-chatのLLMを停止できません: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    directory = ROOT / "strata"
    config = directory / "strata-codex-sc117-iq3_xxs.json"
    template = ROOT / "qwen3.8-flash-next" / "strata-codex-config.json"
    if directory.is_dir() and template.is_file():
        config.write_text(template.read_text().replace("@ROOT@", str(ROOT)))
    if not config.is_file():
        raise SystemExit("先に bash qwen3.8-flash-next/setup.sh を実行してください")
    with socket.socket() as probe:
        if probe.connect_ex((args.host, args.port)) == 0:
            raise SystemExit(
                f"ポート{args.port}は使用中です。起動済みサーバーを確認してください"
            )
    unload_chat()
    path = ROOT / "irodori-tts" / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.seek(0)
        try:
            owner = json.load(stream).get("owner", "別のモデル")
        except ValueError:
            owner = "別のモデル"
        raise SystemExit(
            f"GPUを{owner}が使用中です。音声モデルをOFFにしてから起動してください"
        ) from exc
    stream.seek(0)
    stream.truncate()
    json.dump({"owner": "Codex用Strata Flash Next", "pid": os.getpid()}, stream)
    stream.flush()
    os.set_inheritable(stream.fileno(), True)
    executable = str(directory / ".venv" / "bin" / "python")
    os.execv(
        executable,
        [
            executable,
            str(directory / "serve" / "server.py"),
            "--engine",
            "strata",
            "--config",
            str(config),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
    )


if __name__ == "__main__":
    main()
