"""Start Strata with llm-chat's context selection without editing the installed/Codex config."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def context_config(config: dict, length: int) -> dict:
    updated = dict(config)
    args = list(config.get("args", []))
    if "--max-context" in args:
        index = args.index("--max-context")
        args[index + 1] = str(length)
    else:
        args.extend(("--max-context", str(length)))
    updated["args"] = args
    return updated


def main():
    meta = tomllib.loads((ROOT / "qwen3.8-flash-next/model.toml").read_text())
    try:
        length = int(os.environ.get("CTX_SIZE", str(meta["default_context_length"])))
    except ValueError as exc:
        raise SystemExit("CTX_SIZEは整数で指定してください") from exc
    if not 512 <= length <= meta["max_context_length"]:
        raise SystemExit(
            f"CTX_SIZEは512〜{meta['max_context_length']}で指定してください"
        )
    directory = ROOT / "strata"
    original = directory / "strata-sc117-iq3_xxs.json"
    path = directory / "strata-chat-sc117-iq3_xxs.json"
    config = context_config(json.loads(original.read_text()), length)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    executable = str(directory / ".venv/bin/python")
    print(f"llm-chat: Strata context = {length} tokens", flush=True)
    os.execv(
        executable,
        [
            executable,
            str(directory / "serve/server.py"),
            "--engine",
            "strata",
            "--config",
            str(path),
            *sys.argv[1:],
        ],
    )


if __name__ == "__main__":
    main()
