"""Cross-process GPU ownership for llm-chat and voice-synthesize on Linux/WSL."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from fastapi import HTTPException


class GPUReservation:
    def __init__(self, root: Path):
        self.path = root / "irodori-tts" / "gpu.lock"
        self.file = None

    @property
    def fds(self) -> tuple[int, ...]:
        return (self.file.fileno(),) if self.file else ()

    def acquire(self, owner: str):
        if self.file:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            details = stream.read()
            stream.close()
            try:
                current = json.loads(details)["owner"]
            except (KeyError, ValueError):
                current = "別のモデル"
            raise HTTPException(
                409,
                f"GPUを{current}が使用中です。音声モデルをOFFにしてから切り替えてください",
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"owner": owner, "pid": os.getpid()}))
        stream.flush()
        self.file = stream

    def release(self):
        if self.file:
            # Close only: children inherit this descriptor and retain the lock until exit.
            self.file.close()
            self.file = None
