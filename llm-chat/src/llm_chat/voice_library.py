"""Persistent, atomic voice library. A selected WAV, not its seed, is the identity."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import re
import shutil
import uuid
import wave
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from fastapi import HTTPException

IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")
MAX_WAV_BYTES = 32 * 1024 * 1024


def serialized_mutation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        with self.root.with_suffix(".lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            return method(self, *args, **kwargs)
    return wrapped


def wav_info(data: bytes, *, max_seconds: int = 120, max_bytes: int = MAX_WAV_BYTES) -> dict:
    if not data or len(data) > max_bytes:
        raise HTTPException(422, f"WAVは{max_bytes // (1024 * 1024)} MiB以下にしてください")
    try:
        with wave.open(io.BytesIO(data)) as audio:
            frames, rate = audio.getnframes(), audio.getframerate()
            if rate <= 0 or not 0.2 <= frames / rate <= max_seconds:
                raise ValueError("duration")
            if audio.getnchannels() not in (1, 2) or audio.getsampwidth() not in (
                1,
                2,
                3,
                4,
            ):
                raise ValueError("format")
            # A truncated header-only WAV must never become a reference.
            expected = frames * audio.getnchannels() * audio.getsampwidth()
            if len(audio.readframes(frames)) != expected:
                raise ValueError("truncated")
            return {
                "sha256": hashlib.sha256(data).hexdigest(),
                "seconds": frames / rate,
                "sample_rate": rate,
                "channels": audio.getnchannels(),
                "bytes": len(data),
            }
    except (wave.Error, EOFError, ValueError) as exc:
        raise HTTPException(422, f"0.2～{max_seconds}秒のPCM WAVを指定してください") from exc


class VoiceLibrary:
    def __init__(self, root: Path):
        self.root = root

    def directory(self, identifier: str) -> Path:
        if not IDENTIFIER.fullmatch(identifier):
            raise HTTPException(404, "声が見つかりません")
        return self.root / identifier

    def get(self, identifier: str) -> dict:
        try:
            return json.loads(
                (self.directory(identifier) / "metadata.json").read_text()
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, "声が見つかりません") from exc

    def list(self) -> list[dict]:
        result = []
        for path in self.root.glob("*/metadata.json"):
            if IDENTIFIER.fullmatch(path.parent.name):
                try:
                    result.append(self.get(path.parent.name))
                except HTTPException as exc:
                    if exc.status_code != 404:
                        raise
        return sorted(result, key=lambda item: item["created_at"], reverse=True)

    def save(self, item: dict) -> dict:
        path = self.directory(item["id"]) / "metadata.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
        temp.replace(path)
        return item

    @serialized_mutation
    def create(self, data: bytes, *, name: str, kind: str, provenance: dict) -> dict:
        info = wav_info(data)
        identifier = uuid.uuid4().hex
        directory = self.directory(identifier)
        directory.mkdir(parents=True)
        try:
            (directory / "reference.wav").write_bytes(data)
            return self.save(
                {
                    "id": identifier,
                    "name": name,
                    "kind": kind,
                    "favorite": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "references": [{"file": "reference.wav", **info}],
                    "provenance": provenance,
                }
            )
        except BaseException:
            shutil.rmtree(directory)
            raise

    @serialized_mutation
    def update(
        self, identifier: str, *, name=None, favorite=None, register=False
    ) -> dict:
        item = self.get(identifier)
        if name is not None:
            item["name"] = name
        if favorite is not None:
            item["favorite"] = favorite
        if register:
            item["kind"] = "voice"
        return self.save(item)

    @serialized_mutation
    def append_reference(self, identifier: str, data: bytes) -> dict:
        item = self.get(identifier)
        if item["kind"] != "voice":
            raise HTTPException(409, "先に声を登録してください")
        info = wav_info(data)
        if sum(ref["seconds"] for ref in item["references"]) + info["seconds"] > 120:
            raise HTTPException(422, "同じ話者の参照は合計120秒以下にしてください")
        filename = f"reference-{uuid.uuid4().hex}.wav"
        path = self.directory(identifier) / filename
        path.write_bytes(data)
        item["references"].append(
            {"file": filename, **info, "source": "same-speaker-upload"}
        )
        try:
            return self.save(item)
        except BaseException:
            path.unlink()
            raise

    def audio_path(self, identifier: str, index: int = 0) -> Path:
        item = self.get(identifier)
        if index < 0 or index >= len(item["references"]):
            raise HTTPException(404, "参照が見つかりません")
        path = self.directory(identifier) / item["references"][index]["file"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != item["references"][index]["sha256"]
        ):
            raise HTTPException(409, "参照WAVのSHA256が一致しません")
        return path

    @serialized_mutation
    def delete(self, identifier: str) -> None:
        self.get(identifier)
        shutil.rmtree(self.directory(identifier))
