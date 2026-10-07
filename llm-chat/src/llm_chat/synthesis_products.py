"""Self-contained production artifacts, independent of the chat voice library."""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import HTTPException

from .voice_library import IDENTIFIER, VoiceLibrary


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class ProductStore:
    def __init__(self, root: Path, legacy: Path | None = None):
        self.root, self.legacy = root, legacy

    def directory(self, identifier: str) -> Path:
        if IDENTIFIER.fullmatch(identifier):
            for root in dict.fromkeys((self.root, self.legacy)):
                if root is not None:
                    path = root / identifier
                    if (path / "metadata.json").is_file():
                        return path
        raise HTTPException(404, "作品が見つかりません")

    def get(self, identifier: str) -> dict:
        try:
            return json.loads(
                (self.directory(identifier) / "metadata.json").read_text()
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, "作品が見つかりません") from exc

    def list(self) -> list[dict]:
        items = {}
        # Prefer a new product over a legacy result with the same identifier.
        for root in dict.fromkeys((self.root, self.legacy)):
            if root is None:
                continue
            for path in root.glob("*/metadata.json"):
                if (
                    not IDENTIFIER.fullmatch(path.parent.name)
                    or path.parent.name in items
                ):
                    continue
                try:
                    items[path.parent.name] = json.loads(path.read_text())
                except FileNotFoundError:
                    continue
        return sorted(items.values(), key=lambda item: item["created_at"], reverse=True)

    def stage(self) -> tuple[str, Path]:
        identifier = uuid.uuid4().hex
        directory = self.root / (".pending-" + identifier)
        directory.mkdir(parents=True)
        return identifier, directory

    def publish(self, stage: Path, data: bytes, metadata: dict, settings: dict) -> None:
        (stage / "audio.wav").write_bytes(data)
        (stage / "manuscript.txt").write_text(metadata["text"])
        write_json(stage / "settings.json", settings)
        write_json(stage / "metadata.json", metadata)
        # Include private voice WAVs and their provenance in the portable package.
        with zipfile.ZipFile(
            stage / "product.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if (
                    path.is_file()
                    and path.name != "product.zip"
                    and path.suffix != ".lock"
                ):
                    archive.write(path, arcname=str(path.relative_to(stage)))
        stage.rename(self.root / metadata["id"])

    def delete(self, identifier: str):
        shutil.rmtree(self.directory(identifier))


def copy_voice(source: VoiceLibrary, item: dict, destination: VoiceLibrary) -> dict:
    """Copy actual WAVs; future changes to the source library cannot change a product."""
    copied = destination.create(
        source.audio_path(item["id"]).read_bytes(),
        name=item["name"],
        kind="voice",
        provenance=item["provenance"],
    )
    for index in range(1, len(item["references"])):
        copied = destination.append_reference(
            copied["id"], source.audio_path(item["id"], index).read_bytes()
        )
    return copied
