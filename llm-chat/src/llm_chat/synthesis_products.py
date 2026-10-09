"""Self-contained production artifacts, independent of the chat voice library."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from fastapi import HTTPException

from .voice_library import IDENTIFIER, VoiceLibrary, serialized_mutation


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_archive(directory: Path, target: Path, replacements: dict | None = None):
    replacements = replacements or {}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "product.zip" and path.suffix != ".lock":
                relative = str(path.relative_to(directory))
                archive.write(replacements.get(relative, path), arcname=relative)


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
        write_archive(stage, stage / "product.zip")
        stage.rename(self.root / metadata["id"])

    @serialized_mutation
    def rename(self, identifier: str, name: str) -> dict:
        directory = self.directory(identifier)
        metadata = self.get(identifier)
        metadata["name"] = name
        # Prepare the complete replacement package before changing saved files.
        with tempfile.TemporaryDirectory(prefix=".rename-", dir=directory.parent) as temp:
            pending = Path(temp)
            replacements = {"metadata.json": pending / "metadata.json"}
            write_json(replacements["metadata.json"], metadata)
            if (directory / "settings.json").is_file():
                settings = json.loads((directory / "settings.json").read_text())
                settings["name"] = name
                replacements["settings.json"] = pending / "settings.json"
                write_json(replacements["settings.json"], settings)
            if (directory / "product.zip").is_file():
                write_archive(directory, pending / "product.zip", replacements)
                (pending / "product.zip").replace(directory / "product.zip")
            for filename, path in reversed(list(replacements.items())):
                path.replace(directory / filename)
        return metadata

    @serialized_mutation
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
    for original, reference in zip(item["references"], copied["references"]):
        for key in ("trimmed_from_seconds", "source_sha256"):
            if key in original:
                reference[key] = original[key]
    return destination.save(copied)
