"""Speaker-labelled manuscripts and lossless, bounded PCM WAV assembly."""

from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from .voice_library import wav_info

MAX_OUTPUT_SECONDS = 4 * 3600
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
TAG = re.compile(r"\[([^\[\]\r\n]+)\]")


class SpeakerSpec(BaseModel):
    label: str | None = Field(default=None, max_length=80)
    reference_text: str | None = Field(default=None, max_length=6000)

    @field_validator("label")
    @classmethod
    def valid_label(cls, value):
        if value is None:
            return None
        value = value.strip()
        if any(char in value for char in "[]\r\n"):
            raise ValueError("話者ラベルに角括弧や改行は使えません")
        return value or None


@dataclass
class ReferenceUpload:
    filename: str
    data: bytes
    label: str | None = None
    reference_text: str | None = None


def validate_labels(labels: list[str | None]):
    if len(labels) > 1 and any(not label for label in labels):
        raise HTTPException(
            422, "2ファイル以上では、すべての参照WAVに話者ラベルを付けてください"
        )
    named = [label for label in labels if label]
    if len(set(named)) != len(named):
        raise HTTPException(422, "話者ラベルは重複させないでください")


def parse_script(text: str, labels: list[str | None]) -> list[dict]:
    validate_labels(labels)
    if not labels:
        raise HTTPException(422, "参照WAVを選んでください")
    matches = list(TAG.finditer(text))
    if len(labels) == 1 and (not labels[0] or not matches):
        return [{"speaker_index": 0, "label": labels[0], "text": text.strip()}]
    if not matches or text[: matches[0].start()].strip():
        raise HTTPException(422, "複数話者の原稿は [話者ラベル] から始めてください")
    turns = []
    for index, match in enumerate(matches):
        label = match.group(1).strip()
        if label not in labels:
            raise HTTPException(422, f"未登録の話者ラベルです: [{label}]")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        utterance = text[match.end() : end].strip()
        if not utterance:
            raise HTTPException(422, f"[{label}] の読み上げ文章が空です")
        if "[" in utterance or "]" in utterance:
            raise HTTPException(
                422, "話者タグの角括弧が不正です。[話者ラベル] の形式にしてください"
            )
        turns.append(
            {"speaker_index": labels.index(label), "label": label, "text": utterance}
        )
    return turns


def speech_chunks(text: str, max_chars: int = 160) -> list[str]:
    """Keep all manuscript characters while bounding each inference request."""
    result = []
    while len(text) > max_chars:
        # Prefer a completed sentence, then a word boundary, then a hard bound.
        breaks = [
            i + 1 for i, char in enumerate(text[:max_chars]) if char in "。！？.!?\n"
        ]
        end = breaks[-1] if breaks and breaks[-1] >= max_chars // 2 else 0
        if not end:
            space = text.rfind(" ", max_chars // 2, max_chars)
            end = space + 1 if space >= 0 else max_chars
        result.append(text[:end])
        text = text[end:]
    if text:
        result.append(text)
    return result


def assemble_wavs(
    paths: list[Path], gap_seconds: float = 1.0
) -> tuple[bytes, list[dict]]:
    """Insert silence only between segments; never alter generated PCM samples."""
    if not paths:
        raise HTTPException(422, "読み上げる区間がありません")
    buffer = io.BytesIO()
    timeline, params, frames = [], None, 0
    with wave.open(buffer, "wb") as output:
        for index, path in enumerate(paths):
            data = path.read_bytes()
            wav_info(data, max_seconds=MAX_OUTPUT_SECONDS, max_bytes=MAX_OUTPUT_BYTES)
            with wave.open(io.BytesIO(data)) as source:
                current = (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                )
                if params is None:
                    params = current
                    output.setnchannels(params[0])
                    output.setsampwidth(params[1])
                    output.setframerate(params[2])
                elif current != params:
                    raise HTTPException(
                        503, "生成区間のWAV形式が一致しないため、作品を連結できません"
                    )
                channels, width, rate = params
                gap = round(rate * gap_seconds) if index else 0
                end = frames + gap + source.getnframes()
                if (
                    end / rate > MAX_OUTPUT_SECONDS
                    or end * channels * width + 44 > MAX_OUTPUT_BYTES
                ):
                    raise HTTPException(
                        422, "連結した作品は4時間・1 GiB以下にしてください"
                    )
                if gap:
                    silence = (b"\x80" if width == 1 else b"\0") * (
                        gap * channels * width
                    )
                    output.writeframesraw(silence)
                timeline.append(
                    {
                        "start_sample": frames + gap,
                        "end_sample": end,
                        "start_seconds": (frames + gap) / rate,
                        "end_seconds": end / rate,
                        "gap_before_seconds": gap / rate,
                    }
                )
                output.writeframesraw(source.readframes(source.getnframes()))
                frames = end
    return buffer.getvalue(), timeline
