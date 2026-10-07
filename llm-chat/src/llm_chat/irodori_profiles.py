"""One source for endpoint, checkpoint, precision and GPU reservation per mode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    mode: str
    model_key: str
    repo: str
    model_device: str
    model_precision: str
    codec_device: str
    codec_precision: str
    checkpoint_relative: str
    steps: int

    @property
    def uses_gpu(self):
        return any(device != "cpu" for device in (self.model_device, self.codec_device))

    @property
    def identity(self):
        return (
            self.model_key,
            self.repo,
            self.model_device,
            self.model_precision,
            self.codec_device,
            self.codec_precision,
            self.checkpoint_relative,
        )

    def checkpoint(self, root: Path) -> Path:
        return (
            root / "irodori-tts" / "models" / self.model_key / self.checkpoint_relative
        )


def profile_for(mode: str) -> Profile:
    if mode == "chat":
        return Profile(
            mode,
            "chat",
            "Aratako/Irodori-TTS-v4.1-Small-MF",
            os.environ.get(
                "IRODORI_CHAT_MODEL_DEVICE",
                os.environ.get("IRODORI_MODEL_DEVICE", "cpu"),
            ),
            os.environ.get(
                "IRODORI_CHAT_MODEL_PRECISION",
                os.environ.get("IRODORI_MODEL_PRECISION", "fp32"),
            ),
            os.environ.get(
                "IRODORI_CHAT_CODEC_DEVICE",
                os.environ.get("IRODORI_CODEC_DEVICE", "cpu"),
            ),
            os.environ.get(
                "IRODORI_CHAT_CODEC_PRECISION",
                os.environ.get("IRODORI_CODEC_PRECISION", "fp32"),
            ),
            "model.safetensors",
            4,
        )
    if mode not in ("design", "synthesize"):
        raise ValueError(f"Unknown Irodori mode: {mode}")
    precision = os.environ.get("IRODORI_QUALITY_MODEL_PRECISION", "bf16")
    return Profile(
        mode,
        "large",
        "Aratako/Irodori-TTS-v4-Large",
        os.environ.get("IRODORI_QUALITY_MODEL_DEVICE", "cuda"),
        precision,
        os.environ.get("IRODORI_QUALITY_CODEC_DEVICE", "cpu"),
        os.environ.get("IRODORI_QUALITY_CODEC_PRECISION", "fp32"),
        "model.safetensors" if precision == "fp32" else "bf16/model.safetensors",
        40,
    )
