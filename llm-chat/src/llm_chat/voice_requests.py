"""Validated RF settings for voice design and independent high-quality synthesis."""

from typing import Literal

from pydantic import BaseModel, Field


class RFSettings(BaseModel):
    num_steps: int = Field(default=40, ge=4, le=120)
    cfg_scale_text: float = Field(default=3.0, ge=0, le=10)
    cfg_scale_caption: float = Field(default=3.0, ge=0, le=10)
    cfg_scale_speaker: float = Field(default=5.0, ge=0, le=10)

    def sampling_settings(self) -> dict:
        return {name: getattr(self, name) for name in RFSettings.model_fields}


class SynthesisSettings(RFSettings):
    engine: Literal["irodori", "qwen"] = "irodori"
    temperature: float = Field(default=0.9, gt=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=50, ge=1, le=1000)
    repetition_penalty: float = Field(default=1.05, ge=1, le=2)
    max_new_tokens: int = Field(default=2048, ge=256, le=4096)

    def sampling_settings(self) -> dict:
        if self.engine == "qwen":
            return {
                name: getattr(self, name)
                for name in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "max_new_tokens",
                )
            }
        return super().sampling_settings()
