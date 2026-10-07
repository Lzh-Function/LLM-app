"""Adapter for the isolated Qwen3-TTS service and portable voice references."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import HTTPException

from .irodori import IrodoriClient
from .voice_library import wav_info


class QwenClient(IrodoriClient):
    def http(self):
        key = os.environ.get("LLM_QWEN_TTS_API_KEY")
        return httpx.AsyncClient(
            timeout=httpx.Timeout(1800, connect=5),
            trust_env=False,
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )

    async def load(self, mode):
        async with self.http() as client:
            response = await client.post(
                f"{self.url}/local/runtime", json={"mode": mode}
            )
            response.raise_for_status()
            return response.json()

    async def speech(
        self,
        text,
        *,
        voice="none",
        caption,
        mode,
        seed=None,
        references=None,
        settings=None,
    ):
        if mode == "chat":
            raise HTTPException(422, "QwenはGPUでの声作成・作品制作用です")
        options = {**(settings or {}), **(references or {}), "seed": seed}
        # Clone tone comes from the reference; Base does not accept free-form instructions.
        if not options.get("prompt_id"):
            options["caption"] = caption
        started = time.perf_counter()
        try:
            async with self.http() as client:
                response = await client.post(
                    f"{self.url}/v1/audio/speech",
                    timeout=httpx.Timeout(max(1800, len(text) * 12), connect=5),
                    json={
                        "model": "qwen3-tts",
                        "input": text,
                        "voice": voice,
                        "response_format": "wav",
                        "qwen": options,
                    },
                )
                response.raise_for_status()
            elapsed = time.perf_counter() - started
            info = wav_info(
                response.content, max_seconds=4 * 3600, max_bytes=1024 * 1024 * 1024
            )
            details = json.loads(response.headers.get("X-Qwen-Metrics", "{}"))
            return response.content, {
                **details,
                "elapsed_seconds": elapsed,
                "audio_seconds": info["seconds"],
                "rtf": elapsed / info["seconds"],
                "seed": details.get("seed", seed or 0),
                "settings": options,
                "wav_sha256": info["sha256"],
                "sample_rate": info["sample_rate"],
            }
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(503, f"Qwenの音声合成に失敗しました: {exc}") from exc

    async def prepare(self, library, identifier, *, item=None):
        async with self._lock:
            item = item if item is not None else library.get(identifier)
            transcript = item["provenance"].get(
                "reference_text", item["provenance"].get("text", "")
            )
            signature = hashlib.sha256(
                (item["references"][0]["sha256"] + transcript).encode()
            ).hexdigest()
            if signature in self._references:
                return self._references[signature]
            try:
                data = await asyncio.to_thread(
                    library.audio_path(identifier, 0).read_bytes
                )
                async with self.http() as client:
                    response = await client.post(
                        f"{self.url}/local/voices/prepare",
                        data={"transcript": transcript},
                        files={"file": ("reference.wav", data, "audio/wav")},
                    )
                    response.raise_for_status()
                    prepared = response.json()
                options = {"prompt_id": prepared["prompt_id"]}
                self._references[signature] = options
                return options
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                raise HTTPException(
                    503, f"Qwenの参照準備に失敗しました: {exc}"
                ) from exc


def installed_provenance(root: Path, mode: str, health: dict):
    path = root / "qwen-tts" / "install.json"
    key = health.get("runtime", {}).get("model_key")
    checkpoint = health.get("runtime", {}).get("checkpoint")
    local = (
        checkpoint
        and key in ("design", "base")
        and Path(checkpoint).resolve() == (root / "qwen-tts" / "models" / key).resolve()
    )
    manifest = json.loads(path.read_text()) if path.is_file() and local else {}
    expected = manifest.get("models", {}).get(key, {})
    return {
        "engine": "qwen",
        "mode": mode,
        "model": health.get("model", {}),
        "runtime": health.get("runtime", {}),
        "installation": manifest,
        "checkpoint_revision": expected.get("revision"),
        "checkpoint_sha256": expected.get("weights", {}),
    }
