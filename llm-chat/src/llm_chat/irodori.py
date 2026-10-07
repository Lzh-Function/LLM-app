"""Irodori OpenAI adapter. Captions use named presets independent of style IDs."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx
from fastapi import HTTPException

from .irodori_profiles import profile_for
from .voice_library import VoiceLibrary, wav_info

URL = os.environ.get("LLM_IRODORI_URL", "http://127.0.0.1:8088").rstrip("/")
PRESETS = [
    {"id": "neutral", "name": "自然", "caption": "自然で穏やかな話し方。"},
    {"id": "happy", "name": "喜び", "caption": "明るく楽しそうな話し方。"},
    {
        "id": "calm",
        "name": "落ち着き",
        "caption": "落ち着いて丁寧に、ゆっくり説明する話し方。",
    },
    {
        "id": "sad",
        "name": "悲しみ",
        "caption": "悲しみを感じる、静かで控えめな話し方。",
    },
    {
        "id": "encouraging",
        "name": "励まし",
        "caption": "相手を優しく励ます、温かい話し方。",
    },
    {"id": "surprised", "name": "驚き", "caption": "驚きを感じる、少し弾んだ話し方。"},
]


class IrodoriClient:
    def __init__(self, url: str = URL):
        self.url = url
        self._references: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    def http(self):
        key = os.environ.get("LLM_IRODORI_API_KEY")
        return httpx.AsyncClient(
            timeout=httpx.Timeout(900, connect=5),
            trust_env=False,
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )

    async def health(self) -> dict:
        async with self.http() as client:
            response = await client.get(f"{self.url}/health", timeout=5)
            response.raise_for_status()
            return response.json()

    async def speech(
        self,
        text: str,
        *,
        voice: str = "none",
        caption: str,
        mode: str,
        seed: int | None = None,
        references: dict | None = None,
        settings: dict | None = None,
    ) -> tuple[bytes, dict]:
        options = {
            "caption": caption,
            "num_steps": profile_for(mode).steps,
            "chunking_enabled": False,
            **(references or {}),
            **(settings or {}),
        }
        if seed is not None:
            options["seed"] = seed
        started = time.perf_counter()
        try:
            async with self.http() as client:
                response = await client.post(
                    f"{self.url}/v1/audio/speech",
                    timeout=httpx.Timeout(max(900, len(text) * 2) if mode == "synthesize" else 900, connect=5),
                    json={
                        "model": os.environ.get(
                            "LLM_IRODORI_MODEL_NAME", "irodori-tts"
                        ),
                        "input": text,
                        "voice": voice,
                        "response_format": "wav",
                        "irodori": options,
                    },
                )
                response.raise_for_status()
            elapsed = time.perf_counter() - started
            info = wav_info(response.content, max_seconds=4 * 3600, max_bytes=1024 * 1024 * 1024)
            return response.content, {
                "elapsed_seconds": elapsed,
                "audio_seconds": info["seconds"],
                "rtf": elapsed / info["seconds"],
                "seed": int(response.headers.get("X-Irodori-Seed", seed or 0)),
                "settings": options,
                "wav_sha256": info["sha256"],
                "sample_rate": info["sample_rate"],
            }
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(503, f"Irodoriの音声合成に失敗しました: {exc}") from exc

    async def prepare(self, library: VoiceLibrary, identifier: str, *, item: dict | None = None) -> dict:
        # The server owns latent paths; remote endpoints can use the same protocol.
        async with self._lock:
            item = item if item is not None else library.get(identifier)
            signature = ":".join(ref["sha256"] for ref in item["references"])
            key = f"{identifier}:{signature}"
            if key in self._references:
                return self._references[key]
            try:
                async with self.http() as client:
                    latents = []
                    wavs = []
                    for index in range(len(item["references"])):
                        data = await asyncio.to_thread(
                            library.audio_path(identifier, index).read_bytes
                        )
                        voice_id = f"{identifier}_{index}"
                        upload = await client.post(
                            f"{self.url}/v1/audio/voices",
                            data={"voice_id": voice_id},
                            files={"file": (f"{voice_id}.wav", data, "audio/wav")},
                        )
                        if upload.status_code == 409:
                            upload = await client.put(
                                f"{self.url}/v1/audio/voices/{voice_id}",
                                files={"file": (f"{voice_id}.wav", data, "audio/wav")},
                            )
                        upload.raise_for_status()
                        prepared = await client.post(
                            f"{self.url}/local/voices/{voice_id}/prepare"
                        )
                        if prepared.status_code == 404:
                            # Unextended official servers still work with waveform references.
                            metadata = await client.get(f"{self.url}/v1/audio/voices")
                            metadata.raise_for_status()
                            spec = next(
                                s
                                for s in metadata.json()["data"]
                                if s["id"] == voice_id
                            )
                            wavs.append(spec["ref_wav"])
                        else:
                            prepared.raise_for_status()
                            latents.extend(prepared.json()["ref_latents"])
                    options = {"ref_wavs": wavs} if wavs else {"ref_latents": latents}
                    if wavs and latents:
                        raise ValueError("Mixed reference formats returned by Irodori")
                    self._references[key] = options
                    return options
            except (httpx.HTTPError, KeyError, StopIteration, ValueError) as exc:
                raise HTTPException(
                    503, f"Irodoriの参照登録に失敗しました: {exc}"
                ) from exc

    def clear(self):
        self._references.clear()


def styles() -> list[dict]:
    return [{"id": preset["id"], "name": preset["name"]} for preset in PRESETS]


def caption_for(preset_id: str) -> str:
    for preset in PRESETS:
        if preset["id"] == preset_id:
            return preset["caption"]
    raise HTTPException(422, "利用できない感情プリセットです")


def library_voices(library: VoiceLibrary) -> list[dict]:
    return [
        {
            "key": f"irodori:{item['id']}",
            "id": item["id"],
            "engine": "irodori",
            "name": item["name"],
            "styles": styles(),
        }
        for item in library.list()
        if item["kind"] == "voice"
    ]


def installed_provenance(root: Path, mode: str, health: dict) -> dict:
    import json

    path = root / "irodori-tts" / "install.json"
    checkpoint = health.get("runtime", {}).get("checkpoint")
    profile = profile_for(mode)
    local_checkpoint = profile.checkpoint(root)
    is_local = checkpoint and Path(checkpoint).resolve() == local_checkpoint.resolve()
    installation = json.loads(path.read_text()) if path.is_file() and is_local else {}
    expected = installation.get("models", {}).get(profile.model_key, {})
    return {
        "mode": mode,
        "model": health.get("model", {}),
        "runtime": health.get("runtime", {}),
        "installation": installation,
        "checkpoint_revision": expected.get("revision"),
        "checkpoint_sha256": expected.get("weights", {}),
        "derived_checkpoint": expected.get("derived"),
    }
