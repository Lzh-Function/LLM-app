"""Official API plus a persistent reference-latent cache for registered WAVs."""

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

import irodori_openai_tts.app as official_api
import torch
from chunking import bounded_chunks
from fastapi import Depends, HTTPException
from irodori_openai_tts.app import (
    app,
    require_auth,
    runtime_manager,
    settings,
    voice_registry,
)
from irodori_tts.inference_runtime import _load_audio
from pydantic import Field

# The pinned upstream splitter has a minimum but no maximum chunk length.
# Retain its sentence grouping and cap very long sentences before GPU inference.
_original_split = official_api._split_text_for_speech


def _bounded_split(text, **kwargs):
    return bounded_chunks(_original_split(text, **kwargs))


official_api._split_text_for_speech = _bounded_split


# Extend the HTTP envelope; the upstream bounded splitter still processes small chunks.
class ProductionSpeechRequest(official_api.SpeechRequest):
    input: str = Field(min_length=1, max_length=20000)


app.router.routes = [
    route
    for route in app.router.routes
    if not (
        getattr(route, "path", None) == "/v1/audio/speech"
        and "POST" in getattr(route, "methods", set())
    )
]


@app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
async def create_speech(payload: ProductionSpeechRequest):
    if len(payload.input) > 4096 and not payload.irodori.chunking_enabled:
        raise HTTPException(422, "Long manuscripts require chunking_enabled=true")
    return await official_api.create_speech(payload)


# Preparation and synthesis both use the runtime's internal synthesis lock.
_prepare_lock = threading.Lock()


@lru_cache(maxsize=4)
def codec_fingerprint(location: str, size: int, modified: int) -> str:
    digest = hashlib.sha256()
    with Path(location).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@app.post("/local/voices/{voice_id}/prepare", dependencies=[Depends(require_auth)])
def prepare(voice_id: str):
    try:
        voice = voice_registry.resolve(voice_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    paths = ([voice.ref_wav] if voice.ref_wav else []) + list(voice.ref_wavs or [])
    if not paths:
        raise HTTPException(422, "A registered waveform reference is required")
    with _prepare_lock:
        runtime = runtime_manager.get()
        cache = settings.voices_dir / ".latents"
        cache.mkdir(parents=True, exist_ok=True)
        latents = []
        codec_path = Path(settings.codec_repo)
        codec_identity = str(settings.codec_repo)
        if codec_path.is_file():
            stat = codec_path.stat()
            codec_identity = codec_fingerprint(
                str(codec_path), stat.st_size, stat.st_mtime_ns
            )
        for source in paths:
            path = Path(source)
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            identity = json.dumps(
                [
                    fingerprint,
                    codec_identity,
                    settings.codec_precision,
                    settings.codec_deterministic_encode,
                    -16.0,
                    True,
                ],
                sort_keys=True,
            )
            key = hashlib.sha256(identity.encode()).hexdigest()
            target = cache / f"{key}.pt"
            if not target.is_file():
                # Match SamplingRequest defaults; encode once, then reuse for all sentences.
                with runtime._infer_lock, torch.inference_mode():
                    wav, sr = _load_audio(path)
                    wav = wav[:, : int(sr * 120)]
                    latent = runtime.codec.encode_waveform(
                        wav.unsqueeze(0),
                        sample_rate=sr,
                        normalize_db=-16.0,
                        ensure_max=True,
                    ).cpu()
                temporary = target.with_suffix(".tmp")
                torch.save(latent, temporary)
                temporary.replace(target)
            latents.append(str(target.resolve()))
    return {"ref_latents": latents}
