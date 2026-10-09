"""Local Qwen3-TTS BF16 GPU backend. Holds only one model, including its codec."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import re
import secrets
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import soundfile as sf
import torch
from chunking import chunks
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

ROOT = Path(__file__).resolve().parent
MAX_REFERENCE_SECONDS = 120
MAX_REFERENCE_UPLOAD_BYTES = 256 * 1024 * 1024
MANIFEST = json.loads((ROOT / "install.json").read_text())
LOCK = threading.RLock()
model = None
active_key = None
loading = False
security = HTTPBearer(auto_error=False)


def authenticate(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
):
    key = os.environ.get("LLM_QWEN_TTS_API_KEY")
    if key and (
        credentials is None or not secrets.compare_digest(credentials.credentials, key)
    ):
        raise HTTPException(401, "Invalid API key")


app = FastAPI(title="Local Qwen3-TTS", dependencies=[Depends(authenticate)])


def load(key):
    global model, active_key, loading
    if active_key == key and model is not None:
        return model
    loading = True
    try:
        model = None
        active_key = None
        gc.collect()
        torch.cuda.empty_cache()
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-TTS requires CUDA in this service")
        model = Qwen3TTSModel.from_pretrained(
            str(ROOT / "models" / key),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        active_key = key
        return model
    finally:
        loading = False


@app.get("/health")
def health():
    entry = MANIFEST["models"].get(active_key, {})
    return {
        "engine": "qwen",
        "model": {
            "hf_checkpoint": entry.get("repo"),
            "model_device": "cuda",
            "model_precision": "bfloat16",
            "attention": "sdpa",
        },
        "runtime": {
            "loaded": model is not None,
            "loading": loading,
            "model_key": active_key,
            "checkpoint": str(ROOT / "models" / active_key) if active_key else None,
        },
    }


class RuntimeRequest(BaseModel):
    mode: Literal["design", "synthesize"]


@app.post("/local/runtime")
def runtime(req: RuntimeRequest):
    with LOCK:
        load("design" if req.mode == "design" else "base")
        return health()


@app.post("/local/voices/prepare")
def prepare(
    file: Annotated[UploadFile, File()],
    transcript: Annotated[str, Form(max_length=6000)] = "",
):
    data = file.file.read(MAX_REFERENCE_UPLOAD_BYTES + 1)
    if len(data) > MAX_REFERENCE_UPLOAD_BYTES:
        raise HTTPException(422, "Reference WAV exceeds 256 MiB")
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32")
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Reference must contain finite audio samples")
        source_seconds = len(audio) / rate
        trimmed = source_seconds > MAX_REFERENCE_SECONDS
        audio = audio[: MAX_REFERENCE_SECONDS * rate]
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    # A full-source transcript no longer matches a cropped reference.
    transcript_ignored = trimmed and bool(transcript.strip())
    transcript = "" if trimmed else transcript.strip()
    signature = hashlib.sha256(
        data
        + transcript.encode()
        + json.dumps(MANIFEST["models"]["base"], sort_keys=True).encode()
        + (f"\0trim:{MAX_REFERENCE_SECONDS}".encode() if trimmed else b"")
    ).hexdigest()
    directory = ROOT / "voices" / ".prompts"
    path = directory / f"{signature}.pt"
    with LOCK, torch.inference_mode():
        engine = load("base")
        if not path.is_file():
            items = engine.create_voice_clone_prompt(
                ref_audio=(audio, rate),
                ref_text=transcript or None,
                x_vector_only_mode=not bool(transcript),
            )
            records = [
                {
                    key: value.detach().cpu()
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in asdict(item).items()
                }
                for item in items
            ]
            directory.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".pending")
            torch.save(records, temp)
            temp.replace(path)
    return {
        "prompt_id": signature,
        "clone_mode": "icl" if transcript else "speaker_embedding_only",
        "reference_policy": "primary_clip",
        "reference_sha256": hashlib.sha256(data).hexdigest(),
        "source_reference_seconds": source_seconds,
        "reference_seconds": len(audio) / rate,
        "max_reference_seconds": MAX_REFERENCE_SECONDS,
        "reference_trimmed": trimmed,
        "transcript_ignored_due_to_trim": transcript_ignored,
    }


class QwenOptions(BaseModel):
    caption: str = Field(default="", max_length=1000)
    prompt_id: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    temperature: float = Field(default=0.9, gt=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=50, ge=1, le=1000)
    repetition_penalty: float = Field(default=1.05, ge=1, le=2)
    max_new_tokens: int = Field(default=2048, ge=256, le=4096)


class SpeechRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str = Field(min_length=1, max_length=20000)
    voice: str = "none"
    response_format: Literal["wav"] = "wav"
    qwen: QwenOptions = Field(default_factory=QwenOptions)


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(422, "Text cannot be blank")
    opts = req.qwen
    seed = opts.seed if opts.seed is not None else secrets.randbits(32)
    samples = []
    segments = chunks(req.input)
    started = time.perf_counter()
    with LOCK, torch.inference_mode():
        prompt = None
        if opts.prompt_id:
            if not re.fullmatch(r"[a-f0-9]{64}", opts.prompt_id):
                raise HTTPException(422, "Invalid prompt ID")
            path = ROOT / "voices" / ".prompts" / f"{opts.prompt_id}.pt"
            if not path.is_file():
                raise HTTPException(404, "Voice prompt is not prepared")
            engine = load("base")
            prompt = [
                VoiceClonePromptItem(**record)
                for record in torch.load(path, map_location="cpu", weights_only=True)
            ]
        else:
            if len(req.input) > 160:
                raise HTTPException(
                    422,
                    "Voice design requires at most 160 characters; clone the designed voice for long text",
                )
            if not opts.caption.strip():
                raise HTTPException(422, "Voice design requires a caption")
            engine = load("design")
        options = opts.model_dump(
            include={
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "max_new_tokens",
            }
        )
        options.update(
            do_sample=True,
            subtalker_dosample=True,
            subtalker_top_k=opts.top_k,
            subtalker_top_p=opts.top_p,
            subtalker_temperature=opts.temperature,
        )
        for index, segment in enumerate(segments):
            torch.manual_seed((seed + index) % 2**32)
            if prompt:
                wavs, rate = engine.generate_voice_clone(
                    text=segment,
                    language="Japanese",
                    voice_clone_prompt=prompt,
                    non_streaming_mode=True,
                    **options,
                )
            else:
                wavs, rate = engine.generate_voice_design(
                    text=segment,
                    language="Japanese",
                    instruct=opts.caption,
                    non_streaming_mode=True,
                    **options,
                )
            wave = np.asarray(wavs[0], dtype=np.float32)
            # A budget-exhausted segment is incomplete; never publish it as a complete product.
            if (
                len(wave)
                >= (opts.max_new_tokens - 1)
                * json.loads(
                    (
                        ROOT
                        / "models"
                        / active_key
                        / "speech_tokenizer"
                        / "config.json"
                    ).read_text()
                )["decode_upsample_rate"]
            ):
                raise HTTPException(
                    422,
                    "音声トークン上限に達しました。原稿を短くするか上限を増やしてください",
                )
            if not len(wave) or not np.isfinite(wave).all():
                raise RuntimeError("Qwen returned invalid audio")
            if index:
                samples.append(np.zeros(round(rate * 0.12), dtype=np.float32))
            samples.append(wave)
        output = io.BytesIO()
        sf.write(output, np.concatenate(samples), rate, format="WAV", subtype="PCM_16")
    metrics = {
        "seed": seed,
        "chunks": len(segments),
        "chunk_max_chars": 160,
        "clone_mode": "icl"
        if prompt and prompt[0].icl_mode
        else "speaker_embedding_only"
        if prompt
        else "voice_design",
        "reference_policy": "primary_clip" if prompt else None,
        "backend_seconds": time.perf_counter() - started,
        "streaming": False,
    }
    return Response(
        output.getvalue(),
        media_type="audio/wav",
        headers={"X-Qwen-Metrics": json.dumps(metrics)},
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
