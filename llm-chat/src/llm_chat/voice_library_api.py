"""Voice design, audition, registration and evaluation API for the chat UI."""

from __future__ import annotations

import asyncio
import json
import secrets
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

from . import irodori, qwen_tts, voice
from .voice_library import MAX_REFERENCE_UPLOAD_BYTES, MAX_WAV_BYTES
from .voice_requests import SynthesisSettings

router = APIRouter(prefix="/api/voice/library")
DEFAULT_TEXT = (
    "こんにちは。今日は、近くの公園をゆっくり散歩しました。木々の間から差し込む光が、とてもきれいでした。"
    "帰り道で小さな本屋に立ち寄り、気になっていた物語を一冊選びました。"
    "明日は少し早起きをして、温かいお茶を飲みながら読んでみようと思います。"
    "忙しい毎日でも、こうした時間を大切にしていきたいですね。"
)


class CandidateRequest(SynthesisSettings):
    caption: str = Field(min_length=1, max_length=1000)
    text: str = Field(default=DEFAULT_TEXT, min_length=1, max_length=600)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    count: int = Field(default=1, ge=1, le=8)
    name: str = Field(default="声の候補", min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_design_text(self):
        if self.engine == "qwen" and len(self.text) > 160:
            raise ValueError("Qwenの声作成用の試聴文は160文字以下にしてください")
        return self


class UpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    favorite: bool | None = None
    register_voice: bool = Field(default=False, alias="register")


class PreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    preset: str = "neutral"
    mode: Literal["chat", "design", "synthesize"] = "chat"
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


@router.get("")
async def list_library() -> dict:
    return {
        "items": await asyncio.to_thread(voice.LIBRARY.list),
        "default_text": DEFAULT_TEXT,
        "presets": irodori.styles(),
    }


@router.post("/candidates")
async def generate(req: CandidateRequest, request: Request) -> dict:
    runtime = request.app.state.speech_runtime
    await runtime.require_tts("design", req.engine)
    result = []
    async with voice._tts_request_lock:
        for index in range(req.count):
            if await request.is_disconnected():
                break
            await runtime.require_tts("design", req.engine)
            seed = (
                ((req.seed + index) % 2**32)
                if req.seed is not None
                else secrets.randbits(32)
            )
            data, metrics = await runtime.tts.speech(
                req.text,
                caption=req.caption,
                mode="design",
                seed=seed,
                settings=req.sampling_settings(),
            )
            health = await runtime.tts.health()
            provider = qwen_tts if req.engine == "qwen" else irodori
            provenance = {
                **provider.installed_provenance(runtime.root, "design", health),
                "caption": req.caption,
                "text": req.text,
                "seed": seed,
                **metrics,
            }
            item = await asyncio.to_thread(
                voice.LIBRARY.create,
                data,
                name=f"{req.name} {index + 1}",
                kind="candidate",
                provenance=provenance,
            )
            result.append(item)
    return {"items": result}


@router.post("/import")
async def import_voice(
    request: Request,
    file: Annotated[UploadFile, File()],
    name: Annotated[str, Form(min_length=1, max_length=80)],
    source: Annotated[str, Form(max_length=1000)] = "",
    transcript: Annotated[str, Form(max_length=6000)] = "",
) -> dict:
    trim_reference = getattr(request.app.state, "trim_reference_uploads", False)
    max_bytes = MAX_REFERENCE_UPLOAD_BYTES if trim_reference else MAX_WAV_BYTES
    data = await file.read(max_bytes + 1)
    return await asyncio.to_thread(
        voice.LIBRARY.create,
        data,
        name=name,
        kind="voice",
        trim_reference=trim_reference,
        provenance={
            "source": "import",
            "usage_notes": source,
            "text": transcript.strip(),
        },
    )


@router.patch("/{identifier}")
async def update(identifier: str, req: UpdateRequest) -> dict:
    async with voice._tts_request_lock:
        return await asyncio.to_thread(
            voice.LIBRARY.update,
            identifier,
            name=req.name,
            favorite=req.favorite,
            register=req.register_voice,
        )


@router.post("/{identifier}/references")
async def add_reference(
    identifier: str,
    request: Request,
    file: Annotated[UploadFile, File()],
    same_speaker: Annotated[bool, Form()],
) -> dict:
    if not same_speaker:
        raise HTTPException(422, "同じ話者のクリップであることを確認してください")
    trim_reference = getattr(request.app.state, "trim_reference_uploads", False)
    max_bytes = MAX_REFERENCE_UPLOAD_BYTES if trim_reference else MAX_WAV_BYTES
    data = await file.read(max_bytes + 1)
    async with voice._tts_request_lock:
        return await asyncio.to_thread(
            voice.LIBRARY.append_reference, identifier, data, trim_reference=trim_reference
        )


@router.get("/{identifier}/audio")
async def audio(identifier: str, index: int = 0) -> FileResponse:
    path = await asyncio.to_thread(voice.LIBRARY.audio_path, identifier, index)
    return FileResponse(
        path, media_type="audio/wav", filename=f"{identifier}-{index}.wav"
    )


@router.get("/{identifier}/metadata")
async def metadata(identifier: str) -> dict:
    return await asyncio.to_thread(voice.LIBRARY.get, identifier)


@router.post("/{identifier}/preview")
async def preview(identifier: str, req: PreviewRequest, request: Request) -> Response:
    runtime = request.app.state.speech_runtime
    await runtime.require_irodori(req.mode)
    caption = irodori.caption_for(req.preset)
    async with voice._tts_request_lock:
        await runtime.require_irodori(req.mode)
        references = await runtime.irodori.prepare(voice.LIBRARY, identifier)
        await runtime.require_irodori(req.mode)
        data, metrics = await runtime.irodori.speech(
            req.text,
            voice=identifier + "_0",
            caption=caption,
            mode=req.mode,
            references=references,
            seed=req.seed,
        )
    return Response(
        data,
        media_type="audio/wav",
        headers={
            "X-TTS-Metrics": json.dumps(metrics, ensure_ascii=True),
        },
    )


@router.delete("/{identifier}", status_code=204)
async def delete(identifier: str) -> None:
    async with voice._tts_request_lock:
        await asyncio.to_thread(voice.LIBRARY.delete, identifier)
