"""Local speech input and AivisSpeech/VOICEVOX output for the chat UI."""

from __future__ import annotations

import asyncio
import io
import os
import threading
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import irodori
from .voice_library import VoiceLibrary

router = APIRouter(prefix="/api/voice")

APP_DIR = Path(__file__).resolve().parents[2]
AIVIS_URL = os.environ.get("LLM_AIVIS_URL", "http://127.0.0.1:10101").rstrip("/")
VOICEVOX_URL = os.environ.get("LLM_VOICEVOX_URL", "http://127.0.0.1:50021").rstrip("/")
STT_MODEL = os.environ.get("LLM_STT_MODEL", "medium")
STT_THREADS = int(os.environ.get("LLM_STT_THREADS", "8"))
STT_DIR = Path(os.environ.get("LLM_STT_DIR", APP_DIR / "voice-models"))
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
ALLOWED_SPEAKERS = {
    "e756b8e4-b606-4e15-99b1-3f9c6a1b2317",  # まお
    "5680ac39-43c9-487a-bc3e-018c0d29cc38",  # コハク
}
VOICEVOX_SPEAKERS = {
    "ずんだもん", "冥鳴ひまり", "中国うさぎ", "東北ずん子", "東北きりたん"
}
ENGINE_URLS = {"aivis": AIVIS_URL, "voicevox": VOICEVOX_URL}
LIBRARY = VoiceLibrary(Path(os.environ.get("LLM_VOICE_LIBRARY_DIR", APP_DIR / "voice-library")))
SPEAKERS_TIMEOUT_S = 5

_stt_model = None
_stt_model_lock = threading.Lock()
_stt_request_lock = asyncio.Lock()
_tts_request_lock = asyncio.Lock()


def _model():
    global _stt_model
    with _stt_model_lock:
        if _stt_model is None:
            from faster_whisper import WhisperModel

            STT_DIR.mkdir(parents=True, exist_ok=True)
            _stt_model = WhisperModel(
                STT_MODEL,
                device="cpu",
                compute_type="int8",
                cpu_threads=STT_THREADS,
                num_workers=1,
                download_root=str(STT_DIR),
            )
    return _stt_model


def _transcribe(data: bytes) -> dict:
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    try:
        audio = decode_audio(io.BytesIO(data), sampling_rate=16000)
    except Exception as exc:
        raise ValueError("音声ファイルを読み取れませんでした") from exc
    seconds = len(audio) / 16000
    if seconds < 0.2 or seconds > MAX_AUDIO_SECONDS:
        raise ValueError(f"録音時間は0.2～{MAX_AUDIO_SECONDS}秒にしてください")

    vad = VadOptions(
        min_speech_duration_ms=150, min_silence_duration_ms=400, speech_pad_ms=200
    )
    speech = get_speech_timestamps(audio, vad)
    if not speech:
        return {"text": "", "duration": round(seconds, 2), "speech_detected": False}

    segments, _ = _model().transcribe(
        audio,
        language="ja",
        beam_size=3,
        vad_filter=True,
        vad_parameters=vad,
        condition_on_previous_text=False,
    )
    return {
        "text": "".join(segment.text for segment in segments).strip(),
        "duration": round(seconds, 2),
        "speech_detected": True,
    }


async def _speakers(engine: str) -> list[dict]:
    if engine not in ENGINE_URLS:
        raise HTTPException(422, "利用できない音声エンジンです")
    try:
        # HTTPX のタイムアウトは通信段階ごと。応答が少しずつ届く場合も全体で打ち切る。
        async with (
            asyncio.timeout(SPEAKERS_TIMEOUT_S),
            httpx.AsyncClient(timeout=SPEAKERS_TIMEOUT_S) as client,
        ):
            response = await client.get(f"{ENGINE_URLS[engine]}/speakers")
            response.raise_for_status()
            speakers = response.json()
            if engine == "aivis":
                return [s for s in speakers if s.get("speaker_uuid") in ALLOWED_SPEAKERS]
            return [s for s in speakers if s.get("name") in VOICEVOX_SPEAKERS]
    except (TimeoutError, httpx.HTTPError, ValueError) as exc:
        raise HTTPException(503, f"{engine} に接続できません: {exc}") from exc


async def styles_for_speaker(name: str) -> list[dict]:
    engine, separator, speaker_name = name.partition(":")
    if not separator:  # 古いクライアントからの AivisSpeech 話者名
        engine, speaker_name = "aivis", name
    if engine == "irodori":
        item = LIBRARY.get(speaker_name)
        if item["kind"] != "voice":
            raise HTTPException(422, "先に声を登録してください")
        return irodori.styles()
    for speaker in await _speakers(engine):
        if speaker["name"] == speaker_name:
            return [
                {"id": style["id"], "name": style["name"]}
                for style in speaker["styles"]
            ]
    raise HTTPException(422, "利用できない話者です")


class RuntimeRequest(BaseModel):
    enabled: bool
    engine: Literal["legacy", "irodori"] | None = None
    mode: Literal["chat", "design", "synthesize"] | None = None


@router.get("/runtime")
async def runtime_status(request: Request) -> dict:
    return request.app.state.speech_runtime.snapshot()


@router.put("/runtime")
async def set_runtime(req: RuntimeRequest, request: Request) -> dict:
    return await request.app.state.speech_runtime.set_enabled(req.enabled, req.engine, req.mode)


@router.get("/voices")
async def voices(response: Response = None, request: Request = None) -> list[dict]:
    runtime = getattr(request.app.state, "speech_runtime", None) if request is not None else None
    if runtime is not None and not runtime.enabled:
        return []
    if runtime is not None and runtime.engine == "irodori":
        if runtime.mode != "chat":
            return []
        await runtime.require_irodori("chat")
        return irodori.library_voices(LIBRARY)
    result = []
    errors = []
    engines = list(ENGINE_URLS)
    results = await asyncio.gather(
        *(_speakers(engine) for engine in engines), return_exceptions=True
    )
    for engine, speakers in zip(engines, results):
        if isinstance(speakers, HTTPException):
            errors.append(speakers.detail)
            continue
        if isinstance(speakers, BaseException):
            raise speakers
        for speaker in speakers:
            result.append({
                "key": f"{engine}:{speaker['name']}",
                "engine": engine,
                "name": speaker["name"],
                "styles": [
                    {"id": style["id"], "name": style["name"]}
                    for style in speaker["styles"]
                ],
            })
    if not result and errors:
        raise HTTPException(503, " / ".join(errors))
    if response is not None and runtime is not None and runtime.starting:
        response.headers["X-Voice-Starting"] = "1"
    return result


@router.post("/transcribe")
async def transcribe(file: Annotated[UploadFile, File()]) -> dict:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "音声ファイルが大きすぎます")
    if not data:
        raise HTTPException(400, "音声ファイルが空です")
    async with _stt_request_lock:
        try:
            return await asyncio.to_thread(_transcribe, data)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, f"文字起こしに失敗しました: {exc}") from exc


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    style_id: int | None = None
    engine: Literal["aivis", "voicevox", "irodori"] = "aivis"
    voice_id: str | None = None
    preset: str = "neutral"


@router.post("/synthesize")
async def synthesize(req: SynthesisRequest, request: Request = None) -> Response:
    runtime = getattr(request.app.state, "speech_runtime", None) if request is not None else None
    if runtime is not None and not runtime.enabled:
        raise HTTPException(409, "読み上げをONにしてください")
    if req.engine == "irodori":
        if runtime is None:
            raise HTTPException(503, "音声ランタイムがありません")
        await runtime.require_irodori("chat")
        item = LIBRARY.get(req.voice_id or "")
        if item["kind"] != "voice":
            raise HTTPException(422, "登録済みの声を指定してください")
        caption = irodori.caption_for(req.preset)
        async with _tts_request_lock:
            await runtime.require_irodori("chat")
            references = await runtime.irodori.prepare(LIBRARY, item["id"])
            await runtime.require_irodori("chat")
            data, metrics = await runtime.irodori.speech(req.text, voice=item["id"] + "_0",
                                                       caption=caption, mode="chat", references=references)
        return Response(data, media_type="audio/wav", headers={
            "X-TTS-Seconds": str(metrics["elapsed_seconds"]), "X-TTS-RTF": str(metrics["rtf"]),
        })
    if runtime is not None and runtime.engine != "legacy":
        raise HTTPException(409, "AivisSpeech / VOICEVOXへ切り替えてください")
    allowed = {style["id"] for speaker in await _speakers(req.engine) for style in speaker["styles"]}
    if req.style_id not in allowed:
        raise HTTPException(422, "利用できない音声スタイルです")
    async with _tts_request_lock:
        if runtime is not None and not runtime.enabled:
            raise HTTPException(409, "読み上げをONにしてください")
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(120, connect=5)
            ) as client:
                query = await client.post(
                    f"{ENGINE_URLS[req.engine]}/audio_query",
                    params={"text": req.text, "speaker": req.style_id},
                )
                query.raise_for_status()
                audio = await client.post(
                    f"{ENGINE_URLS[req.engine]}/synthesis",
                    params={"speaker": req.style_id},
                    json=query.json(),
                )
                audio.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(503, f"音声合成に失敗しました: {exc}") from exc
    return Response(audio.content, media_type="audio/wav")
