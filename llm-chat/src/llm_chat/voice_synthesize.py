"""Independent Large GPU voice creation and synthesis service, no LLM runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from . import irodori, qwen_tts, voice, voice_library_api
from .synthesis_products import ProductStore, copy_voice
from .voice_library import VoiceLibrary
from .voice_requests import SynthesisSettings
from .voice_runtime import VoiceRuntime, local_address

ROOT = Path(os.environ.get("LLM_ROOT", voice.APP_DIR.parent))
STATIC = Path(__file__).parent / "static"
OUTPUT_DIR = Path(
    os.environ.get(
        "VOICE_SYNTHESIZE_PRODUCTION_DIR",
        os.environ.get(
            "VOICE_SYNTHESIZE_OUTPUT_DIR", ROOT / "voice-synthesize" / "productions"
        ),
    )
)
LEGACY_OUTPUT_DIR = ROOT / "voice-synthesize" / "outputs"
DEFAULT_VOICE_CAPTION = "自然で落ち着いた声。聞き取りやすく、明瞭で穏やかな話し方。"


def products() -> ProductStore:
    return ProductStore(OUTPUT_DIR, LEGACY_OUTPUT_DIR)


LLM_CHAT_URL = os.environ.get("LLM_CHAT_URL", "http://127.0.0.1:5070").rstrip("/")


async def unload_chat_llm():
    if local_address(LLM_CHAT_URL) is None:
        raise HTTPException(422, "LLM_CHAT_URLはローカルのllm-chatを指定してください")
    try:
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            response = await client.post(f"{LLM_CHAT_URL}/api/unload")
            response.raise_for_status()
            state = response.json()
            if state.get("active") is not None or state.get("status") != "stopped":
                raise ValueError("LLMアンロードを確認できません")
    except httpx.ConnectError:
        # A stopped llm-chat does not need to be started just to unload an LLM.
        return
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(503, f"llm-chatのLLMをアンロードできません: {exc}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = VoiceRuntime(
        ROOT,
        {
            "qwen": os.environ.get(
                "VOICE_SYNTHESIZE_QWEN_URL", "http://127.0.0.1:8090"
            ),
            "irodori": os.environ.get(
                "VOICE_SYNTHESIZE_TTS_URL", "http://127.0.0.1:8089"
            ),
        },
        before_gpu_start=unload_chat_llm,
    )
    runtime.engine, runtime.mode = "irodori", "synthesize"
    app.state.speech_runtime = runtime
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(title="Voice Synthesize", lifespan=lifespan)
app.state.trim_reference_uploads = True
app.include_router(voice_library_api.router)


@app.get("/")
async def index():
    return FileResponse(STATIC / "voice-synthesize.html")


@app.get("/api/voice/runtime")
async def runtime_status(request: Request):
    return request.app.state.speech_runtime.snapshot()


class RuntimeRequest(BaseModel):
    enabled: bool
    engine: Literal["irodori", "qwen"] = "irodori"
    mode: Literal["design", "synthesize"] = "synthesize"


@app.put("/api/voice/runtime")
async def runtime_update(req: RuntimeRequest, request: Request):
    return await request.app.state.speech_runtime.set_enabled(
        req.enabled, req.engine, req.mode
    )


class SynthesisRequest(SynthesisSettings):
    reference_text: str | None = Field(default=None, max_length=6000)
    text: str = Field(min_length=1, max_length=20000)
    voice_id: str | None = None
    source_product_id: str | None = None
    voice_caption: str = Field(
        default=DEFAULT_VOICE_CAPTION, min_length=1, max_length=1000
    )
    name: str = Field(default="音声合成", min_length=1, max_length=80)
    caption: str | None = Field(default=None, max_length=1000)
    preset: str = "neutral"
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_source(self):
        if self.voice_id and self.source_product_id:
            raise ValueError("声ライブラリと作品の声はどちらか一方を選んでください")
        if (
            not self.text.strip()
            or not self.name.strip()
            or not self.voice_caption.strip()
        ):
            raise ValueError("原稿・作品名・声の説明に空白だけは指定できません")
        if self.engine == "qwen" and (
            self.preset != "neutral" or (self.caption and self.caption.strip())
        ):
            raise ValueError(
                "Qwen Baseの話し方は参照音声から引き継ぎます。作品の話し方の変更は声の説明で指定してください"
            )
        return self


@app.get("/api/synthesis/options")
async def synthesis_options():
    return {
        "engines": [
            {"id": "irodori", "name": "Irodori v4 Large"},
            {"id": "qwen", "name": "Qwen3-TTS 1.7B"},
        ],
        "presets": irodori.styles(),
        "default_voice_caption": DEFAULT_VOICE_CAPTION,
        "max_text_chars": 20000,
        "storage": "voice-synthesize/productions",
    }


def installed_provenance(runtime, mode, health):
    provider = qwen_tts if runtime.engine == "qwen" else irodori
    return provider.installed_provenance(ROOT, mode, health)


async def production_voice(req: SynthesisRequest, stage: Path, runtime):
    private = VoiceLibrary(stage / "voice")
    if req.source_product_id:
        previous = products().get(req.source_product_id)
        identifier = previous.get("production_voice_id")
        if not identifier:
            raise HTTPException(
                409,
                "この旧形式の音声には作品専用の声が保存されていません。声を選び直してください",
            )
        source = VoiceLibrary(products().directory(req.source_product_id) / "voice")
        item = await asyncio.to_thread(
            copy_voice, source, source.get(identifier), private
        )
        original = previous.get("voice_source", {})
        while original.get("kind") == "product":
            original = original.get("original", {})
        provenance = {
            "kind": "product",
            "product_id": req.source_product_id,
            "original": original,
        }
    elif req.voice_id:
        original = voice.LIBRARY.get(req.voice_id)
        if original["kind"] != "voice":
            raise HTTPException(422, "声ライブラリから登録済みの声を選んでください")
        item = await asyncio.to_thread(copy_voice, voice.LIBRARY, original, private)
        provenance = {
            "kind": "library",
            "library_id": req.voice_id,
            "snapshot": original,
        }
    else:
        seed = req.seed if req.seed is not None else secrets.randbits(32)
        reference_text = (
            voice_library_api.DEFAULT_TEXT[
                : voice_library_api.DEFAULT_TEXT.index("帰り道")
            ]
            if req.engine == "qwen"
            else voice_library_api.DEFAULT_TEXT
        )
        data, metrics = await runtime.tts.speech(
            reference_text,
            caption=req.voice_caption.strip(),
            mode="synthesize",
            seed=seed,
            settings=req.sampling_settings(),
        )
        health = await runtime.tts.health()
        item = await asyncio.to_thread(
            private.create,
            data,
            name="作品専用の声",
            kind="voice",
            provenance={
                "caption": req.voice_caption.strip(),
                "text": reference_text,
                **installed_provenance(
                    runtime, "design" if req.engine == "qwen" else "synthesize", health
                ),
                **metrics,
            },
        )
        provenance = {
            "kind": "generated",
            "caption": req.voice_caption.strip(),
            "seed": metrics["seed"],
        }
    if req.reference_text is not None:
        # Preserve the user-supplied transcript inside this product, never change the shared voice.
        item["provenance"]["reference_text"] = req.reference_text.strip()
        (private.directory(item["id"]) / "metadata.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2)
        )
    return private, item, provenance


@app.post("/api/synthesis")
async def synthesize(req: SynthesisRequest, request: Request):
    runtime = request.app.state.speech_runtime
    await runtime.require_tts("synthesize", req.engine)
    caption = (
        ""
        if req.engine == "qwen"
        else (
            req.caption.strip()
            if req.caption and req.caption.strip()
            else irodori.caption_for(req.preset)
        )
    )
    store = products()
    async with voice._tts_request_lock:
        await runtime.require_tts("synthesize", req.engine)
        identifier, stage = await asyncio.to_thread(store.stage)
        started = time.perf_counter()
        try:
            private, item, source = await production_voice(req, stage, runtime)
            await runtime.require_tts("synthesize", req.engine)
            references = await runtime.tts.prepare(private, item["id"], item=item)
            await runtime.require_tts("synthesize", req.engine)
            health = await runtime.tts.health()
            data, metrics = await runtime.tts.speech(
                req.text,
                voice=item["id"] + "_0",
                caption=caption,
                mode="synthesize",
                seed=req.seed,
                references=references,
                settings={
                    **req.sampling_settings(),
                    **(
                        {
                            "chunking_enabled": True,
                            "chunk_min_chars": 80,
                            "chunk_max_chars": 160,
                        }
                        if req.engine == "irodori"
                        else {}
                    ),
                },
            )
            result = {
                "schema_version": 2,
                "engine": req.engine,
                "id": identifier,
                "name": req.name,
                "text": req.text,
                "voice_id": req.voice_id,
                "source_product_id": req.source_product_id,
                "voice_name": item["name"],
                "voice_source": source,
                "production_voice_id": item["id"],
                "caption": caption,
                "references": item["references"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "provenance": installed_provenance(runtime, "synthesize", health),
                "production_seconds": time.perf_counter() - started,
                "metrics": metrics,
            }
            settings = {
                **req.model_dump(),
                "seed": metrics["seed"],
                "caption": caption,
                "voice_source": source,
                "production_voice_id": item["id"],
                "reference_sha256": [ref["sha256"] for ref in item["references"]],
            }
            # Wait for atomic publication even if the HTTP client disconnects.
            publish = asyncio.create_task(
                asyncio.to_thread(store.publish, stage, data, result, settings)
            )
            try:
                await asyncio.shield(publish)
            except asyncio.CancelledError:
                await publish
                raise
        finally:
            if stage.exists():
                await asyncio.to_thread(shutil.rmtree, stage)
    return result


@app.get("/api/synthesis")
async def outputs():
    return {"items": await asyncio.to_thread(products().list)}


@app.get("/api/synthesis/{identifier}/audio")
async def audio(identifier: str):
    return FileResponse(
        products().directory(identifier) / "audio.wav",
        media_type="audio/wav",
        filename=f"{identifier}.wav",
    )


@app.get("/api/synthesis/{identifier}/metadata")
async def metadata(identifier: str):
    return await asyncio.to_thread(products().get, identifier)


@app.get("/api/synthesis/{identifier}/manuscript")
async def manuscript(identifier: str):
    path = products().directory(identifier) / "manuscript.txt"
    if not path.is_file():
        # Legacy outputs kept their manuscript inside metadata only.
        from fastapi.responses import Response

        return Response(
            products().get(identifier)["text"],
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{identifier}.txt"'},
        )
    return FileResponse(
        path, media_type="text/plain; charset=utf-8", filename=f"{identifier}.txt"
    )


@app.get("/api/synthesis/{identifier}/settings")
async def output_settings(identifier: str):
    path = products().directory(identifier) / "settings.json"
    if not path.is_file():
        raise HTTPException(
            404, "旧形式の音声には設定ファイルがありません。生成条件を参照してください"
        )
    return json.loads(path.read_text())


@app.get("/api/synthesis/{identifier}/archive")
async def archive(identifier: str):
    path = products().directory(identifier) / "product.zip"
    if not path.is_file():
        raise HTTPException(404, "旧形式の音声には作品ZIPがありません")
    return FileResponse(
        path, media_type="application/zip", filename=f"{identifier}.zip"
    )


@app.delete("/api/synthesis/{identifier}", status_code=204)
async def delete(identifier: str) -> None:
    await asyncio.to_thread(products().delete, identifier)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.environ.get("VOICE_SYNTHESIZE_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("VOICE_SYNTHESIZE_PORT", "5080"))
    )
    args = parser.parse_args()
    uvicorn.run("llm_chat.voice_synthesize:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
