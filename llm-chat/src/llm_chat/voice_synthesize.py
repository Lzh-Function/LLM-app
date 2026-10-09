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
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

from . import irodori, qwen_tts, voice, voice_library_api
from .dialogue import (
    MAX_OUTPUT_BYTES,
    MAX_OUTPUT_SECONDS,
    ReferenceUpload,
    SpeakerSpec,
    assemble_wavs,
    parse_script,
    speech_chunks,
    validate_labels,
)
from .synthesis_products import ProductStore, copy_voice
from .voice_library import MAX_REFERENCE_UPLOAD_BYTES, VoiceLibrary, wav_info
from .voice_requests import RenameRequest, SynthesisSettings
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
    speakers: list[SpeakerSpec] | None = None
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


async def production_voice(
    req: SynthesisRequest,
    stage: Path,
    runtime,
    upload: tuple[str, bytes] | None = None,
):
    private = VoiceLibrary(stage / "voice")
    if upload is not None:
        filename, data = upload
        item = await asyncio.to_thread(
            private.create,
            data,
            name=Path(filename).stem[:80] or "アップロードした参照音声",
            kind="voice",
            trim_reference=True,
            provenance={
                "source": "upload",
                "filename": filename,
                "text": (req.reference_text or "").strip(),
            },
        )
        provenance = {
            "kind": "upload",
            "filename": filename,
            "transcript_ignored_due_to_trim": bool(
                item["provenance"].get("source_text")
            ),
        }
    elif req.source_product_id:
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
    if req.reference_text is not None and upload is None:
        # Preserve the user-supplied transcript inside this product, never change the shared voice.
        item["provenance"]["reference_text"] = req.reference_text.strip()
        (private.directory(item["id"]) / "metadata.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2)
        )
    return private, item, provenance


@app.post("/api/synthesis")
async def synthesize(req: SynthesisRequest, request: Request):
    return await create_production(req, request)


@app.post("/api/synthesis/upload")
async def synthesize_upload(
    request: Request,
    settings: Annotated[str, Form()],
    file: Annotated[UploadFile | None, File()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
):
    try:
        req = SynthesisRequest.model_validate_json(settings)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    if req.voice_id or req.source_product_id:
        raise HTTPException(
            422, "参照WAVのアップロードと登録済みの声・作品の声は同時に指定できません"
        )
    if file is not None and files:
        raise HTTPException(422, "file と files はどちらか一方を指定してください")
    selected = [file] if file is not None else (files or [])
    if not selected:
        raise HTTPException(422, "参照WAVを選んでください")
    if req.speakers is not None and len(req.speakers) != len(selected):
        raise HTTPException(422, "参照WAVと話者設定の数を一致させてください")
    specs = (
        req.speakers if req.speakers is not None else [SpeakerSpec() for _ in selected]
    )
    labels = [spec.label for spec in specs]
    validate_labels(labels)
    parse_script(req.text, labels)
    if len(selected) > 1 and req.reference_text is not None:
        raise HTTPException(
            422, "複数話者の参照文章は各 speakers の reference_text に指定してください"
        )
    uploads, total = [], 0
    for selected_file, spec in zip(selected, specs):
        data = await selected_file.read(MAX_REFERENCE_UPLOAD_BYTES - total + 1)
        total += len(data)
        if total > MAX_REFERENCE_UPLOAD_BYTES:
            raise HTTPException(422, "参照WAVは合計256 MiB以下にしてください")
        await asyncio.to_thread(
            wav_info, data, max_seconds=None, max_bytes=MAX_REFERENCE_UPLOAD_BYTES
        )
        uploads.append(
            ReferenceUpload(
                Path(selected_file.filename or "reference.wav").name,
                data,
                spec.label,
                spec.reference_text
                if spec.reference_text is not None
                else req.reference_text,
            )
        )
    return await create_production(req, request, uploads=uploads)


async def dialogue_voices(stage, uploads, previous, source_product_id):
    private, speakers = VoiceLibrary(stage / "voice"), []
    if uploads is not None:
        for upload in uploads:
            item = await asyncio.to_thread(
                private.create,
                upload.data,
                name=upload.label or Path(upload.filename).stem[:80],
                kind="voice",
                trim_reference=True,
                provenance={
                    "source": "upload",
                    "filename": upload.filename,
                    "text": (upload.reference_text or "").strip(),
                },
            )
            speakers.append(
                {
                    "label": upload.label,
                    "production_voice_id": item["id"],
                    "voice_name": item["name"],
                    "references": item["references"],
                    "voice_source": {
                        "kind": "upload",
                        "filename": upload.filename,
                        "transcript_ignored_due_to_trim": bool(
                            item["provenance"].get("source_text")
                        ),
                    },
                }
            )
    else:
        source = VoiceLibrary(products().directory(source_product_id) / "voice")
        for speaker in previous["speakers"]:
            item = await asyncio.to_thread(
                copy_voice, source, source.get(speaker["production_voice_id"]), private
            )
            original = speaker["voice_source"]
            while original.get("kind") == "product":
                original = original.get("original", {})
            speakers.append(
                {
                    **speaker,
                    "production_voice_id": item["id"],
                    "references": item["references"],
                    "voice_source": {
                        "kind": "product",
                        "product_id": source_product_id,
                        "original": original,
                        "transcript_ignored_due_to_trim": bool(
                            item["provenance"].get("source_text")
                        ),
                    },
                }
            )
    return private, speakers


async def render_dialogue(req, runtime, stage, private, speakers, turns, caption):
    started = time.perf_counter()
    prepared = {}
    base_seed = req.seed if req.seed is not None else secrets.randbits(32)
    segments = []
    directory = stage / "segments"
    directory.mkdir()
    for turn_index, turn in enumerate(turns):
        speaker = speakers[turn["speaker_index"]]
        identifier = speaker["production_voice_id"]
        if identifier not in prepared:
            await runtime.require_tts("synthesize", req.engine)
            prepared[identifier] = await runtime.tts.prepare(private, identifier)
        for text in speech_chunks(turn["text"]):
            if not text.strip():
                continue
            await runtime.require_tts("synthesize", req.engine)
            data, metrics = await runtime.tts.speech(
                text,
                voice=identifier + "_0",
                caption=caption,
                mode="synthesize",
                seed=(base_seed + len(segments)) % 2**32,
                references=prepared[identifier],
                settings={
                    **req.sampling_settings(),
                    **({"chunking_enabled": False} if req.engine == "irodori" else {}),
                },
            )
            info = wav_info(
                data, max_seconds=MAX_OUTPUT_SECONDS, max_bytes=MAX_OUTPUT_BYTES
            )
            # Reject excessive output before asking the model for another segment.
            if (
                sum(s["metrics"]["audio_seconds"] for s in segments)
                + info["seconds"]
                + len(segments)
                > MAX_OUTPUT_SECONDS
            ):
                raise HTTPException(422, "連結した作品は4時間以下にしてください")
            if sum(s["bytes"] for s in segments) + len(data) > MAX_OUTPUT_BYTES:
                raise HTTPException(422, "連結した作品は1 GiB以下にしてください")
            file = f"segments/{len(segments) + 1:04d}.wav"
            await asyncio.to_thread((stage / file).write_bytes, data)
            segments.append(
                {
                    "label": speaker["label"],
                    "production_voice_id": identifier,
                    "turn_index": turn_index,
                    "text": text,
                    "file": file,
                    "bytes": len(data),
                    "metrics": {**metrics, "audio_seconds": info["seconds"]},
                }
            )
    data, timeline = await asyncio.to_thread(
        assemble_wavs, [stage / s["file"] for s in segments]
    )
    for segment, timing in zip(segments, timeline):
        segment.update(timing)
    info = wav_info(data, max_seconds=MAX_OUTPUT_SECONDS, max_bytes=MAX_OUTPUT_BYTES)
    elapsed = time.perf_counter() - started
    return (
        data,
        {
            "elapsed_seconds": elapsed,
            "audio_seconds": info["seconds"],
            "rtf": elapsed / info["seconds"],
            "seed": base_seed,
            "settings": req.sampling_settings(),
            "wav_sha256": info["sha256"],
            "sample_rate": info["sample_rate"],
            "chunks": len(segments),
            "turns": len(turns),
            "speaker_count": len(speakers),
            "gap_seconds": 1.0,
            "reference_cache": "ref_latents"
            if req.engine == "irodori"
            else "prompt_id",
        },
        segments,
    )


async def create_production(
    req: SynthesisRequest,
    request: Request,
    uploads: list[ReferenceUpload] | None = None,
):
    previous = products().get(req.source_product_id) if req.source_product_id else None
    if req.speakers is not None and uploads is None:
        raise HTTPException(422, "話者設定は参照WAVアップロード時に指定してください")
    multiple = bool(
        (uploads and (len(uploads) > 1 or uploads[0].label))
        or (previous and previous.get("speakers"))
    )
    if (
        multiple
        and req.reference_text is not None
        and len(uploads or previous["speakers"]) > 1
    ):
        raise HTTPException(422, "複数話者の再制作では保存済みの参照文章を使います")
    turns = (
        parse_script(
            req.text,
            [u.label for u in uploads]
            if uploads
            else [s["label"] for s in previous["speakers"]],
        )
        if multiple
        else None
    )
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
            if multiple:
                private, speakers = await dialogue_voices(
                    stage, uploads, previous, req.source_product_id
                )
                if (
                    len(speakers) == 1
                    and uploads is None
                    and req.reference_text is not None
                ):
                    saved_item = private.get(speakers[0]["production_voice_id"])
                    saved_item["provenance"]["reference_text"] = (
                        req.reference_text.strip()
                    )
                    await asyncio.to_thread(private.save, saved_item)
                item = private.get(speakers[0]["production_voice_id"])
                source = {
                    "kind": "multi_upload" if uploads else "multi_product",
                    "transcript_ignored_due_to_trim": any(
                        s["voice_source"].get("transcript_ignored_due_to_trim")
                        for s in speakers
                    ),
                }
                if req.source_product_id:
                    source["product_id"] = req.source_product_id
                data, metrics, segments = await render_dialogue(
                    req, runtime, stage, private, speakers, turns, caption
                )
            else:
                if uploads:
                    req = req.model_copy(
                        update={"reference_text": uploads[0].reference_text}
                    )
                upload = (uploads[0].filename, uploads[0].data) if uploads else None
                private, item, source = await production_voice(
                    req, stage, runtime, upload
                )
                await runtime.require_tts("synthesize", req.engine)
                references = await runtime.tts.prepare(private, item["id"], item=item)
                await runtime.require_tts("synthesize", req.engine)
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
            health = await runtime.tts.health()
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
            if multiple:
                result.update(
                    {
                        "schema_version": 3,
                        "speakers": speakers,
                        "segments": segments,
                        "voice_name": " / ".join(
                            s["label"] or s["voice_name"] for s in speakers
                        ),
                        "references": [
                            {**ref, "speaker": s["label"]}
                            for s in speakers
                            for ref in s["references"]
                        ],
                    }
                )
                settings.update(
                    {
                        "speakers": speakers,
                        "gap_seconds": 1.0,
                        "reference_sha256": [
                            ref["sha256"] for ref in result["references"]
                        ],
                    }
                )
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


@app.patch("/api/synthesis/{identifier}")
async def rename_output(identifier: str, req: RenameRequest):
    return await asyncio.to_thread(products().rename, identifier, req.name)


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


@app.get("/api/synthesis/{identifier}/speakers/{index}/audio")
async def speaker_audio(identifier: str, index: int):
    store = products()
    speakers = store.get(identifier).get("speakers", [])
    if index < 0 or index >= len(speakers):
        raise HTTPException(404, "話者が見つかりません")
    library = VoiceLibrary(store.directory(identifier) / "voice")
    path = await asyncio.to_thread(
        library.audio_path, speakers[index]["production_voice_id"]
    )
    return FileResponse(
        path, media_type="audio/wav", filename=f"{identifier}-speaker-{index + 1}.wav"
    )


@app.get("/api/synthesis/{identifier}/segments/{index}/audio")
async def segment_audio(identifier: str, index: int):
    store = products()
    segments = store.get(identifier).get("segments", [])
    if index < 0 or index >= len(segments):
        raise HTTPException(404, "生成区間が見つかりません")
    return FileResponse(
        store.directory(identifier) / segments[index]["file"],
        media_type="audio/wav",
        filename=f"{identifier}-segment-{index + 1}.wav",
    )


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
