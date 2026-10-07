"""Qwen engine coordination, transcript-aware cloning and product portability."""

import asyncio
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from llm_chat import voice, voice_synthesize
from llm_chat.gpu_reservation import GPUReservation
from llm_chat.voice_library import VoiceLibrary
from llm_chat.voice_library_api import CandidateRequest
from llm_chat.voice_runtime import VoiceRuntime
from test_irodori import wav


class ChunkingTest(unittest.TestCase):
    def test_preserves_punctuation_newlines_and_bounds_unbroken_text(self):
        path = Path(__file__).resolve().parents[2] / "qwen-tts" / "chunking.py"
        spec = importlib.util.spec_from_file_location("qwen_chunking", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for text in (
            "句読点なし" * 4000,
            "短い文。\n\n次の文！" * 60,
            "word " * 4000,
            "。！？\n",
        ):
            parts = mod.chunks(text)
            self.assertEqual("".join(parts), text)
            self.assertLessEqual(max(map(len, parts)), 160)

    def test_rejects_incompatible_instructions_and_unbounded_design(self):
        with self.assertRaises(ValueError):
            voice_synthesize.SynthesisRequest(
                engine="qwen", text="原稿", preset="happy"
            )
        with self.assertRaises(ValueError):
            CandidateRequest(engine="qwen", caption="明瞭", text="声" * 161)
        with self.assertRaises(ValueError):
            voice_synthesize.SynthesisRequest(engine="qwen", text="原稿", temperature=0)


class RuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_switch_preserves_gpu_exclusion_and_only_one_managed_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = AsyncMock()
            runtime = VoiceRuntime(root, {}, before_gpu_start=before)

            async def ready():
                runtime.ready = True

            try:
                with (
                    patch.object(runtime, "_start_qwen", ready),
                    patch.object(runtime, "_start_irodori", ready),
                ):
                    await runtime.set_enabled(True, "qwen", "design")
                    await asyncio.gather(*runtime.tasks)
                    task = runtime.tasks[0]
                    await runtime.set_enabled(True, "qwen", "synthesize")
                    self.assertIs(task, runtime.tasks[0])
                    self.assertTrue(runtime.ready)
                    self.assertEqual(before.await_count, 1)
                    with self.assertRaises(HTTPException):
                        GPUReservation(root).acquire("LLM")
                    await runtime.set_enabled(True, "irodori", "synthesize")
                    await asyncio.gather(*runtime.tasks)
                    self.assertEqual(runtime.engine, "irodori")
                    self.assertEqual(before.await_count, 2)
                    self.assertTrue(runtime.gpu.fds)
                    with self.assertRaises(ValueError):
                        await runtime.set_enabled(True, "qwen", "chat")
            finally:
                await runtime.stop()
            guard = GPUReservation(root)
            guard.acquire("LLM")
            guard.release()

    async def test_failed_qwen_install_releases_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = VoiceRuntime(
                Path(temp), {"qwen": "http://127.0.0.1:1"}, before_gpu_start=AsyncMock()
            )
            await runtime.set_enabled(True, "qwen", "synthesize")
            await asyncio.gather(*runtime.tasks)
            self.assertIn("未導入", runtime.errors["qwen"])
            self.assertFalse(runtime.gpu.fds)
            await runtime.stop()


class ProductTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = VoiceLibrary(self.root / "library")
        self.calls = []
        self.patches = [
            patch.object(voice_synthesize, "ROOT", self.root),
            patch.object(voice_synthesize, "OUTPUT_DIR", self.root / "products"),
            patch.object(voice_synthesize, "LEGACY_OUTPUT_DIR", self.root / "legacy"),
            patch.object(voice, "LIBRARY", self.library),
        ]
        for p in self.patches:
            p.start()
        self.life = voice_synthesize.lifespan(voice_synthesize.app)
        await self.life.__aenter__()
        self.runtime = voice_synthesize.app.state.speech_runtime
        self.runtime.engine = "qwen"
        self.runtime.enabled = self.runtime.ready = True

        def upstream(req):
            self.calls.append(req)
            if req.url.path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "model": {},
                        "runtime": {"loaded": True, "model_key": "base"},
                    },
                )
            if req.url.path == "/local/voices/prepare":
                return httpx.Response(200, json={"prompt_id": "a" * 64})
            if req.url.path == "/v1/audio/speech":
                opts = json.loads(req.content)["qwen"]
                return httpx.Response(
                    200,
                    content=wav(2),
                    headers={
                        "X-Qwen-Metrics": json.dumps(
                            {
                                "seed": opts.get("seed") or 42,
                                "clone_mode": "icl"
                                if opts.get("prompt_id")
                                else "voice_design",
                                "chunks": 3,
                            }
                        )
                    },
                )
            return httpx.Response(404)

        self.runtime.qwen.http = lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=voice_synthesize.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    async def test_20000_character_manuscript_is_forwarded_and_preserved_in_zip(self):
        text = "長文の原稿です。" * 2500
        self.assertEqual(len(text), 20000)
        response = await self.client.post(
            "/api/synthesis", json={"engine": "qwen", "text": text}
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(
            json.loads(
                [r for r in self.calls if r.url.path == "/v1/audio/speech"][-1].content
            )["input"],
            text,
        )
        self.assertEqual(result["text"], text)
        package = await self.client.get(f"/api/synthesis/{result['id']}/archive")
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            self.assertEqual(archive.read("manuscript.txt").decode(), text)
        self.assertEqual(
            (await self.client.get("/api/synthesis/options")).json()["max_text_chars"],
            20000,
        )
        too_long = await self.client.post(
            "/api/synthesis", json={"engine": "qwen", "text": text + "字"}
        )
        self.assertEqual(too_long.status_code, 422)
        self.assertEqual(
            len((await self.client.get("/api/synthesis")).json()["items"]), 1
        )

    async def test_generated_voice_transcript_is_portable_and_reused_after_original_product_deleted(
        self,
    ):
        response = await self.client.post(
            "/api/synthesis",
            json={"engine": "qwen", "text": "長い文章。" * 60, "seed": 42},
        )
        self.assertEqual(response.status_code, 200, response.text)
        first = response.json()
        self.assertEqual(first["engine"], "qwen")
        self.assertEqual(first["metrics"]["clone_mode"], "icl")
        self.assertFalse(self.library.list())
        speech = [
            json.loads(r.content)
            for r in self.calls
            if r.url.path == "/v1/audio/speech"
        ]
        self.assertIn("caption", speech[0]["qwen"])
        self.assertIn("prompt_id", speech[1]["qwen"])
        self.assertNotIn("caption", speech[1]["qwen"])
        self.assertNotIn("num_steps", speech[1]["qwen"])
        prepared = next(r for r in self.calls if r.url.path == "/local/voices/prepare")
        self.assertIn("こんにちは。".encode(), prepared.content)
        archive = await self.client.get(f"/api/synthesis/{first['id']}/archive")
        with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
            ref = next(
                n
                for n in z.namelist()
                if n.startswith("voice/") and n.endswith("metadata.json")
            )
            self.assertTrue(json.loads(z.read(ref))["provenance"]["text"])
        second = await self.client.post(
            "/api/synthesis",
            json={
                "engine": "qwen",
                "source_product_id": first["id"],
                "text": "別の原稿",
            },
        )
        self.assertEqual(second.status_code, 200, second.text)
        second = second.json()
        await self.client.delete(f"/api/synthesis/{first['id']}")
        third = await self.client.post(
            "/api/synthesis",
            json={
                "engine": "qwen",
                "source_product_id": second["id"],
                "text": "もう一度",
            },
        )
        self.assertEqual(third.status_code, 200, third.text)
        self.assertEqual(first["references"], third.json()["references"])

    async def test_transcript_override_is_private_and_only_primary_clip_is_sent(self):
        item = self.library.create(wav(1), name="取り込み", kind="voice", provenance={})
        self.library.append_reference(item["id"], wav(2))
        response = await self.client.post(
            "/api/synthesis",
            json={
                "engine": "qwen",
                "voice_id": item["id"],
                "text": "原稿",
                "reference_text": "正確な参照文章",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("reference_text", self.library.get(item["id"])["provenance"])
        prepares = [r for r in self.calls if r.url.path == "/local/voices/prepare"]
        self.assertEqual(len(prepares), 1)
        self.assertIn("正確な参照文章".encode(), prepares[0].content)
        private = VoiceLibrary(self.root / "products" / response.json()["id"] / "voice")
        self.assertEqual(
            private.get(response.json()["production_voice_id"])["provenance"][
                "reference_text"
            ],
            "正確な参照文章",
        )

    async def test_engine_mismatch_is_rejected_before_any_upstream_calls(self):
        response = await self.client.post(
            "/api/synthesis", json={"engine": "irodori", "text": "原稿"}
        )
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.calls)
