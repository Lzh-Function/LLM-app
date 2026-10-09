"""GPU ownership, loading cancellation and standalone artifact integration."""

import asyncio
import importlib.util
import io
import json
import shlex
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from llm_chat import server, voice, voice_synthesize
from llm_chat.gpu_reservation import GPUReservation
from llm_chat.irodori_profiles import profile_for
from llm_chat.voice_library import VoiceLibrary
from llm_chat.voice_runtime import VoiceRuntime
from test_irodori import wav


class LongTextTest(unittest.TestCase):
    def test_long_sentences_are_bounded_without_losing_text(self):
        path = Path(__file__).resolve().parents[2] / "irodori-tts" / "chunking.py"
        spec = importlib.util.spec_from_file_location("irodori_chunking", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for text in ("声のライブラリから音声を合成します" * 350, "word " * 1500):
            chunks = module.bounded_chunks([text])
            self.assertEqual("".join(chunks), text)
            self.assertLessEqual(max(map(len, chunks)), 160)
        self.assertEqual(module.bounded_chunks(["短い文章です。"]), ["短い文章です。"])


class GPUOwnershipTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.before = AsyncMock()
        self.runtime = VoiceRuntime(
            self.root, {"irodori": "http://127.0.0.1:1"}, before_gpu_start=self.before
        )

    async def asyncTearDown(self):
        await self.runtime.stop()
        self.temp.cleanup()

    async def test_gpu_unloads_llm_first_and_missing_model_releases_slot(self):
        self.before.side_effect = lambda: self.assertFalse(self.runtime.gpu.fds)
        await self.runtime.set_enabled(True, "irodori", "design")
        self.before.assert_awaited_once()
        self.assertTrue(self.runtime.gpu.fds)
        await asyncio.gather(*self.runtime.tasks)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(self.runtime.errors)
        other = GPUReservation(self.root)
        other.acquire("LLM")
        other.release()

    async def test_cpu_conversation_does_not_unload_llm(self):
        await self.runtime.set_enabled(True, "irodori", "chat")
        await asyncio.gather(*self.runtime.tasks)
        self.before.assert_not_awaited()
        self.assertFalse(self.runtime.gpu.fds)

    async def test_design_to_synthesis_reuses_model_and_blocks_llm(self):
        async def ready():
            self.runtime.ready = True

        with patch.object(self.runtime, "_start_irodori", ready):
            await self.runtime.set_enabled(True, "irodori", "design")
            await asyncio.gather(*self.runtime.tasks)
            task = self.runtime.tasks[0]
            await self.runtime.set_enabled(True, "irodori", "synthesize")
            self.assertIs(task, self.runtime.tasks[0])
            self.assertTrue(self.runtime.ready)
            self.before.assert_awaited_once()
            other = GPUReservation(self.root)
            with self.assertRaises(HTTPException) as exc:
                other.acquire("LLM")
            self.assertEqual(exc.exception.status_code, 409)
            await self.runtime.stop()
            other.acquire("LLM")
            other.release()

    async def test_unload_failure_never_starts_gpu(self):
        self.before.side_effect = HTTPException(503, "unload failed")
        with self.assertRaises(HTTPException):
            await self.runtime.set_enabled(True, "irodori", "design")
        self.assertFalse(self.runtime.gpu.fds)
        self.assertFalse(self.runtime.tasks)

    async def test_unmanaged_gpu_server_is_not_reused(self):
        with patch("llm_chat.voice_runtime.port_in_use", AsyncMock(return_value=True)):
            await self.runtime.set_enabled(True, "irodori", "design")
            await asyncio.gather(*self.runtime.tasks)
        self.assertIn("管理外", self.runtime.errors["irodori"])
        self.assertFalse(self.runtime.gpu.fds)

    async def test_inherited_fd_retains_lock_after_parent_closes(self):
        slot = GPUReservation(self.root)
        slot.acquire("TTS")
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)", pass_fds=slot.fds
        )
        try:
            slot.release()
            other = GPUReservation(self.root)
            with self.assertRaises(HTTPException):
                other.acquire("LLM")
            child.terminate()
            await child.wait()
            other.acquire("LLM")
            other.release()
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()

    def test_profile_defaults_and_full_large_checkpoint(self):
        design, synth, chat = (
            profile_for(mode) for mode in ("design", "synthesize", "chat")
        )
        self.assertEqual(design.identity, synth.identity)
        self.assertEqual(design.repo, "Aratako/Irodori-TTS-v4-Large")
        self.assertEqual(design.model_device, "cuda")
        self.assertEqual(design.model_precision, "bf16")
        self.assertEqual(design.codec_device, "cpu")
        self.assertEqual(chat.model_device, "cpu")


class LoadingCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        with patch.object(server, "LLM_ROOT", self.root):
            self.manager = server.ModelManager()
        script = self.root / "serve.sh"
        script.write_text(
            "exec "
            + shlex.quote(sys.executable)
            + " -c 'import time; time.sleep(60)'\n"
        )
        self.manager.models = {
            name: server.ModelInfo(name, name, "", 1, self.root) for name in ("a", "b")
        }
        self.children = []

        async def waiting(proc):
            self.children.append(proc)
            await asyncio.sleep(60)

        self.manager._wait_healthy = waiting

    async def asyncTearDown(self):
        await self.manager.unload()
        self.temp.cleanup()

    async def wait_children(self, count):
        async with asyncio.timeout(3):
            while len(self.children) < count:
                await asyncio.sleep(0.01)

    async def test_unload_during_llm_load_stops_process_and_releases_gpu(self):
        await self.manager.request_load("a")
        await self.wait_children(1)
        await asyncio.wait_for(self.manager.unload(), timeout=3)
        self.assertIsNotNone(self.children[0].returncode)
        self.assertEqual(self.manager.status, "stopped")
        self.assertFalse(self.manager.gpu.fds)
        slot = GPUReservation(self.root)
        slot.acquire("TTS")
        slot.release()

    async def test_switch_waits_for_cancelled_loader_cleanup(self):
        await self.manager.request_load("a")
        await self.wait_children(1)
        await self.manager.request_load("b")
        await self.wait_children(2)
        self.assertIsNotNone(self.children[0].returncode)
        self.assertIsNone(self.children[1].returncode)
        self.assertEqual(self.manager.active, "b")
        await self.manager.unload()
        self.assertIsNotNone(self.children[1].returncode)

    async def test_cancelled_stop_retains_process_for_cleanup(self):
        started, finish = asyncio.Event(), asyncio.Event()

        async def wait():
            started.set()
            await finish.wait()
            return 0

        proc = SimpleNamespace(pid=999999, wait=wait)
        self.manager._proc = proc
        with patch.object(server.os, "killpg"):
            task = asyncio.create_task(self.manager._stop_proc())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIs(self.manager._proc, proc)
            finish.set()
            await self.manager._stop_proc()
        self.assertIsNone(self.manager._proc)

    async def test_tts_ownership_prevents_llm_load(self):
        slot = GPUReservation(self.root)
        slot.acquire("TTS")
        try:
            with self.assertRaises(HTTPException):
                await self.manager.request_load("a")
            self.assertEqual(self.manager.status, "stopped")
            self.assertFalse(self.children)
        finally:
            slot.release()


class StandaloneTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = VoiceLibrary(self.root / "library")
        self.payloads = []
        self.unloader = patch.object(voice_synthesize, "unload_chat_llm", AsyncMock())
        self.patches = [
            self.unloader,
            patch.object(voice_synthesize, "ROOT", self.root),
            patch.object(voice_synthesize, "OUTPUT_DIR", self.root / "productions"),
            patch.object(voice_synthesize, "LEGACY_OUTPUT_DIR", self.root / "legacy"),
            patch.object(voice, "LIBRARY", self.library),
        ]
        for p in self.patches:
            p.start()
        self.life = voice_synthesize.lifespan(voice_synthesize.app)
        await self.life.__aenter__()
        self.runtime = voice_synthesize.app.state.speech_runtime
        self.runtime.enabled = self.runtime.ready = True

        def upstream(request):
            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "model": {"hf_checkpoint": profile_for("synthesize").repo},
                        "runtime": {"loaded": True},
                    },
                )
            if request.url.path.endswith("/prepare"):
                return httpx.Response(200, json={"ref_latents": ["/cache/speaker.pt"]})
            if request.url.path == "/v1/audio/speech":
                self.payloads.append(json.loads(request.content))
                return httpx.Response(
                    200,
                    content=wav(2 if self.payloads[-1]["voice"] == "none" else 130),
                    headers={"X-Irodori-Seed": "123"},
                )
            return httpx.Response(201, json={"id": "voice"})

        self.runtime.irodori.http = lambda: httpx.AsyncClient(
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

    async def test_long_import_is_cropped_and_transcript_is_not_reused(self):
        response = await self.client.post(
            "/api/voice/library/import",
            data={"name": "長い録音", "transcript": "カット前の全文"},
            files={"file": ("long.wav", wav(121), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()
        self.assertEqual(item["references"][0]["seconds"], 120)
        self.assertEqual(item["references"][0]["trimmed_from_seconds"], 121)
        self.assertEqual(item["provenance"]["text"], "")
        self.assertEqual(item["provenance"]["source_text"], "カット前の全文")
        audio = await self.client.get(f'/api/voice/library/{item["id"]}/audio')
        self.assertEqual(audio.content, wav(120))

    async def test_extra_reference_uses_remaining_duration_and_full_group_rejects(self):
        item = self.library.create(wav(100), name="声", kind="voice", provenance={})
        endpoint = f'/api/voice/library/{item["id"]}/references'
        payload = {"same_speaker": "true"}
        files = {"file": ("extra.wav", wav(121), "audio/wav")}
        response = await self.client.post(endpoint, data=payload, files=files)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["references"][1]["seconds"], 20)
        self.assertEqual(self.library.audio_path(item["id"], 1).read_bytes(), wav(20))
        response = await self.client.post(endpoint, data=payload, files=files)
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(len(self.library.get(item["id"])["references"]), 2)

    async def test_import_larger_than_previous_upload_limit_is_cropped(self):
        data = wav(1100)
        self.assertGreater(len(data), 32 * 1024 * 1024)
        response = await self.client.post(
            "/api/voice/library/import",
            data={"name": "長い録音"},
            files={"file": ("long.wav", data, "audio/wav")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()
        self.assertEqual(item["references"][0]["seconds"], 120)
        self.assertEqual(self.library.audio_path(item["id"]).read_bytes(), wav(120))

    async def test_20000_character_manuscript_is_forwarded_and_preserved_in_zip(self):
        text = "長文の原稿です。" * 2500
        self.assertEqual(len(text), 20000)
        response = await self.client.post("/api/synthesis", json={"engine": "irodori", "text": text})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(self.payloads[-1]["input"], text)
        self.assertEqual(result["text"], text)
        package = await self.client.get(f"/api/synthesis/{result['id']}/archive")
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            self.assertEqual(archive.read("manuscript.txt").decode(), text)
        self.assertEqual((await self.client.get("/api/synthesis/options")).json()["max_text_chars"], 20000)
        too_long = await self.client.post("/api/synthesis", json={"engine": "irodori", "text": text + "字"})
        self.assertEqual(too_long.status_code, 422)
        self.assertEqual(len((await self.client.get("/api/synthesis")).json()["items"]), 1)

    async def test_registered_voice_synthesis_saves_downloadable_wav_and_metadata(self):
        item = self.library.create(wav(1), name="声A", kind="voice", provenance={})
        response = await self.client.post(
            "/api/synthesis",
            json={
                "voice_id": item["id"],
                "text": "こんにちは。" * 30,
                "seed": 123,
                "num_steps": 80,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["voice_id"], item["id"])
        self.assertEqual(result["references"], item["references"])
        self.assertEqual(self.payloads[-1]["irodori"]["num_steps"], 80)
        self.assertTrue(self.payloads[-1]["irodori"]["chunking_enabled"])
        self.assertEqual(
            (await self.client.get(f"/api/synthesis/{result['id']}/audio")).content,
            wav(130),
        )
        self.assertEqual(
            (await self.client.get(f"/api/synthesis/{result['id']}/metadata")).json(),
            result,
        )
        self.assertEqual(
            len((await self.client.get("/api/synthesis")).json()["items"]), 1
        )
        self.assertEqual(
            (await self.client.delete(f"/api/synthesis/{result['id']}")).status_code,
            204,
        )
        self.assertEqual(
            (await self.client.get(f"/api/synthesis/{result['id']}/audio")).status_code,
            404,
        )

    async def test_production_needs_no_chat_library_and_saves_portable_artifacts(self):
        with patch.object(
            self.library, "get", side_effect=AssertionError("chat library accessed")
        ):
            response = await self.client.post(
                "/api/synthesis",
                json={
                    "text": "この原稿から、独立した音声作品を作ります。",
                    "name": "ナレーション作品",
                    "voice_caption": "穏やかな低めの女性の声。",
                    "seed": 123,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIsNone(result["voice_id"])
        self.assertEqual(result["voice_source"]["kind"], "generated")
        self.assertEqual(len(self.payloads), 2)
        self.assertEqual(self.payloads[0]["voice"], "none")
        self.assertNotEqual(self.payloads[1]["voice"], "none")
        self.assertEqual(self.library.list(), [])
        directory = self.root / "productions" / result["id"]
        self.assertTrue(
            (
                directory / "voice" / result["production_voice_id"] / "reference.wav"
            ).is_file()
        )
        text = await self.client.get(f"/api/synthesis/{result['id']}/manuscript")
        self.assertEqual(text.text, result["text"])
        settings = (
            await self.client.get(f"/api/synthesis/{result['id']}/settings")
        ).json()
        self.assertEqual(settings["voice_caption"], "穏やかな低めの女性の声。")
        self.assertEqual(settings["seed"], 123)
        package = await self.client.get(f"/api/synthesis/{result['id']}/archive")
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            self.assertEqual(archive.read("audio.wav"), wav(130))
            self.assertEqual(archive.read("manuscript.txt").decode(), result["text"])
            self.assertIn(
                f"voice/{result['production_voice_id']}/reference.wav",
                archive.namelist(),
            )
        self.assertFalse(list((self.root / "productions").glob(".pending-*")))

    async def test_saved_product_reuses_private_voice_after_source_library_deletion(
        self,
    ):
        source = self.library.create(
            wav(), name="共有の声", kind="voice", provenance={"source": "import"}
        )
        self.library.append_reference(source["id"], wav(2))
        first = (
            await self.client.post(
                "/api/synthesis", json={"voice_id": source["id"], "text": "最初の作品"}
            )
        ).json()
        self.library.delete(source["id"])
        with patch.object(
            self.library, "get", side_effect=AssertionError("deleted library accessed")
        ):
            response = await self.client.post(
                "/api/synthesis",
                json={"source_product_id": first["id"], "text": "同じ声で別の作品"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        second = response.json()
        self.assertEqual(second["voice_source"]["kind"], "product")
        self.assertEqual(
            [r["sha256"] for r in first["references"]],
            [r["sha256"] for r in second["references"]],
        )
        self.assertNotEqual(first["production_voice_id"], second["production_voice_id"])
        self.assertEqual(len(self.payloads), 2)  # No voice design during regeneration.
        await self.client.delete(f"/api/synthesis/{first['id']}")
        self.assertEqual(
            (
                await self.client.get(f"/api/synthesis/{second['id']}/archive")
            ).status_code,
            200,
        )

    async def test_failed_synthesis_never_publishes_incomplete_product(self):
        with patch.object(
            self.runtime.irodori,
            "speech",
            AsyncMock(side_effect=HTTPException(503, "failed")),
        ):
            response = await self.client.post(
                "/api/synthesis", json={"text": "失敗確認"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual((await self.client.get("/api/synthesis")).json()["items"], [])
        self.assertFalse(list((self.root / "productions").glob("*")))
        self.assertEqual(self.library.list(), [])

    async def test_legacy_outputs_remain_readable_and_invalid_source_is_rejected(self):
        identifier = "a" * 32
        directory = self.root / "legacy" / identifier
        directory.mkdir(parents=True)
        legacy = {
            "id": identifier,
            "name": "旧合成",
            "text": "以前の原稿",
            "created_at": "2026-01-01",
        }
        (directory / "metadata.json").write_text(json.dumps(legacy))
        (directory / "audio.wav").write_bytes(wav())
        self.assertEqual(
            (await self.client.get("/api/synthesis")).json()["items"], [legacy]
        )
        self.assertEqual(
            (await self.client.get(f"/api/synthesis/{identifier}/manuscript")).text,
            "以前の原稿",
        )
        self.assertEqual(
            (await self.client.get(f"/api/synthesis/{identifier}/audio")).content, wav()
        )
        for body in (
            {
                "text": " ",
            },
            {"text": "原稿", "voice_id": "b" * 32, "source_product_id": identifier},
        ):
            self.assertEqual(
                (await self.client.post("/api/synthesis", json=body)).status_code, 422
            )
        self.assertEqual(
            (
                await self.client.post(
                    "/api/synthesis",
                    json={"text": "再制作", "source_product_id": identifier},
                )
            ).status_code,
            409,
        )
        self.assertFalse(list((self.root / "productions").glob("*")))

    async def test_candidates_disabled_runtime_and_invalid_settings_rejected(self):
        item = self.library.create(wav(), name="候補", kind="candidate", provenance={})
        body = {"voice_id": item["id"], "text": "テスト"}
        self.assertEqual(
            (await self.client.post("/api/synthesis", json=body)).status_code, 422
        )
        self.library.update(item["id"], register=True)
        self.runtime.ready = False
        self.assertEqual(
            (await self.client.post("/api/synthesis", json=body)).status_code, 503
        )
        self.runtime.enabled = False
        self.assertEqual(
            (await self.client.post("/api/synthesis", json=body)).status_code, 409
        )
        for settings in (
            {"num_steps": 121},
            {"cfg_scale_text": -1},
            {"text": "x" * 20001},
        ):
            self.assertEqual(
                (
                    await self.client.post("/api/synthesis", json={**body, **settings})
                ).status_code,
                422,
            )
        self.assertEqual(
            (await self.client.get("/api/synthesis/not-an-id/audio")).status_code, 404
        )
        self.assertEqual((await self.client.get("/api/status")).status_code, 404)


class StandaloneUnloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_absent_llm_chat_is_skipped_but_failed_unload_blocks(self):
        for outcome in (
            httpx.ConnectError("offline"),
            httpx.Response(200, json={"active": None, "status": "stopped"}),
            httpx.Response(200, json={"active": "model", "status": "ready"}),
            httpx.Response(500),
        ):

            def upstream(request, outcome=outcome):
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            with patch.object(
                voice_synthesize.httpx, "AsyncClient", return_value=client
            ):
                if (
                    isinstance(outcome, httpx.ConnectError)
                    or outcome.status_code == 200
                    and outcome.json().get("active") is None
                ):
                    await voice_synthesize.unload_chat_llm()
                else:
                    with self.assertRaises(HTTPException) as exc:
                        await voice_synthesize.unload_chat_llm()
                    self.assertEqual(exc.exception.status_code, 503)
