import asyncio
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from llm_chat import irodori, server, voice
from llm_chat.voice_control import separate_style, sse
from llm_chat.voice_library import VoiceLibrary, wav_info
from llm_chat.voice_runtime import VoiceRuntime


def wav(seconds=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * int(seconds * 16000))
    return buffer.getvalue()


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = VoiceLibrary(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_registration_preserves_wav_id_and_provenance_after_restart(self):
        item = self.library.create(
            wav(), name="候補", kind="candidate", provenance={"seed": 42}
        )
        self.library.update(item["id"], name="声A", register=True, favorite=True)
        reopened = VoiceLibrary(self.root)
        saved = reopened.get(item["id"])
        self.assertEqual(saved["kind"], "voice")
        self.assertEqual(saved["provenance"]["seed"], 42)
        self.assertEqual(reopened.audio_path(item["id"]).read_bytes(), wav())
        self.assertEqual(
            irodori.library_voices(reopened)[0]["key"], "irodori:" + item["id"]
        )

    def test_reference_group_only_accepts_registered_voice(self):
        item = self.library.create(wav(), name="候補", kind="candidate", provenance={})
        with self.assertRaises(HTTPException):
            self.library.append_reference(item["id"], wav())
        self.library.update(item["id"], register=True)
        saved = self.library.append_reference(item["id"], wav(2))
        self.assertEqual(len(saved["references"]), 2)
        self.assertEqual(self.library.audio_path(item["id"], 1).read_bytes(), wav(2))

    def test_corrupt_truncated_and_oversized_duration_references_are_rejected(self):
        for data in (b"invalid", wav()[:-100], wav(121)):
            with self.subTest(data=len(data)), self.assertRaises(HTTPException):
                wav_info(data)
        item = self.library.create(wav(), name="声", kind="voice", provenance={})
        self.library.audio_path(item["id"]).write_bytes(wav(2))
        with self.assertRaises(HTTPException) as caught:
            self.library.audio_path(item["id"])
        self.assertEqual(caught.exception.status_code, 409)

    def test_paths_cannot_escape_library(self):
        for identifier in ("../test", "A" * 32, "none", "../../.env"):
            with self.assertRaises(HTTPException):
                self.library.get(identifier)


class IrodoriAPITest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.library = VoiceLibrary(Path(self.temp.name) / "library")
        self.runtime = VoiceRuntime(
            Path(self.temp.name), {"irodori": "http://127.0.0.1:8088"}
        )
        self.runtime.engine = "irodori"
        self.runtime.enabled = self.runtime.ready = True
        self.requests = []
        self.uploads = {}
        self.prepared = 0

        async def upstream(request):
            self.requests.append(request)
            path = request.url.path
            if path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "model": {
                            "hf_checkpoint": "Aratako/Irodori-TTS-v4.1-Small-MF"
                            if self.runtime.mode == "chat"
                            else "Aratako/Irodori-TTS-v4-Large"
                        },
                        "runtime": {"loaded": True},
                    },
                )
            if path == "/v1/audio/speech":
                payload = json.loads(request.content)
                return httpx.Response(
                    200,
                    content=wav(),
                    headers={"X-Irodori-Seed": str(payload["irodori"].get("seed", 9))},
                )
            if path == "/v1/audio/voices" and request.method == "POST":
                # Multipart IDs occur after the voice_id field header.
                identifier = (
                    request.content.split(b'name="voice_id"\r\n\r\n')[1]
                    .split(b"\r\n")[0]
                    .decode()
                )
                if identifier in self.uploads:
                    return httpx.Response(409)
                self.uploads[identifier] = True
                return httpx.Response(201, json={"id": identifier})
            if path.startswith("/v1/audio/voices/") and request.method == "PUT":
                return httpx.Response(200, json={"id": path.split("/")[-1]})
            if path.endswith("/prepare"):
                self.prepared += 1
                return httpx.Response(
                    200, json={"ref_latents": [f"/server/{path.split('/')[-2]}.pt"]}
                )
            raise AssertionError((request.method, path))

        self.runtime.irodori.http = lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )
        self.patch = patch.object(voice, "LIBRARY", self.library)
        self.patch.start()
        self.old_runtime = getattr(server.app.state, "speech_runtime", None)
        server.app.state.speech_runtime = self.runtime
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.runtime.stop()
        server.app.state.speech_runtime = self.old_runtime
        self.patch.stop()
        self.temp.cleanup()

    async def test_design_register_and_mf_chat_use_distinct_samplers_and_fixed_reference(
        self,
    ):
        self.runtime.mode = "design"
        result = await self.client.post(
            "/api/voice/library/candidates",
            json={
                "caption": "落ち着いた声",
                "text": "こんにちは。",
                "seed": 42,
                "count": 2,
            },
        )
        self.assertEqual(result.status_code, 200, result.text)
        items = result.json()["items"]
        self.assertEqual([item["provenance"]["seed"] for item in items], [42, 43])
        self.assertEqual(items[0]["provenance"]["settings"]["num_steps"], 40)
        identifier = items[0]["id"]
        result = await self.client.patch(
            f"/api/voice/library/{identifier}", json={"register": True, "name": "声A"}
        )
        self.assertEqual(result.json()["kind"], "voice")
        self.runtime.mode = "chat"
        voices = (await self.client.get("/api/voice/voices")).json()
        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0]["styles"][1]["id"], "happy")
        for text in ("今日は良い天気です。", "次の文です。"):
            result = await self.client.post(
                "/api/voice/synthesize",
                json={
                    "engine": "irodori",
                    "voice_id": identifier,
                    "preset": "happy",
                    "text": text,
                },
            )
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.content, wav())
        self.assertEqual(self.prepared, 1)
        speech = [
            json.loads(r.content)
            for r in self.requests
            if r.url.path.endswith("/speech")
        ]
        self.assertEqual(speech[-1]["irodori"]["num_steps"], 4)
        self.assertEqual(speech[-1]["irodori"]["caption"], irodori.caption_for("happy"))
        self.assertEqual(
            speech[-1]["irodori"]["ref_latents"], speech[-2]["irodori"]["ref_latents"]
        )
        self.assertNotIn("style_id", speech[-1])

    async def test_prepare_reuploads_on_restart_and_uses_put_for_existing_file(self):
        item = self.library.create(wav(), name="声", kind="voice", provenance={})
        await self.runtime.irodori.prepare(self.library, item["id"])
        self.runtime.irodori.clear()
        await self.runtime.irodori.prepare(self.library, item["id"])
        self.assertEqual(self.prepared, 2)
        self.assertTrue(any(r.method == "PUT" for r in self.requests))

    async def test_off_cold_wrong_mode_invalid_preset_and_legacy_engine_are_rejected(
        self,
    ):
        item = self.library.create(wav(), name="声", kind="voice", provenance={})
        payload = {"engine": "irodori", "voice_id": item["id"], "text": "テスト"}
        self.runtime.enabled = False
        self.assertEqual(
            (await self.client.post("/api/voice/synthesize", json=payload)).status_code,
            409,
        )
        self.runtime.enabled = True
        self.runtime.ready = False
        self.assertEqual(
            (await self.client.post("/api/voice/synthesize", json=payload)).status_code,
            503,
        )
        self.runtime.ready = True
        self.runtime.mode = "design"
        self.assertEqual(
            (await self.client.post("/api/voice/synthesize", json=payload)).status_code,
            409,
        )
        self.runtime.mode = "chat"
        self.assertEqual(
            (
                await self.client.post(
                    "/api/voice/synthesize", json={**payload, "preset": "bogus"}
                )
            ).status_code,
            422,
        )
        self.assertEqual(
            (
                await self.client.post(
                    "/api/voice/synthesize", json={"text": "テスト", "style_id": 3}
                )
            ).status_code,
            409,
        )
        self.assertEqual(self.requests, [])

    async def test_import_group_preview_and_delete(self):
        result = await self.client.post(
            "/api/voice/library/import",
            data={"name": "録音", "source": "自分の声"},
            files={"file": ("reference.wav", wav(), "audio/wav")},
        )
        self.assertEqual(result.status_code, 200, result.text)
        identifier = result.json()["id"]
        endpoint = f"/api/voice/library/{identifier}"
        self.assertEqual((await self.client.get(endpoint + "/audio")).content, wav())
        rejected = await self.client.post(
            endpoint + "/references",
            data={"same_speaker": "false"},
            files={"file": ("extra.wav", wav(), "audio/wav")},
        )
        self.assertEqual(rejected.status_code, 422)
        appended = await self.client.post(
            endpoint + "/references",
            data={"same_speaker": "true"},
            files={"file": ("extra.wav", wav(), "audio/wav")},
        )
        self.assertEqual(len(appended.json()["references"]), 2)
        result = await self.client.post(
            endpoint + "/preview", json={"text": "別の文章", "preset": "sad"}
        )
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn("rtf", json.loads(result.headers["X-TTS-Metrics"]))
        self.assertEqual(self.prepared, 2)
        self.assertEqual((await self.client.delete(endpoint)).status_code, 204)
        self.assertEqual((await self.client.get(endpoint + "/audio")).status_code, 404)

    async def test_string_style_event_and_history(self):
        async def source():
            yield sse(
                {
                    "choices": [
                        {"delta": {"content": "[[VOICE_STYLE:喜び]]こんにちは。"}}
                    ]
                }
            )
            yield b"data: [DONE]\n\n"

        parts = [p async for p in separate_style(source(), irodori.styles())]
        self.assertEqual(json.loads(parts[0][6:])["voice_style"]["id"], "happy")
        message = server.HistoryMessage(
            role="assistant", content="本文", voice_style_id="happy", voice_id="abc"
        )
        self.assertEqual(message.voice_style_id, "happy")


class IrodoriRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_health_is_followed_by_warmup_before_ready(self):
        runtime = VoiceRuntime(Path("/tmp"), {"irodori": "https://speech.example"})
        runtime.engine = "irodori"
        loaded = False
        calls = []

        async def upstream(req):
            nonlocal loaded
            calls.append(req.url.path)
            if req.url.path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "model": {"hf_checkpoint": "Aratako/Irodori-TTS-v4.1-Small-MF"},
                        "runtime": {"loaded": loaded},
                    },
                )
            self.assertFalse(runtime.ready)
            loaded = True
            return httpx.Response(200, content=wav())

        runtime.irodori.http = lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )
        try:
            await runtime.set_enabled(True)
            await asyncio.gather(*runtime.tasks)
            self.assertTrue(runtime.ready)
            self.assertEqual(calls, ["/health", "/v1/audio/speech", "/health"])
            self.assertEqual(runtime.processes, {})
        finally:
            await runtime.stop()

    async def test_external_wrong_checkpoint_is_not_warmed_or_stopped(self):
        runtime = VoiceRuntime(Path("/tmp"), {"irodori": "https://speech.example"})
        runtime.engine = "irodori"

        async def upstream(req):
            self.assertEqual(req.url.path, "/health")
            return httpx.Response(200, json={"model": {"hf_checkpoint": "wrong"}})

        runtime.irodori.http = lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )
        try:
            await runtime.set_enabled(True)
            await asyncio.gather(*runtime.tasks)
            self.assertFalse(runtime.ready)
            self.assertIn("モデルが異なります", runtime.errors["irodori"])
            self.assertEqual(runtime.processes, {})
        finally:
            await runtime.stop()


if __name__ == "__main__":
    unittest.main()
