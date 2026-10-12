"""Reference reuse across uploads, products, processes and downloads."""

import errno
import json
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx
from llm_chat.irodori import IrodoriClient
from llm_chat.qwen_tts import QwenClient
from llm_chat.synthesis_products import ProductStore, copy_voice
from llm_chat.voice_library import VoiceLibrary
from llm_chat.voice_synthesize import TemporaryArchiveResponse
from test_irodori import wav


class ReferenceStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ProductStore(Path(self.temp.name) / "products")

    def publish(self, name, data=None, previous=None):
        identifier, stage = self.store.stage()
        private = self.store.voice_library(stage)
        if previous:
            library, original = previous
            item = copy_voice(library, original, private)
        else:
            item = private.create(data, name=name, kind="voice", provenance={"text": name})
        self.store.publish(stage, wav(2), {"id": identifier, "name": name, "text": name}, {"name": name})
        return identifier, self.store.voice_library(self.store.directory(identifier)), item

    def test_repeated_uploads_and_regeneration_share_storage_and_survive_deletion(self):
        first, library, item = self.publish("初回", wav())
        second, again, duplicate = self.publish("再アップロード", wav())
        third, clone, copied = self.publish("再制作", previous=(library, item))
        paths = [library.audio_path(item["id"]), again.audio_path(duplicate["id"]), clone.audio_path(copied["id"])]
        self.assertTrue(all(paths[0].samefile(path) for path in paths))
        self.assertEqual(len(list(self.store.references.root.glob("*.wav"))), 1)
        clone.update(copied["id"], name="変更後")
        self.assertEqual(library.get(item["id"])["name"], "初回")
        for identifier in (first, second):
            self.store.delete(identifier)
        self.assertEqual(clone.audio_path(copied["id"]).read_bytes(), wav())
        package = self.store.archive(third)
        try:
            with zipfile.ZipFile(package) as archive:
                self.assertEqual(archive.read(f'voice/{copied["id"]}/reference.wav'), wav())
        finally:
            package.unlink()
        self.assertFalse(list(self.store.root.glob("*/product.zip")))
        self.store.delete(third)
        self.assertFalse(list(self.store.references.root.glob("*.wav")))

    def test_concurrent_identical_uploads_write_one_reference_object(self):
        def create(index):
            library = self.store.voice_library(self.store.root / f"pending-{index}")
            item = library.create(wav(), name="声", kind="voice", provenance={})
            return library.audio_path(item["id"])

        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(create, range(8)))
        self.assertTrue(all(paths[0].samefile(path) for path in paths))
        self.assertEqual(paths[0].read_bytes(), wav())
        self.assertEqual(len(list(self.store.references.root.glob("*.wav"))), 1)

    def test_filesystems_without_hardlinks_keep_usable_products(self):
        with patch("llm_chat.voice_library.os.link", side_effect=OSError(errno.EXDEV, "cross-device")):
            identifier, library, item = self.publish("声", wav())
        self.store.references.prune()
        self.assertEqual(library.audio_path(item["id"]).read_bytes(), wav())
        self.store.delete(identifier)

    def test_legacy_zip_without_settings_is_still_downloadable(self):
        identifier, _, _ = self.publish("旧作品", wav())
        directory = self.store.directory(identifier)
        existing = directory / "product.zip"
        existing.write_bytes(b"legacy package")
        (directory / "settings.json").unlink()
        temporary = self.store.archive(identifier)
        try:
            self.assertEqual(temporary.read_bytes(), b"legacy package")
        finally:
            temporary.unlink()
        self.assertEqual(existing.read_bytes(), b"legacy package")


class PersistentReferenceClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_restart_looks_up_prompt_without_reupload_and_text_change_prepares(self):
        with tempfile.TemporaryDirectory() as directory:
            library = VoiceLibrary(Path(directory))
            item = library.create(wav(), name="声", kind="voice", provenance={"text": "  参照文章  "})
            calls, saved = [], {}

            def upstream(request):
                calls.append(request)
                if request.url.path.endswith("/lookup"):
                    text = json.loads(request.content)["transcript"]
                    return httpx.Response(200, json={"prompt_id": saved[text]}) if text in saved else httpx.Response(404)
                self.assertTrue(request.url.path.endswith("/prepare"))
                text = request.content.split(b'name="transcript"\r\n\r\n')[1].split(b"\r\n")[0].decode()
                saved[text] = str(len(saved) + 1) * 64
                return httpx.Response(200, json={"prompt_id": saved[text]})

            def client():
                adapter = QwenClient("http://tts")
                adapter.http = lambda: httpx.AsyncClient(transport=httpx.MockTransport(upstream))
                return adapter

            prepared = await client().prepare(library, item["id"])
            with patch.object(library, "audio_path", side_effect=AssertionError("WAV read again")):
                self.assertEqual(await client().prepare(library, item["id"]), prepared)
            item["provenance"]["text"] = "別の参照文章"
            library.save(item)
            self.assertNotEqual(await client().prepare(library, item["id"]), prepared)
            self.assertEqual(len([r for r in calls if r.url.path.endswith("/prepare")]), 2)

    async def test_official_irodori_fallback_does_not_replace_or_duplicate_wavs(self):
        with tempfile.TemporaryDirectory() as directory:
            library = VoiceLibrary(Path(directory))
            item = library.create(wav(), name="声", kind="voice", provenance={})
            registered, calls = {}, []

            def upstream(request):
                calls.append(request)
                if request.url.path.endswith("/prepare"):
                    return httpx.Response(404)
                self.assertEqual(request.url.path, "/v1/audio/voices")
                if request.method == "POST":
                    identifier = request.content.split(b'name="voice_id"\r\n\r\n')[1].split(b"\r\n")[0].decode()
                    if identifier in registered:
                        return httpx.Response(409)
                    registered[identifier] = {"id": identifier, "ref_wav": f"/voices/{identifier}.wav"}
                    return httpx.Response(201)
                return httpx.Response(200, json={"data": list(registered.values())})

            results = []
            for _ in range(2):
                client = IrodoriClient("http://tts")
                client.http = lambda: httpx.AsyncClient(transport=httpx.MockTransport(upstream))
                results.append(await client.prepare(library, item["id"]))
            self.assertEqual(results[0], results[1])
            self.assertEqual(len(registered), 1)
            self.assertIn("ref_wavs", results[0])
            self.assertFalse(any(r.method == "PUT" for r in calls))

    async def test_temporary_zip_is_removed_when_download_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download.zip"
            path.write_bytes(b"archive")

            async def fail(message):
                raise OSError("client disconnected")

            with self.assertRaises(OSError):
                await TemporaryArchiveResponse(path)({"type": "http", "method": "GET", "headers": []}, None, fail)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
