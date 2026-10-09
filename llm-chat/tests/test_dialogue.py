"""Multi-speaker production: actual adapters, PCM samples and portable artifacts."""

import hashlib
import io
import json
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

import httpx
import test_voice_gpu as fixtures
from fastapi import HTTPException
from llm_chat import dialogue, voice_synthesize


def pcm(seconds=0.25, sample=123, *, width=2, channels=1, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(
            sample.to_bytes(width, "little", signed=width > 1)
            * int(seconds * rate)
            * channels
        )
    return buffer.getvalue()


def samples(data):
    with wave.open(io.BytesIO(data)) as audio:
        return audio.readframes(audio.getnframes())


class ScriptTest(unittest.TestCase):
    def test_tags_are_removed_and_repeated_speakers_preserve_order(self):
        result = dialogue.parse_script(
            "\n[speaker A] 一人目。\n[speaker B] 二人目。[speaker A] 再登場。",
            ["speaker A", "speaker B"],
        )
        self.assertEqual([turn["speaker_index"] for turn in result], [0, 1, 0])
        self.assertEqual(
            [turn["text"] for turn in result], ["一人目。", "二人目。", "再登場。"]
        )
        self.assertEqual(
            dialogue.parse_script("[演出]普通の単一話者原稿", [None])[0]["text"],
            "[演出]普通の単一話者原稿",
        )
        self.assertEqual(
            dialogue.parse_script("タグ不要", ["名前"])[0]["text"], "タグ不要"
        )

    def test_invalid_tags_labels_and_empty_turns_are_rejected(self):
        for text, labels in (
            ("[A]文章", ["A", None]),
            ("[A]文章", ["A", "A"]),
            ("文章", ["A", "B"]),
            ("先頭文[A]文章", ["A", "B"]),
            ("[A][B]文章", ["A", "B"]),
            ("[C]文章", ["A", "B"]),
            ("[A]文章[B", ["A", "B"]),
            ("[A]文章]", ["A", "B"]),
        ):
            with (
                self.subTest(text=text, labels=labels),
                self.assertRaises(HTTPException),
            ):
                dialogue.parse_script(text, labels)

    def test_long_turn_chunks_keep_every_character_and_stay_bounded(self):
        for text in (
            "あ" * 20000,
            "長い文章。\n" * 100,
            "word " * 1000,
            "あ" * 160 + " " + "い" * 200,
        ):
            parts = dialogue.speech_chunks(text)
            self.assertEqual("".join(parts), text)
            self.assertTrue(all(0 < len(part) <= 160 for part in parts))

    def test_pcm_is_preserved_with_exact_silence_and_no_padding_at_ends(self):
        with tempfile.TemporaryDirectory() as temp:
            for width, channels, sample in (
                (1, 1, 200),
                (2, 2, 1000),
                (3, 1, 123456),
                (4, 2, 12345678),
            ):
                with self.subTest(width=width, channels=channels):
                    one, two = (
                        pcm(sample=sample, width=width, channels=channels),
                        pcm(sample=sample - 1, width=width, channels=channels),
                    )
                    paths = [Path(temp) / "one.wav", Path(temp) / "two.wav"]
                    paths[0].write_bytes(one)
                    paths[1].write_bytes(two)
                    result, timeline = dialogue.assemble_wavs(paths)
                    gap = (b"\x80" if width == 1 else b"\0") * (
                        16000 * width * channels
                    )
                    self.assertEqual(samples(result), samples(one) + gap + samples(two))
                    self.assertEqual(timeline[0]["start_sample"], 0)
                    self.assertEqual(timeline[1]["start_seconds"], 1.25)
                    self.assertEqual(timeline[1]["end_seconds"], 1.5)

    def test_incompatible_or_excessive_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / "one.wav", Path(temp) / "two.wav"]
            paths[0].write_bytes(pcm())
            paths[1].write_bytes(pcm(rate=8000))
            with self.assertRaises(HTTPException) as caught:
                dialogue.assemble_wavs(paths)
            self.assertEqual(caught.exception.status_code, 503)
            paths[1].write_bytes(pcm())
            with (
                patch.object(dialogue, "MAX_OUTPUT_SECONDS", 1),
                self.assertRaises(HTTPException) as caught,
            ):
                dialogue.assemble_wavs(paths)
            self.assertEqual(caught.exception.status_code, 422)


class DialogueAPITest(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.StandaloneTest.asyncSetUp
    asyncTearDown = fixtures.StandaloneTest.asyncTearDown

    def configure(self, engine, fail_at=None):
        self.runtime.engine = engine
        self.calls, self.spoken, self.prepared, self.upload_ids = [], [], [], {}
        self.ref_a, self.ref_b = pcm(sample=111), pcm(seconds=121, sample=222)
        self.ref_c = pcm(sample=333)
        self.output_a, self.output_b = pcm(sample=1001), pcm(sample=-2002)
        reference_data = [self.ref_a, pcm(seconds=120, sample=222), self.ref_c]
        signatures = [hashlib.sha256(data).hexdigest() for data in reference_data]

        def upstream(request):
            self.calls.append(request)
            path = request.url.path
            if path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "model": {},
                        "runtime": {"loaded": True, "model_key": "base"},
                    },
                )
            if path == "/v1/audio/voices" and request.method == "POST":
                index = next(
                    i
                    for i, data in enumerate(reference_data)
                    if data in request.content
                )
                identifier = (
                    request.content.split(b'name="voice_id"\r\n\r\n')[1]
                    .split(b"\r\n")[0]
                    .decode()
                )
                self.upload_ids[identifier] = index
                return httpx.Response(201, json={"id": identifier})
            if path.endswith("/prepare"):
                if engine == "qwen":
                    index = next(
                        i
                        for i, data in enumerate(reference_data)
                        if data in request.content
                    )
                    self.assertEqual(
                        "一人目の参照文章".encode() in request.content, index == 0
                    )
                    self.assertNotIn("二人目のカット前の全文".encode(), request.content)
                    options = {"prompt_id": signatures[index]}
                else:
                    index = self.upload_ids[path.split("/")[-2]]
                    options = {"ref_latents": [f"/cache/{signatures[index]}.pt"]}
                self.prepared.append(index)
                return httpx.Response(200, json=options)
            if path == "/v1/audio/speech":
                body = json.loads(request.content)
                self.spoken.append(body)
                if fail_at == len(self.spoken):
                    return httpx.Response(503, json={"detail": "model failed"})
                options = body["qwen" if engine == "qwen" else "irodori"]
                identity = options.get("prompt_id") or options["ref_latents"][0]
                is_a = signatures[0] in identity
                used_seed = options.get("seed", 42)
                headers = {
                    "X-Irodori-Seed": str(used_seed),
                    "X-Qwen-Metrics": json.dumps({"seed": used_seed, "chunks": 1}),
                }
                return httpx.Response(
                    200,
                    content=self.output_a if is_a else self.output_b,
                    headers=headers,
                )
            raise AssertionError((request.method, path))

        self.runtime.tts.http = lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )

    async def upload(
        self,
        text="[speaker A]こんにちは。[speaker B]返答。[speaker A]ありがとう。",
        specs=None,
        **settings,
    ):
        if specs is None:
            specs = [
                {"label": "speaker A", "reference_text": "一人目の参照文章"},
                {"label": "speaker B", "reference_text": "二人目のカット前の全文"},
            ]
        return await self.client.post(
            "/api/synthesis/upload",
            data={
                "settings": json.dumps(
                    {
                        "engine": self.runtime.engine,
                        "text": text,
                        "name": "対話作品",
                        "seed": 2**32 - 1,
                        "speakers": specs,
                        **settings,
                    }
                )
            },
            files=[
                ("files", ("A.wav", self.ref_a, "audio/wav")),
                ("files", ("B.wav", self.ref_b, "audio/wav")),
            ],
        )

    async def exercise_dialogue(self, engine):
        self.configure(engine)
        response = await self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        first = response.json()
        self.assertEqual(first["schema_version"], 3)
        self.assertEqual(
            [s["label"] for s in first["speakers"]], ["speaker A", "speaker B"]
        )
        self.assertEqual(
            [s["text"] for s in first["segments"]],
            ["こんにちは。", "返答。", "ありがとう。"],
        )
        self.assertEqual(
            [body["input"] for body in self.spoken],
            ["こんにちは。", "返答。", "ありがとう。"],
        )
        key = "qwen" if engine == "qwen" else "irodori"
        self.assertEqual([body[key]["seed"] for body in self.spoken], [2**32 - 1, 0, 1])
        self.assertEqual(self.prepared, [0, 1])
        self.assertEqual(
            self.spoken[0][key].get("prompt_id") or self.spoken[0][key]["ref_latents"],
            self.spoken[2][key].get("prompt_id") or self.spoken[2][key]["ref_latents"],
        )
        self.assertEqual(self.library.list(), [])
        endpoint = f"/api/synthesis/{first['id']}"
        audio = await self.client.get(endpoint + "/audio")
        silence = b"\0" * (16000 * 2)
        self.assertEqual(
            samples(audio.content),
            samples(self.output_a)
            + silence
            + samples(self.output_b)
            + silence
            + samples(self.output_a),
        )
        self.assertEqual(first["metrics"]["audio_seconds"], 2.75)
        self.assertEqual(
            [s["start_seconds"] for s in first["segments"]], [0, 1.25, 2.5]
        )
        self.assertEqual(
            (await self.client.get(endpoint + "/speakers/1/audio")).content,
            pcm(seconds=120, sample=222),
        )
        self.assertEqual(
            (await self.client.get(endpoint + "/segments/1/audio")).content,
            self.output_b,
        )
        self.assertEqual(
            (await self.client.get(endpoint + "/segments/-1/audio")).status_code, 404
        )
        self.assertEqual(
            (await self.client.get(endpoint + "/speakers/2/audio")).status_code, 404
        )
        package = await self.client.get(endpoint + "/archive")
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            saved = json.loads(archive.read("settings.json"))
            self.assertEqual(saved["speakers"], first["speakers"])
            self.assertEqual(saved["gap_seconds"], 1.0)
            self.assertEqual(archive.read("manuscript.txt").decode(), first["text"])
            for speaker in first["speakers"]:
                self.assertIn(
                    f"voice/{speaker['production_voice_id']}/reference.wav",
                    archive.namelist(),
                )
            for segment in first["segments"]:
                self.assertIn(segment["file"], archive.namelist())

        response = await self.client.post(
            "/api/synthesis",
            json={
                "engine": engine,
                "source_product_id": first["id"],
                "text": "[speaker B]別の原稿。[speaker A]続き。",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        second = response.json()
        self.assertEqual(
            self.prepared,
            [0, 1],
            "cached references must survive copying into another product",
        )
        self.assertEqual(
            [s["label"] for s in second["segments"]], ["speaker B", "speaker A"]
        )
        self.assertNotEqual(
            [s["production_voice_id"] for s in first["speakers"]],
            [s["production_voice_id"] for s in second["speakers"]],
        )
        await self.client.delete(endpoint)
        renamed = await self.client.patch(
            f"/api/synthesis/{second['id']}", json={"name": "対話作品の新しい名前"}
        )
        self.assertEqual(renamed.status_code, 200, renamed.text)
        package = await self.client.get(f"/api/synthesis/{second['id']}/archive")
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            self.assertEqual(
                json.loads(archive.read("metadata.json"))["name"],
                "対話作品の新しい名前",
            )
            self.assertEqual(
                len(
                    [
                        path
                        for path in archive.namelist()
                        if path.endswith("reference.wav")
                    ]
                ),
                2,
            )
        third = await self.client.post(
            "/api/synthesis",
            json={
                "engine": engine,
                "source_product_id": second["id"],
                "text": "[speaker A]元作品削除後。[speaker B]同じ話者。",
            },
        )
        self.assertEqual(third.status_code, 200, third.text)
        self.assertEqual(self.prepared, [0, 1])

    async def test_irodori_separate_inferences_latent_reuse_silence_zip_and_regeneration(
        self,
    ):
        await self.exercise_dialogue("irodori")

    async def test_qwen_separate_inferences_prompt_reuse_silence_zip_and_regeneration(
        self,
    ):
        await self.exercise_dialogue("qwen")

    async def test_three_uploaded_speakers_with_japanese_labels(self):
        for engine in ("irodori", "qwen"):
            with self.subTest(engine=engine):
                self.configure(engine)
                response = await self.client.post(
                    "/api/synthesis/upload",
                    data={
                        "settings": json.dumps(
                            {
                                "engine": engine,
                                "text": "[ナレーター]物語。[案内役]案内。[客]返答。[ナレーター]終幕。",
                                "speakers": [
                                    {
                                        "label": "案内役",
                                        "reference_text": "一人目の参照文章",
                                    },
                                    {"label": "客"},
                                    {"label": "ナレーター"},
                                ],
                            }
                        )
                    },
                    files=[
                        ("files", (f"{index}.wav", data, "audio/wav"))
                        for index, data in enumerate(
                            (self.ref_a, self.ref_b, self.ref_c)
                        )
                    ],
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(len(result["speakers"]), 3)
                self.assertEqual(
                    [s["label"] for s in result["segments"]],
                    ["ナレーター", "案内役", "客", "ナレーター"],
                )
                self.assertEqual(self.prepared, [2, 0, 1])
                key = "qwen" if engine == "qwen" else "irodori"
                identities = [
                    body[key].get("prompt_id") or body[key]["ref_latents"][0]
                    for body in self.spoken
                ]
                self.assertEqual(len(set(identities)), 3)
                self.assertEqual(identities[0], identities[3])
                self.assertEqual(result["metrics"]["audio_seconds"], 4)

    async def test_twenty_thousand_character_dialogue_keeps_the_entire_manuscript(self):
        self.configure("irodori")
        manuscript = "[speaker A]" + "あ" * 19975 + "[speaker B]返答。"
        self.assertEqual(len(manuscript), 20000)
        response = await self.upload(manuscript)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["text"], manuscript)
        self.assertEqual(len(self.spoken), 126)
        self.assertEqual(
            "".join(body["input"] for body in self.spoken), "あ" * 19975 + "返答。"
        )
        self.assertEqual(result["metrics"]["audio_seconds"], 156.5)
        self.assertEqual(self.prepared, [0, 1])

    async def test_long_turns_and_repeated_adjacent_labels_are_independent(self):
        self.configure("irodori")
        response = await self.upload(
            "[speaker A]" + "あ" * 321 + "[speaker A]同じ話者の次。[speaker B]返答。"
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(self.spoken), 5)
        self.assertEqual("".join(body["input"] for body in self.spoken[:3]), "あ" * 321)
        self.assertTrue(
            all(
                len(body["input"]) <= 160 and not body["irodori"]["chunking_enabled"]
                for body in self.spoken
            )
        )
        self.assertEqual(result["metrics"]["audio_seconds"], 5 * 0.25 + 4)

    async def test_invalid_dialogue_is_rejected_before_runtime_or_upstream_calls(self):
        self.configure("irodori")
        self.runtime.enabled = False
        for specs, text in (
            ([{}, {}], "[A]原稿"),
            ([{"label": "A"}, {}], "[A]原稿"),
            ([{"label": "A"}, {"label": "A"}], "[A]原稿"),
            ([{"label": "A"}], "[A]原稿"),
            ([{"label": "A"}, {"label": "B"}], "[C]原稿"),
            ([{"label": "A"}, {"label": "B"}], "話者なし"),
            ([{"label": "[A]"}, {"label": "B"}], "[A]原稿"),
        ):
            with self.subTest(specs=specs, text=text):
                response = await self.upload(text, specs)
                self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse(self.calls)
        self.assertEqual((await self.client.get("/api/synthesis")).json()["items"], [])

    async def test_failed_later_inference_does_not_publish_partial_product(self):
        self.configure("irodori", fail_at=2)
        response = await self.upload()
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(len(self.spoken), 2)
        self.assertEqual((await self.client.get("/api/synthesis")).json()["items"], [])
        self.assertFalse(list((self.root / "productions").glob("*")))

    async def test_single_plural_file_needs_no_label_and_combined_upload_size_is_bounded(
        self,
    ):
        self.configure("irodori")
        response = await self.client.post(
            "/api/synthesis/upload",
            data={"settings": json.dumps({"text": "単一話者"})},
            files=[("files", ("A.wav", self.ref_a, "audio/wav"))],
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("speakers", response.json())
        with patch.object(
            voice_synthesize, "MAX_REFERENCE_UPLOAD_BYTES", len(self.ref_a) + 100
        ):
            response = await self.upload()
        self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
