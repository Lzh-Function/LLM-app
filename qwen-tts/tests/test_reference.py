"""Exercise reference preparation with a mocked model and real WAV decoding."""

import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch
from fastapi import HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import service


def wav(audio, rate=8000, subtype="PCM_16"):
    output = io.BytesIO()
    sf.write(output, audio, rate, format="WAV", subtype=subtype)
    return output.getvalue()


class ReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root_patch = patch.object(service, "ROOT", Path(self.temp.name))
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.engine = Mock()
        self.engine.create_voice_clone_prompt.return_value = [
            service.VoiceClonePromptItem(
                ref_code=None,
                ref_spk_embedding=torch.zeros(4),
                x_vector_only_mode=True,
                icl_mode=False,
                ref_text=None,
            )
        ]
        load_patch = patch.object(service, "load", return_value=self.engine)
        self.load = load_patch.start()
        self.addCleanup(load_patch.stop)

    def prepare(self, data, transcript=""):
        return service.prepare(UploadFile(filename="reference.wav", file=io.BytesIO(data)), transcript)

    def test_duration_boundary_preserves_short_transcript_and_crops_long_audio(self):
        for seconds in (1, 120, 121):
            with self.subTest(seconds=seconds):
                source = np.linspace(-0.8, 0.8, seconds * 8000, dtype=np.float32)
                data = wav(source)
                result = self.prepare(data, "  参照の全文  ")
                options = self.engine.create_voice_clone_prompt.call_args.kwargs
                audio, rate = options["ref_audio"]
                expected, _ = sf.read(io.BytesIO(data), dtype="float32")
                np.testing.assert_array_equal(audio, expected[:120 * rate])
                trimmed = seconds > 120
                self.assertEqual(result["source_reference_seconds"], seconds)
                self.assertEqual(result["reference_seconds"], min(seconds, 120))
                self.assertEqual(result["reference_trimmed"], trimmed)
                self.assertEqual(result["transcript_ignored_due_to_trim"], trimmed)
                self.assertEqual(options["ref_text"], None if trimmed else "参照の全文")
                self.assertEqual(options["x_vector_only_mode"], trimmed)
                self.assertEqual(result["clone_mode"], "speaker_embedding_only" if trimmed else "icl")
                self.assertEqual(result["reference_sha256"], hashlib.sha256(data).hexdigest())

    def test_long_reference_without_transcript_uses_embedding_and_reuses_cache(self):
        data = wav(np.full(121 * 8000, 0.2, dtype=np.float32))
        first = self.prepare(data)
        second = self.prepare(data, "全文は切り取り音声に対応しない")
        self.assertEqual(first["prompt_id"], second["prompt_id"])
        self.assertEqual(first["clone_mode"], "speaker_embedding_only")
        self.assertFalse(first["transcript_ignored_due_to_trim"])
        self.assertTrue(second["transcript_ignored_due_to_trim"])
        self.engine.create_voice_clone_prompt.assert_called_once()
        path = service.ROOT / "voices" / ".prompts" / f'{first["prompt_id"]}.pt'
        records = torch.load(path, weights_only=True)
        self.assertEqual(len(records), 1)

    def test_source_over_old_size_limit_is_cropped_before_model_and_downmixed(self):
        data = wav(np.full((121 * 48000, 2), 0.25, dtype=np.float32), 48000, "FLOAT")
        self.assertGreater(len(data), 32 * 1024 * 1024)
        result = self.prepare(data)
        audio, rate = self.engine.create_voice_clone_prompt.call_args.kwargs["ref_audio"]
        self.assertEqual(rate, 48000)
        self.assertEqual(audio.shape, (120 * 48000,))
        np.testing.assert_array_equal(audio, np.full(audio.shape, 0.25, dtype=np.float32))
        self.assertTrue(result["reference_trimmed"])

    def test_invalid_source_is_rejected_even_when_nan_is_past_cutoff(self):
        source = np.zeros(121 * 8000, dtype=np.float32)
        source[-1] = np.nan
        for data in (b"invalid WAV", wav(np.zeros(0)), wav(source, subtype="FLOAT")):
            with self.subTest(length=len(data)), self.assertRaises(HTTPException) as raised:
                self.prepare(data)
            self.assertEqual(raised.exception.status_code, 422)
        self.load.assert_not_called()

    def test_upload_read_is_bounded_before_decoding(self):
        class OversizeData:
            def __len__(self):
                return service.MAX_REFERENCE_UPLOAD_BYTES + 1

        stream = Mock()
        # Oversize data is rejected before any decoder or model sees it.
        stream.read.return_value = OversizeData()
        with self.assertRaises(HTTPException) as raised:
            service.prepare(UploadFile(filename="large.wav", file=stream), "")
        self.assertEqual(raised.exception.status_code, 422)
        stream.read.assert_called_once_with(service.MAX_REFERENCE_UPLOAD_BYTES + 1)
        self.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
