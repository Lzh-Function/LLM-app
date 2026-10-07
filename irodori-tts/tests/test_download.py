"""Conversion provenance and setup after removing unused checkpoints."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "irodori_download", Path(__file__).resolve().parents[1] / "download.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "models"
        self.large = self.output / "large"
        self.checkpoint = self.large / "bf16" / "model.safetensors"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"converted checkpoint")
        identity = {
            "source_sha256": "original-checkpoint-sha",
            "precision": "bf16",
            "converter_version": 1,
        }
        self.checkpoint.with_suffix(".json").write_text(json.dumps(identity))
        self.previous = {
            "repo": download.MODELS["large"],
            "revision": "pinned",
            "weights": {"model.safetensors": identity["source_sha256"]},
            "derived": {
                "path": "bf16/model.safetensors",
                "sha256": download.sha256(self.checkpoint),
                **identity,
            },
        }
        (self.output.parent / "install.json").write_text(
            json.dumps({"models": {"large": self.previous}})
        )

    def test_reuse_requires_matching_revision_marker_and_checkpoint_hash(self):
        def reuse(revision="pinned"):
            return download.installed_bf16(
                self.large, self.previous, download.MODELS["large"], revision
            )

        self.assertEqual(reuse(), self.previous["derived"])
        self.assertIsNone(reuse("new-revision"))
        self.checkpoint.write_bytes(b"corrupt")
        self.assertIsNone(reuse())
        self.checkpoint.with_suffix(".json").write_text("invalid json")
        self.assertIsNone(reuse())

    def run_setup(self, extra=()):
        def snapshot(repo, *, revision, local_dir, ignore_patterns):
            self.assertEqual(revision, "pinned")
            if not ignore_patterns:
                (local_dir / "model.safetensors").write_bytes(b"original checkpoint")
            return str(local_dir)

        with (
            patch.object(download, "MODELS", {"large": download.MODELS["large"]}),
            patch.object(download, "snapshot_download", side_effect=snapshot) as fetch,
            patch.object(
                download,
                "bf16_checkpoint",
                return_value=self.previous["derived"],
            ) as convert,
            patch.object(
                download.importlib.metadata,
                "distribution",
                return_value=SimpleNamespace(version="test", read_text=lambda _: None),
            ),
            patch.object(
                sys, "argv", ["download", "--output", str(self.output), *extra]
            ),
        ):
            download.main()
        manifest = json.loads((self.output.parent / "install.json").read_text())
        return fetch.call_args.kwargs, convert, manifest["models"]["large"]

    def test_setup_after_cleanup_does_not_fetch_fp32_or_comparison_small(self):
        self.assertNotIn("design", download.MODELS)
        args, convert, installed = self.run_setup()
        self.assertEqual(args["ignore_patterns"], ["model.safetensors"])
        convert.assert_not_called()
        self.assertFalse((self.large / "model.safetensors").exists())
        self.assertTrue(self.checkpoint.exists())
        self.assertEqual(installed["weights"], {})
        self.assertEqual(
            installed["source_weights"]["model.safetensors"],
            self.previous["derived"]["source_sha256"],
        )
        # A subsequent setup must also work with the new, source-free manifest.
        args, convert, _ = self.run_setup()
        self.assertEqual(args["ignore_patterns"], ["model.safetensors"])
        convert.assert_not_called()

    def test_missing_bf16_fetches_and_converts_then_removes_source(self):
        self.checkpoint.unlink()
        args, convert, _ = self.run_setup()
        self.assertIsNone(args["ignore_patterns"])
        convert.assert_called_once()
        self.assertFalse((self.large / "model.safetensors").exists())

    def test_explicit_fp32_option_retains_conversion_source(self):
        args, convert, installed = self.run_setup(["--keep-large-fp32"])
        self.assertIsNone(args["ignore_patterns"])
        convert.assert_not_called()
        self.assertTrue((self.large / "model.safetensors").exists())
        self.assertIn("model.safetensors", installed["weights"])

    def test_corrupt_bf16_forces_reconversion_despite_unchanged_marker(self):
        self.checkpoint.write_bytes(b"corrupt")
        args, convert, _ = self.run_setup()
        self.assertIsNone(args["ignore_patterns"])
        self.assertTrue(convert.call_args.kwargs["force"])


if __name__ == "__main__":
    unittest.main()
