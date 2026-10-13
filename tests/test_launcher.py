"""Exercise the root launcher with harmless service stubs and no GPU or Colab."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "llm"
STUB = '''#!/usr/bin/env bash
printf 'cwd=%s\\nscript=%s\\n' "$PWD" "$0"
printf 'arg=%s\\n' "$@"
printf 'host=%s\\nport=%s\\nctx=%s\\n' "${HOST-}" "${PORT-}" "${CTX_SIZE-}"
exit "${STUB_EXIT_CODE:-0}"
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "outside"
        self.outside.mkdir()
        shutil.copy2(SOURCE, self.root / "llm")
        (self.root / "llm").chmod(0o755)
        for name in (
            "aivisspeech/serve.sh", "voicevox/serve.sh", "irodori-tts/serve.sh",
            "qwen-tts/serve.sh", "qwen3.5-9b/serve.sh",
            "qwen3.5-9b/serve-codex.sh", "qwen3.8-flash-next/serve-codex.sh",
            "ternary-bonsai-2-27b-abliterated/serve-codex.sh",
        ):
            path = self.root / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(STUB)
        (self.root / "qwen3.5-9b/model.toml").write_text('id = "qwen3.5-9b"\n')
        (self.root / "llm-chat").mkdir()
        binary = self.root / "bin"
        binary.mkdir()
        for name in ("uv", "python3"):
            (binary / name).write_text(STUB)
            (binary / name).chmod(0o755)
        self.binary = binary
        self.environment = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"]}

    def launch(self, *args):
        return subprocess.run((self.root / "llm", *args), cwd=self.outside,
                              env=self.environment, capture_output=True, text=True, check=False)

    def test_every_service_reaches_its_backend_and_preserves_arguments(self):
        routes = (
            (("studio",), self.binary / "uv", ("run", "voice-synthesize")),
            (("dashboard",), self.binary / "python3", ("-B", str(self.root / "resource-dashboard/server.py"))),
            (("model", "qwen3.5-9b"), self.root / "qwen3.5-9b/serve.sh", ()),
            (("codex", "flash-next"), self.root / "qwen3.8-flash-next/serve-codex.sh", ()),
            (("codex", "qwen"), self.root / "qwen3.8-flash-next/serve-codex.sh", ()),
            (("codex", "qwen-9b"), self.root / "qwen3.5-9b/serve-codex.sh", ()),
            (("codex", "bonsai"), self.root / "ternary-bonsai-2-27b-abliterated/serve-codex.sh", ()),
            (("colab", "qwen", "codex"), self.binary / "python3", (str(self.root / "codex/colab-qwen/session.py"),)),
            (("colab", "glm", "start"), self.binary / "python3", (str(self.root / "codex/colab-glm/session.py"),)),
            (("voice", "aivisspeech"), self.root / "aivisspeech/serve.sh", ()),
            (("voice", "voicevox"), self.root / "voicevox/serve.sh", ()),
            (("voice", "irodori"), self.root / "irodori-tts/serve.sh", ()),
            (("voice", "qwen"), self.root / "qwen-tts/serve.sh", ()),
        )
        for command, script, prefix in routes:
            with self.subTest(command=command):
                result = self.launch(*command, "two words", "--port", "1234")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"script={script}\n", result.stdout)
                for item in prefix:
                    self.assertIn(f"arg={item}\n", result.stdout)
                self.assertIn("arg=two words\narg=--port\narg=1234\n", result.stdout)

    def test_studio_and_dashboard_keep_their_required_working_directories(self):
        self.assertIn(f"cwd={self.root / 'llm-chat'}\n", self.launch("studio").stdout)
        self.assertIn(f"cwd={self.root}\n", self.launch("dashboard").stdout)

    def test_chat_runs_in_its_project_directory(self):
        result = self.launch("chat", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"cwd={self.root / 'llm-chat'}\n", result.stdout)
        self.assertIn("arg=run\narg=llm-chat\narg=--help\n", result.stdout)

    def test_model_environment_is_passed_to_the_existing_script(self):
        self.environment.update(HOST="127.0.0.1", PORT="5071", CTX_SIZE="131072")
        result = self.launch("model", "qwen3.5-9b")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("host=127.0.0.1\nport=5071\nctx=131072\n", result.stdout)

    def test_model_without_id_lists_configured_local_models(self):
        second = self.root / "gemma4-26b-a4b"
        second.mkdir()
        (second / "model.toml").write_text('id = "gemma4-26b-a4b"\n')
        (second / "serve.sh").write_text(STUB)
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        (incomplete / "model.toml").write_text('id = "incomplete"\n')

        result = self.launch("model")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Configured local models:\n", result.stdout)
        self.assertIn("  gemma4-26b-a4b\n", result.stdout)
        self.assertIn("  qwen3.5-9b\n", result.stdout)
        self.assertNotIn("  incomplete\n", result.stdout)
        self.assertNotIn("script=", result.stdout)

    def test_model_list_handles_empty_repository(self):
        (self.root / "qwen3.5-9b/model.toml").unlink()
        result = self.launch("model")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("  (none)\n", result.stdout)

    def test_backend_failure_exit_code_is_preserved(self):
        self.environment["STUB_EXIT_CODE"] = "37"
        result = self.launch("model", "qwen3.5-9b")
        self.assertEqual(result.returncode, 37)

    def test_invalid_or_unsafe_targets_never_dispatch(self):
        for command in (("model", "../other"), ("model", "unknown"),
                        ("codex",), ("codex", "other"),
                        ("colab", "other"), ("voice", "other"), ("other",)):
            with self.subTest(command=command):
                result = self.launch(*command)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Usage:", result.stderr)
                self.assertNotIn("script=", result.stdout)

    def test_help_does_not_start_a_service(self):
        result = self.launch("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)
        self.assertNotIn("script=", result.stdout)


if __name__ == "__main__":
    unittest.main()
