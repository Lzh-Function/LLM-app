"""Check that installing Qwen never makes it the default or loses unrelated settings."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import tomllib

SETUP = Path(__file__).with_name("setup.py")
MODEL = "qwen3.8-flash-next-sc117-abliterated-iq3_xxs"


class SetupTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def install(self):
        subprocess.run(
            ["uv", "run", str(SETUP), "--config-home", str(self.root)],
            check=True,
            capture_output=True,
            text=True,
        )
        return tomllib.loads((self.root / "config.toml").read_text())

    def test_existing_gpt_and_other_profiles_are_preserved(self):
        (self.root / "config.toml").write_text(
            'model = "gpt-6.1-sol"\nmodel_reasoning_effort = "medium"\n'
            '[features]\napps = true\n[projects."/tmp/project"]\ntrust_level = "trusted"\n'
            '[sandbox_workspace_write]\nwritable_roots = ["/tmp/project"]\n'
        )
        other = self.root / "review.config.toml"
        other.write_text('model = "gpt-6.1-sol"\n')
        (self.root / "qwen.config.toml").write_text(
            '[projects."/tmp/qwen-project"]\ntrust_level = "trusted"\n'
            '[mcp_servers.other]\ncommand = "other-tool"\n'
        )
        (self.root / "bonsai.config.toml").write_text(
            '[projects."/tmp/bonsai-project"]\ntrust_level = "trusted"\n'
            '[mcp_servers.other]\ncommand = "other-tool"\n'
        )
        result = self.install()
        self.assertEqual(result["model"], "gpt-6.1-sol")
        self.assertEqual(result["model_reasoning_effort"], "medium")
        self.assertTrue(result["features"]["apps"])
        self.assertEqual(result["projects"]["/tmp/project"]["trust_level"], "trusted")
        self.assertEqual(
            result["sandbox_workspace_write"]["writable_roots"], ["/tmp/project"]
        )
        self.assertTrue(result["sandbox_workspace_write"]["network_access"])
        self.assertTrue(other.exists())
        before = (self.root / "config.toml").read_text()
        self.install()
        self.assertEqual((self.root / "config.toml").read_text(), before)
        profile = tomllib.loads((self.root / "qwen.config.toml").read_text())
        renamed = tomllib.loads((self.root / "flash-next.config.toml").read_text())
        self.assertEqual(renamed["model"], MODEL)
        self.assertEqual(renamed["projects"], profile["projects"])
        self.assertEqual(renamed["mcp_servers"], profile["mcp_servers"])
        self.assertEqual(profile["model"], MODEL)
        self.assertEqual(profile["model_context_window"], 131072)
        self.assertEqual(profile["model_auto_compact_token_limit"], 98304)
        self.assertFalse(profile["features"]["apps"])
        self.assertEqual(profile["web_search"], "live")
        self.assertEqual(profile["projects"]["/tmp/qwen-project"]["trust_level"], "trusted")
        self.assertEqual(profile["mcp_servers"]["other"]["command"], "other-tool")
        search = profile["mcp_servers"]["local_web_search"]
        self.assertTrue(Path(search["command"]).is_file())
        self.assertTrue(Path(search["args"][0]).is_file())
        self.assertTrue(search["required"])
        self.assertEqual(search["enabled_tools"], ["web_search", "fetch_page"])
        self.assertNotIn("mcp_servers", result)
        bundled = json.loads(subprocess.run(
            ["codex", "debug", "models", "--bundled"],
            check=True, capture_output=True, text=True,
        ).stdout)
        standard = next(m for m in bundled["models"] if m["slug"] == "gpt-6.1-sol")
        local = json.loads((self.root / "flash-next-models.json").read_text())["models"][0]
        for key in ("base_instructions", "model_messages"):
            self.assertEqual(local[key], standard[key])
        self.assertNotIn("model_catalog_json", result)
        bonsai = tomllib.loads((self.root / "bonsai.config.toml").read_text())
        self.assertEqual(bonsai["model"], "ternary-bonsai-2-27b-abliterated")
        self.assertEqual(bonsai["model_context_window"], 98304)
        self.assertEqual(bonsai["model_auto_compact_token_limit"], 73728)
        self.assertEqual(bonsai["model_reasoning_effort"], "medium")
        self.assertEqual(bonsai["model_providers"]["prism"]["wire_api"], "responses")
        self.assertEqual(bonsai["mcp_servers"]["local_web_search"], search)
        self.assertEqual(bonsai["mcp_servers"]["other"]["command"], "other-tool")
        self.assertEqual(bonsai["projects"]["/tmp/bonsai-project"]["trust_level"], "trusted")
        model = json.loads((self.root / "bonsai-models.json").read_text())["models"][0]
        runtime = json.loads(SETUP.with_name("bonsai-runtime.json").read_text())
        self.assertEqual(model["context_window"], runtime["context_window"])
        for key in ("base_instructions", "model_messages"):
            self.assertEqual(model[key], standard[key])
        qwen9 = tomllib.loads((self.root / "qwen-9b.config.toml").read_text())
        self.assertEqual(qwen9["model"], "qwen3.5-9b")
        self.assertEqual(qwen9["model_providers"]["llamacpp"]["base_url"], "http://127.0.0.1:8083/v1")
        self.assertEqual(qwen9["mcp_servers"]["local_web_search"], search)
        model9 = json.loads((self.root / "qwen-9b-models.json").read_text())["models"][0]
        runtime9 = json.loads(SETUP.with_name("qwen-9b-runtime.json").read_text())
        self.assertEqual(qwen9["model_context_window"], runtime9["context_window"])
        self.assertEqual(model9["context_window"], runtime9["context_window"])
        for key in ("base_instructions", "model_messages"):
            self.assertEqual(model9[key], standard[key])

    def test_named_flash_profile_keeps_its_own_settings(self):
        (self.root / "flash-next.config.toml").write_text(
            '[projects."/tmp/flash"]\ntrust_level = "trusted"\n'
        )
        (self.root / "qwen.config.toml").write_text(
            '[projects."/tmp/legacy"]\ntrust_level = "trusted"\n'
        )
        self.install()
        renamed = tomllib.loads((self.root / "flash-next.config.toml").read_text())
        self.assertIn("/tmp/flash", renamed["projects"])
        self.assertNotIn("/tmp/legacy", renamed["projects"])

    def test_migrate_global_qwen_restores_gpt_and_keeps_new_settings(self):
        backup = self.root / "config-backups" / "flash-next-20261007-000000"
        backup.mkdir(parents=True)
        (backup / "config.toml").write_text(
            'model = "gpt-6.1-sol"\nmodel_reasoning_effort = "high"\n'
            "[features]\nmulti_agent = true\n"
        )
        (self.root / "config.toml").write_text(
            f'model = "{MODEL}"\nmodel_provider = "strata"\n'
            "model_context_window = 131072\nmodel_auto_compact_token_limit = 98304\n"
            'web_search = "disabled"\nshow_raw_agent_reasoning = true\n'
            "[features]\napps = false\nmulti_agent = false\nshell_snapshot = true\n"
            '[mcp_servers.example]\ncommand = "example"\n'
            '[model_providers.strata]\nbase_url = "http://127.0.0.1:8080/v1"\n'
        )
        legacy = self.root / "qwen-local.config.toml"
        legacy.write_text('model = "old-qwen"\n')
        (self.root / "flash-next-models.json").write_text('{"previous": true}\n')
        result = self.install()
        self.assertEqual(result["model"], "gpt-6.1-sol")
        self.assertEqual(result["model_reasoning_effort"], "high")
        for key in (
            "model_provider",
            "model_context_window",
            "model_auto_compact_token_limit",
            "web_search",
            "show_raw_agent_reasoning",
            "model_providers",
        ):
            self.assertNotIn(key, result)
        self.assertNotIn("apps", result["features"])
        self.assertTrue(result["features"]["multi_agent"])
        self.assertTrue(result["features"]["shell_snapshot"])
        self.assertEqual(result["mcp_servers"]["example"]["command"], "example")
        self.assertFalse(legacy.exists())
        newest = max((self.root / "config-backups").iterdir())
        self.assertTrue((newest / legacy.name).exists())
        self.assertEqual(
            json.loads((newest / "flash-next-models.json").read_text()),
            {"previous": True},
        )

    def test_fresh_configuration_leaves_builtin_default(self):
        result = self.install()
        self.assertNotIn("model", result)
        self.assertNotIn("model_provider", result)
        self.assertNotIn("features", result)
        self.assertTrue(result["sandbox_workspace_write"]["network_access"])


if __name__ == "__main__":
    unittest.main()
