"""Context limits, restart behavior, launch propagation and API compatibility."""

import asyncio
import importlib.util
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from llm_chat import server


class ContextTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        with patch.object(server, "LLM_ROOT", self.root):
            self.manager = server.ModelManager()
        self.manager.models = {
            "wide": server.ModelInfo(
                "wide",
                "Wide",
                "",
                1,
                self.root,
                default_context_length=131072,
                max_context_length=262144,
            ),
            "small": server.ModelInfo(
                "small",
                "Small",
                "",
                2,
                self.root,
                default_context_length=32768,
                max_context_length=131072,
            ),
        }
        script = self.root / "serve.sh"
        script.write_text(
            "exec "
            + shlex.quote(sys.executable)
            + " -c 'import json,os,time; print(json.dumps({k:os.environ[k] for k in "
            + '["CTX_SIZE","LLAMA_ARG_N_PARALLEL"]}),flush=True); time.sleep(60)'
            + "'\n"
        )
        self.children = []

        async def healthy(proc):
            self.children.append(proc)
            async with asyncio.timeout(3):
                while (self.root / "server.log").read_text().count('"CTX_SIZE"') < len(
                    self.children
                ):
                    await asyncio.sleep(0.01)
            return True

        self.manager._wait_healthy = healthy
        self.manager._effective_context_length = AsyncMock(
            side_effect=lambda value: value
        )

    async def asyncTearDown(self):
        await self.manager.unload()
        self.temp.cleanup()

    async def ready(self):
        await asyncio.wait_for(self.manager._task, timeout=3)
        self.assertEqual(self.manager.status, "ready", self.manager.error)

    async def test_128k_and_256k_same_model_reload_and_environment(self):
        await self.manager.request_load("wide")
        await self.ready()
        self.assertEqual(self.manager.snapshot()["active_context_length"], 131072)
        await self.manager.request_load("wide", 131072)
        self.assertEqual(len(self.children), 1)
        await self.manager.request_load("wide", 262144)
        await self.ready()
        self.assertEqual(len(self.children), 2)
        self.assertIsNotNone(self.children[0].returncode)
        self.assertEqual(self.manager.context_length, 262144)
        async with asyncio.timeout(3):
            while (
                not self.root.joinpath("server.log").read_text().count('"CTX_SIZE"')
                >= 2
            ):
                await asyncio.sleep(0.01)
        text = (self.root / "server.log").read_text()
        self.assertIn('"CTX_SIZE": "131072"', text)
        self.assertIn('"CTX_SIZE": "262144"', text)
        self.assertIn('"LLAMA_ARG_N_PARALLEL": "1"', text)
        await self.manager.unload()
        await self.manager.request_load("wide")
        await self.ready()
        self.assertEqual(self.manager.context_length, 262144)

    async def test_invalid_context_does_not_stop_current_model(self):
        await self.manager.request_load("small")
        await self.ready()
        task = self.manager._task
        for value in (131073, 262144, 0, 511, True, "65536", 65536.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await self.manager.request_load("small", value)
            self.assertIs(self.manager._task, task)
            self.assertIsNone(self.children[0].returncode)

    async def test_load_api_without_body_and_strict_bounds(self):
        lock = SimpleNamespace(_lock=asyncio.Lock())
        with (
            patch.object(server, "manager", self.manager),
            patch.object(server.app.state, "speech_runtime", lock, create=True),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                result = await client.post("/api/models/wide/load")
                self.assertEqual(result.status_code, 200)
                await self.ready()
                self.assertEqual(result.json()["active_context_length"], 131072)
                for value in (262145, 0, True, "131072", 131072.5):
                    result = await client.post(
                        "/api/models/wide/load", json={"context_length": value}
                    )
                    self.assertEqual(result.status_code, 422, result.text)
                    self.assertEqual(self.manager.context_length, 131072)
                result = await client.post(
                    "/api/models/wide/load", json={"context_length": 196608}
                )
                self.assertEqual(result.status_code, 200)
                await self.ready()
                self.assertEqual(self.manager.context_length, 196608)

    async def test_startup_failure_releases_gpu_for_retry(self):
        self.manager._wait_healthy = AsyncMock(return_value=False)
        await self.manager.request_load("wide", 262144)
        await asyncio.wait_for(self.manager._task, timeout=3)
        self.assertEqual(self.manager.status, "error")
        self.assertFalse(self.manager.gpu.fds)
        self.assertIsNone(self.manager._proc)

    async def test_effective_backend_context_is_read_from_props(self):
        del self.manager._effective_context_length
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"default_generation_settings": {"n_ctx": 65536}}
            )
        )
        with patch.object(
            server.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=transport, **kw),
        ):
            self.assertEqual(
                await self.manager._effective_context_length(131072), 65536
            )


class StrataContextTest(unittest.TestCase):
    def test_chat_launcher_preserves_installed_and_codex_configs(self):
        path = Path(__file__).resolve().parents[2] / "qwen3.8-flash-next/chat_server.py"
        spec = importlib.util.spec_from_file_location("strata_chat_launcher", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "strata").mkdir()
            (root / "qwen3.8-flash-next").mkdir()
            (root / "qwen3.8-flash-next/model.toml").write_text(
                "default_context_length = 131072\nmax_context_length = 262144\n"
            )
            original = {
                "args": ["--max-context", "32768", "--kv", "int8"],
                "env": {"HEADROOM": "6"},
            }
            base = root / "strata/strata-sc117-iq3_xxs.json"
            base.write_text(json.dumps(original))
            codex = root / "strata/strata-codex-sc117-iq3_xxs.json"
            codex.write_text('{"separate": true}')
            with (
                patch.object(module, "ROOT", root),
                patch.dict(module.os.environ, {"CTX_SIZE": "262144"}),
                patch.object(module.os, "execv") as execute,
            ):
                module.main()
                execute.assert_called_once()
            result = json.loads(
                (root / "strata/strata-chat-sc117-iq3_xxs.json").read_text()
            )
            self.assertEqual(
                result["args"], ["--max-context", "262144", "--kv", "int8"]
            )
            self.assertEqual(result["env"], original["env"])
            self.assertEqual(json.loads(base.read_text()), original)
            self.assertEqual(codex.read_text(), '{"separate": true}')


if __name__ == "__main__":
    unittest.main()
