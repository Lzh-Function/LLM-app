"""Fault injection with real process groups; no real GPU allocation or OOM is required."""

import asyncio
import errno
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from llm_chat import server
from llm_chat.gpu_reservation import GPUReservation

BACKEND = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path.cwd()
context = int(os.environ['CTX_SIZE'])
if context > 65536:
    # A surviving engine/vision child holds some RAM and the inherited GPU reservation.
    code = """
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
memory = bytearray(16 * 1024 * 1024)
Path('child-ready').write_text(str(os.getpid()))
time.sleep(60)
"""
    child = subprocess.Popen([sys.executable, '-c', code], close_fds=False)
    while not root.joinpath('child-ready').exists():
        time.sleep(.01)
    print('CUDA error: out of memory while allocating KV cache', flush=True)
    os._exit(1)
root.joinpath('ready').write_text(json.dumps({'pid': os.getpid(), 'context': context}))
time.sleep(60)
'''


class ModelFailureTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [
            patch.object(server, "LLM_ROOT", self.root),
            patch.object(server, "STOP_TIMEOUT_S", 0.15),
            patch.object(server, "KILL_TIMEOUT_S", 2),
        ]
        for item in self.patches:
            item.start()
        self.manager = server.ModelManager()
        self.manager.models = {}
        for name in ("wide", "other"):
            directory = self.root / name
            directory.mkdir()
            (directory / "backend.py").write_text(BACKEND)
            (directory / "serve.sh").write_text(
                "exec " + shlex.quote(sys.executable) + " backend.py\n"
            )
            self.manager.models[name] = server.ModelInfo(
                name,
                name,
                "",
                1,
                directory,
                default_context_length=131072,
                max_context_length=262144,
            )

        async def healthy(proc):
            directory = self.manager.models[self.manager.active].dir
            async with asyncio.timeout(3):
                while proc.returncode is None:
                    ready = directory / "ready"
                    if (
                        ready.exists()
                        and json.loads(ready.read_text())["pid"] == proc.pid
                    ):
                        return True
                    await asyncio.sleep(0.01)
            return False

        self.manager._wait_healthy = healthy
        self.manager._effective_context_length = AsyncMock(
            side_effect=lambda value: value
        )

    async def asyncTearDown(self):
        try:
            await self.manager.unload()
        finally:
            for item in reversed(self.patches):
                item.stop()
            self.temp.cleanup()

    async def settled(self):
        await asyncio.wait_for(self.manager._task, 4)

    def assert_released(self):
        self.assertIsNone(self.manager._proc)
        self.assertFalse(self.manager.gpu.fds)
        slot = GPUReservation(self.root)
        slot.acquire("another service")
        slot.release()

    async def test_oom_kills_orphan_child_and_allows_shorter_context(self):
        await self.manager.request_load("wide", 262144)
        await self.settled()
        result = self.manager.snapshot()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "out_of_memory")
        self.assertIn("262144", result["error"])
        self.assertIn("コンテキスト長を下げる", result["error"])
        self.assertIn("CUDA error", result["error_detail"])
        child = int((self.root / "wide/child-ready").read_text())
        stat = Path(f"/proc/{child}/stat")
        if stat.exists():
            self.assertIn(stat.read_text().rsplit(") ", 1)[1].split()[0], ("Z", "X"))
        self.assert_released()
        await self.manager.request_load("wide", 32768)
        await self.settled()
        self.assertEqual(self.manager.status, "ready")
        self.assertEqual(self.manager.context_length, 32768)
        self.assertIsNone(self.manager.error_code)

    async def test_oom_api_then_other_model_and_stop_remain_usable(self):
        runtime = SimpleNamespace(_lock=asyncio.Lock(), snapshot=dict)
        with (
            patch.object(server, "manager", self.manager),
            patch.object(server.app.state, "speech_runtime", runtime, create=True),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                self.assertEqual(
                    (
                        await client.post(
                            "/api/models/wide/load", json={"context_length": 262144}
                        )
                    ).status_code,
                    200,
                )
                await self.settled()
                response = await client.get("/api/status")
                self.assertEqual(response.json()["error_code"], "out_of_memory")
                self.assertEqual(
                    (
                        await client.post(
                            "/api/models/other/load", json={"context_length": 8192}
                        )
                    ).status_code,
                    200,
                )
                await self.settled()
                self.assertEqual(self.manager.active, "other")
                self.assertEqual(self.manager.status, "ready")
                self.assertEqual((await client.post("/api/unload")).status_code, 200)
                self.assertEqual(self.manager.status, "stopped")
                self.assert_released()

    async def test_error_is_published_only_after_cleanup(self):
        started, finish = asyncio.Event(), asyncio.Event()
        original = self.manager._stop_proc

        async def delayed_stop():
            if self.manager._proc is not None:
                started.set()
                await finish.wait()
            await original()

        self.manager._stop_proc = delayed_stop
        try:
            await self.manager.request_load("wide", 262144)
            await asyncio.wait_for(started.wait(), 3)
            self.assertEqual(self.manager.snapshot()["status"], "cleaning")
            with self.assertRaises(HTTPException):
                GPUReservation(self.root).acquire("other")
        finally:
            finish.set()
        await self.settled()
        self.assertEqual(self.manager.status, "error")
        self.assert_released()

    async def test_sigkill_reports_possible_oom_and_cleans_children(self):
        path = self.root / "wide/backend.py"
        text = BACKEND.replace(
            "print('CUDA error: out of memory while allocating KV cache', flush=True)",
            "print('engine was killed', flush=True)",
        )
        path.write_text(
            text.replace("os._exit(1)", "os.kill(os.getpid(), signal.SIGKILL)")
        )
        await self.manager.request_load("wide", 131072)
        await self.settled()
        self.assertEqual(self.manager.error_code, "process_killed")
        self.assertIn("可能性", self.manager.error)
        self.assert_released()

    async def test_startup_timeout_terminates_hanging_backend(self):
        # Even a hung parent that ignores graceful shutdown must be forcibly stopped.
        (self.root / "wide/backend.py").write_text(
            BACKEND.replace(
                "root.joinpath('ready').write_text",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\nroot.joinpath('ready').write_text",
            )
        )
        del self.manager._wait_healthy
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(503))
        with (
            patch.object(server, "LOAD_TIMEOUT_S", 0.05),
            patch.object(
                server.httpx,
                "AsyncClient",
                lambda **kw: real_client(transport=transport, **kw),
            ),
        ):
            await self.manager.request_load("wide", 32768)
            await self.settled()
        self.assertEqual(self.manager.error_code, "startup_timeout")
        self.assert_released()

    async def test_retry_during_failure_cleanup_waits_without_overlapping_models(self):
        started, finish = asyncio.Event(), asyncio.Event()
        original = self.manager._stop_proc

        async def delayed_stop():
            if self.manager._proc is not None:
                started.set()
                await finish.wait()
            await original()

        self.manager._stop_proc = delayed_stop
        try:
            await self.manager.request_load("wide", 262144)
            await asyncio.wait_for(started.wait(), 3)
            previous = self.manager._proc.pid
            await asyncio.wait_for(self.manager.request_load("other", 8192), 3)
            self.assertFalse((self.root / "other/ready").exists())
        finally:
            finish.set()
        await self.settled()
        self.assertFalse(server.process_group_alive(previous))
        self.assertEqual(self.manager.active, "other")
        self.assertEqual(self.manager.status, "ready")

    async def test_stale_oom_log_does_not_misclassify_next_failure(self):
        (self.root / "wide/server.log").write_text("OLD ERROR: out of memory\n")
        (self.root / "wide/backend.py").write_text(
            "print('invalid model file', flush=True)\n"
        )
        await self.manager.request_load("wide", 32768)
        await self.settled()
        self.assertEqual(self.manager.error_code, "process_exit")
        self.assertIn("invalid model file", self.manager.error_detail)
        self.assertNotIn("OLD ERROR", self.manager.error_detail)
        self.assert_released()

    async def test_spawn_memory_error_releases_reservation(self):
        with patch.object(
            server.asyncio,
            "create_subprocess_exec",
            AsyncMock(side_effect=OSError(errno.ENOMEM, "Cannot allocate memory")),
        ):
            await self.manager.request_load("wide", 32768)
            await self.settled()
        self.assertEqual(self.manager.error_code, "out_of_memory")
        self.assert_released()

    async def test_ready_server_exit_does_not_leave_ready_state_or_lock(self):
        await self.manager.request_load("wide", 32768)
        await self.settled()
        self.manager._proc.terminate()
        await self.manager._proc.wait()
        self.assertEqual(self.manager.snapshot()["status"], "cleaning")
        await self.settled()
        self.assertEqual(self.manager.status, "error")
        self.assert_released()
        await self.manager.request_load("wide", 8192)
        await self.settled()
        self.assertEqual(self.manager.status, "ready")

    async def test_retry_same_context_after_exit_without_status_poll(self):
        await self.manager.request_load("wide", 32768)
        await self.settled()
        previous = self.manager._proc
        previous.terminate()
        await previous.wait()
        await self.manager.request_load("wide", 32768)
        await self.settled()
        self.assertEqual(self.manager.status, "ready")
        self.assertNotEqual(self.manager._proc.pid, previous.pid)

    async def test_cleanup_failure_is_reported_without_claiming_memory_release(self):
        original = self.manager._stop_proc

        async def cannot_stop():
            if self.manager._proc is not None:
                raise RuntimeError("cannot confirm process termination")
            await original()

        self.manager._stop_proc = cannot_stop
        try:
            await self.manager.request_load("wide", 262144)
            await self.settled()
            self.assertEqual(self.manager.status, "error")
            self.assertEqual(self.manager.error_code, "cleanup_failed")
            self.assertIn("停止", self.manager.error)
            self.assertNotIn("メモリを解放しました", self.manager.error)
            self.assertTrue(self.manager.gpu.fds)
        finally:
            self.manager._stop_proc = original
        await self.manager.unload()
        self.assert_released()


if __name__ == "__main__":
    unittest.main()
