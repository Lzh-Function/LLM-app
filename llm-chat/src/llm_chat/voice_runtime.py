"""Start installed local speech engines and stop the processes this app owns."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .gpu_reservation import GPUReservation
from .irodori import IrodoriClient
from .irodori_profiles import profile_for
from .qwen_tts import QwenClient

logger = logging.getLogger("uvicorn.error")
START_TIMEOUT_S = 120
STOP_TIMEOUT_S = 15
ENGINE_PATHS = {
    "aivis": ("aivisspeech", "Linux-x64/run"),
    "voicevox": ("voicevox", "linux-cpu-x64/run"),
}


def local_address(url: str) -> tuple[str, int] | None:
    try:
        parsed = urlsplit(url)
        port = parsed.port if parsed.port is not None else 80
    except ValueError:
        return None
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or port < 1
    ):
        return None
    return parsed.hostname, port


async def port_in_use(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=1
        )
    except (OSError, TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


class VoiceRuntime:
    def __init__(self, root: Path, urls: dict[str, str], *, before_gpu_start=None):
        self.root = root
        self.urls = urls
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.tasks: list[asyncio.Task] = []
        self.enabled = False
        self.stopping = False
        self.engine = "legacy"
        self.mode = "chat"
        self.ready = False
        self.errors: dict[str, str] = {}
        self.startup_seconds: float | None = None
        self.warmup: dict | None = None
        self.irodori = IrodoriClient(urls.get("irodori", "http://127.0.0.1:8088"))
        self.qwen = QwenClient(urls.get("qwen", "http://127.0.0.1:8090"))
        self._lock = asyncio.Lock()
        self.before_gpu_start = before_gpu_start
        self.gpu = GPUReservation(root)

    @property
    def starting(self) -> bool:
        return any(not task.done() for task in self.tasks)

    def start(self) -> None:
        if (
            self.engine == "qwen"
            or (self.engine == "irodori" and profile_for(self.mode).uses_gpu)
        ) and not self.gpu.fds:
            raise RuntimeError("GPUモードは await set_enabled(True) で起動してください")
        if self.enabled and (
            self.engine not in ("irodori", "qwen") or self.starting or self.ready
        ):
            return
        self.enabled = True
        self.errors.clear()
        self.ready = False
        self.startup_seconds = None
        self.warmup = None
        if self.engine == "qwen":
            self.tasks = [asyncio.create_task(self._start_qwen())]
        elif self.engine == "irodori":
            self.tasks = [asyncio.create_task(self._start_irodori())]
        else:
            self.tasks = [
                asyncio.create_task(self._start_engine(engine, url))
                for engine, url in self.urls.items()
                if engine in ENGINE_PATHS
            ]

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "starting": self.starting,
            "stopping": self.stopping,
            "engine": self.engine,
            "mode": self.mode,
            "ready": self.ready,
            "errors": self.errors.copy(),
            "startup_seconds": self.startup_seconds,
            "warmup": self.warmup,
            "profile": vars(profile_for(self.mode))
            if self.engine == "irodori"
            else {
                "model_device": "cuda",
                "model_precision": "bfloat16",
                "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-"
                + ("VoiceDesign" if self.mode == "design" else "Base"),
            }
            if self.engine == "qwen"
            else None,
            "exclusive_gpu": bool(self.gpu.fds),
            "pid": self.processes["irodori"].pid
            if "irodori" in self.processes
            else self.processes["qwen"].pid
            if "qwen" in self.processes
            else None,
        }

    async def set_enabled(
        self, enabled: bool, engine: str | None = None, mode: str | None = None
    ) -> dict:
        engine = engine or self.engine
        mode = mode or self.mode
        if engine not in ("legacy", "irodori", "qwen") or mode not in (
            "chat",
            "design",
            "synthesize",
        ):
            raise ValueError("Invalid speech engine or mode")
        if engine == "qwen" and mode == "chat":
            raise ValueError("Qwen supports GPU design and synthesis only")
        # Serialize toggles so OFF completes before another ON can spawn children.
        async with self._lock:
            same_backend = (
                engine == self.engine
                and self.enabled
                and (
                    engine == "qwen"
                    or (
                        engine == "irodori"
                        and profile_for(mode).identity
                        == profile_for(self.mode).identity
                    )
                )
            )
            if same_backend:
                # Qwen swaps VoiceDesign/Base inside its serialized backend on demand.
                self.mode = mode
            elif (engine, mode) != (self.engine, self.mode):
                self.enabled = False
                self.stopping = True
                try:
                    await self._stop()
                finally:
                    self.stopping = False
                self.engine, self.mode = engine, mode
            if enabled:
                if engine == "qwen" and not self.ready and not self.starting:
                    if self.before_gpu_start:
                        await self.before_gpu_start()
                    self.gpu.acquire("Qwen3-TTSの声作成・高品質合成")
                if engine == "irodori" and not self.ready and not self.starting:
                    profile = profile_for(mode)
                    if profile.uses_gpu:
                        if self.before_gpu_start:
                            await self.before_gpu_start()
                        self.gpu.acquire("Irodori Largeの声作成・高品質合成")
                    self.irodori.url = (
                        self.urls.get(
                            "irodori_quality",
                            self.urls.get("irodori", self.irodori.url),
                        )
                        if mode != "chat"
                        else self.urls.get("irodori", self.irodori.url)
                    )
                self.start()
            else:
                self.enabled = False
                self.stopping = True
                try:
                    await self._stop()
                finally:
                    self.stopping = False
            return self.snapshot()

    @property
    def tts(self):
        return self.qwen if self.engine == "qwen" else self.irodori

    async def require_tts(self, mode: str, engine: str):
        from fastapi import HTTPException

        if engine == "irodori":
            return await self.require_irodori(mode)
        if not self.enabled or self.engine != engine or self.mode != mode:
            raise HTTPException(409, "選択した音声エンジンのモードをONにしてください")
        if not self.ready:
            raise HTTPException(
                503, " / ".join(self.errors.values()) or "Qwenのモデルロード中です"
            )

    async def _start_qwen(self):
        started = time.perf_counter()
        address = local_address(self.qwen.url)
        directory = self.root / "qwen-tts"
        try:
            if address and await port_in_use(*address):
                raise RuntimeError(
                    "管理外のQwenが起動済みです。停止してからGPUモードをONにしてください"
                )
            if address:
                script = directory / "serve.sh"
                if not script.is_file() or not (directory / "install.json").is_file():
                    raise RuntimeError(
                        "Qwen未導入です。bash qwen-tts/setup.sh を実行してください"
                    )
                with (directory / "server.log").open("ab") as log:
                    spawn = asyncio.create_task(
                        asyncio.create_subprocess_exec(
                            "bash",
                            str(script),
                            "--host",
                            address[0],
                            "--port",
                            str(address[1]),
                            cwd=directory,
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                            pass_fds=self.gpu.fds,
                        )
                    )
                    try:
                        self.processes["qwen"] = await asyncio.shield(spawn)
                    except asyncio.CancelledError:
                        self.processes["qwen"] = await spawn
                        raise
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                proc = self.processes.get("qwen")
                if proc and proc.returncode is not None:
                    raise RuntimeError(
                        f"Qwenが終了しました ({proc.returncode})。qwen-tts/server.logを確認してください"
                    )
                try:
                    await self.qwen.health()
                    break
                except (httpx.HTTPError, ValueError):
                    await asyncio.sleep(0.5)
            else:
                raise RuntimeError("Qwenの起動がタイムアウトしました")
            health = await self.qwen.load(self.mode)
            expected = "Qwen/Qwen3-TTS-12Hz-1.7B-" + (
                "VoiceDesign" if self.mode == "design" else "Base"
            )
            if (
                health.get("model", {}).get("hf_checkpoint") != expected
                or not health.get("runtime", {}).get("loaded")
                or health.get("model", {}).get("model_precision") != "bfloat16"
                or health.get("model", {}).get("model_device") != "cuda"
            ):
                raise RuntimeError("Qwen CUDA BF16のモデルロードを確認できません")
            self.ready = True
            self.startup_seconds = time.perf_counter() - started
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.errors["qwen"] = str(exc)
            logger.exception("Qwen startup failed")
            proc = self.processes.get("qwen")
            if proc:
                await self._stop_process("qwen", proc)
                self.processes.pop("qwen", None)
            self.gpu.release()

    async def require_irodori(self, mode: str) -> None:
        from fastapi import HTTPException

        if not self.enabled or self.engine != "irodori" or self.mode != mode:
            raise HTTPException(
                409,
                f"Irodoriの{'会話' if mode == 'chat' else '高品質合成' if mode == 'synthesize' else '声作成'}モードをONにしてください",
            )
        if not self.ready:
            detail = (
                " / ".join(self.errors.values())
                or "Irodoriのロード・ウォームアップ中です"
            )
            raise HTTPException(503, detail)

    async def _start_irodori(self) -> None:
        started = time.perf_counter()
        url = self.irodori.url
        address = local_address(url)
        directory = self.root / "irodori-tts"
        profile = profile_for(self.mode)
        expected = profile.repo
        try:
            occupied = bool(address and await port_in_use(*address))
            if profile.uses_gpu and occupied and "irodori" not in self.processes:
                raise RuntimeError(
                    "管理外のIrodoriが起動済みです。そのサーバーを停止してからGPUモードをONにしてください"
                )
            if address and not occupied:
                script = directory / "serve.sh"
                if not script.is_file() or not profile.checkpoint(self.root).is_file():
                    raise RuntimeError(
                        "Irodori未導入です。bash irodori-tts/setup.sh を実行してください"
                    )
                with (directory / "server.log").open("ab") as log:
                    spawn = asyncio.create_task(
                        asyncio.create_subprocess_exec(
                            "bash",
                            str(script),
                            "--host",
                            address[0],
                            "--port",
                            str(address[1]),
                            "--mode",
                            self.mode,
                            cwd=directory,
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                            pass_fds=self.gpu.fds,
                        )
                    )
                    try:
                        self.processes["irodori"] = await asyncio.shield(spawn)
                    except asyncio.CancelledError:
                        self.processes["irodori"] = await spawn
                        raise
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                proc = self.processes.get("irodori")
                if proc and proc.returncode is not None:
                    raise RuntimeError(
                        f"Irodoriが終了しました ({proc.returncode})。irodori-tts/server.log を確認してください"
                    )
                try:
                    health = await self.irodori.health()
                    break
                except (httpx.HTTPError, ValueError):
                    await asyncio.sleep(0.5)
            else:
                raise RuntimeError("Irodoriの起動がタイムアウトしました")
            if health.get("model", {}).get("hf_checkpoint") != expected:
                raise RuntimeError(
                    f"起動済みIrodoriのモデルが異なります。{expected} を指定して起動してください"
                )
            for field in (
                "model_device",
                "codec_device",
                "model_precision",
                "codec_precision",
            ):
                actual = health.get("model", {}).get(field)
                if actual is not None and actual != getattr(profile, field):
                    raise RuntimeError(f"Irodoriの{field}が設定と異なります: {actual}")
            # /health alone does not load either model or codec.
            _, self.warmup = await self.irodori.speech(
                "こんにちは。", caption="自然な話し方。", mode=self.mode, seed=0
            )
            health = await self.irodori.health()
            if not health.get("runtime", {}).get("loaded"):
                raise RuntimeError("Irodoriのモデルロードを確認できません")
            self.ready = True
            self.startup_seconds = time.perf_counter() - started
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.errors["irodori"] = str(exc)
            logger.exception("Irodori startup failed")
            proc = self.processes.get("irodori")
            if proc:
                await self._stop_process("irodori", proc)
                self.processes.pop("irodori", None)
            self.gpu.release()

    async def _start_engine(self, engine: str, url: str) -> None:
        address = local_address(url)
        if address is None:
            logger.info(
                "%s: using configured speech endpoint; local autostart skipped", engine
            )
            return
        host, port = address
        folder, binary = ENGINE_PATHS[engine]
        directory = self.root / folder
        script = directory / "serve.sh"
        if await port_in_use(host, port):
            logger.info(
                "%s: port %s already in use; leaving the existing service running",
                engine,
                port,
            )
            return
        if not script.is_file() or not (directory / binary).is_file():
            logger.info(
                "%s: speech engine is not installed in %s; autostart skipped",
                engine,
                directory,
            )
            return
        log_path = directory / "server.log"
        try:
            with log_path.open("ab") as log:
                log.write(
                    f"\n===== llm-chat start {time.strftime('%F %T')} =====\n".encode()
                )
                log.flush()
                # Shield creation so cancellation cannot lose ownership of a newly spawned process.
                spawn = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        "bash",
                        str(script),
                        "--host",
                        host,
                        "--port",
                        str(port),
                        cwd=directory,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                )
                try:
                    proc = await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    self.processes[engine] = await spawn
                    raise
            self.processes[engine] = proc
            logger.info(
                "%s: starting speech engine on port %s (log: %s)",
                engine,
                port,
                log_path,
            )
            deadline = time.monotonic() + START_TIMEOUT_S
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if proc.returncode is not None:
                        logger.error(
                            "%s: speech engine exited (%s); see %s",
                            engine,
                            proc.returncode,
                            log_path,
                        )
                        return
                    try:
                        response = await client.get(f"{url.rstrip('/')}/speakers")
                        response.raise_for_status()
                        if isinstance(response.json(), list):
                            logger.info("%s: speech engine ready", engine)
                            return
                    except (httpx.HTTPError, ValueError):
                        pass
                    await asyncio.sleep(0.5)
            logger.warning(
                "%s: speech engine is still starting; see %s", engine, log_path
            )
        except OSError:
            logger.exception(
                "%s: could not start speech engine (log: %s)", engine, log_path
            )

    async def stop(self) -> None:
        await self.set_enabled(False)

    async def _stop(self) -> None:
        self.ready = False
        self.irodori.clear()
        self.qwen.clear()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        await asyncio.gather(
            *(
                self._stop_process(engine, proc)
                for engine, proc in self.processes.items()
            )
        )
        self.processes.clear()
        self.gpu.release()
        self.errors.clear()
        self.startup_seconds = None
        self.warmup = None

    async def _stop_process(
        self, engine: str, proc: asyncio.subprocess.Process
    ) -> None:
        # The parent can exit while its worker still holds the listening port.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=STOP_TIMEOUT_S)
        except TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
        # Packaged engines can have a launcher that exits before its worker.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        logger.info("%s: managed speech engine stopped", engine)
