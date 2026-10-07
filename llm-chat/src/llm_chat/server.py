"""統合チャットサーバー。

LLM_ROOT (既定: このアプリの親ディレクトリ /workspace/LLM) 直下の
model.toml を持つディレクトリを LLM として検出し、
選択されたモデルの serve.sh (llama-server / Strata) を内部ポートで1つだけ起動する。
VRAM 12GB では複数モデルの同時常駐はできないため、切り替え時は
旧プロセスを停止してから新しいモデルを起動する。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO

import httpx
import tomllib
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import agent, irodori, voice, voice_control, voice_library_api
from .gpu_reservation import GPUReservation
from .images import ChatMessage
from .voice_runtime import VoiceRuntime

APP_DIR = Path(__file__).resolve().parents[2]  # .../LLM/llm-chat
LLM_ROOT = Path(os.environ.get("LLM_ROOT", APP_DIR.parent))
HISTORY_DIR = Path(os.environ.get("LLM_HISTORY_DIR", APP_DIR / "history"))
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = int(os.environ.get("LLM_BACKEND_PORT", "5071"))
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
LOAD_TIMEOUT_S = 900
STOP_TIMEOUT_S = 20
KILL_TIMEOUT_S = 5
STATIC_DIR = Path(__file__).parent / "static"


class ModelStartError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def process_group_alive(group: int) -> bool:
    """Ignore zombies, which no longer hold memory or the inherited GPU lock."""
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) == group and fields[0] not in ("Z", "X"):
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    order: int
    dir: Path
    thinking_mode: str = "enable_thinking"
    mmproj: str | None = None
    default_context_length: int = 32768
    max_context_length: int = 32768

    @property
    def images_ready(self) -> bool:
        return bool(self.mmproj and (self.dir / self.mmproj).is_file())

    @property
    def log_path(self) -> Path:
        return self.dir / "server.log"


def discover_models() -> dict[str, ModelInfo]:
    models: dict[str, ModelInfo] = {}
    for toml_path in sorted(LLM_ROOT.glob("*/model.toml")):
        d = toml_path.parent
        if not (d / "serve.sh").exists():
            continue
        meta = tomllib.loads(toml_path.read_text())
        mid = meta.get("id", d.name)
        models[mid] = ModelInfo(
            id=mid,
            name=meta.get("name", mid),
            description=meta.get("description", ""),
            order=int(meta.get("order", 100)),
            dir=d,
            thinking_mode=meta.get("thinking_mode", "enable_thinking"),
            mmproj=meta.get("mmproj"),
            default_context_length=int(meta.get("default_context_length", 32768)),
            max_context_length=int(meta.get("max_context_length", 32768)),
        )
        if not 512 <= models[mid].default_context_length <= models[mid].max_context_length:
            raise ValueError(f"{toml_path}: invalid context length range")
    return dict(sorted(models.items(), key=lambda kv: kv[1].order))


def tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def open_log(path: Path) -> BinaryIO:
    log = path.open("ab")
    log.write(f"\n===== start {time.strftime('%F %T')} =====\n".encode())
    log.flush()
    return log


class ModelManager:
    """llama-server / Strataのモデルサーバーを1つだけ管理する。"""

    def __init__(self) -> None:
        self.models = discover_models()
        self.active: str | None = None
        self.status = "stopped"  # stopped | loading | ready | error
        self.error = ""
        self.error_code: str | None = None
        self.error_detail = ""
        self.loaded_at: float | None = None
        self.context_length: int | None = None
        self._context_lengths: dict[str, int] = {}
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._log_path: Path | None = None
        self._log_offset = 0
        self.gpu = GPUReservation(LLM_ROOT)

    def snapshot(self) -> dict:
        # A ready server can also be killed by the OS; never leave it stuck as ready.
        if (self.status == "ready" and self._proc is not None
                and self._proc.returncode is not None):
            self.status = "cleaning"
            self._task = asyncio.create_task(self._failed(RuntimeError("モデルサーバーが終了しました")))
        return {
            "active": self.active,
            "status": self.status,
            "error": self.error,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "loaded_at": self.loaded_at,
            "active_context_length": self.context_length,
            "models": [
                {"id": m.id, "name": m.name, "description": m.description,
                 "supports_images": bool(m.mmproj), "images_ready": m.images_ready,
                 "default_context_length": m.default_context_length,
                 "max_context_length": m.max_context_length,
                 "context_length": self._context_lengths.get(m.id, m.default_context_length)}
                for m in self.models.values()
            ],
        }

    async def request_load(self, model_id: str, context_length: int | None = None) -> None:
        async with self._control_lock:
            if model_id not in self.models:
                raise KeyError(model_id)
            model = self.models[model_id]
            selected = (
                self._context_lengths.get(model_id, model.default_context_length)
                if context_length is None else context_length
            )
            if type(selected) is not int or not 512 <= selected <= model.max_context_length:
                raise ValueError(f"コンテキスト長は512〜{model.max_context_length}トークンで指定してください")
            if (
                self.active == model_id and self.status in ("loading", "ready")
                and self._context_lengths.get(model_id) == selected
                and (self.status == "loading" or (
                    self._proc is not None and self._proc.returncode is None
                ))
            ):
                return
            self.gpu.acquire("llm-chatの言語モデル")
            # Finish the old task's process cleanup before creating its successor.
            if self._task and not self._task.done():
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            self.active, self.status, self.error = model_id, "loading", ""
            self.loaded_at = None
            self.error_code, self.error_detail = None, ""
            self._log_path, self._log_offset = None, 0
            self.context_length = selected
            self._context_lengths[model_id] = selected
            self._task = asyncio.create_task(self._load(model_id, selected))

    async def _load(self, model_id: str, context_length: int) -> None:
        try:
            await self._load_model(model_id, context_length)
        except asyncio.CancelledError:
            async with self._lock:
                await self._stop_proc()
            raise
        except Exception as exc:  # noqa: BLE001 - surface background startup failures to the UI
            await self._failed(exc)

    def _current_log(self) -> str:
        if self._log_path is None:
            return ""
        try:
            with self._log_path.open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(self._log_offset, log.tell() - 8192))
                return log.read().decode(errors="replace").strip()
        except OSError:
            return ""

    async def _failed(self, exc: Exception) -> None:
        self.status = "cleaning"
        self.loaded_at = None
        details = self._current_log()
        returncode = self._proc.returncode if self._proc is not None else None
        code = exc.code if isinstance(exc, ModelStartError) else "startup_failed"
        evidence = (str(exc) + "\n" + details).lower()
        memory_errors = (
            "out of memory", "cannot allocate memory", "failed to allocate",
            "cudaerrormemoryallocation", "hiperroroutofmemory", "bad_alloc",
            "cublas_status_alloc_failed", "not enough memory", "メモリ不足",
        )
        if any(text in evidence for text in memory_errors):
            code = "out_of_memory"
            reason = "GPUまたはRAMのメモリが不足しました。"
        elif returncode in (-signal.SIGKILL, 137):
            code = "process_killed"
            reason = "モデルが強制終了されました。メモリ不足の可能性があります。"
        elif code == "startup_timeout":
            reason = "モデルの起動が制限時間内に完了しませんでした。"
        else:
            reason = "モデルサーバーの起動または動作に失敗しました。"
        requested = self.context_length
        model = self.models.get(self.active)
        prefix = f"{model.name if model else 'モデル'}（コンテキスト{requested}トークン）: "
        # Publish error only once the full process group has stopped and its memory is gone.
        try:
            async with self._lock:
                await self._stop_proc()
        except Exception as cleanup_exc:  # noqa: BLE001 - keep recovery controls usable
            self.error_code = "cleanup_failed"
            self.error = prefix + "モデルの停止を確認できませんでした。「停止」で再試行してください。"
            self.error_detail = f"{exc}\n{details}\n停止エラー: {cleanup_exc}".strip()
            self.status = "error"
            return  # Retain the reservation until cleanup is confirmed; do not overlap models.
        self.gpu.release()
        self.loaded_at, self.context_length = None, None
        self.error_code = code
        self.error_detail = f"{exc}\n終了コード: {returncode}\n{details}".strip()
        self.error = prefix + reason + " モデルのメモリを解放しました。コンテキスト長を下げるか、別のモデルを選んで「切り替え」を押してください。"
        self.status = "error"

    async def _load_model(self, model_id: str, context_length: int) -> None:
        async with self._lock:
            if self.active != model_id:  # 待っている間に別モデルが選択された
                return
            await self._stop_proc()
            self.active, self.status, self.error = model_id, "loading", ""
            m = self.models[model_id]
            env = {**os.environ, "HOST": BACKEND_HOST, "PORT": str(BACKEND_PORT),
                   "CTX_SIZE": str(context_length), "LLAMA_ARG_N_PARALLEL": "1"}
            with open_log(m.log_path) as log:
                self._log_path, self._log_offset = m.log_path, log.tell()
                spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                    "bash",
                    str(m.dir / "serve.sh"),
                    cwd=m.dir,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    pass_fds=self.gpu.fds,
                ))
                try:
                    self._proc = await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    self._proc = await spawn
                    raise
            ok = await self._wait_healthy(self._proc)
            if self.active != model_id:
                return
            if ok:
                self.context_length = await self._effective_context_length(context_length)
                self.status, self.loaded_at = "ready", time.time()
            else:
                raise ModelStartError("process_exit", "モデルが起動完了前に終了しました")

    async def _effective_context_length(self, requested: int) -> int:
        # Both llama.cpp and Strata expose the actual running context through /props.
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{BACKEND_URL}/props")
                response.raise_for_status()
                value = response.json().get("default_generation_settings", {}).get("n_ctx")
            if type(value) is int and value > 0:
                return value
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return requested

    async def _wait_healthy(self, proc: asyncio.subprocess.Process) -> bool:
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        async with httpx.AsyncClient(timeout=2) as client:
            while time.monotonic() < deadline:
                if proc.returncode is not None:
                    return False
                try:
                    r = await client.get(f"{BACKEND_URL}/health")
                    if r.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        raise ModelStartError("startup_timeout", f"起動待機が{LOAD_TIMEOUT_S}秒を超えました")

    async def _stop_proc(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=STOP_TIMEOUT_S)
        except TimeoutError:
            pass
        # Even an exited parent may leave a vision/engine child with RAM, VRAM or the GPU lock.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await asyncio.wait_for(proc.wait(), timeout=KILL_TIMEOUT_S)
        deadline = time.monotonic() + KILL_TIMEOUT_S
        while process_group_alive(proc.pid):
            if time.monotonic() >= deadline:
                raise RuntimeError("モデルの子プロセスが停止しませんでした")
            await asyncio.sleep(0.02)
        self._proc = None

    async def unload(self) -> None:
        async with self._control_lock:
            # Cancel loading before waiting on the process lock.
            self.active = None
            task, self._task = self._task, None
            if task and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            async with self._lock:
                await self._stop_proc()
                self.gpu.release()
                self.active, self.status, self.error, self.loaded_at = (
                    None, "stopped", "", None,
                )
                self.context_length = None
                self.error_code, self.error_detail = None, ""


manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    speech = VoiceRuntime(LLM_ROOT, {**voice.ENGINE_URLS, "irodori": irodori.URL},
                          before_gpu_start=manager.unload)
    app.state.speech_runtime = speech
    try:
        yield
    finally:
        try:
            await manager.unload()
        finally:
            await speech.stop()


app = FastAPI(title="LLM Chat", lifespan=lifespan)


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    enable_thinking: bool = True
    web_search: bool = False
    voice_speaker: str | None = None


class LoadRequest(BaseModel):
    context_length: int | None = Field(default=None, ge=512, strict=True)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/voice-library.js")
async def library_script() -> FileResponse:
    return FileResponse(STATIC_DIR / "voice-library.js", media_type="text/javascript")


app.include_router(voice.router)
app.include_router(voice_library_api.router)


@app.get("/api/status")
async def status() -> dict:
    return {**manager.snapshot(), "voice": app.state.speech_runtime.snapshot()}


@app.post("/api/models/{model_id}/load")
async def load(model_id: str, req: Annotated[LoadRequest | None, Body()] = None) -> dict:
    try:
        async with app.state.speech_runtime._lock:
            await manager.request_load(model_id, req.context_length if req else None)
    except KeyError:
        raise HTTPException(404, f"unknown model: {model_id}")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return manager.snapshot()


@app.post("/api/unload")
async def unload() -> dict:
    async with app.state.speech_runtime._lock:
        await manager.unload()
    return manager.snapshot()


@app.get("/api/models/{model_id}/log", response_class=PlainTextResponse)
async def log(model_id: str, n: int = 200) -> str:
    m = manager.models.get(model_id)
    if m is None:
        raise HTTPException(404, f"unknown model: {model_id}")
    return tail(m.log_path, n)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if manager.active != req.model or manager.status != "ready":
        raise HTTPException(409, "model is not ready")
    if any(message.images for message in req.messages):
        model = manager.models[req.model]
        if not model.mmproj:
            raise HTTPException(400, "このモデルは画像入力に対応していません。画像対応モデルへ切り替えてください")
        if not model.images_ready:
            raise HTTPException(409, "画像用モデルが未取得です。画像用モデルを取得して再起動してください")

    voice_styles = None
    if req.voice_speaker:
        try:
            voice_styles = await voice.styles_for_speaker(req.voice_speaker)
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
    messages = [message.backend_message() for message in req.messages]
    if voice_styles:
        speaker_name = req.voice_speaker.partition(":")[2] or req.voice_speaker
        if req.voice_speaker.startswith("irodori:"):
            speaker_name = voice.LIBRARY.get(speaker_name)["name"]
        voice_prompt = voice_control.prompt(speaker_name, voice_styles)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = voice_prompt + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": voice_prompt})

    thinking_mode = manager.models[req.model].thinking_mode
    if thinking_mode == "reasoning_effort":
        thinking_kwargs = (
            {"reasoning_effort": "medium"}
            if req.enable_thinking
            else {"enable_thinking": False}
        )
    else:
        thinking_kwargs = {"enable_thinking": req.enable_thinking}
    payload: dict = {
        "model": req.model,
        "messages": messages,
        "stream": True,
        "chat_template_kwargs": thinking_kwargs,
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    if req.web_search:
        stream = agent.run(BACKEND_URL, payload)
        if voice_styles:
            stream = voice_control.separate_style(stream, voice_styles)
        return StreamingResponse(stream, media_type="text/event-stream")

    async def relay():
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5)) as client:
            try:
                async with client.stream(
                    "POST", f"{BACKEND_URL}/v1/chat/completions", json=payload
                ) as r:
                    if r.status_code != 200:
                        body = (await r.aread()).decode(errors="replace")
                        yield f"data: {json.dumps({'error': body})}\n\n".encode()
                        return
                    async for chunk in r.aiter_raw():
                        yield chunk
            except httpx.HTTPError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

    stream = relay()
    if voice_styles:
        stream = voice_control.separate_style(stream, voice_styles)
    return StreamingResponse(stream, media_type="text/event-stream")


# ---------------------------------------------------------------------------
# 会話履歴: HISTORY_DIR/<id>.json (再読み込み用) と <id>.md (閲覧用) を保存する
# ---------------------------------------------------------------------------

HISTORY_ID = re.compile(r"^[0-9A-Za-z_-]{1,80}$")


class HistoryMessage(ChatMessage):
    model: str | None = None
    reasoning: str | None = None
    tools: list[dict] | None = None
    sources: list[dict] | None = None
    voice_style_id: int | str | None = None
    voice_style_name: str | None = None
    voice_engine: str | None = None
    voice_id: str | None = None


class Conversation(BaseModel):
    id: str
    title: str = ""
    created: str
    updated: str
    system: str = ""
    messages: list[HistoryMessage]


def history_path(conv_id: str, suffix: str) -> Path:
    if not HISTORY_ID.match(conv_id):
        raise HTTPException(400, "invalid conversation id")
    return HISTORY_DIR / f"{conv_id}{suffix}"


def to_markdown(conv: Conversation) -> str:
    out = [
        f"# {conv.title or conv.id}",
        "",
        f"- created: {conv.created}",
        f"- updated: {conv.updated}",
    ]
    if conv.system:
        out += ["", "## System", "", conv.system]
    for m in conv.messages:
        if m.role == "user":
            out += ["", "## User", "", m.content]
            for image in m.images:
                label = re.sub(r"[\[\]\\\r\n]", "_", image.name)
                out += ["", f"![{label}]({image.data_url})"]
        else:
            out += ["", f"## Assistant ({m.model or '?'})", ""]
            if m.reasoning:
                out += [
                    "<details><summary>思考</summary>",
                    "",
                    m.reasoning,
                    "",
                    "</details>",
                    "",
                ]
            for t in m.tools or []:
                arg = t.get("args", {}).get("query") or t.get("args", {}).get("url", "")
                out.append(f"> {t.get('name')}: {arg} ({t.get('status')})")
            if m.tools:
                out.append("")
            out.append(m.content)
            if m.sources:
                out += ["", "**出典**", ""]
                out += [
                    f"{x['n']}. [{x.get('title') or x['url']}]({x['url']})"
                    for x in m.sources
                ]
    return "\n".join(out) + "\n"


@app.get("/api/history")
def list_history() -> list[dict]:
    items = []
    for p in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            conv = Conversation.model_validate_json(p.read_text())
        except ValueError:
            continue
        items.append(
            {
                "id": conv.id,
                "title": conv.title,
                "updated": conv.updated,
                "n": len(conv.messages),
            }
        )
    return items


@app.get("/api/history/{conv_id}")
def get_history(conv_id: str) -> Conversation:
    p = history_path(conv_id, ".json")
    if not p.exists():
        raise HTTPException(404, "not found")
    return Conversation.model_validate_json(p.read_text())


@app.put("/api/history/{conv_id}")
def save_history(conv_id: str, conv: Conversation) -> dict:
    if conv.id != conv_id:
        raise HTTPException(400, "id mismatch")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path(conv_id, ".json").write_text(conv.model_dump_json(indent=2))
    history_path(conv_id, ".md").write_text(to_markdown(conv))
    return {"saved": str(history_path(conv_id, ".json"))}
