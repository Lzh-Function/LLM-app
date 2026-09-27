"""統合チャットサーバー。

LLM_ROOT (既定: このアプリの親ディレクトリ /workspace/LLM) 直下の
model.toml を持つディレクトリを LLM として検出し、
選択されたモデルの serve.sh (llama-server) を内部ポートで 1 つだけ起動する。
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
import tomllib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from . import agent, voice, voice_control

APP_DIR = Path(__file__).resolve().parents[2]  # .../LLM/llm-chat
LLM_ROOT = Path(os.environ.get("LLM_ROOT", APP_DIR.parent))
HISTORY_DIR = Path(os.environ.get("LLM_HISTORY_DIR", APP_DIR / "history"))
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = int(os.environ.get("LLM_BACKEND_PORT", "5071"))
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
LOAD_TIMEOUT_S = 900
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    order: int
    dir: Path
    thinking_mode: str = "enable_thinking"

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
        )
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
    """llama-server プロセスを 1 つだけ管理する。"""

    def __init__(self) -> None:
        self.models = discover_models()
        self.active: str | None = None
        self.status = "stopped"  # stopped | loading | ready | error
        self.error = ""
        self.loaded_at: float | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "status": self.status,
            "error": self.error,
            "loaded_at": self.loaded_at,
            "models": [
                {"id": m.id, "name": m.name, "description": m.description}
                for m in self.models.values()
            ],
        }

    def request_load(self, model_id: str) -> None:
        if model_id not in self.models:
            raise KeyError(model_id)
        if self.active == model_id and self.status in ("loading", "ready"):
            return
        # 先に状態を更新しておき、ポーリング側が即座に loading を見られるようにする
        self.active, self.status, self.error = model_id, "loading", ""
        self._task = asyncio.create_task(self._load(model_id))

    async def _load(self, model_id: str) -> None:
        async with self._lock:
            if self.active != model_id:  # 待っている間に別モデルが選択された
                return
            await self._stop_proc()
            self.active, self.status, self.error = model_id, "loading", ""
            m = self.models[model_id]
            env = {**os.environ, "HOST": BACKEND_HOST, "PORT": str(BACKEND_PORT)}
            with await asyncio.to_thread(open_log, m.log_path) as log:
                self._proc = await asyncio.create_subprocess_exec(
                    "bash",
                    str(m.dir / "serve.sh"),
                    cwd=m.dir,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            ok = await self._wait_healthy(self._proc)
            if self.active != model_id:
                return
            if ok:
                self.status, self.loaded_at = "ready", time.time()
            else:
                self.status = "error"
                self.error = tail(m.log_path, 25) or "llama-server の起動に失敗しました"
                await self._stop_proc()

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
        return False

    async def _stop_proc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            await asyncio.wait_for(proc.wait(), timeout=20)
        except (TimeoutError, ProcessLookupError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

    async def unload(self) -> None:
        async with self._lock:
            await self._stop_proc()
            self.active, self.status, self.error, self.loaded_at = (
                None,
                "stopped",
                "",
                None,
            )


manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await manager.unload()


app = FastAPI(title="LLM Chat", lifespan=lifespan)


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None
    enable_thinking: bool = True
    web_search: bool = False
    voice_speaker: str | None = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.include_router(voice.router)


@app.get("/api/status")
async def status() -> dict:
    return manager.snapshot()


@app.post("/api/models/{model_id}/load")
async def load(model_id: str) -> dict:
    try:
        manager.request_load(model_id)
    except KeyError:
        raise HTTPException(404, f"unknown model: {model_id}")
    return manager.snapshot()


@app.post("/api/unload")
async def unload() -> dict:
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

    voice_styles = None
    if req.voice_speaker:
        try:
            voice_styles = await voice.styles_for_speaker(req.voice_speaker)
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
    messages = [dict(message) for message in req.messages]
    if voice_styles:
        voice_prompt = voice_control.prompt(req.voice_speaker.partition(":")[2] or req.voice_speaker, voice_styles)
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


class HistoryMessage(BaseModel):
    role: str
    content: str
    model: str | None = None
    reasoning: str | None = None
    tools: list[dict] | None = None
    sources: list[dict] | None = None
    voice_style_id: int | None = None
    voice_style_name: str | None = None
    voice_engine: str | None = None


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
