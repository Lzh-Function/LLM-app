"""Persist Codex API traffic while keeping Strata's HTTP and streaming behavior."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parent
API_PATHS = {"/v1/responses", "/v1/chat/completions", "/v1/messages"}


def json_text(value):
    return json.dumps(value, ensure_ascii=False)


def private_file(path, mode="a"):
    stream = path.open(mode, encoding="utf-8")
    path.chmod(0o600)
    return stream


class TraceStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.session = root / f"{stamp}-{os.getpid()}"
        self.session.mkdir(mode=0o700)
        temporary = root / f".latest-{uuid.uuid4().hex}"
        temporary.symlink_to(self.session.name, target_is_directory=True)
        temporary.replace(root / "latest")
        self.lock = threading.Lock()
        self.live = private_file(root / "live.log")
        self.last_channel = None
        self.message(f"SERVER START · {self.session}")

    def message(self, message):
        with self.lock:
            self.live.write(f"\n\n[{datetime.now(timezone.utc).isoformat()}] {message}\n")
            self.live.flush()
            self.last_channel = None

    def text(self, identifier, channel, text):
        with self.lock:
            key = (identifier, channel)
            if key != self.last_channel:
                self.live.write(f"\n\n[{identifier}] {channel}\n")
                self.last_channel = key
            self.live.write(text)
            self.live.flush()

    def begin(self, path, body):
        return RequestTrace(self, path, body)

    def close(self):
        self.live.close()


class RequestTrace:
    def __init__(self, store, path, body):
        self.store = store
        self.identifier = uuid.uuid4().hex
        self.directory = store.session / self.identifier
        self.directory.mkdir(mode=0o700)
        self.started = time.time()
        self.failed = False
        self.path = path
        raw = self.directory / "request.body"
        raw.write_bytes(body)
        raw.chmod(0o600)
        try:
            request = json.loads(body)
        except (ValueError, UnicodeError):
            request = None
        if request is not None:
            with private_file(self.directory / "request.json", "w") as stream:
                json.dump(request, stream, ensure_ascii=False, indent=2)
        self.streams = {
            name: private_file(self.directory / name, "w")
            for name in ("events.jsonl", "reasoning.txt", "output.txt", "tools.jsonl")
        }
        self.wire = (self.directory / "response.http").open("wb")
        (self.directory / "response.http").chmod(0o600)
        store.message(f"REQUEST {self.identifier} · {path}")
        store.text(self.identifier, "INPUT", body.decode("utf-8", errors="replace"))

    def event(self, event):
        stream = self.streams["events.jsonl"]
        stream.write(json_text({"time": time.time(), "event": event}) + "\n")
        stream.flush()
        if not isinstance(event, dict):
            return
        kind = event.get("type", "")
        if kind == "response.reasoning_text.delta":
            self.text("reasoning.txt", "THINKING", event.get("delta", ""))
        elif kind == "response.output_text.delta":
            self.text("output.txt", "OUTPUT", event.get("delta", ""))
        elif kind == "response.output_item.done":
            item = event.get("item", {})
            if item.get("type") in ("function_call", "custom_tool_call"):
                self.tool(item)
        elif "choices" in event:
            for choice in event["choices"]:
                delta = choice.get("delta", {})
                self.text("reasoning.txt", "THINKING", delta.get("reasoning_content", ""))
                self.text("output.txt", "OUTPUT", delta.get("content", ""))
                if delta.get("tool_calls"):
                    self.tool(delta["tool_calls"])

    def text(self, filename, channel, value):
        if not value:
            return
        self.streams[filename].write(value)
        self.streams[filename].flush()
        self.store.text(self.identifier, channel, value)

    def tool(self, item):
        stream = self.streams["tools.jsonl"]
        stream.write(json_text(item) + "\n")
        stream.flush()
        self.store.text(self.identifier, "TOOL CALL", json_text(item) + "\n")

    def response(self, code, value):
        with private_file(self.directory / "response.json", "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        if code >= 400:
            self.store.text(self.identifier, "ERROR", json_text(value) + "\n")

    def finish(self, status, record):
        metadata = {
            "id": self.identifier, "path": self.path,
            "started_at": self.started, "finished_at": time.time(),
            "http_status": status, "trace_failed": self.failed,
            "state": "error" if status is None or status >= 400 else "completed",
            **{k: v for k, v in (record or {}).items()
               if k in ("state", "outcome", "error", "usage", "timings", "wallclock_s")},
        }
        try:
            temporary = self.directory / "metadata.tmp"
            with private_file(temporary, "w") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
            temporary.replace(self.directory / "metadata.json")
            self.store.message(f"REQUEST END {self.identifier} · HTTP {status}")
        finally:
            for stream in self.streams.values():
                stream.close()
            self.wire.close()


class ResponseCapture:
    def __init__(self, original, trace):
        self.original, self.trace = original, trace

    def write(self, data):
        try:
            self.trace.wire.write(data)
            self.trace.wire.flush()
        except OSError as exc:
            if not self.trace.failed:
                print(f"[codex trace] HTTP capture failed: {exc}", file=sys.stderr, flush=True)
            self.trace.failed = True
        return self.original.write(data)

    def __getattr__(self, name):
        return getattr(self.original, name)


def tracing_handler(base, store):
    class Handler(base):
        trace = None
        trace_status = None

        def _save(self, method, *args):
            if self.trace is None:
                return
            try:
                getattr(self.trace, method)(*args)
            except OSError as exc:
                if not self.trace.failed:
                    print(f"[codex trace] write failed: {exc}", file=sys.stderr, flush=True)
                self.trace.failed = True

        def _body(self):
            body = super()._body()
            if self.trace is None and self.path.split("?")[0].rstrip("/") in API_PATHS:
                try:
                    self.trace = store.begin(self.path, body)
                    self.wfile = ResponseCapture(self.wfile, self.trace)
                except OSError as exc:
                    print(f"[codex trace] capture could not start: {exc}", file=sys.stderr, flush=True)
            return body

        def send_response(self, code, message=None):
            self.trace_status = code
            return super().send_response(code, message)

        def _json(self, code, obj):
            self._save("response", code, obj)
            return super()._json(code, obj)

        def _capture(self, items, api):
            def recorded():
                try:
                    for item in items:
                        if item is not None:
                            self._save("event", item)
                        yield item
                finally:
                    items.close()
            return super()._capture(recorded(), api)

        def do_POST(self):
            self.trace = None
            self.trace_status = None
            original = self.wfile
            try:
                return super().do_POST()
            finally:
                self._save("finish", self.trace_status, self.record)
                self.wfile = original
                self.trace = None

    return Handler


class Tee:
    def __init__(self, terminal, stream, lock):
        self.terminal, self.stream, self.lock = terminal, stream, lock

    def write(self, text):
        with self.lock:
            self.stream.write(text)
            self.stream.flush()
            return self.terminal.write(text)

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def main():
    sys.path.insert(0, str(ROOT / "strata"))
    from serve import server

    root = Path(os.environ.get("CODEX_TRACE_DIR", str(TASK_DIR / "codex-traces")))
    store = TraceStore(root)
    factory = server.make_handler
    server.make_handler = lambda svc: tracing_handler(factory(svc), store)
    os.environ["STRATA_DEBUG"] = "1"  # Strata also logs generated text before API parsing.
    log = private_file(TASK_DIR / "server-codex.log")
    lock = threading.Lock()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_stdout, log, lock), Tee(old_stderr, log, lock)
    try:
        port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8095"
        print(f"[codex trace] Monitor: http://127.0.0.1:{port}/api-monitor", flush=True)
        print(f"[codex trace] Full files: {store.session}", flush=True)
        print(f"[codex trace] Live log: {root / 'live.log'}", flush=True)
        return server.main()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log.close()
        store.close()


if __name__ == "__main__":
    sys.exit(main())
