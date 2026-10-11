"""Adapt Codex Responses tools and history to llama.cpp Chat Completions."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "strata"))
from serve.frontend import Event, ToolCall
from serve.responses import Assembler, ResponsesError, check_request, input_messages, request_tools, template_kwargs, text_format


def chat_request(request):
    check_request(request)
    messages = input_messages(request)
    pending = deque()
    for index, message in enumerate(messages):
        if isinstance(message.get("content"), list):
            message["content"] = [
                {"type": "image_url", "image_url": {"url": p["source"]}}
                if p["type"] == "image" else p for p in message["content"]
            ]
        if message.get("tool_calls"):
            pending.clear()
            for number, call in enumerate(message["tool_calls"]):
                call["id"] = f"call_{index}_{number}"
                call["type"] = "function"
                pending.append(call["id"])
                call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
        elif message["role"] == "tool":
            if not pending:
                raise ResponsesError("tool result has no preceding call", "input")
            message["tool_call_id"] = pending.popleft()
    tools, names, skipped = request_tools(request)
    body = {
        "model": request.get("model", "ternary-bonsai-2-27b-abliterated"),
        "messages": messages, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": template_kwargs(request, {"reasoning_effort": "medium"}),
        "max_tokens": request.get("max_output_tokens", 16384),
    }
    if tools:
        body["tools"] = [{"type": "function", "function": tool} for tool in tools]
    for key in ("temperature", "top_p", "parallel_tool_calls"):
        if key in request:
            body[key] = request[key]
    choice = request.get("tool_choice")
    if isinstance(choice, str):
        body["tool_choice"] = choice
    elif isinstance(choice, dict) and choice.get("type") == "function":
        name = choice["name"]
        if choice.get("namespace"):
            name = f"{choice['namespace']}.{name}"
        body["tool_choice"] = {"type": "function", "function": {"name": name}}
    fmt = text_format(request)
    if fmt:
        body["response_format"] = fmt
    return body, names, skipped


def response_events(request, names, chunks):
    assembler = Assembler(request, request.get("model", "ternary-bonsai-2-27b-abliterated"), 0, names,
                          "reasoning.encrypted_content" in (request.get("include") or []))
    yield from assembler.start()
    calls, usage, timings = {}, {}, {}
    finish = "stop"
    try:
        for chunk in chunks:
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            usage.update(chunk.get("usage") or {})
            timings.update(chunk.get("timings") or {})
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                for field, kind in (("reasoning_content", "reasoning"), ("content", "content")):
                    if delta.get(field):
                        yield from assembler.feed(Event(kind, text=delta[field]))
                for call in delta.get("tool_calls") or []:
                    target = calls.setdefault(call.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        target["id"] = call["id"]
                    function = call.get("function") or {}
                    target["name"] += function.get("name") or ""
                    args = function.get("arguments") or ""
                    target["arguments"] += args if isinstance(args, str) else json.dumps(args)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        if finish != "length":
            for call in calls.values():
                arguments = json.loads(call["arguments"] or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                tool = ToolCall(call["name"], arguments)
                if call["id"]:
                    tool.id = call["id"]
                yield from assembler.feed(Event("tool_call", call=tool))
        final_events = assembler.finish({
            "finish": finish, "prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n", 0)),
            "completion_tokens": usage.get("completion_tokens", timings.get("predicted_n", 0)),
            "reused": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", timings.get("cache_n", 0)),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        })
        for event in final_events:
            if "response" in event:
                event["response"]["timings"] = timings
            yield event
    except Exception as exc:
        yield assembler.failed(str(exc))


def sse_chunks(response):
    for line in response:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        yield json.loads(data)
    raise OSError("backend stream ended without [DONE]")


def make_handler(backend):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    class Handler(BaseHTTPRequestHandler):
        record = None

        def _body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", "0")))

        def _json(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _capture(self, events, api):
            return events

        def do_GET(self):
            path = self.path.split("?")[0]
            if path not in ("/health", "/props", "/v1/models"):
                self._json(404, {"error": "not found"})
                return
            try:
                with opener.open(backend + path, timeout=5) as response:
                    self._json(response.status, json.load(response))
            except (OSError, ValueError) as exc:
                self._json(503, {"error": str(exc)})

        def do_POST(self):
            self.record = {}
            streaming = False
            if self.path.rstrip("/") != "/v1/responses":
                self._json(404, {"error": "not found"})
                return
            try:
                request = json.loads(self._body())
                body, names, skipped = chat_request(request)
                if skipped:
                    print("[llama] hosted tools omitted: " + ", ".join(skipped), flush=True)
                upstream = urllib.request.Request(backend + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                  headers={"Content-Type": "application/json"})
                # Open before sending headers so backend errors retain their HTTP status.
                with opener.open(upstream, timeout=1800) as response:
                    events = self._capture(response_events(request, names, sse_chunks(response)), "responses")
                    if request.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        streaming = True
                        for event in events:
                            if "response" in event:
                                self.record = {**event["response"], "state": event["response"]["status"]}
                            self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode())
                            self.wfile.flush()
                    else:
                        for event in events:
                            if "response" in event:
                                self.record = {**event["response"], "state": event["response"]["status"]}
                        self._json(200, self.record)
            except urllib.error.HTTPError as exc:
                try:
                    error = json.loads(exc.read())
                except ValueError:
                    error = {"error": {"message": str(exc)}}
                self._json(exc.code, error)
            except ResponsesError as exc:
                self._json(exc.status, exc.body())
            except (ValueError, OSError) as exc:
                if streaming:
                    self.record.update(state="error", error=str(exc))
                    print(f"[llama API] stream disconnected: {exc}", flush=True)
                else:
                    self._json(502, {"error": {"message": str(exc)}})

        def log_message(self, format, *args):
            print("[llama API] " + format % args, flush=True)

    return Handler
