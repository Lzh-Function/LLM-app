"""Check the Codex protocol features absent from Prism's native Responses API."""

import json
import http.client
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llama_responses import ROOT, chat_request, make_handler, response_events

sys.path.insert(0, str(ROOT / "qwen3.8-flash-next"))
from trace_server import TraceStore, tracing_handler
from bonsai_server import check_port


TOOLS = [{"type": "namespace", "name": "functions", "tools": [
    {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
    {"type": "custom", "name": "apply_patch"},
]}]


def chunk(delta=None, finish=None, **extra):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}], **extra}


class ResponsesTest(unittest.TestCase):
    def test_split_function_and_custom_calls_round_trip_with_reasoning(self):
        request = {"input": "edit", "tools": TOOLS, "include": ["reasoning.encrypted_content"]}
        _, names, _ = chat_request(request)
        patch = "*** Begin Patch\n*** End Patch"
        chunks = [chunk({"reasoning_content": "First inspect."}), chunk({"tool_calls": [
            {"index": 0, "id": "cmd1", "function": {"name": "functions.exec_command", "arguments": '{"cmd":'}},
            {"index": 1, "id": "patch1", "function": {"name": "functions__apply_patch", "arguments": json.dumps({"input": patch})}},
        ]}), chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"pwd"}'}}]}),
            chunk(finish="tool_calls", usage={"prompt_tokens": 100, "completion_tokens": 20,
                                             "prompt_tokens_details": {"cached_tokens": 40}})]
        events = list(response_events(request, names, chunks))
        final = events[-1]["response"]
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["usage"]["input_tokens"], 100)
        calls = [item for item in final["output"] if item["type"].endswith("tool_call") or item["type"] == "function_call"]
        self.assertEqual(calls[0]["namespace"], "functions")
        self.assertEqual(calls[0]["name"], "exec_command")
        self.assertEqual(calls[1]["type"], "custom_tool_call")
        self.assertEqual(calls[1]["input"], patch)
        history = [{"role": "user", "content": "edit"}, *final["output"],
                   {"type": "custom_tool_call_output", "call_id": "patch1", "output": "patched"},
                   {"type": "function_call_output", "call_id": "cmd1", "output": "/tmp"}]
        body, _, _ = chat_request({**request, "input": history})
        assistant = body["messages"][1]
        self.assertEqual(assistant["reasoning_content"], "First inspect.")
        self.assertEqual([m["content"] for m in body["messages"][2:]], ["/tmp", "patched"])
        self.assertEqual([m["tool_call_id"] for m in body["messages"][2:]],
                         [c["id"] for c in assistant["tool_calls"]])
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"cmd": "pwd"})

    def test_image_and_hosted_search_are_translated(self):
        body, _, skipped = chat_request({"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "describe"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc"}]}],
            "tools": [{"type": "web_search"}], "reasoning": {"effort": "none"}})
        self.assertEqual(body["messages"][0]["content"][1],
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}})
        self.assertEqual(skipped, ["web_search"])
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

    def test_length_limit_does_not_execute_partial_tool(self):
        request = {"input": "hi", "tools": TOOLS}
        _, names, _ = chat_request(request)
        events = list(response_events(request, names, [chunk({"tool_calls": [
            {"index": 0, "function": {"name": "functions.exec_command", "arguments": '{"cmd":'}}]}, "length")]))
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertEqual(events[-1]["response"]["output"], [])

    def test_backend_error_is_reported(self):
        events = list(response_events({"input": "hi"}, {}, [{"error": "GPU failed"}]))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertIn("GPU failed", str(events[-1]["response"]["error"]))


class HTTPTraceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = TraceStore(Path(self.directory.name))
        self.finished = threading.Event()
        begin = self.store.begin

        def begin_with_notification(*args):
            trace = begin(*args)
            finish = trace.finish

            def finish_with_notification(*args):
                try:
                    finish(*args)
                finally:
                    self.finished.set()

            trace.finish = finish_with_notification
            return trace

        self.store.begin = begin_with_notification
        self.upstream_body = None
        self.upstream_status = 200
        test = self

        class Backend(BaseHTTPRequestHandler):
            def do_POST(self):
                test.upstream_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(test.upstream_status)
                self.end_headers()
                if test.upstream_status != 200:
                    self.wfile.write(b'{"error":{"message":"context exceeded"}}')
                    return
                for value in [chunk({"reasoning_content": "Inspecting."}), chunk({"content": "確認OK"}),
                              chunk(finish="stop", usage={"prompt_tokens": 99, "completion_tokens": 10},
                                    timings={"predicted_per_second": 42.5})]:
                    self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *_):
                pass

        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        address = f"http://127.0.0.1:{self.backend.server_port}"
        self.api = ThreadingHTTPServer(("127.0.0.1", 0), tracing_handler(make_handler(address), self.store))
        for server in (self.backend, self.api):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    def tearDown(self):
        for server in (self.api, self.backend):
            server.shutdown()
            server.server_close()
        self.store.close()

    def post(self, request):
        self.finished.clear()
        connection = http.client.HTTPConnection("127.0.0.1", self.api.server_port, timeout=10)
        connection.request("POST", "/v1/responses", json.dumps(request), {"Content-Type": "application/json"})
        response = connection.getresponse()
        code, data = response.status, response.read()
        connection.close()
        self.assertTrue(self.finished.wait(timeout=5), "trace finalization did not finish")
        return code, data

    def test_stream_and_nonstream_capture_input_reasoning_output_http_and_timings(self):
        for streaming in (True, False):
            code, data = self.post({"input": "hi", "stream": streaming})
            self.assertEqual(code, 200)
            self.assertIn("確認OK", data.decode() if not streaming else
                          str([json.loads(line[5:]) for line in data.decode().splitlines() if line.startswith("data:")]))
        self.assertTrue(self.upstream_body["stream"])
        for directory in self.store.session.iterdir():
            self.assertEqual(json.loads((directory / "request.json").read_text())["input"], "hi")
            self.assertEqual((directory / "reasoning.txt").read_text(), "Inspecting.")
            self.assertEqual((directory / "output.txt").read_text(), "確認OK")
            self.assertIn(b"HTTP/1.0 200", (directory / "response.http").read_bytes())
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["usage"]["input_tokens"], 99)
            self.assertEqual(metadata["timings"]["predicted_per_second"], 42.5)
            self.assertFalse(metadata["trace_failed"])

    def test_backend_rejection_retains_status_and_is_traced(self):
        self.upstream_status = 400
        code, data = self.post({"input": "too long", "stream": True})
        self.assertEqual(code, 400)
        self.assertIn("context exceeded", json.loads(data)["error"]["message"])
        directory = next(self.store.session.iterdir())
        self.assertEqual(json.loads((directory / "metadata.json").read_text())["state"], "error")

    def test_port_check_rejects_listener_and_allows_immediate_restart(self):
        port = self.api.server_port
        with self.assertRaises(OSError):
            check_port(port)
        self.post({"input": "hi"})
        self.api.shutdown()
        self.api.server_close()
        check_port(port)


if __name__ == "__main__":
    unittest.main()
