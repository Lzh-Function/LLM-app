"""Exercise the Codex trace wrapper against Strata's real HTTP path and mock engine."""

import contextlib
import http.client
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "strata"))
from serve import server
from serve.frontend import ChatTemplate

spec = importlib.util.spec_from_file_location("codex_trace_server", ROOT / "qwen3.8-flash-next" / "trace_server.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)

SCRIPT = "まず確認します。\n</think>\n\n回答です。"
TOOLS = [{"type": "function", "name": "exec_command", "parameters": {
    "type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]
CALL = ('確認します。\n</think>\n<tool_call>\n<function=exec_command>\n'
        '<parameter=cmd>\ncat a.txt\n</parameter>\n</function>\n</tool_call>')


class TraceHTTPTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = trace.TraceStore(Path(self.temp.name) / "traces")
        self.addCleanup(self.store.close)
        tokenizer = server.ByteTokenizer()
        self.engine = server.MockEngine(tokenizer, SCRIPT, max_context=16384)
        self.service = server.Service(
            self.engine, tokenizer, ChatTemplate(ROOT / "strata/serve/chat_template.jinja")
        )
        self.service.api_monitor = True
        factory = server.make_handler
        with patch.object(server, "make_handler", lambda svc: trace.tracing_handler(factory(svc), self.store)):
            self.httpd = server.serve(self.service, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        try:
            raw = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
            with contextlib.redirect_stdout(io.StringIO()):
                connection.request(method, path, body=raw, headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                content = response.read()
            return response.status, content
        finally:
            connection.close()

    def requests(self):
        directories = sorted(p for p in self.store.session.iterdir() if p.is_dir())
        deadline = time.monotonic() + 3
        while not all((p / "metadata.json").is_file() for p in directories):
            if time.monotonic() >= deadline:
                self.fail("Trace did not finish")
            time.sleep(0.01)
        return directories

    def test_stream_preserves_wire_events_and_full_reasoning(self):
        body = {"input": "テスト", "stream": True}
        code, raw = self.request("POST", "/v1/responses", body)
        self.assertEqual(code, 200)
        directory = self.requests()[0]
        self.assertEqual(json.loads((directory / "request.json").read_text()), body)
        events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: ")]
        saved = [json.loads(line)["event"] for line in (directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(saved, events)
        self.assertEqual((directory / "reasoning.txt").read_text(), "まず確認します。\n")
        self.assertEqual((directory / "output.txt").read_text(), "回答です。")
        self.assertEqual((directory / "response.http").read_bytes().split(b"\r\n\r\n", 1)[1], raw)
        metadata = json.loads((directory / "metadata.json").read_text())
        self.assertEqual(metadata["state"], "completed")
        self.assertFalse(metadata["trace_failed"])
        code, monitor = self.request("GET", "/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn(b"monitor.js", monitor)
        self.assertTrue((self.store.root / "latest").is_symlink())
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual((directory / "request.json").stat().st_mode & 0o777, 0o600)

    def test_nonstream_response_and_overlong_inputs_are_saved_without_truncation(self):
        code, raw = self.request("POST", "/v1/responses", {"input": "短文"})
        self.assertEqual(code, 200)
        first = self.requests()[0]
        self.assertEqual(json.loads((first / "response.json").read_text()), json.loads(raw))
        body = {"input": "x" * 300000}
        code, raw = self.request("POST", "/v1/responses", body)
        self.assertEqual(code, 400)
        second = next(p for p in self.requests() if p != first)
        self.assertEqual(json.loads((second / "request.json").read_text()), body)
        self.assertEqual(json.loads((second / "response.json").read_text()), json.loads(raw))
        self.assertGreater((second / "request.body").stat().st_size, 262144)

    def test_tool_calls_and_returned_results_are_preserved(self):
        self.engine.script = self.engine.tok.encode(CALL + server.IM_END, parse_special=True)
        code, raw = self.request("POST", "/v1/responses", {"input": "読んで", "tools": TOOLS})
        self.assertEqual(code, 200, raw)
        response = json.loads(raw)
        first = self.requests()[0]
        calls = [json.loads(line) for line in (first / "tools.jsonl").read_text().splitlines()]
        self.assertEqual(calls[0]["name"], "exec_command")
        self.engine.script = self.engine.tok.encode(SCRIPT + server.IM_END, parse_special=True)
        body = {"input": [{"role": "user", "content": "読んで"}, *response["output"], {
            "type": "function_call_output", "call_id": calls[0]["call_id"], "output": "ファイルの内容"}],
            "tools": TOOLS}
        code, raw = self.request("POST", "/v1/responses", body)
        self.assertEqual(code, 200, raw)
        second = next(p for p in self.requests() if p != first)
        self.assertEqual(json.loads((second / "request.json").read_text()), body)

    def test_log_failure_does_not_break_model_response(self):
        with patch.object(trace.RequestTrace, "event", side_effect=OSError("disk full")), \
                contextlib.redirect_stderr(io.StringIO()):
            code, raw = self.request("POST", "/v1/responses", {"input": "短文"})
        self.assertEqual(code, 200, raw)
        directory = self.requests()[0]
        self.assertTrue(json.loads((directory / "metadata.json").read_text())["trace_failed"])

    def test_unauthorized_request_does_not_record_body(self):
        self.service.api_key = "test-key"
        code, _ = self.request("POST", "/v1/responses", {"input": "秘密"})
        self.assertEqual(code, 401)
        self.assertEqual(self.requests(), [])


class LauncherTest(unittest.TestCase):
    def test_entrypoint_saves_live_trace_and_raw_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "qwen3.8-flash-next/trace_server.py"),
                 "--engine", "mock", "--port", str(port), "--api-monitor", "--script", SCRIPT],
                env={**os.environ, "CODEX_TRACE_DIR": str(root / "traces")},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 10
                while True:
                    if process.poll() is not None:
                        self.fail(process.stdout.read().decode())
                    try:
                        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                        connection.request("GET", "/health")
                        response = connection.getresponse()
                        response.read()
                        connection.close()
                        break
                    except OSError:
                        connection.close()
                        if time.monotonic() >= deadline:
                            self.fail("Mock trace server did not start")
                        time.sleep(0.05)
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                connection.request("POST", "/v1/responses", body=json.dumps({"input": "確認"}),
                                   headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                connection.close()
            finally:
                process.terminate()
                try:
                    output, _ = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    output, _ = process.communicate()
            self.assertEqual(process.returncode, 0, output.decode())
            self.assertIn(b"[strata] raw:", output)
            self.assertIn("THINKING", (root / "traces/live.log").read_text())
            self.assertEqual(len(list((root / "traces/latest").glob("*/request.json"))), 1)


if __name__ == "__main__":
    unittest.main()
