"""Own a local llama.cpp backend, GPU reservation, Responses adapter and full traces."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qwen3.8-flash-next"))
from codex_server import unload_chat
from trace_server import Tee, TraceStore, private_file, tracing_handler
from llama_responses import make_handler


def check_port(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def interrupted(*_):
    raise KeyboardInterrupt


def main(settings):
    name = settings["profile"]
    directory = ROOT / settings["model_directory"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=settings["port"])
    parser.add_argument("--backend-port", type=int, default=settings["backend_port"])
    parser.add_argument("--context", type=int, default=settings["context_window"])
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.port, args.backend_port)) or args.port == args.backend_port:
        parser.error("use two different valid ports")
    if not 4096 <= args.context <= settings["max_context_window"]:
        parser.error(f"context must be between 4096 and {settings['max_context_window']}")
    for port in (args.port, args.backend_port):
        check_port(port)
    unload_chat()
    lock_path = ROOT / "irodori-tts/gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            raise SystemExit("GPUを別のモデルが使用中です: " + lock.read())
        lock.seek(0)
        lock.truncate()
        json.dump({"owner": f"Codex {name}", "pid": os.getpid()}, lock)
        lock.flush()
        trace = TraceStore(Path(os.environ.get("CODEX_TRACE_DIR", str(directory / "codex-traces"))))
        process, httpd = None, None
        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        with (directory / "codex-engine.log").open("a") as log:
            try:
                command = ["bash", str(ROOT / "llm"), "model", directory.name, "--port", str(args.backend_port),
                           "--ctx-size", str(args.context), "--gpu-layers", settings["gpu_layers"],
                           "--fit", settings["fit"], "--cache-type-k", settings["cache_type"],
                           "--cache-type-v", settings["cache_type"], "--reasoning-preserve",
                           "--ubatch-size", str(settings["ubatch_size"]), "--parallel", "1"]
                if not settings["kv_offload"]:
                    command.append("--no-kv-offload")
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(lock.fileno(),))
                backend = f"http://127.0.0.1:{args.backend_port}"
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + 300
                print(f"[{name}] loading {args.context} tokens; engine log: {directory / 'codex-engine.log'}", flush=True)
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} engine exited ({process.returncode}); see {log.name}")
                    try:
                        with opener.open(backend + "/health", timeout=2) as response:
                            if json.load(response).get("status") == "ok":
                                break
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{name} engine startup timed out")
                    time.sleep(1)
                with opener.open(backend + "/props", timeout=5) as response:
                    actual = json.load(response)["default_generation_settings"]["n_ctx"]
                if actual != args.context:
                    raise RuntimeError(f"engine context {actual} differs from requested {args.context}")
                httpd = ThreadingHTTPServer(("127.0.0.1", args.port), tracing_handler(make_handler(backend), trace))
                httpd.daemon_threads = True
                threading.Thread(target=httpd.serve_forever, daemon=True).start()
                print(f"[{name}] ready: http://127.0.0.1:{args.port}/v1; context={actual}", flush=True)
                print(f"[{name}] full traces: {trace.session}; Ctrl+C stops the server and engine", flush=True)
                while process.poll() is None:
                    time.sleep(1)
                raise RuntimeError(f"{name} engine exited ({process.returncode})")
            except KeyboardInterrupt:
                print(f"[{name}] stopping ...", flush=True)
            finally:
                if httpd:
                    httpd.shutdown()
                    httpd.server_close()
                if process and process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                trace.close()
                signal.signal(signal.SIGTERM, previous_handler)
                print(f"[{name}] stopped; engine exited and GPU reservation released", flush=True)


def run(profile):
    settings = json.loads((ROOT / f"codex/{profile}-runtime.json").read_text())
    log = private_file(ROOT / settings["model_directory"] / "server-codex.log")
    terminal_out, terminal_err = sys.stdout, sys.stderr
    log_lock = threading.Lock()
    sys.stdout, sys.stderr = Tee(terminal_out, log, log_lock), Tee(terminal_err, log, log_lock)
    try:
        main(settings)
    finally:
        sys.stdout, sys.stderr = terminal_out, terminal_err
        log.close()


if __name__ == "__main__":
    profile = sys.argv.pop(1) if len(sys.argv) > 1 else ""
    if profile not in ("bonsai", "qwen-9b"):
        raise SystemExit("usage: llama_server.py {bonsai|qwen-9b} [options]")
    run(profile)
