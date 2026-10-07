"""Compare CPU Small MF4 / GPU Large RF40 via llm-chat, optionally during a real LLM response.

Run in llm-chat's environment; does not load models into this process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


async def checked(client, method, path, **kwargs):
    response = await client.request(method, path, **kwargs)
    response.raise_for_status()
    return response


def resources(pid):
    result = {}
    for key, path, fields in (
        ("tts", Path(f"/proc/{pid}/status"), {"VmRSS", "VmHWM"}),
        (
            "wsl",
            Path("/proc/meminfo"),
            {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"},
        ),
    ):
        if path.is_file():
            values = {}
            for line in path.read_text().splitlines():
                name, _, value = line.partition(":")
                if name in fields:
                    values[name + "_kib"] = int(value.strip().split()[0])
            result[key] = values
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            result["whole_gpu_used_mib"] = [
                int(value) for value in proc.stdout.splitlines()
            ]
    return result


async def llm_response(client, model):
    started = time.perf_counter()
    first = None
    timings = None
    async with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "CPUとGPUの役割について、日本語で詳しく説明してください。",
                }
            ],
            "enable_thinking": False,
            "max_tokens": 256,
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            content = ((event.get("choices") or [{}])[0].get("delta") or {}).get(
                "content"
            )
            if content and first is None:
                first = time.perf_counter() - started
            timings = event.get("timings") or timings
    return {
        "first_token_seconds": first,
        "elapsed_seconds": time.perf_counter() - started,
        "timings": timings,
    }


async def run(args):
    if args.concurrent_chat and args.modes != "chat":
        raise ValueError("Large GPU unloads the LLM; use --modes chat with --concurrent-chat")
    texts = json.loads(args.texts.read_text())[: args.count]
    if not texts or any(
        not isinstance(text, str) or not 1 <= len(text) <= 200 for text in texts
    ):
        raise ValueError("Use a JSON array of 1–200 character sentences")
    args.output.mkdir(parents=True, exist_ok=False)
    async with httpx.AsyncClient(
        base_url=args.url, timeout=900, trust_env=False
    ) as client:
        initial = (await checked(client, "GET", "/api/voice/runtime")).json()
        library = (
            await checked(client, "GET", f"/api/voice/library/{args.voice}/metadata")
        ).json()
        status = (await checked(client, "GET", "/api/status")).json()
        if args.concurrent_chat and status["status"] != "ready":
            raise ValueError("Load a chat model before --concurrent-chat")
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "voice": library,
            "runs": [],
        }
        try:
            if args.concurrent_chat:
                await checked(
                    client, "PUT", "/api/voice/runtime", json={"enabled": False}
                )
                report["llm_without_tts"] = await llm_response(client, status["active"])
            for mode in args.modes.split(","):
                if mode not in ("chat", "design"):
                    raise ValueError("Modes must be chat,design")
                started = time.perf_counter()
                await checked(
                    client,
                    "PUT",
                    "/api/voice/runtime",
                    json={"enabled": True, "engine": "irodori", "mode": mode},
                )
                samples = []
                deadline = time.monotonic() + 900
                while True:
                    runtime = (
                        await checked(client, "GET", "/api/voice/runtime")
                    ).json()
                    samples.append(
                        await asyncio.to_thread(resources, runtime.get("pid"))
                    )
                    if runtime["errors"]:
                        raise RuntimeError(runtime["errors"])
                    if runtime["ready"]:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("Irodori startup timed out")
                    await asyncio.sleep(0.5)
                startup = {
                    "mode": mode,
                    "seconds_to_ready": time.perf_counter() - started,
                    "runtime": runtime,
                    "resource_samples": samples,
                }
                report["runs"].append(startup)
                for index, text in enumerate(texts):
                    samples = []
                    synthesis = asyncio.create_task(
                        checked(
                            client,
                            "POST",
                            f"/api/voice/library/{args.voice}/preview",
                            json={
                                "text": text,
                                "mode": mode,
                                "preset": args.preset,
                                "seed": args.seed,
                            },
                        )
                    )
                    llm = (
                        asyncio.create_task(llm_response(client, status["active"]))
                        if args.concurrent_chat
                        else None
                    )
                    try:
                        while not synthesis.done() or (llm and not llm.done()):
                            samples.append(
                                await asyncio.to_thread(resources, runtime.get("pid"))
                            )
                            await asyncio.sleep(0.2)
                        audio = await synthesis
                        # Audio completion is separate from a potentially longer LLM response.
                        elapsed = audio.elapsed.total_seconds()
                        filename = f"{mode}-{index:02}.wav"
                        (args.output / filename).write_bytes(audio.content)
                        row = {
                            "mode": mode,
                            "index": index,
                            "text": text,
                            "wav": filename,
                            "request_seconds": elapsed,
                            "tts": json.loads(audio.headers["X-TTS-Metrics"]),
                            "llm": await llm if llm else None,
                            "resource_samples": samples,
                            "human_review": {
                                "reading_errors": None,
                                "noise": None,
                                "speaker_match": None,
                                "extra_tail": None,
                            },
                        }
                        report["runs"].append(row)
                        (args.output / "report.json").write_text(
                            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                        )
                        print(
                            f"{mode} {index + 1}/{len(texts)}: RTF={row['tts']['rtf']:.3f}",
                            flush=True,
                        )
                    finally:
                        pending = [
                            task
                            for task in (synthesis, llm)
                            if task and not task.done()
                        ]
                        for task in pending:
                            task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
        finally:
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            await checked(
                client,
                "PUT",
                "/api/voice/runtime",
                json={
                    "enabled": initial["enabled"],
                    "engine": initial["engine"],
                    "mode": initial["mode"],
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5070")
    parser.add_argument(
        "--voice", required=True, help="Registered voice's permanent ID"
    )
    parser.add_argument(
        "--texts", type=Path, default=Path(__file__).with_name("evaluation-texts.json")
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--modes", default="chat,design")
    parser.add_argument("--preset", default="neutral")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrent-chat", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "evaluation"
        / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
    )
    asyncio.run(run(parser.parse_args()))
