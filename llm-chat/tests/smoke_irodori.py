"""Explicit real-model integration check; uses a temporary library, no LLM load.

uv run python tests/smoke_irodori.py --output ../irodori-tts/evaluation/smoke
"""

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import httpx
from llm_chat import irodori, server, voice
from llm_chat.voice_library import VoiceLibrary
from llm_chat.voice_runtime import port_in_use


async def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    with tempfile.TemporaryDirectory(prefix="irodori smoke ") as temp:
        library = VoiceLibrary(Path(temp))
        with (
            patch.object(voice, "LIBRARY", library),
            patch.object(irodori, "URL", "http://127.0.0.1:18089"),
        ):
            async with server.lifespan(server.app):
                runtime = server.app.state.speech_runtime
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server.app),
                    base_url="http://test",
                    timeout=900,
                ) as client:

                    async def switch(mode):
                        before = runtime.processes.get("irodori")
                        started = time.perf_counter()
                        res = await client.put(
                            "/api/voice/runtime",
                            json={"enabled": True, "engine": "irodori", "mode": mode},
                        )
                        res.raise_for_status()
                        await asyncio.wait_for(
                            asyncio.gather(*runtime.tasks), timeout=900
                        )
                        assert runtime.ready, runtime.errors
                        if before:
                            assert before.returncode is not None
                        state = runtime.snapshot()
                        state["wall_seconds_to_ready"] = time.perf_counter() - started
                        report[f"startup_{mode}"] = state
                        print(
                            f"{mode} ready: {state['wall_seconds_to_ready']:.2f}s",
                            flush=True,
                        )

                    assert not runtime.enabled
                    await switch("design")
                    res = await client.post(
                        "/api/voice/library/candidates",
                        json={
                            "caption": "落ち着いた低めの女性の声。自然で穏やかな話し方。",
                            "seed": 42,
                            "text": "こんにちは。今日は公園をゆっくり散歩しました。木々の間から差し込む光がとてもきれいでした。明日も良い一日になるといいですね。",
                        },
                    )
                    res.raise_for_status()
                    candidate = res.json()["items"][0]
                    identifier = candidate["id"]
                    report["candidate"] = candidate
                    (output / "candidate.wav").write_bytes(
                        library.audio_path(identifier).read_bytes()
                    )
                    registered = await client.patch(
                        f"/api/voice/library/{identifier}",
                        json={"register": True, "name": "検証用の声"},
                    )
                    registered.raise_for_status()
                    assert registered.json()["references"] == candidate["references"]
                    await switch("chat")
                    voices = (await client.get("/api/voice/voices")).json()
                    assert voices[0]["id"] == identifier
                    for index, (text, preset) in enumerate(
                        (
                            (
                                "明日の予定を一緒に確認しましょう。午前十時に駅前で待ち合わせます。",
                                "neutral",
                            ),
                            ("ありがとうございます。とても助かりました。", "happy"),
                        )
                    ):
                        res = await client.post(
                            f"/api/voice/library/{identifier}/preview",
                            json={"text": text, "preset": preset},
                        )
                        res.raise_for_status()
                        (output / f"mf-{index}.wav").write_bytes(res.content)
                        report[f"mf_{index}"] = json.loads(res.headers["X-TTS-Metrics"])
                        print(
                            f"MF preview {index}: {report[f'mf_{index}']['rtf']:.3f} RTF",
                            flush=True,
                        )
                    references = await runtime.irodori.prepare(library, identifier)
                    mtimes = {
                        p: Path(p).stat().st_mtime_ns for p in references["ref_latents"]
                    }
                    runtime.irodori.clear()
                    again = await runtime.irodori.prepare(library, identifier)
                    assert again == references
                    assert {
                        p: Path(p).stat().st_mtime_ns for p in again["ref_latents"]
                    } == mtimes
                    report["persistent_latent_cache_reused"] = True
                    before = runtime.processes["irodori"]
                    report["tts_process_memory"] = Path(
                        f"/proc/{before.pid}/status"
                    ).read_text()
                    await client.put("/api/voice/runtime", json={"enabled": False})
                    assert before.returncode is not None
                    assert not await port_in_use("127.0.0.1", 18089)
                    assert not runtime.ready and not runtime.enabled
                    report["off_released_process"] = True
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"PASS: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
