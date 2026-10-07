"""Launch the official server with the same profile definitions used by the UI."""

import argparse
import os
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "llm-chat" / "src"))
from llm_chat.irodori_profiles import profile_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("chat", "design", "synthesize"), default="chat"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    profile = profile_for(args.mode)
    checkpoint = profile.checkpoint(TASK_DIR.parent)
    if not checkpoint.is_file():
        raise SystemExit(
            "Run bash irodori-tts/setup.sh first (including the Large BF16 conversion)"
        )
    os.environ.update(
        {
            "IRODORI_CHECKPOINT": str(checkpoint),
            "IRODORI_HF_CHECKPOINT": profile.repo,
            "IRODORI_DEFAULT_NUM_STEPS": str(profile.steps),
            "IRODORI_CODEC_REPO": str(TASK_DIR / "models" / "codec" / "weights.pth"),
            "IRODORI_MODEL_DEVICE": profile.model_device,
            "IRODORI_MODEL_PRECISION": profile.model_precision,
            "IRODORI_CODEC_DEVICE": profile.codec_device,
            "IRODORI_CODEC_PRECISION": profile.codec_precision,
            "IRODORI_MODEL_NAME": os.environ.get(
                "LLM_IRODORI_MODEL_NAME", "irodori-tts"
            ),
            "IRODORI_VOICES_DIR": os.environ.get(
                "IRODORI_VOICES_DIR", str(TASK_DIR / "voices" / profile.model_key)
            ),
            "IRODORI_PRELOAD": "false",
            "IRODORI_MAX_CONCURRENT_SYNTHESIS": "1",
            "IRODORI_ALLOW_NO_REF_VOICE": "true",
            "OMP_NUM_THREADS": os.environ.get("IRODORI_CPU_THREADS", "4"),
            "MKL_NUM_THREADS": os.environ.get("IRODORI_CPU_THREADS", "4"),
        }
    )
    if os.environ.get("LLM_IRODORI_API_KEY"):
        os.environ["IRODORI_API_KEY"] = os.environ["LLM_IRODORI_API_KEY"]
    sys.path.insert(0, str(TASK_DIR))
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
    os.chdir(TASK_DIR / "runtime")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "service:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
    )


if __name__ == "__main__":
    main()
