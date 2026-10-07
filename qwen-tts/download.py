"""Pin official Qwen3-TTS 1.7B snapshots, including their speech tokenizers."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = {
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
MODEL_REVISIONS = {
    "design": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
    "base": "fd4b254389122332181a7c3db7f27e918eec64e3",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-models", action="store_true")
    args = parser.parse_args()
    target = ROOT / "install.json"
    previous = (
        json.loads(target.read_text())
        if target.is_file() and not args.update_models
        else {}
    )
    result = {"engine": "qwen", "source_revision": REVISION, "models": {}}
    for key, repo in MODELS.items():
        revision = previous.get("models", {}).get(key, {}).get("revision") or (
            HfApi().model_info(repo).sha if args.update_models else MODEL_REVISIONS[key]
        )
        print(f"Downloading {repo}@{revision}", flush=True)
        directory = Path(
            snapshot_download(repo, revision=revision, local_dir=ROOT / "models" / key)
        )
        result["models"][key] = {
            "repo": repo,
            "revision": revision,
            "weights": {
                str(p.relative_to(directory)): digest(p)
                for p in sorted(directory.rglob("*.safetensors"))
            },
        }
    result["dependencies"] = {
        name: importlib.metadata.version(name)
        for name in ("qwen-tts", "torch", "torchaudio", "transformers")
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target)
    print(f"Installed: {target}", flush=True)


if __name__ == "__main__":
    main()
