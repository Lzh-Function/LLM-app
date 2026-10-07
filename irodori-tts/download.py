"""Pin model snapshots and record exact weights and dependency revisions."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "chat": "Aratako/Irodori-TTS-v4.1-Small-MF",
    "large": "Aratako/Irodori-TTS-v4-Large",
    "codec": "Aratako/Semantic-DACVAE-Japanese-32dim",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def installed_bf16(directory, previous, repo, revision):
    """Reuse verified derived weights without fetching their 13 GB FP32 source."""
    derived = previous.get("derived", {})
    if (
        previous.get("repo") != repo
        or previous.get("revision") != revision
        or derived.get("path") != "bf16/model.safetensors"
        or derived.get("precision") != "bf16"
        or derived.get("converter_version") != 1
        or not derived.get("source_sha256")
    ):
        return None
    checkpoint = directory / "bf16" / "model.safetensors"
    identity = {
        key: derived[key] for key in ("source_sha256", "precision", "converter_version")
    }
    try:
        if json.loads(
            checkpoint.with_suffix(".json").read_text()
        ) == identity and sha256(checkpoint) == derived.get("sha256"):
            return derived.copy()
    except (OSError, ValueError):
        pass
    return None


def bf16_checkpoint(directory, source_sha, *, force=False):
    """Convert on CPU so the official loader never stages 13 GB of FP32 on CUDA."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    target = directory / "bf16" / "model.safetensors"
    marker = target.with_suffix(".json")
    identity = {
        "source_sha256": source_sha,
        "precision": "bf16",
        "converter_version": 1,
    }
    if (
        force
        or not target.is_file()
        or not marker.is_file()
        or json.loads(marker.read_text()) != identity
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        print(
            "Converting Large FP32 to BF16 on CPU (no INT8/INT4 quantization)",
            flush=True,
        )
        tensors = {}
        with safe_open(
            directory / "model.safetensors", framework="pt", device="cpu"
        ) as handle:
            metadata = handle.metadata()
            for name in handle.keys():  # noqa: SIM118 (safe_open is not iterable)
                tensor = handle.get_tensor(name)
                tensors[name] = (
                    tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                )
        temporary = target.with_suffix(".tmp")
        save_file(tensors, str(temporary), metadata=metadata)
        temporary.replace(target)
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    return {"path": "bf16/model.safetensors", "sha256": sha256(target), **identity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--update-models", action="store_true", help="Resolve newer model revisions"
    )
    parser.add_argument(
        "--keep-large-fp32",
        action="store_true",
        help="Retain the Large conversion source for explicit FP32 use",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output.parent / "install.json"
    previous = (
        json.loads(target.read_text())
        if target.is_file() and not args.update_models
        else {}
    )
    manifest = {
        "models": {},
        "server_revision": "61012c760f22f7b4a6c21c5c5f8f9e148120b6f9",
    }
    conversion_sources = []
    for mode, repo in MODELS.items():
        installed = previous.get("models", {}).get(mode, {})
        revision = installed.get("revision") or HfApi().model_info(repo).sha
        directory = args.output / mode
        derived = (
            installed_bf16(directory, installed, repo, revision)
            if mode == "large"
            else None
        )
        directory = Path(
            snapshot_download(
                repo,
                revision=revision,
                local_dir=directory,
                ignore_patterns=["model.safetensors"]
                if derived and not args.keep_large_fp32
                else None,
            )
        )
        weights = {}
        for path in sorted(
            p
            for p in directory.rglob("*")
            if p.suffix in (".safetensors", ".pth", ".pt", ".bin")
        ):
            if "bf16" not in path.relative_to(directory).parts:
                weights[str(path.relative_to(directory))] = sha256(path)
        manifest["models"][mode] = {
            "repo": repo,
            "revision": revision,
            "weights": weights,
        }
        if mode == "large":
            derived = derived or bf16_checkpoint(
                directory, weights["model.safetensors"], force=True
            )
            manifest["models"][mode]["derived"] = derived
            manifest["models"][mode]["source_weights"] = {
                "model.safetensors": derived["source_sha256"]
            }
            if not args.keep_large_fp32:
                weights.pop("model.safetensors", None)
                conversion_sources.append(directory / "model.safetensors")
    for name in ("irodori-tts", "torch", "torchao", "torchaudio"):
        dist = importlib.metadata.distribution(name)
        manifest[name] = {
            "version": dist.version,
            "source": dist.read_text("direct_url.json"),
        }
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temp.replace(target)
    for path in conversion_sources:
        path.unlink(missing_ok=True)
    print(f"Installed model revisions and SHA256: {target}")


if __name__ == "__main__":
    main()
