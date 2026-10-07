"""Install the pinned SC117 IQ3_XXS into an existing Strata installation.

Downloads resume in place; no Hugging Face cache or experts.bin copy is made.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATA = ROOT / "strata"
DATA = ROOT / "Strata-data"
MODEL = DATA / "models/sc117_iq3_xxs"
PACK = DATA / "packs/sc117_iq3_xxs"
REPO = "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
REVISION = "eec4e10a2c29b440a2ae85591fa11569ce752963"
FILES = [
    ("Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf",
     47342144896, "03b11926b41a3d7f7a008882b1bc7a6330a3c9c43d3cad4598d31dc73042fdc5"),
    ("Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf",
     28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def download(name: str, size: int, digest: str) -> None:
    path = MODEL / name
    marker = path.with_suffix(".gguf.sha256")
    if path.exists():
        if path.stat().st_size != size:
            raise RuntimeError(f"Unexpected file size: {path}")
        if not marker.exists() or marker.read_text().strip() != digest:
            print(f"Verifying SHA-256: {name}", flush=True)
            if sha256(path) != digest:
                raise RuntimeError(f"SHA-256 mismatch: {path}")
            marker.write_text(digest + "\n")
        print(f"Reusing verified file: {name}", flush=True)
        return
    part = path.with_suffix(".gguf.part")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/IQ3_XXS/{name}?download=true"
    print(f"Downloading {size / 1e9:.2f} GB: {name}", flush=True)
    proc = subprocess.Popen([
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "6", "--retry-delay", "3", "--retry-all-errors",
        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
        "--continue-at", "-", "--output", str(part), url,
    ])
    last_report = 0.0
    try:
        while proc.poll() is None:
            now = time.monotonic()
            if now - last_report >= 20:
                received = part.stat().st_size if part.exists() else 0
                print(f"Download: {received / 1e9:.2f}/{size / 1e9:.2f} GB ({received / size:.1%})", flush=True)
                last_report = now
            time.sleep(1)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    if proc.returncode:
        raise RuntimeError(f"Download failed ({proc.returncode}); rerun setup.sh to resume")
    if part.stat().st_size != size:
        raise RuntimeError(f"Incomplete download: {part}")
    print("Verifying downloaded SHA-256", flush=True)
    if sha256(part) != digest:
        raise RuntimeError(f"SHA-256 mismatch: {part}; retained for inspection")
    part.rename(path)
    marker.write_text(digest + "\n")
    print("Downloaded file SHA-256 verified", flush=True)


def install() -> None:
    for p in (STRATA / "engine/strata", STRATA / "engine/strata-vision",
              STRATA / ".venv/bin/python", DATA / "mtp/rt/experts.bin",
              DATA / "models/mmproj-Qwen3.8-Flash-Next-BF16.gguf"):
        if not p.is_file():
            raise RuntimeError(f"Existing Strata engine, MTP and vision installation required: {p}")
    first, second = (MODEL / f[0] for f in FILES)
    subprocess.run([
        str(STRATA / ".venv/bin/python"), str(STRATA / "tools/iq_pack.py"),
        "--gguf", str(first), "--out", str(PACK),
    ], check=True)
    assert not (PACK / "experts.bin").exists(), "An experts.bin copy would waste SSD space"
    # Preserve the old tokenizer only if it is byte-identical to the SC117 export.
    shared = DATA / "shared/qwen3.8-flash-next-tokenizer"
    exported = PACK / "tokenizer"
    if shared.is_dir() and not exported.is_symlink():
        names = {p.name for p in exported.iterdir()}
        if names == {p.name for p in shared.iterdir()} and all(
            sha256(exported / n) == sha256(shared / n) for n in names
        ):
            shutil.rmtree(exported)
            exported.symlink_to(shared, target_is_directory=True)
            print("Reusing byte-identical tokenizer", flush=True)
        else:
            shutil.rmtree(shared)
            print("Tokenizer differs; using SC117 export", flush=True)
    cfg = json.loads((ROOT / "qwen3.8-flash-next/strata-config.json").read_text().replace("@ROOT@", str(ROOT)))
    assert str(first) in cfg["args"] and str(second) in cfg["args"]
    config = STRATA / "strata-sc117-iq3_xxs.json"
    temporary = config.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(config)
    (MODEL / "source.json").write_text(json.dumps({
        "repo": REPO, "revision": REVISION,
        "files": [{"name": n, "bytes": s, "sha256": h} for n, s, h in FILES],
        "shared": ["second shard", "stock mmproj", "stock MTP", "tokenizer if identical"],
    }, indent=2) + "\n")
    print(f"Ready: {config}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    MODEL.mkdir(parents=True, exist_ok=True)
    with (MODEL / ".install.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another SC117 installation is already running") from None
        for entry in FILES:
            download(*entry)
        if not args.download_only:
            install()


if __name__ == "__main__":
    main()
