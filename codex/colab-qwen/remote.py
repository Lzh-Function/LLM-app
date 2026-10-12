"""Colab bootstrap: persistent weights/pack on mounted Drive, build in /content."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

WORK = Path('/content/strata-qwen')
DRIVE = Path('/content/drive')


def run(*args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def ensure_drive(data):
    if not DRIVE.is_mount() or not (DRIVE / 'MyDrive').is_dir():
        raise RuntimeError('GDrive is not mounted. Run colab drivemount -s strata-qwen in your terminal.')
    data = data.resolve()
    if not data.is_relative_to((DRIVE / 'MyDrive').resolve()):
        raise RuntimeError('Model destination must be inside /content/drive/MyDrive; SSD fallback is disabled.')
    data.mkdir(parents=True, exist_ok=True)
    probe = data / '.write-test'
    probe.write_text('ok')
    probe.unlink()
    return data


def check_server_port(port):
    if not isinstance(port, int) or not 1 <= port <= 65535 or port == 8080:
        raise RuntimeError('Use a dedicated Strata port, not Colab/Jupyter port 8080.')
    try:
        with socket.socket() as stream:
            stream.bind(('127.0.0.1', port))
    except OSError as exc:
        raise RuntimeError(f'Strata port {port} is unavailable; refusing to build/download first.') from exc


def select_variant(manifest, requested):
    """Resolve the startup model variant; every download and pack path comes from it."""
    options = manifest['variants']
    name = requested or manifest.get('default_variant')
    if name not in options:
        raise RuntimeError(f'Unknown model variant {name!r}; choose from: ' + ', '.join(sorted(options)))
    return name, options[name]


def check_drive_space(data, profile):
    missing = 0
    for item in profile['files']:
        path = data / item['destination']
        partial = path.with_name(path.name + '.part')
        present = path.stat().st_size if path.is_file() else partial.stat().st_size if partial.is_file() else 0
        missing += max(0, item['size'] - present)
        if item['destination'].endswith('.gguf'):
            print(f'GGUF cache: {path.name}: {present / 1e9:.2f}/{item["size"] / 1e9:.2f} GB', flush=True)
    # Leave space for the dense pack, tokenizer and small engine artifacts.
    reserve = 128 * 1024 * 1024 if (data / profile['pack'] / '.complete.json').is_file() else 5_000_000_000
    free = shutil.disk_usage(data).free
    print(f'Drive reported free space: {free / 1e9:.2f} GB; need about {(missing + reserve) / 1e9:.2f} GB', flush=True)
    if free < missing + reserve:
        raise RuntimeError('Insufficient Drive free space for the missing weights and pack; download not started.')


def fetch_verified(url, path, size, sha256, *, reverify=False):
    """Resume in destination directory, validate range and hash before publishing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_name(path.name + '.verified.json')
    if path.exists():
        stat = path.stat()
        expected = {'sha256': sha256, 'size': size, 'mtime_ns': stat.st_mtime_ns}
        if not reverify and stat.st_size == size and marker.is_file():
            try:
                if json.loads(marker.read_text()) == expected:
                    print('Verified cache:', path.name, flush=True)
                    return
            except ValueError:
                pass
        if stat.st_size != size or digest(path) != sha256:
            raise RuntimeError(f'Existing file fails verification: {path}. Move it aside before retrying.')
    else:
        partial = path.with_name(path.name + '.part')
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise RuntimeError(f'Partial file is too large: {partial}')
            if offset == size:
                break
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if offset:
                        value = response.headers.get('Content-Range', '')
                        if response.status != 206 or not value.startswith(f'bytes {offset}-'):
                            raise RuntimeError('Server did not honor resume Range; keeping partial file unchanged.')
                    elif response.status != 200:
                        raise RuntimeError(f'Unexpected download status: {response.status}')
                    print(f'Download {path.name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB', flush=True)
                    last_report = time.monotonic()
                    with partial.open('ab') as output:
                        while block := response.read(8 * 1024 * 1024):
                            if offset + len(block) > size:
                                raise RuntimeError('Download exceeds pinned file size.')
                            output.write(block)
                            offset += len(block)
                            if time.monotonic() - last_report >= 20:
                                output.flush()
                                print(f'{path.name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB', flush=True)
                                last_report = time.monotonic()
                if offset == size:
                    break
                raise OSError(f'Incomplete download: {offset}/{size}')
            except (OSError, urllib.error.URLError) as exc:
                if attempt == 5:
                    raise
                print(f'Download retry {attempt + 1}: {exc}', flush=True)
                time.sleep(min(5 * (attempt + 1), 20))
        print('SHA-256 verification:', path.name, flush=True)
        if partial.stat().st_size != size or digest(partial) != sha256:
            raise RuntimeError(f'SHA-256 mismatch: {partial}. Move it aside before retrying.')
        partial.replace(path)
    atomic_json(marker, {'sha256': sha256, 'size': size, 'mtime_ns': path.stat().st_mtime_ns})
    print('Downloaded and SHA-256 verified (Drive cloud sync is separate):', path.name, flush=True)


def source_tree(repo, commit, destination):
    if destination.is_dir():
        return
    archive = WORK / (repo.replace('/', '-') + '.tar.gz')
    urllib.request.urlretrieve(f'https://codeload.github.com/{repo}/tar.gz/{commit}', archive)
    staging = WORK / (destination.name + '-extract')
    staging.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        # Explicit checks work on Colab Python versions without tarfile's filter=.
        for member in bundle.getmembers():
            if member.issym() or member.islnk() or not (staging / member.name).resolve().is_relative_to(staging.resolve()):
                raise RuntimeError('Unsafe source archive member')
        bundle.extractall(staging)
    top = next(staging.iterdir())
    destination.parent.mkdir(parents=True, exist_ok=True)
    top.rename(destination)
    staging.rmdir()
    archive.unlink()


def engine_config(source, engine, data, profile, context, workers, port):
    # iq_pack.py and the engine resolve sibling shards by name inside the shard's own
    # directory, so each variant keeps its own models/ and packs/ subtree.
    shards = [data / item['destination'] for item in profile['files'][:2]]
    pack = data / profile['pack']
    return {
        'exe': str(engine), 'cwd': str(source),
        'args': ['--pack', str(pack), '--native', str(shards[0]),
                 '--ple-gguf', str(shards[1]), '--ple-io', 'ram',
                 '--expert-profile', str(source / 'data/expert-profile.bin'),
                 '--expert-cache', 'auto', '--prefill', 'auto',
                 '--resident-experts', '--max-context', str(context), '--kv', 'int8',
                 '--spec', '4', '--spec-min-p', '0.5', '--mtp', str(data / 'mtp/rt'),
                 '--pool-workers', str(workers)],
        'tokenizer': str(pack / 'tokenizer'), 'model_name': profile['model'],
        'log': str(data / 'logs/engine.log'), 'host': '127.0.0.1', 'port': port,
        'gpu': 0, 'gpus_asked': True, 'open_browser': False, 'draft_vocab': 'cjk',
        'api_monitor': True, 'env': {'STRATA_RESIDENT_HEADROOM_GIB': '8'},
        'lib_dirs': ['/usr/local/cuda/lib64'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/Strata-data')
    parser.add_argument('--context', type=int, default=131072)
    parser.add_argument('--variant')
    parser.add_argument('--reverify', action='store_true')
    args = parser.parse_args()
    data = ensure_drive(Path(args.data_dir))  # Must precede every weight/cache download.
    WORK.mkdir(exist_ok=True)
    manifest = json.loads((WORK / 'manifest.json').read_text())
    variant, profile = select_variant(manifest, args.variant)
    check_server_port(manifest['server_port'])  # Fail before building or downloading.
    check_drive_space(data, profile)
    print('Model variant:', variant, flush=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader'], text=True)
    print('GPU:', gpu.strip(), flush=True)
    if 'A100' not in gpu or '8.0' not in gpu:
        raise RuntimeError('Expected the requested A100 (sm_80); refusing a different accelerator.')
    ram_kib = int(re.search(r'MemTotal:\s+(\d+)', Path('/proc/meminfo').read_text()).group(1))
    if ram_kib < 96 * 1024 * 1024:
        raise RuntimeError('Expected high-mem RAM >=96 GiB; avoiding Drive expert streaming.')
    nvcc = shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('Colab CUDA toolkit nvcc is missing.')
    cuda_version = subprocess.check_output([nvcc, '--version'], text=True)
    print(cuda_version, flush=True)
    source = WORK / 'Strata'
    source_tree('Niko1221/Strata', manifest['strata_commit'], source)
    python = source / '.venv/bin/python'
    if not python.exists():
        result = subprocess.run([sys.executable, '-m', 'venv', str(source / '.venv')])
        if result.returncode:
            # Colab's system Python can omit ensurepip/python3-venv.
            run(sys.executable, '-m', 'pip', 'install', 'uv')
            run(sys.executable, '-m', 'uv', 'venv', source / '.venv',
                '--python', sys.executable, '--seed', '--allow-existing')
    run(python, '-m', 'pip', 'install', '-r', source / 'requirements.txt')
    llama = source / 'third_party/llama.cpp'
    source_tree('ggml-org/llama.cpp', manifest['llama_commit'], llama)
    # Build artifacts are small; weights and generated packs never leave Drive.
    signature = hashlib.sha256((manifest['strata_commit'] + cuda_version + platform.libc_ver().__repr__()
                                + Path('/proc/cpuinfo').read_text().split('flags', 1)[-1].split('\n', 1)[0]).encode()).hexdigest()[:20]
    cached = data / 'artifacts' / signature / 'strata'
    engine = source / 'engine/strata'
    engine.parent.mkdir(exist_ok=True)
    if cached.is_file():
        shutil.copy2(cached, engine)
        engine.chmod(0o755)
        print('Cached A100 engine:', cached, flush=True)
    else:
        env = os.environ.copy()
        env['PATH'] = str(python.parent) + os.pathsep + env['PATH']
        toolkit_major = int(re.search(r'release (\d+)\.', cuda_version).group(1))
        # Use upstream's toolkit/compiler/CPU detection, without its model downloader.
        code = ('import setup; llama=setup.get_llama_cpp(); '
                f'print(setup.build_engine({{"arch":80,"archs":[80]}},"none",True,llama,toolkit={toolkit_major}))')
        run(python, '-c', code, cwd=source, env=env)
        built = source / ('engine-cuda12' if toolkit_major == 12 else 'engine') / 'strata'
        if built != engine:
            shutil.copy2(built, engine)
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(engine, cached)
    for item in profile['files']:
        url = f'https://huggingface.co/{item["repo"]}/resolve/{item["revision"]}/{item["source"]}'
        fetch_verified(url, data / item['destination'], item['size'], item['sha256'], reverify=args.reverify)
    pack = data / profile['pack']
    pack_marker = pack / '.complete.json'
    pack_identity = {'strata': manifest['strata_commit'], 'gguf': profile['files'][0]['sha256']}
    complete = pack_marker.is_file() and json.loads(pack_marker.read_text()) == pack_identity
    complete = complete and all((pack / name).is_file() for name in ['dense.bin', 'dense.txt', 'native_experts.txt', 'tokenizer/vocab.json'])
    if not complete:
        run(python, source / 'tools/iq_pack.py', '--gguf', data / profile['files'][0]['destination'], '--out', pack)
        atomic_json(pack_marker, pack_identity)
    (data / 'logs').mkdir(exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 1)
    config = engine_config(source, engine, data, profile, args.context, workers, manifest['server_port'])
    config_path = WORK / 'strata.json'
    atomic_json(config_path, config)
    port = manifest['server_port']
    print(f'Starting Strata IQ3_S {variant}, localhost:{port}, context', args.context, flush=True)
    os.chdir(source)
    os.execv(str(python), [str(python), '-m', 'serve.server', '--engine', 'strata', '--config', str(config_path), '--port', str(port)])


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if WORK.is_dir():
            (WORK / 'failed').write_text('Bootstrap/server failed; inspect bootstrap.log\n')
        raise
