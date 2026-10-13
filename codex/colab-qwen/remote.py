"""Colab bootstrap: persistent Drive cache, runtime SSD model data, private inference."""
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
SSD_RESERVE = 5_000_000_000
PACK_FILES = (
    'dense.bin', 'index.txt', 'native_experts.txt', 'conversions.json',
    'compat-bf16.json', 'extra.bin', 'experts.bin', 'experts.bin.src.json',
    'tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/token_type.json',
    'tokenizer/tokenizer.json', 'tokenizer/chat_template.jinja', '.complete.json',
)


def privacy_launcher():
    uploaded = Path(__file__).with_name('colab_privacy.py')
    return uploaded if uploaded.is_file() else Path(__file__).resolve().parents[1] / 'colab_privacy.py'


def check_privacy():
    command = [sys.executable, str(privacy_launcher()), '--check']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        lines = result.stdout.splitlines()
        reason = next((line for line in reversed(lines) if line.startswith('Colab privacy guard refused startup:')),
                      '\n'.join(lines[-12:]))
        raise RuntimeError(f'Privacy preflight failed: {reason}')
    print(result.stdout, end='', flush=True)


def serve_private(argv, cwd):
    command = [sys.executable, str(privacy_launcher()), '--cwd', str(cwd), '--', *map(str, argv)]
    os.execv(command[0], command)


def run(*args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    total = path.stat().st_size
    read = 0
    started = last_report = time.monotonic()
    print(f'SHA-256 verification: {path.name}: 0.00/{total / 1e9:.2f} GB', flush=True)
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
            read += len(block)
            now = time.monotonic()
            if now - last_report >= 20:
                elapsed = max(0.001, now - started)
                print(f'SHA-256 {path.name}: {read / 1e9:.2f}/{total / 1e9:.2f} GB '
                      f'({read / elapsed / 1e6:.1f} MB/s)', flush=True)
                last_report = now
    print(f'SHA-256 read complete: {path.name}: {read / 1e9:.2f}/{total / 1e9:.2f} GB', flush=True)
    return h.hexdigest()


def ensure_drive(data):
    if not DRIVE.is_mount() or not (DRIVE / 'MyDrive').is_dir():
        raise RuntimeError('GDrive is not mounted. Run colab drivemount -s strata-qwen in your terminal.')
    data = data.resolve()
    if not data.is_relative_to((DRIVE / 'MyDrive').resolve()):
        raise RuntimeError('Persistent model cache must be inside /content/drive/MyDrive; SSD fallback for this cache is disabled.')
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
    for item in drive_files(profile):
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


def hf_files(profile):
    """Model bodies fetched straight onto the runtime SSD; nothing about them touches Drive."""
    return [item for item in profile['files'] if item.get('fetch') == 'huggingface']


def drive_files(profile):
    """Small parts and generated artifacts that keep using the persistent Drive cache."""
    return [item for item in profile['files'] if item.get('fetch', 'drive') != 'huggingface']


def hf_token():
    """The uploaded HuggingFace token. Its value is never printed, and its absence still works."""
    path = WORK / '.hf_token'
    try:
        value = path.read_text(encoding='ascii').strip()
    except OSError:
        print(f'No HuggingFace token at {path}; downloading anonymously.', flush=True)
        return None
    if not value:
        print(f'HuggingFace token at {path} is empty; downloading anonymously.', flush=True)
        return None
    try:
        path.chmod(0o600)
    except OSError:
        pass
    print(f'HuggingFace token loaded: {path.name} ({len(value)} chars)', flush=True)
    return value


def hf_response(url, offset, token):
    headers = {}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    if token:
        headers['Authorization'] = 'Bearer ' + token
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def fetch_hf_to_ssd(url, destination, size, sha256, token, *, reverify=False):
    """Stream one pinned model body onto the runtime SSD, SHA-256 checked in flight.

    The Drive copy path is gone for model bodies: the only copy inference reads is this one.
    A published file keeps a verification marker, so a server restart on the same runtime
    transfers nothing.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_name(destination.name + '.verified.json')
    if destination.is_file():
        stat = destination.stat()
        if stat.st_size != size:
            raise RuntimeError(f'Existing runtime model file has the wrong size: {destination}. '
                               'Move it aside before retrying.')
        if not reverify and marker.is_file():
            try:
                record = json.loads(marker.read_text())
            except ValueError:
                record = None
            if isinstance(record, dict) and record.get('sha256') == sha256 and record.get('size') == size:
                print('Verified runtime SSD copy (no transfer):', destination.name, flush=True)
                return
        print(f'Runtime SSD copy needs a SHA-256 recheck: {destination.name}', flush=True)
        if digest(destination) != sha256:
            raise RuntimeError(f'Existing runtime model file fails verification: {destination}')
        atomic_json(marker, {'sha256': sha256, 'size': size, 'mtime_ns': destination.stat().st_mtime_ns})
        return
    free = shutil.disk_usage(WORK).free
    if free < size + SSD_RESERVE:
        raise RuntimeError(f'Not enough runtime SSD space for {destination.name}: {free / 1e9:.2f} GB free, '
                           f'need {(size + SSD_RESERVE) / 1e9:.2f} GB.')
    partial = destination.with_name(destination.name + '.part')
    started = time.monotonic()
    computed = None
    for attempt in range(6):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > size:
            raise RuntimeError(f'Partial file is too large: {partial}')
        if offset == size:
            break
        checksum = hashlib.sha256()
        if offset:  # A resumed prefix has to enter the digest before the new bytes do.
            with partial.open('rb') as existing:
                while block := existing.read(8 * 1024 * 1024):
                    checksum.update(block)
        try:
            response = hf_response(url, offset, token)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403) and token is not None:
                exc.close()
                token = None
                print('HuggingFace refused the token; retrying this file anonymously.', flush=True)
                continue
            if exc.code == 429:
                exc.close()
                wait = min(60, max(1, int(exc.headers.get('Retry-After') or 5)))
                print(f'HuggingFace throttled the download (429); waiting {wait}s.', flush=True)
                time.sleep(wait)
                continue
            raise
        try:
            with response, partial.open('ab') as output:
                if offset:
                    value = response.headers.get('Content-Range', '')
                    if response.status != 206 or not value.startswith(f'bytes {offset}-'):
                        raise RuntimeError('Server did not honor resume Range; keeping the partial file.')
                elif response.status != 200:
                    raise RuntimeError(f'Unexpected download status: {response.status}')
                print(f'Download {destination.name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB', flush=True)
                last_report = time.monotonic()
                while block := response.read(8 * 1024 * 1024):
                    if offset + len(block) > size:
                        raise RuntimeError(f'Download exceeds the pinned file size: {destination.name}')
                    output.write(block)
                    checksum.update(block)
                    offset += len(block)
                    now = time.monotonic()
                    if now - last_report >= 20:
                        output.flush()
                        elapsed = max(0.001, now - started)
                        print(f'Download {destination.name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB '
                              f'({offset / elapsed / 1e6:.1f} MB/s)', flush=True)
                        last_report = now
            if offset != size:
                raise OSError(f'Incomplete download: {offset}/{size}')
            computed = checksum.hexdigest()
            break
        except (OSError, urllib.error.URLError) as exc:
            if attempt == 5:
                raise
            print(f'Download retry {attempt + 1}: {exc}', flush=True)
            time.sleep(min(5 * (attempt + 1), 20))
    if partial.stat().st_size != size:
        raise RuntimeError(f'Incomplete download: {partial}')
    if computed is None:
        computed = digest(partial)
    if computed != sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'SHA-256 mismatch: {destination.name}. Partial file discarded; the retry re-downloads.')
    partial.replace(destination)
    atomic_json(marker, {'sha256': sha256, 'size': size, 'mtime_ns': destination.stat().st_mtime_ns})
    elapsed = max(0.001, time.monotonic() - started)
    print(f'Runtime SSD model body ready: {destination.name}: {size / 1e9:.2f} GB in {elapsed:.1f} s '
          f'({size / elapsed / 1e6:.1f} MB/s), SHA-256 verified', flush=True)


def copy_model_file(source, destination, size, *, sha256=None):
    """Sequential copy with optional in-flight SHA check and atomic publication."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free
    if free < size + SSD_RESERVE:
        raise RuntimeError(f'Not enough runtime disk/cache space to copy {source.name}: '
                           f'{free / 1e9:.2f} GB free, need {(size + SSD_RESERVE) / 1e9:.2f} GB.')
    partial = destination.with_name(destination.name + '.part')
    digest = hashlib.sha256() if sha256 is not None else None
    copied = 0
    started = last_report = time.monotonic()
    check = '; SHA-256 checked during copy' if digest else ''
    print(f'Copy model: {source.name}: 0.00/{size / 1e9:.2f} GB{check}', flush=True)
    try:
        with source.open('rb') as incoming, partial.open('wb') as outgoing:
            while block := incoming.read(8 * 1024 * 1024):
                copied += len(block)
                if copied > size:
                    raise RuntimeError(f'Model copy size mismatch: {source}')
                outgoing.write(block)
                if digest:
                    digest.update(block)
                now = time.monotonic()
                if now - last_report >= 20:
                    elapsed = max(0.001, now - started)
                    print(f'Copy model: {source.name}: {copied / 1e9:.2f}/{size / 1e9:.2f} GB '
                          f'({copied / elapsed / 1e6:.1f} MB/s)', flush=True)
                    last_report = now
        if copied != size:
            raise RuntimeError(f'Model copy size mismatch: {source}')
        if digest and digest.hexdigest() != sha256:
            raise RuntimeError(f'SHA-256 mismatch during model copy: {source}')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    elapsed = max(0.001, time.monotonic() - started)
    print(f'Copied model: {source.name}: {copied / 1e9:.2f} GB in {elapsed:.1f} s '
          f'({copied / elapsed / 1e6:.1f} MB/s){check}', flush=True)


def model_tree_files(folder):
    """Known pack artifacts only; exclude logs and reject linked model paths."""
    files = []
    for relative in PACK_FILES:
        path = folder / relative
        parts = Path(relative).parts
        if any(folder.joinpath(*parts[:depth]).is_symlink() for depth in range(len(parts) + 1)):
            raise RuntimeError(f'Model pack symlink is not supported: {path}')
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda p: (p.name == '.complete.json', str(p.relative_to(folder))))


def copy_model_tree(source, destination):
    # Publish the pack completion marker last, after every payload is copied.
    if not source.is_dir():
        raise RuntimeError(f'Model pack directory is missing: {source}')
    files = model_tree_files(source)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / '.complete.json').unlink(missing_ok=True)
    for path in files:
        copy_model_file(path, destination / path.relative_to(source), path.stat().st_size)


def check_runtime_space(data, profile):
    pack = data / profile['pack']
    pack_bytes = sum(p.stat().st_size for p in model_tree_files(pack)) if pack.is_dir() else SSD_RESERVE
    required = sum(item['size'] for item in profile['files']) + pack_bytes + SSD_RESERVE
    free = shutil.disk_usage(WORK).free
    print(f'Runtime SSD free: {free / 1e9:.2f} GB; model bodies, pack and reserve: '
          f'{required / 1e9:.2f} GB (DriveFS cache also uses runtime disk)', flush=True)
    if free < required:
        raise RuntimeError('Not enough runtime SSD space for the selected Qwen model; copy not started.')


def fetch_verified(url, path, size, sha256, *, reverify=False, skip_cache_mtime=True,
                   runtime_path=None):
    """Resume in destination directory, validate range and hash before publishing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_name(path.name + '.verified.json')
    existed = path.exists()
    if existed:
        stat = path.stat()
        expected = {'sha256': sha256, 'size': size, 'mtime_ns': stat.st_mtime_ns}
        if not reverify and stat.st_size == size and marker.is_file():
            try:
                record = json.loads(marker.read_text())
                if skip_cache_mtime:
                    matches = (isinstance(record, dict) and record.get('sha256') == sha256
                               and record.get('size') == size)
                else:
                    matches = record == expected
                if matches:
                    label = 'Verified cache (mtime check skipped):' if skip_cache_mtime else 'Verified cache:'
                    print(label, path.name, flush=True)
                    if runtime_path is not None:
                        copy_model_file(path, runtime_path, size)
                    return
            except ValueError:
                pass
        if stat.st_size != size:
            raise RuntimeError(f'Existing file fails verification: {path}. Move it aside before retrying.')
        if runtime_path is not None:
            # Revalidation reads Drive once: hash while copying, not before it.
            copy_model_file(path, runtime_path, size, sha256=sha256)
        elif digest(path) != sha256:
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
        if partial.stat().st_size != size:
            raise RuntimeError(f'SHA-256 mismatch: {partial}. Move it aside before retrying.')
        if runtime_path is not None:
            copy_model_file(partial, runtime_path, size, sha256=sha256)
        elif digest(partial) != sha256:
            raise RuntimeError(f'SHA-256 mismatch: {partial}. Move it aside before retrying.')
        partial.replace(path)
    atomic_json(marker, {'sha256': sha256, 'size': size, 'mtime_ns': path.stat().st_mtime_ns})
    action = 'Reverified cache' if existed else 'Downloaded and SHA-256 verified'
    print(action + ' (Drive cloud sync is separate):', path.name, flush=True)


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
        'log': '/dev/null', 'host': '127.0.0.1', 'port': port,
        'gpu': 0, 'gpus_asked': True, 'open_browser': False, 'draft_vocab': 'cjk',
        'api_monitor': False, 'env': {'STRATA_RESIDENT_HEADROOM_GIB': '8'},
        'lib_dirs': ['/usr/local/cuda/lib64'],
    }


def pack_complete(pack, identity):
    """IQ packs use index.txt; dense.txt belongs to the separate MTP pack."""
    try:
        return (json.loads((pack / '.complete.json').read_text()) == identity
                and all((pack / name).is_file() for name in
                        ('dense.bin', 'index.txt', 'native_experts.txt',
                         'tokenizer/vocab.json', 'tokenizer/chat_template.jinja')))
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/Strata-data')
    parser.add_argument('--context', type=int, default=262144)
    parser.add_argument('--variant')
    parser.add_argument('--reverify', action='store_true')
    parser.add_argument('--skip-cache-mtime', action=argparse.BooleanOptionalAction, default=True,
                        help='Ignore modification time of size/SHA-256 verified cache (default); --reverify wins')
    parser.add_argument('--engine-ready-seconds', type=float, default=180 * 60,
                        help='Strata internal READY wait; bounded by the local supervisor too')
    args = parser.parse_args()
    if args.engine_ready_seconds <= 0:
        parser.error('--engine-ready-seconds must be positive')
    data = ensure_drive(Path(args.data_dir))  # Must precede every weight/cache download.
    WORK.mkdir(exist_ok=True)
    manifest = json.loads((WORK / 'manifest.json').read_text())
    variant, profile = select_variant(manifest, args.variant)
    check_server_port(manifest['server_port'])  # Fail before building or downloading.
    check_privacy()  # Refuse an unsupported kernel before downloading/building the model.
    check_drive_space(data, profile)
    check_runtime_space(data, profile)
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
    # Persist the small engine artifact in Drive; inference uses a runtime copy.
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
    runtime_data = WORK / 'model-data'
    token = hf_token()
    print('Model bodies download straight onto the runtime SSD:', runtime_data, flush=True)
    for item in profile['files']:
        url = f'https://huggingface.co/{item["repo"]}/resolve/{item["revision"]}/{item["source"]}'
        if item.get('fetch') == 'huggingface':
            fetch_hf_to_ssd(url, runtime_data / item['destination'], item['size'], item['sha256'], token,
                            reverify=args.reverify)
        else:
            fetch_verified(url, data / item['destination'], item['size'], item['sha256'],
                           reverify=args.reverify, skip_cache_mtime=args.skip_cache_mtime,
                           runtime_path=runtime_data / item['destination'])
    pack = data / profile['pack']
    pack_identity = {'strata': manifest['strata_commit'], 'gguf': profile['files'][0]['sha256']}
    runtime_pack = runtime_data / profile['pack']
    if pack_complete(pack, pack_identity):
        print('Copying cached IQ pack from Drive to runtime SSD', flush=True)
        copy_model_tree(pack, runtime_pack)
    else:
        # Build using SSD shards; retain a persistent pack for the next runtime.
        run(python, source / 'tools/iq_pack.py',
            '--gguf', runtime_data / profile['files'][0]['destination'], '--out', runtime_pack)
        atomic_json(runtime_pack / '.complete.json', pack_identity)
        print('Saving generated IQ pack to Drive', flush=True)
        copy_model_tree(runtime_pack, pack)
    if not pack_complete(runtime_pack, pack_identity):
        raise RuntimeError('Runtime SSD pack is incomplete; inference not started.')
    workers = max(1, (os.cpu_count() or 2) - 1)
    config = engine_config(source, engine, runtime_data, profile, args.context, workers, manifest['server_port'])
    config_path = WORK / 'strata.json'
    atomic_json(config_path, config)
    port = manifest['server_port']
    # serve.server reads this at module import, before reading config.env.
    # Its 900s default kills a healthy cold Drive load during GPU cache fill.
    os.environ['STRATA_ENGINE_READY_S'] = str(args.engine_ready_seconds)
    print(f'Starting Strata IQ3_S {variant}, localhost:{port}, context', args.context, flush=True)
    print(f'Strata READY timeout: {args.engine_ready_seconds:g} s (local startup deadline also applies)', flush=True)
    serve_private([str(python), '-m', 'serve.server', '--engine', 'strata', '--config', str(config_path),
                   '--port', str(port)], source)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if WORK.is_dir():
            (WORK / 'failed').write_text('Bootstrap/server failed; inspect bootstrap.log\n')
        raise
