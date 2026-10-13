"""Colab bootstrap: persistent GLM-5.3 weights on Drive, SSD-first inference."""
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

WORK = Path('/content/glm53-a100')
DRIVE = Path('/content/drive')
MIN_RAM_KIB = 150 * 1024 * 1024
SSD_RESERVE = 5_000_000_000
MAYA_PACK_FILES = ('index.txt', 'native_experts.txt', 'dense.bin', 'tokenizer/vocab.json',
                   'tokenizer/merges.txt', 'tokenizer/token_type.json', 'tokenizer/chat_template.jinja')
MAYA_PACK_INDEX = 'engines/maya-packs.json'


class ModelCopyIntegrityError(RuntimeError):
    """The verified Drive source changed or failed SHA-256 during SSD staging."""


class InsufficientRuntimeSpace(RuntimeError):
    """The runtime SSD cannot hold the model and a small operating reserve."""


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
        raise RuntimeError('GDrive is not mounted. Run colab drivemount -s glm53-a100 in your terminal.')
    data = data.resolve()
    if not data.is_relative_to((DRIVE / 'MyDrive').resolve()):
        raise RuntimeError('Persistent model destination must be inside /content/drive/MyDrive.')
    data.mkdir(parents=True, exist_ok=True)
    probe = data / '.write-test'
    probe.write_text('ok')
    probe.unlink()
    return data


def check_server_port(port):
    if not isinstance(port, int) or not 1 <= port <= 65535 or port == 8080:
        raise RuntimeError('Use a dedicated GLM port, not Colab/Jupyter port 8080.')
    try:
        with socket.socket() as stream:
            stream.bind(('127.0.0.1', port))
    except OSError as exc:
        raise RuntimeError(f'GLM port {port} is unavailable; refusing to build/download first.') from exc


def select_variant(manifest, requested):
    """Resolve the startup model variant; every download and engine path comes from it."""
    options = manifest['variants']
    name = requested or manifest.get('default_variant')
    if name not in options:
        raise RuntimeError(f'Unknown model variant {name!r}; choose from: ' + ', '.join(sorted(options)))
    return name, options[name]


def select_context(profile, requested):
    """Context length is a per-variant choice: the model's own budget decides the options."""
    options = profile['contexts']
    value = requested or profile['default_context']
    if value not in options:
        raise RuntimeError(f'Unknown context length {value}; choose from: ' + ', '.join(map(str, options)))
    return value


def drive_free_bytes():
    """Drive cloud quota, not DriveFS statvfs (which reports this VM's cache disk)."""
    try:
        address = os.environ.get('TBE_EPHEM_CREDS_ADDR')
        if not address:
            raise RuntimeError('Drive metadata credential endpoint is unavailable')
        if '://' not in address:
            address = 'http://' + address
        request = urllib.request.Request(
            address.rstrip('/') + '/computeMetadata/v1/instance/service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'})
        local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with local.open(request, timeout=10) as response:
            token = json.load(response)['access_token']
        request = urllib.request.Request('https://www.googleapis.com/drive/v3/about?fields=storageQuota',
                                         headers={'Authorization': 'Bearer ' + token})
        with urllib.request.urlopen(request, timeout=15) as response:
            quota = json.load(response)['storageQuota']
        if 'limit' not in quota:
            return float('inf')
        return max(0, int(quota['limit']) - int(quota['usage']))
    except Exception as exc:
        print(f'Drive cloud quota unavailable ({type(exc).__name__}); '
              'mounted filesystem free space will not be treated as Drive quota.', flush=True)
        return None


def check_drive_space(data, profile):
    missing = 0
    for item in drive_files(profile):
        path = data / item['destination']
        partial = path.with_name(path.name + '.part')
        present = path.stat().st_size if path.is_file() else partial.stat().st_size if partial.is_file() else 0
        missing += max(0, item['size'] - present)
        if item['destination'].endswith('.gguf'):
            print(f'GGUF cache: {path.name}: {present / 1e9:.2f}/{item["size"] / 1e9:.2f} GB', flush=True)
    cache_free = shutil.disk_usage(data).free
    print(f'Runtime DriveFS cache disk free: {cache_free / 1e9:.2f} GB; '
          f'missing weights: {missing / 1e9:.2f} GB', flush=True)
    free = drive_free_bytes()
    if free is not None:
        reserve = 2_000_000_000 if profile.get('engine') == 'maya' else 0
        print(f'Drive API cloud free: {free / 1e9:.2f} GB; required: {(missing + reserve) / 1e9:.2f} GB', flush=True)
        if free < missing + reserve:
            raise RuntimeError('Drive cloud storage reports less free space than the missing weights and pack; '
                               'download not started.')


def check_runtime_space(profile, runtime_data):
    """The runtime SSD is the only place a model body can live now, so capacity is checked up front."""
    required = sum(item['size'] for item in profile['files']) + SSD_RESERVE
    if profile.get('engine') == 'maya':
        required += SSD_RESERVE  # Maya's generated pack is built beside the shards.
    free = shutil.disk_usage(WORK).free
    print(f'Runtime SSD free: {free / 1e9:.2f} GB; model bodies, pack and reserve: {required / 1e9:.2f} GB '
          '(the DriveFS cache also uses runtime disk)', flush=True)
    if free < required:
        raise RuntimeError(f'Not enough runtime SSD space for this variant; download into {runtime_data} not started.')


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


def fetch_variant(profile, data, runtime_data, token, reverify, *, skip_cache_mtime=True):
    """Every pinned file of one variant, resumed and SHA-256 verified before publishing."""
    for item in profile['files']:
        url = f'https://huggingface.co/{item["repo"]}/resolve/{item["revision"]}/{item["source"]}'
        if item.get('fetch') == 'huggingface':
            fetch_hf_to_ssd(url, runtime_data / item['destination'], item['size'], item['sha256'], token,
                            reverify=reverify)
        else:
            fetch_verified(url, data / item['destination'], item['size'], item['sha256'],
                           reverify=reverify, skip_cache_mtime=skip_cache_mtime)


def prepare_variants(manifest, requested):
    """--prepare without --variant covers every variant, so one rented runtime pays for all caches."""
    if not requested:
        return sorted(manifest['variants'])
    select_variant(manifest, requested)  # Same unknown-name error as a start.
    return [requested]


def prepare_variant(manifest, variant, profile, data, context, nvcc, cuda_version, reverify, *,
                    skip_cache_mtime=True):
    """Compiled engine and Maya's pack land on Drive; model bodies are re-downloaded per runtime."""
    runtime_data = WORK / 'model-data'
    check_drive_space(data, profile)
    check_runtime_space(profile, runtime_data)
    signature = engine_signature(manifest, profile, nvcc, cuda_version)
    print(f'Preparing {variant} ({profile["model"]}), engine {profile["engine"]}, context {context}', flush=True)
    fetch_variant(profile, data, runtime_data, hf_token(), reverify, skip_cache_mtime=skip_cache_mtime)
    if profile['engine'] == 'maya':
        # Maya's setup writes the pack next to the GGUF files and caches the engine under the same signature.
        _, source, config = maya_setup(manifest, profile, data, runtime_data, context,
                                       manifest['server_port'], signature)
        persist_maya_pack(data, runtime_data, config, profile, context)
    else:
        llama_server(manifest, data, signature, nvcc)
    release_model_bodies(profile, runtime_data)
    print(f'Prepared {variant}: engine and pack cached on Drive, model bodies released from the runtime SSD',
          flush=True)


def release_model_bodies(profile, runtime_data):
    """Free the downloaded bodies so the next prepared variant fits on the same runtime disk."""
    for item in hf_files(profile):
        path = runtime_data / item['destination']
        if path.is_file():
            path.unlink()
            path.with_name(path.name + '.verified.json').unlink(missing_ok=True)
            print(f'Released the runtime SSD copy: {path.name}', flush=True)


def fetch_verified(url, path, size, sha256, *, reverify=False, skip_cache_mtime=True):
    """Resume in destination directory, validate range and hash before publishing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_name(path.name + '.verified.json')
    if path.exists():
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
                    return
            except ValueError:
                pass
        print(f'Cached file needs SHA-256 recheck: {path.name} '
              '(verification record, update time or --reverify); no new download.', flush=True)
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
        if partial.stat().st_size != size or digest(partial) != sha256:
            raise RuntimeError(f'SHA-256 mismatch: {partial}. Move it aside before retrying.')
        partial.replace(path)
    atomic_json(marker, {'sha256': sha256, 'size': size, 'mtime_ns': path.stat().st_mtime_ns})
    print('Downloaded and SHA-256 verified (Drive cloud sync is separate):', path.name, flush=True)


def maya_pack_path(config):
    settings = json.loads(config.read_text())
    args = settings.get('args') or []
    if args.count('--glm-pack') != 1 or args.index('--glm-pack') == len(args) - 1:
        raise RuntimeError('Maya config has no unique --glm-pack path.')
    return Path(args[args.index('--glm-pack') + 1])


def maya_pack_key(profile, context):
    """Index key for the pack Maya built: its model build and the context it was tuned for."""
    return f"{profile['maya']['model']}|{context}"


def maya_pack_index(data):
    """Drive-side pointer from a prepared build to the pack it generated."""
    try:
        record = json.loads((data / MAYA_PACK_INDEX).read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def seed_maya_pack(data, runtime_data, profile, context):
    """Copy a previously built pack back onto the runtime SSD so Maya can skip its rebuild."""
    relative = maya_pack_index(data).get(maya_pack_key(profile, context))
    if not relative:
        return
    source = data / relative
    if not source.is_dir() or not (source / 'index.txt').is_file():
        print(f'Cached Maya pack is incomplete; Maya will rebuild it: {relative}', flush=True)
        return
    print(f'Seeding the cached Maya pack onto the runtime SSD: {relative}', flush=True)
    try:
        for name in MAYA_PACK_FILES:
            cached = source / name
            if cached.is_file():
                copy_model_file(cached, runtime_data / relative / name, cached.stat().st_size)
    except (OSError, InsufficientRuntimeSpace) as exc:
        print(f'Maya pack seeding failed ({type(exc).__name__}); Maya will rebuild it.', flush=True)


def persist_maya_pack(data, runtime_data, config, profile, context):
    """Keep the generated pack on Drive; the model bodies around it are re-downloaded each runtime."""
    pack = maya_pack_path(config)
    try:
        relative = pack.relative_to(runtime_data)
    except ValueError as exc:
        raise RuntimeError(f'Maya pack is outside the runtime model directory: {pack}') from exc
    for name in MAYA_PACK_FILES:
        if not (pack / name).is_file():
            raise RuntimeError(f'Maya pack is missing {name}: {pack}')
    target = data / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    (data / MAYA_PACK_INDEX).parent.mkdir(parents=True, exist_ok=True)
    if (target / 'index.txt').is_file():
        atomic_json(data / MAYA_PACK_INDEX, {**maya_pack_index(data), maya_pack_key(profile, context): str(relative)})
        return
    try:
        for name in MAYA_PACK_FILES:
            source = pack / name
            copy_model_file(source, target / name, source.stat().st_size)
    except (OSError, InsufficientRuntimeSpace) as exc:
        print(f'Maya pack was not cached on Drive ({type(exc).__name__}); '
              'the next runtime rebuilds it on the SSD.', flush=True)
        return
    atomic_json(data / MAYA_PACK_INDEX, {**maya_pack_index(data), maya_pack_key(profile, context): str(relative)})
    print(f'Cached the Maya pack on Drive: {relative}', flush=True)


def copy_model_file(source, destination, size, sha256=None):
    """Copy one Drive file to an unpublished SSD path, checking the GGUF hash in flight."""
    free = shutil.disk_usage(WORK).free
    if free < SSD_RESERVE:
        raise InsufficientRuntimeSpace(f'{source.name}: only {free / 1e9:.2f} GB free')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    checksum = hashlib.sha256() if sha256 else None
    copied = 0
    started = last_report = time.monotonic()
    print(f'Copy to runtime SSD: {source.name}: 0.00/{size / 1e9:.2f} GB', flush=True)
    try:
        with source.open('rb') as incoming, partial.open('wb') as outgoing:
            while block := incoming.read(8 * 1024 * 1024):
                copied += len(block)
                if copied > size:
                    raise ModelCopyIntegrityError(f'Model source grew during SSD copy: {source}')
                outgoing.write(block)
                if checksum:
                    checksum.update(block)
                if shutil.disk_usage(WORK).free < SSD_RESERVE:
                    raise InsufficientRuntimeSpace(f'{source.name}: runtime SSD reserve reached during copy')
                now = time.monotonic()
                if now - last_report >= 20:
                    elapsed = max(0.001, now - started)
                    print(f'Copy to runtime SSD: {source.name}: {copied / 1e9:.2f}/{size / 1e9:.2f} GB '
                          f'({copied / elapsed / 1e6:.1f} MB/s)', flush=True)
                    last_report = now
        if copied != size or (checksum and checksum.hexdigest() != sha256):
            raise ModelCopyIntegrityError(f'Model source changed or failed SHA-256 during SSD copy: {source}')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    elapsed = max(0.001, time.monotonic() - started)
    print(f'Copied to runtime SSD: {source.name}: {copied / 1e9:.2f} GB in {elapsed:.1f} s', flush=True)


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


def ensure_venv(root):
    """A private Python for the engine's build and runtime packages (Colab's system Python refuses pip installs)."""
    python = root / '.venv/bin/python'
    if python.exists():
        return python
    result = subprocess.run([sys.executable, '-m', 'venv', str(root / '.venv')])
    if result.returncode:
        # Colab's system Python can omit ensurepip/python3-venv.
        run(sys.executable, '-m', 'pip', 'install', 'uv')
        run(sys.executable, '-m', 'uv', 'venv', root / '.venv',
            '--python', sys.executable, '--seed', '--allow-existing')
    return python


def engine_signature(manifest, profile, nvcc, cuda_version):
    """Cache key for a compiled engine: which engine and commit, plus the toolchain it was compiled against."""
    commit = manifest['maya_commit'] if profile['engine'] == 'maya' else manifest['llama_commit']
    cpu = Path('/proc/cpuinfo').read_text().split('flags', 1)[-1].split('\n', 1)[0]
    parts = (profile['engine'], commit, nvcc, cuda_version, platform.libc_ver().__repr__(), cpu)
    return hashlib.sha256('\0'.join(parts).encode()).hexdigest()[:20]


def build_jobs():
    # nvcc takes 2-4 GB per job on the CUDA kernels; Colab gives about 12 vCPUs.
    return max(2, min((os.cpu_count() or 4) // 2, 8))


def llama_server(manifest, data, signature, nvcc):
    """llama-server for sm_80, built once and cached on Drive under the toolchain signature."""
    source = WORK / 'llama.cpp'
    source_tree('ggml-org/llama.cpp', manifest['llama_commit'], source)
    binary = source / 'build' / 'bin' / 'llama-server'
    cached = data / 'engines' / f'llama-{signature}' / 'llama-server'
    if cached.is_file():
        binary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, binary)
        binary.chmod(0o755)
        print('Cached A100 llama-server:', cached, flush=True)
        return binary
    python = ensure_venv(source)
    run(python, '-m', 'pip', 'install', 'cmake', 'ninja')
    env = os.environ.copy()
    env['PATH'] = str(python.parent) + os.pathsep + env['PATH']
    run('cmake', '-S', source, '-B', source / 'build', '-DCMAKE_BUILD_TYPE=Release',
        '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=80', '-DBUILD_SHARED_LIBS=OFF',
        '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON',
        '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_APP=OFF',
        '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF',
        f'-DCMAKE_CUDA_COMPILER={nvcc}', cwd=source, env=env)
    run('cmake', '--build', source / 'build', '--target', 'llama-server', '-j', build_jobs(), cwd=source, env=env)
    if not binary.is_file():
        raise RuntimeError(f'llama.cpp build finished without {binary}.')
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, cached)
    print('Cached A100 llama-server for the next run:', cached, flush=True)
    return binary


def maya_config_path(source):
    """The config Maya's setup wrote: newest maya-*.json, as Maya itself picks it at a start."""
    configs = sorted((p for p in source.glob('maya-*.json') if not p.name.endswith('.shared-settings.json')),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        raise RuntimeError('Project Maya finished its setup without writing a maya-*.json config.')
    return configs[0]


def maya_setup(manifest, profile, data, runtime_data, context, port, signature):
    """Set Maya up against the runtime SSD model folder, then hand back the config to start.

    --no-start performs the build and tuning. Serving uses the prepared config
    directly, avoiding Maya's startup touches, rebuilds and update/restart hooks.

    The shards live on the SSD; only the compiled engine and the generated pack
    keep using the persistent Drive cache.
    """
    source = WORK / 'project-maya'
    source_tree('mw00/project-maya', manifest['maya_commit'], source)
    python = ensure_venv(source)
    llama = WORK / 'maya-llama.cpp'
    source_tree('ggml-org/llama.cpp', manifest['maya_llama_commit'], llama)
    build = source / 'build'
    cached = data / 'engines' / f'maya-{signature}'
    if (cached / 'strata').is_file() and (cached / 'MAYA-BUILD.json').is_file():
        # Maya skips the compile when build/strata and its MAYA-BUILD.json stamp match the source.
        build.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached / 'strata', build / 'strata')
        shutil.copy2(cached / 'MAYA-BUILD.json', build / 'MAYA-BUILD.json')
        (build / 'strata').chmod(0o755)
        print('Cached A100 Maya engine:', cached, flush=True)
    env = os.environ.copy()
    env['PATH'] = str(python.parent) + os.pathsep + env['PATH']
    maya = profile['maya']
    seed_maya_pack(data, runtime_data, profile, context)
    command = [python, source / 'maya.py', '--yes', '--model', maya['model'], '--draft', maya.get('draft', 'mtp'),
               '--context', context, '--models-dir', runtime_data / 'models', '--llama-dir', llama,
               '--no-start', '--port', port, '--host', '127.0.0.1']
    if maya.get('no_vision'):
        command.append('--no-vision')
    for key, value in (maya.get('env') or {}).items():
        command += ['--env', f'{key}={value}']
    run(*command, cwd=source, env=env)
    if (build / 'strata').is_file() and (build / 'MAYA-BUILD.json').is_file():
        cached.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build / 'strata', cached / 'strata')
        shutil.copy2(build / 'MAYA-BUILD.json', cached / 'MAYA-BUILD.json')
        print('Cached A100 Maya engine for the next run:', cached, flush=True)
    config = maya_config_path(source)
    settings = json.loads(config.read_text())
    # Maya reports its own MODEL_NAME for every build; the Codex catalog and the health probe use ours.
    settings['model_name'] = profile['model']
    settings.update(log='/dev/null', api_monitor=False, open_browser=False)
    settings.pop('slot_save_path', None)
    atomic_json(config, settings)
    return python, source, config


def check_accelerator():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap',
                                   '--format=csv,noheader'], text=True)
    print('GPU:', gpu.strip(), flush=True)
    if 'A100' not in gpu or '8.0' not in gpu:
        raise RuntimeError('Expected the requested A100 (sm_80); refusing a different accelerator.')


def check_ram():
    ram_kib = int(re.search(r'MemTotal:\s+(\d+)', Path('/proc/meminfo').read_text()).group(1))
    print(f'RAM: {ram_kib / 1024 / 1024:.2f} GiB', flush=True)
    if ram_kib < MIN_RAM_KIB:
        raise RuntimeError(f'Expected high-mem RAM >= {MIN_RAM_KIB // 1024 // 1024} GiB; '
                           'the experts would not fit and would stream from Drive.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/GLM53-data')
    parser.add_argument('--context', type=int)
    parser.add_argument('--variant')
    parser.add_argument('--reverify', action='store_true')
    parser.add_argument('--skip-cache-mtime', action=argparse.BooleanOptionalAction, default=True,
                        help='Ignore modification time of size/SHA-256 verified cache (default); --reverify wins')
    parser.add_argument('--prepare', action='store_true',
                        help='Fetch weights and build the engines for the selected variant(s), then exit without serving')
    args = parser.parse_args()
    data = ensure_drive(Path(args.data_dir))  # Must precede every weight/cache download.
    WORK.mkdir(exist_ok=True)
    manifest = json.loads((WORK / 'manifest.json').read_text())
    names = prepare_variants(manifest, args.variant) if args.prepare else [select_variant(manifest, args.variant)[0]]
    port = manifest['server_port']
    check_server_port(port)  # Fail before building or downloading.
    if not args.prepare:
        check_privacy()
    print('Model variants: ' + ', '.join(names), flush=True)
    check_accelerator()
    check_ram()
    nvcc = shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('Colab CUDA toolkit nvcc is missing.')
    cuda_version = subprocess.check_output([nvcc, '--version'], text=True)
    print(cuda_version, flush=True)
    if args.prepare:
        for name in names:
            profile = manifest['variants'][name]
            prepare_variant(manifest, name, profile, data, select_context(profile, args.context),
                            nvcc, cuda_version, args.reverify, skip_cache_mtime=args.skip_cache_mtime)
        print('GLM_PREPARE_COMPLETE ' + json.dumps(names), flush=True)
        return
    variant = names[0]
    profile = manifest['variants'][variant]
    context = select_context(profile, args.context)
    runtime_data = WORK / 'model-data'
    check_drive_space(data, profile)
    check_runtime_space(profile, runtime_data)
    print(f'Model variant: {variant} ({profile["model"]}), engine {profile["engine"]}, context {context}', flush=True)
    fetch_variant(profile, data, runtime_data, hf_token(), args.reverify, skip_cache_mtime=args.skip_cache_mtime)
    signature = engine_signature(manifest, profile, nvcc, cuda_version)
    if profile['engine'] == 'maya':
        python, source, config = maya_setup(manifest, profile, data, runtime_data, context, port, signature)
        persist_maya_pack(data, runtime_data, config, profile, context)
        argv = [str(python), str(source / 'serve/server.py'), '--engine', 'strata', '--config', str(config),
                '--port', str(port), '--host', '127.0.0.1']
        print(f'Starting Maya {variant} from the runtime SSD, localhost:{port}, context {context}', flush=True)
        serve_private(argv, source)
        return
    binary = llama_server(manifest, data, signature, nvcc)
    argv = [str(binary), '-m', str(runtime_data / profile['llama']['first_shard']), '-c', context,
            '--host', '127.0.0.1', '--port', port, '--alias', profile['model'], *profile['llama']['args'],
            '--log-disable', '--metrics']
    print(f'Starting llama.cpp {variant} from the runtime SSD, localhost:{port}, context {context}', flush=True)
    serve_private(argv, WORK / 'llama.cpp')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if WORK.is_dir():
            (WORK / 'failed').write_text('Bootstrap/server failed; inspect bootstrap.log\n')
        raise
