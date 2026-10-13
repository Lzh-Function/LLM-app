"""A100 Colab/Strata lifecycle and local Codex client. No third party dependencies."""
from __future__ import annotations

import argparse
import codecs
import fcntl
import json
import os
import selectors
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE = HERE / '.runtime'
SESSION = 'strata-qwen'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BOOTSTRAP_DIAGNOSTICS = r"""
import pathlib, re
path = pathlib.Path('/content/strata-qwen') / 'bootstrap.log'
text = path.read_text(errors='replace') if path.exists() else ''
print('\n'.join(text.splitlines()[-80:]))
matches = re.findall(r'^COLAB_PRIVACY_DIAGNOSTIC=(/dev/shm/colab-privacy-[^\s/]+/inference\.log)$', text, re.M)
if matches:
    log = pathlib.Path(matches[-1])
    if log.exists():
        print('\n--- Inference diagnostics (copied from RAM) ---')
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 65536))
            print(stream.read().decode(errors='replace'))
"""
BOOTSTRAP_READER = r"""
import json, re, sys
from pathlib import Path
work = Path(sys.argv[1])
try:
    pid = int((work / 'bootstrap.pid').read_text())
    alive = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
except (OSError, ValueError, IndexError):
    alive = False

def new_lines(path, offset):
    if not path or not path.is_file():
        return '', 0, 0
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        if size < offset:
            offset = 0
        stream.seek(offset)
        lines = []
        start_offset = offset
        while stream.tell() - start_offset < 1024 * 1024:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n') and alive:
                stream.seek(start)
                break
            lines.append(line)
        return b''.join(lines).decode('utf-8', errors='replace'), stream.tell(), size

bootstrap = work / 'bootstrap.log'
text, offset, size = new_lines(bootstrap, int(sys.argv[2]))
with bootstrap.open('rb') as stream:
    stream.seek(max(0, size - 65536))
    recent = stream.read().decode(errors='replace')
matches = re.findall(r'^COLAB_PRIVACY_DIAGNOSTIC=(/dev/shm/colab-privacy-[^\s/]+/inference\.log)$',
                     recent, re.M)
ram_path = matches[-1] if matches else ''
ram_offset = int(sys.argv[3]) if ram_path == sys.argv[4] else 0
ram_text, ram_offset, ram_size = new_lines(Path(ram_path) if ram_path else None, ram_offset)
# Inference output travels over SSH to the local terminal, never to bootstrap.log.
failed = ((work / 'failed').exists()
          or (not alive and offset >= size and ram_offset >= ram_size))
print(json.dumps({'offset': offset, 'ram_offset': ram_offset, 'ram_path': ram_path,
                  'text': text + ram_text, 'failed': failed}))
"""


def say(message):
    print('[strata-qwen]', message, flush=True)


def manifest():
    return json.loads((HERE / 'manifest.json').read_text())


def select_variant(name):
    """Pick the startup variant; the default comes from manifest.json."""
    data = manifest()
    options = data['variants']
    chosen = name or data.get('default_variant')
    if chosen not in options:
        raise RuntimeError('--variant は ' + ' / '.join(sorted(options)) + ' のどれかにしてください。')
    return chosen, options[chosen]


def profile_for_model(model):
    for name, profile in manifest()['variants'].items():
        if profile['model'] == model:
            return name, profile
    return None, None


def proc_identity(pid):
    try:
        # Fields after the parenthesized command start at proc stat field 3.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def health(port, model=None):
    try:
        with OPENER.open(f'http://127.0.0.1:{port}/health', timeout=3) as response:
            info = json.load(response)
        if info.get('loaded') is not True:
            return None
        served = {profile['model'] for profile in manifest()['variants'].values()}
        return info if info.get('model') in (served if model is None else {model}) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def available_port(port):
    with socket.socket() as stream:
        stream.bind(('127.0.0.1', port))


def colab_interpreter():
    entrypoint = Path(shutil.which('colab'))
    with entrypoint.open() as entry:
        interpreter = shlex.split(entry.readline().strip().removeprefix('#!'))
    if not interpreter or 'python' not in interpreter[-1]:
        raise RuntimeError('colabのPython起動環境を判定できません。--drive-auth browser を使用してください。')
    return interpreter


def catalog(context, profile):
    result = json.loads((ROOT / 'codex/models.json').read_text())
    model = result['models'][0]
    bundled = subprocess.run(['codex', 'debug', 'models', '--bundled'], capture_output=True, text=True, check=True)
    models = json.loads(bundled.stdout)['models']
    standard = next((entry for entry in models if entry['slug'] == 'gpt-6.1-sol'), None)
    if standard is None:
        standard = next(entry for entry in models if entry.get('base_instructions'))
    model.update({key: standard[key] for key in ('base_instructions', 'model_messages')})
    model.update(slug=profile['model'], display_name=profile['display_name'],
                 description='Strata on Colab A100, Google Drive weights.',
                 context_window=context, max_context_window=context, input_modalities=['text'],
                 supports_image_detail_original=False)
    path = STATE / 'models.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return path


def codex_command(port, context, arguments, profile):
    settings = {
        'model': profile['model'], 'model_provider': 'strata_colab',
        'model_catalog_json': str(STATE / 'models.json'),
        'model_context_window': context, 'model_auto_compact_token_limit': context * 3 // 4,
        'model_reasoning_effort': 'high', 'show_raw_agent_reasoning': True,
        'model_providers.strata_colab.name': profile['provider_name'],
        'model_providers.strata_colab.base_url': f'http://127.0.0.1:{port}/v1',
        'model_providers.strata_colab.wire_api': 'responses',
        'model_providers.strata_colab.requires_openai_auth': False,
        'model_providers.strata_colab.supports_websockets': False,
        'model_providers.strata_colab.stream_idle_timeout_ms': 600000,
        'features.apps': False, 'features.multi_agent': False,
        'sandbox_mode': 'workspace-write', 'sandbox_workspace_write.network_access': True,
    }
    search = ROOT / 'codex/search-mcp'
    if (search / '.venv/bin/python').is_file():
        settings.update({
            'mcp_servers.local_web_search.command': str(search / '.venv/bin/python'),
            'mcp_servers.local_web_search.args': [str(search / 'server.py')],
            'mcp_servers.local_web_search.enabled': True,
            'mcp_servers.local_web_search.startup_timeout_sec': 30,
            'mcp_servers.local_web_search.tool_timeout_sec': 45,
        })
    command = ['codex', '--no-daemon']
    for key, value in settings.items():
        command += ['-c', key + '=' + json.dumps(value, ensure_ascii=False)]
    return command + list(arguments)


class CommandTimeout(RuntimeError):
    """A local command deadline, distinguishable from remote Python failures."""


class Session:
    def __init__(self, args):
        self.args = args
        self.variant, self.profile = select_variant(getattr(args, 'variant', None))
        self.children = []
        self.owned = False
        self.drive_mounted = False
        self.master = None
        self.bootstrap_offset = 0
        self.inference_offset = 0
        self.inference_log = ''
        self.socket = f'/tmp/strata-qwen-{os.getuid()}-{uuid.uuid4().hex[:12]}.sock'

    def run(self, command, *, timeout=None, stream_output=False, progress=None, **kwargs):
        process = subprocess.Popen(command, **kwargs)
        self.children.append(process)
        try:
            if stream_output:
                stdout = self.stream(process, timeout)
            else:
                if progress:
                    deadline = time.monotonic() + timeout
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, timeout)
                        try:
                            stdout, _ = process.communicate(timeout=min(30, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() >= deadline:
                                raise
                            say(progress)
                else:
                    stdout, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            raise CommandTimeout(f'{command[0]} の待機時間上限（{timeout}秒）を超えました。') from None
        finally:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout)
        return stdout

    @staticmethod
    def stream(process, timeout):
        deadline = time.monotonic() + timeout
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        chunks = []
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                for key, _ in selector.select(min(1, remaining)):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    text = decoder.decode(data)
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
        tail = decoder.decode(b'', final=True)
        chunks.append(tail)
        sys.stdout.write(tail)
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
        return ''.join(chunks)

    def colab(self, *arguments, **kwargs):
        return self.run(['colab', *arguments], **kwargs)

    def save_code_failure(self, stage, attempt, timeout, error_type):
        # Control metadata only; never copy authentication headers or VM output.
        path = STATE / 'setup-error.json'
        record = {'time_unix': time.time(), 'stage': stage, 'attempt': attempt,
                  'timeout_seconds': timeout, 'error_type': error_type}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        path.chmod(0o600)

    def code(self, code, *, quiet=False, timeout=60, progress=None, retry_safe=False,
             stage='Colabセル実行'):
        # Only explicitly idempotent setup may be retried. In particular, never
        # retry Popen/bootstrap or Drive flush after an ambiguous client timeout.
        attempts = 3 if retry_safe else 1
        for attempt in range(1, attempts + 1):
            # colab-cli can exit 0 even when notebook Python raises. A marker
            # printed AFTER the whole script proves completion; source echoes do not.
            marker = 'STRATA_EXEC_OK_' + uuid.uuid4().hex
            path = STATE / 'execute.py'
            path.write_text(f'exec(compile({code!r}, "strata-setup", "exec"))\nprint({marker!r})\n')
            try:
                output = self.colab('exec', '-s', SESSION, '-f', str(path),
                                    '--timeout', str(max(1, timeout - 5)), timeout=timeout,
                                    progress=progress, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            except CommandTimeout as exc:
                self.save_code_failure(stage, attempt, timeout, type(exc).__name__)
                if attempt < attempts:
                    say(f'{stage}: {timeout:g}秒で完了を確認できませんでした。接続を作り直して再試行します（{attempt + 1}/{attempts}）。')
                    continue
                raise CommandTimeout(f'{stage}: {exc} 診断: {STATE / "setup-error.json"}') from None
            except subprocess.CalledProcessError as exc:
                self.save_code_failure(stage, attempt, timeout, type(exc).__name__)
                raise
            lines = output.splitlines()
            completed = marker in lines
            output = '\n'.join(line for line in lines if line != marker)
            if output and not quiet:
                print(output, flush=True)
            if not completed:
                self.save_code_failure(stage, attempt, timeout, 'IncompleteExecution')
                raise RuntimeError(f'{stage}: Colab側のPython実行に失敗しました（実行完了を確認できません）。')
            return output

    def allocate(self):
        if not shutil.which('colab'):
            raise RuntimeError('colab is required.')
        status = self.colab('status', '-s', SESSION, stdout=subprocess.PIPE, text=True, timeout=30)
        if f"Session '{SESSION}' not found." not in status:
            raise RuntimeError(f'Session {SESSION} already exists. Use ./colab-qwen stop first.')
        self.owned = True  # Also clean up an allocation interrupted before READY.
        say('ColabにA100 high-memの割当を要求します（最大5分）。')
        try:
            output = self.colab('new', '-s', SESSION, '--gpu', 'A100', '--high-mem', timeout=300,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as exc:
            output = exc.output or ''
            unavailable = 'Service Unavailable' in output
            record = {'stage': 'allocation', 'time_unix': time.time(), 'gpu': 'A100', 'high_mem': True,
                      'returncode': exc.returncode,
                      'reason': 'service_unavailable' if unavailable else 'unknown'}
            path = STATE / 'allocation-error.json'
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
            path.chmod(0o600)
            if unavailable:
                raise RuntimeError(
                    'ColabがA100 high-memの割当要求に503 Service Unavailableを返しました。'
                    'GDrive認証には進んでいません。空き状況またはサービス側の問題が考えられますが、'
                    '詳細理由は未確定です。時間を置いて同じコマンドを再実行してください。'
                ) from None
            if output:
                print(output, flush=True)
            raise
        if output:
            print(output.rstrip(), flush=True)

    def save_drive_diagnostics(self, info):
        path = STATE / 'drive-mount-error.json'
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
        path.chmod(0o600)
        say(f'GDriveマウント診断を保存しました: {path}')

    def mount_drive_browser(self, probe):
        say('次のURLは、今回作成したruntimeをColabのブラウザ画面で開きます。')
        self.colab('url', '-s', SESSION, timeout=20)
        say('開いたノートブックのセルで次を実行して、Driveアクセスを許可してください。')
        print("from google.colab import drive\ndrive.mount('/content/drive')", flush=True)
        say('A100 runtimeは割当済みです。ブラウザで新しいruntimeを作成しないでください。')
        say('マウント成功後にこのターミナルでEnter。操作待ちは最大5分です。')
        self.run([sys.executable, '-c', 'input()'], timeout=300)
        say('GDriveを確認します（最大20秒）。')
        output = self.code(probe, quiet=True, timeout=20)
        if 'STRATA_DRIVE=true' not in output.splitlines():
            raise RuntimeError('Drive mount failed')

    def mount_drive(self):
        probe = "import os, json\nprint('STRATA_DRIVE=' + json.dumps(os.path.ismount('/content/drive') and os.path.isdir('/content/drive/MyDrive')))\n"
        say('GDriveのマウント状態を確認します（最大20秒）。')
        output = self.code(probe, quiet=True, timeout=20)
        if 'STRATA_DRIVE=true' in output.splitlines():
            self.drive_mounted = True
            return
        if not sys.stdin.isatty():
            raise RuntimeError('GDrive authorization needs a terminal. Run ./colab-qwen start in your terminal.')
        try:
            if getattr(self.args, 'drive_auth', 'cli') == 'browser':
                self.mount_drive_browser(probe)
                self.drive_mounted = True
                return
            say('Google Driveの認証画面が出たら、表示された要求権限をすべて選択して許可し、ターミナルに戻ってください。')
            # Launch with the CLI's own environment so its SDK is available.
            output = self.run([*colab_interpreter(), str(HERE / 'mount_helper.py'), '-s', SESSION],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              stream_output=True, timeout=300)
            records = [line.removeprefix('STRATA_DRIVE_RESULT=') for line in output.splitlines()
                       if line.startswith('STRATA_DRIVE_RESULT=')]
            info = json.loads(records[-1]) if records else None
            if not info or not info.get('mounted') or not info.get('mydrive_exists') or info.get('mount_error_type'):
                if info:
                    self.save_drive_diagnostics(info)
                raise RuntimeError('Drive mount failed')
            self.drive_mounted = True
        except Exception as exc:
            # Never reattach with another exec after a failed CLI mount: that
            # was the silent wait observed here. Diagnostics come from the cell.
            say(str(exc))
            raise RuntimeError('GDriveのマウントに失敗しました。モデル準備を中止してruntimeを停止します。') from exc

    def check_drive(self):
        self.allocate()
        self.mount_drive()
        say('GDriveのマウントとMyDriveを確認しました。')

    def ssh(self, remote_command):
        result = subprocess.run(['ssh', '-S', self.socket, '-o', 'BatchMode=yes',
                                 'root@colab-runtime', remote_command],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError('SSH connection lost: ' + result.stderr.strip())
        return result.stdout

    def bootstrap_output(self):
        result = json.loads(self.ssh(shlex.join([
            'python3', '-c', BOOTSTRAP_READER, '/content/strata-qwen',
            str(self.bootstrap_offset), str(self.inference_offset), self.inference_log])))
        self.bootstrap_offset = result['offset']
        self.inference_offset = result['ram_offset']
        self.inference_log = result['ram_path']
        return result['text'] + ('\nSTRATA_BOOTSTRAP_FAILED\n' if result['failed'] else '')

    def start(self):
        remote_port = json.loads((HERE / 'manifest.json').read_text())['server_port']
        if not isinstance(remote_port, int) or not 1 <= remote_port <= 65535 or remote_port == 8080:
            raise RuntimeError('manifest must select a dedicated remote Strata port, not Colab port 8080.')
        for tool in ('colab', 'ssh', 'ssh-keygen', 'codex'):
            if not shutil.which(tool):
                raise RuntimeError(f'{tool} is required.')
        available_port(self.args.port)
        catalog(self.args.context, self.profile)  # Verify Codex integration before renting a GPU.
        identity = STATE / 'id_ed25519'
        if not identity.is_file():
            self.run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-f', str(identity), '-q'])
        self.allocate()
        self.mount_drive()
        say('Drive認証後のruntime再接続・準備を確認します（1回最大120秒、タイムアウト時は最大3回）。')
        self.code("from pathlib import Path\nimport os\nassert os.path.ismount('/content/drive') and Path('/content/drive/MyDrive').is_dir(), 'Drive mount failed'\nPath('/content/strata-qwen').mkdir(exist_ok=True)\n",
                  timeout=120, retry_safe=True, stage='runtime再接続・準備',
                  progress='runtime再接続・準備の完了を待っています。')
        for name in ('remote.py', 'manifest.json'):
            self.colab('upload', '-s', SESSION, str(HERE / name), 'content/strata-qwen/' + name, timeout=45)
        self.colab('upload', '-s', SESSION, str(HERE.parent / 'colab_privacy.py'),
                   'content/strata-qwen/colab_privacy.py', timeout=45)
        remote_args = ['python3', '-B', '-u', '/content/strata-qwen/remote.py', '--data-dir', self.args.data_dir,
                       '--context', str(self.args.context), '--variant', self.variant,
                       '--engine-ready-seconds', str(self.args.startup_minutes * 60)]
        if self.args.reverify:
            remote_args.append('--reverify')
        remote_args.append('--skip-cache-mtime' if getattr(self.args, 'skip_cache_mtime', True)
                           else '--no-skip-cache-mtime')
        self.code('import subprocess\nfrom pathlib import Path\n'
                  "log=open('/content/strata-qwen/bootstrap.log','ab',buffering=0)\n"
                  f'p=subprocess.Popen({remote_args!r},stdout=log,stderr=log,start_new_session=True)\n'
                  "Path('/content/strata-qwen/bootstrap.pid').write_text(str(p.pid))\nprint('BOOTSTRAP_PID',p.pid)\n", stage='モデル準備プロセス起動')
        proxy = shlex.join(['colab', 'ssh', '--proxy-mode', '-s', SESSION, '--identity', str(identity)])
        ssh_log = (STATE / 'ssh.log').open('ab', buffering=0)
        command = ['ssh', '-M', '-S', self.socket, '-N', '-i', str(identity),
                   '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60',
                   '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
                   '-o', 'ExitOnForwardFailure=yes', '-o', 'StrictHostKeyChecking=accept-new',
                   '-o', f'UserKnownHostsFile={STATE / "known_hosts"}',
                   '-o', 'HostKeyAlias=strata-qwen-' + uuid.uuid4().hex,
                   '-o', 'ProxyCommand=' + proxy,
                   '-L', f'127.0.0.1:{self.args.port}:127.0.0.1:{remote_port}', 'root@colab-runtime']
        self.master = subprocess.Popen(command, stdout=ssh_log, stderr=ssh_log, start_new_session=True)
        ssh_log.close()
        self.children.append(self.master)
        deadline = time.monotonic() + self.args.startup_minutes * 60
        while time.monotonic() < deadline:
            if self.master.poll() is not None:
                raise RuntimeError(f'SSH tunnel failed. See {STATE / "ssh.log"}')
            if health(self.args.port, self.profile['model']):
                say(f'準備完了: http://127.0.0.1:{self.args.port}/v1 ({self.variant})')
                return
            if Path(self.socket).exists():
                tail = self.bootstrap_output()
                if tail:
                    print(tail.rstrip(), flush=True)
                if 'STRATA_BOOTSTRAP_FAILED' in tail:
                    full = self.ssh('python3 -c ' + shlex.quote(BOOTSTRAP_DIAGNOSTICS))
                    (STATE / 'bootstrap-error.log').write_text(full)
                    raise RuntimeError(f'Bootstrap failed. See {STATE / "bootstrap-error.log"}')
            time.sleep(5)
        raise RuntimeError('Startup timed out; attempting Drive sync before stopping the runtime.')

    def flush_drive(self):
        minutes = getattr(self.args, 'flush_minutes', 1440.0)
        seconds = max(1, int(minutes * 60))
        say(f'GDriveの未同期データを同期してから停止します（最大{minutes:g}分）。')
        helper = (HERE / 'shutdown_remote.py').read_text()
        code = (f'ns={{"__name__":"strata_shutdown"}}\nexec(compile({helper!r}, "strata-shutdown", "exec"), ns)\n'
                f'result=ns["shutdown"]({seconds})\nprint("STRATA_DRIVE_SYNC=" + result)\n')
        record = {'time_unix': time.time(), 'timeout_seconds': seconds, 'confirmed': False}
        path = STATE / 'drive-sync-status.json'
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        path.chmod(0o600)
        try:
            output = self.code(code, quiet=True, timeout=seconds + 180,
                               progress='GDriveの同期完了を待っています。同期中もruntimeは稼働しています。',
                               stage='GDrive同期')
            if 'STRATA_DRIVE_SYNC=flushed' not in output.splitlines():
                raise RuntimeError('Drive was not mounted; cloud sync could not be confirmed.')
            record['confirmed'] = True
            self.drive_mounted = False
            say('GDriveの同期完了を確認しました。')
        except Exception as exc:
            record['error_type'] = type(exc).__name__
            say('GDriveの同期完了を確認できませんでした。未同期データを保護するためruntimeを残します。')
            say('再試行: ./colab-qwen stop（待機中もruntimeの利用時間を消費します）。')
        record['finished_at'] = time.time()
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        return record['confirmed']

    def close(self):
        # Ignore further Ctrl+C while returning the billable runtime.
        signal.alarm(0)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        for child in reversed(self.children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        sync_confirmed = True
        if self.owned:
            if self.drive_mounted:
                sync_confirmed = self.flush_drive()
            if sync_confirmed:
                say('runtimeを停止します。')
                stop_runtime()
        return sync_confirmed


def stop_runtime():
    subprocess.run(['colab', 'stop', '-s', SESSION], check=True, timeout=120)
    result = subprocess.run(['colab', 'status', '-s', SESSION], check=True,
                            capture_output=True, text=True, timeout=60)
    if f"Session '{SESSION}' not found." not in result.stdout:
        raise RuntimeError('Runtime stop could not be confirmed. Run colab stop -s strata-qwen manually.')
    say('runtime停止を確認しました。')


def stop(args=None):
    path = STATE / 'session.json'
    state = json.loads(path.read_text()) if path.is_file() else {}
    pid = state.get('pid')
    owner_alive = bool(pid and pid != os.getpid() and state.get('identity') is not None
                       and proc_identity(pid) == state['identity'])
    if getattr(args, 'force', False):
        # --force explicitly chooses runtime release over outstanding writes.
        if owner_alive:
            os.kill(pid, signal.SIGKILL)
        stop_runtime()
        path.unlink(missing_ok=True)
        return
    minutes = getattr(args, 'flush_minutes', 1440.0)
    if owner_alive:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + max(minutes, state.get('flush_minutes', minutes)) * 60 + 240
        notice = time.monotonic() + 30
        say('起動元の終了処理とGDrive同期を待ちます。')
        while proc_identity(pid) == state['identity'] and path.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('起動元の同期待ちが上限に達しました。未同期データ保護のためruntimeを残しています。')
            if time.monotonic() >= notice:
                say('起動元の同期・停止処理を待っています。')
                notice = time.monotonic() + 30
            time.sleep(0.25)
    # A crashed owner or a previous flush failure still needs a verified flush.
    result = subprocess.run(['colab', 'status', '-s', SESSION], check=True,
                            capture_output=True, text=True, timeout=60)
    if f"Session '{SESSION}' not found." in result.stdout:
        path.unlink(missing_ok=True)
        say('runtime停止を確認しました。')
        return
    instance = Session(argparse.Namespace(flush_minutes=minutes))
    if not instance.flush_drive():
        raise RuntimeError('GDriveの同期を確認できないためruntimeを停止していません。')
    stop_runtime()
    path.unlink(missing_ok=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'stop', 'codex', 'client', 'status', 'doctor', 'check-drive'])
    parser.add_argument('--port', type=int, default=8085)
    parser.add_argument('--context', type=int, choices=[32768, 65536, 131072, 262144], default=262144)
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/Strata-data')
    parser.add_argument('--variant', help='Model variant: standard or uncensored (default from manifest.json)')
    parser.add_argument('--max-hours', type=float, default=4.0, help='Hard runtime lifetime, including setup (default 4 hours)')
    parser.add_argument('--startup-minutes', type=float, default=180.0)
    parser.add_argument('--flush-minutes', type=float, default=1440.0,
                        help='Drive flush wait (default 24 hours, as in Colab; failure preserves runtime; extra to max-hours)')
    parser.add_argument('--force', action='store_true', help='stop only: skip Drive sync and discard outstanding writes')
    parser.add_argument('--reverify', action='store_true', help='Read and hash all cached model files again')
    parser.add_argument('--skip-cache-mtime', action=argparse.BooleanOptionalAction, default=True,
                        help='Ignore modification time of size/SHA-256 verified cache (default); --reverify wins')
    parser.add_argument('--drive-auth', choices=['cli', 'browser'], default='cli',
                        help='Drive mounting via CLI or the browser UI of the existing runtime')
    args, extra = parser.parse_known_args()
    if args.force and args.command != 'stop':
        parser.error('--force is supported only with stop')
    if extra[:1] == ['--']:
        extra = extra[1:]
    if extra and args.command not in ('client', 'codex'):
        parser.error('Extra arguments are supported only after client/codex --')
    if not 1 <= args.port <= 65535 or args.max_hours <= 0 or args.startup_minutes <= 0 or args.flush_minutes <= 0:
        parser.error('port and time limits must be positive and valid')
    data = Path(args.data_dir)
    if '..' in data.parts or not data.is_relative_to('/content/drive/MyDrive'):
        parser.error('--data-dir must be within /content/drive/MyDrive')
    STATE.mkdir(mode=0o700, exist_ok=True)
    STATE.chmod(0o700)
    if args.command == 'doctor':
        say('Required tools: ' + ', '.join(f'{name}={shutil.which(name) or "MISSING"}' for name in ('colab', 'ssh', 'ssh-keygen', 'codex')))
        say('Drive destination: ' + args.data_dir)
        variant, profile = select_variant(args.variant)
        say('Model variants: ' + ', '.join(sorted(manifest()['variants'])))
        say(f'Selected variant: {variant} ({profile["model"]})')
        say('GGUF bytes: ' + str(sum(x['size'] for x in profile['files'][:2])))
        catalog(args.context, profile)
        say('Codex model catalog generated; no runtime allocated.')
        return 0
    if args.command == 'stop':
        stop(args)
        return 0
    if args.command == 'status':
        subprocess.run(['colab', 'status', '-s', SESSION], check=True)
        say('API: ' + json.dumps(health(args.port), ensure_ascii=False))
        return 0
    if args.command == 'client':
        info = health(args.port)
        if not info:
            raise RuntimeError('Start ./colab-qwen start in another terminal first.')
        running, profile = profile_for_model(info['model'])
        if args.variant and args.variant != running:
            raise RuntimeError(f'稼働中の変種は {running} です。{args.variant} を使うには先に stop して起動し直す。')
        context = info['max_context']
        catalog(context, profile)
        say('接続先の変種: ' + running)
        return subprocess.call(codex_command(args.port, context, extra, profile))
    lock = (STATE / 'session.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Supervisor already running. Use client or stop.') from None
    state_path = STATE / 'session.json'
    session = Session(args)
    state_path.write_text(json.dumps({'pid': os.getpid(), 'identity': proc_identity(os.getpid()),
                                     'port': args.port, 'variant': session.variant,
                                     'flush_minutes': args.flush_minutes}))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.alarm(max(1, int(args.max_hours * 3600)))
    result = 0
    try:
        if args.command == 'check-drive':
            session.check_drive()
            return 0
        session.start()
        child = None
        if args.command == 'codex':
            child = subprocess.Popen(codex_command(args.port, args.context, extra, session.profile))
            session.children.append(child)
        else:
            say('このターミナルは起動したまま使います。Ctrl+C または別ターミナルの ./colab-qwen stop で終了。')
            say('Codex: ./colab-qwen client -- [Codex arguments]')
            say(f'起動した変種: {session.variant} ({session.profile["model"]})')
        while True:
            if session.master.poll() is not None:
                raise RuntimeError('SSH tunnel closed; stopping runtime.')
            if child and child.poll() is not None:
                result = child.returncode
                break
            time.sleep(1)
    except KeyboardInterrupt:
        say('終了要求または時間上限により停止します。')
        result = 130
    finally:
        sync_confirmed = None
        try:
            sync_confirmed = session.close()
        except Exception:
            say('終了処理に失敗しました。runtimeを確認し、./colab-qwen stop で同期・停止を再試行してください。')
            raise
        else:
            if sync_confirmed is not False:
                state_path.unlink(missing_ok=True)
        finally:
            lock.close()
        if sync_confirmed is False and sys.exc_info()[0] is None:
            raise RuntimeError('GDriveの同期完了を確認できずruntimeを残しています。drive-sync-status.jsonを確認してください。')
    return result


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        say(str(exc))
        sys.exit(1)
