"""A100 Colab/GLM-5.3 lifecycle and local Codex client. No third party dependencies."""
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
SESSION = 'glm53-a100'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def say(message):
    print('[glm53-a100]', message, flush=True)


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


def select_context(profile, requested):
    """Each variant declares its own context options; the manifest default wins when none is asked for."""
    value = requested or profile['default_context']
    if value not in profile['contexts']:
        raise RuntimeError('--context は ' + ' / '.join(map(str, profile['contexts'])) + ' のどれかにしてください。')
    return value


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
    """Readiness for either engine.

    Maya answers /health with its model and max_context; llama.cpp answers /health with only a status
    and keeps the model name and n_ctx in /props. Both serve /props, so probe /health for readiness and
    /props for identity. A loading engine reports max_context 0, which is not ready.
    """
    try:
        with OPENER.open(f'http://127.0.0.1:{port}/health', timeout=3) as response:
            info = json.load(response)
        if info.get('status') != 'ok':
            return None
        with OPENER.open(f'http://127.0.0.1:{port}/props', timeout=3) as response:
            props = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    served = {profile['model'] for profile in manifest()['variants'].values()}
    name = props.get('model_alias') or info.get('model')
    if name not in served or (model is not None and name != model):
        return None
    context = (props.get('default_generation_settings') or {}).get('n_ctx') or info.get('max_context')
    if not isinstance(context, int) or context <= 0:
        return None
    info['model'] = name
    info['max_context'] = context
    return info


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
                 description='GLM-5.3 on Colab A100, Google Drive weights.',
                 context_window=context, max_context_window=context, input_modalities=['text'],
                 supports_image_detail_original=False)
    path = STATE / 'models.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return path


def codex_command(port, context, arguments, profile):
    settings = {
        'model': profile['model'], 'model_provider': 'glm53_colab',
        'model_catalog_json': str(STATE / 'models.json'),
        'model_context_window': context, 'model_auto_compact_token_limit': context * 3 // 4,
        'model_reasoning_effort': 'high', 'show_raw_agent_reasoning': True,
        'model_providers.glm53_colab.name': profile['provider_name'],
        'model_providers.glm53_colab.base_url': f'http://127.0.0.1:{port}/v1',
        'model_providers.glm53_colab.wire_api': 'responses',
        'model_providers.glm53_colab.requires_openai_auth': False,
        'model_providers.glm53_colab.supports_websockets': False,
        'model_providers.glm53_colab.stream_idle_timeout_ms': 600000,
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


class Session:
    def __init__(self, args):
        self.args = args
        self.variant, self.profile = select_variant(getattr(args, 'variant', None))
        self.children = []
        self.owned = False
        self.drive_mounted = False
        self.master = None
        self.socket = f'/tmp/glm53-a100-{os.getuid()}-{uuid.uuid4().hex[:12]}.sock'

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
            raise RuntimeError(f'{command[0]} の待機時間上限（{timeout}秒）を超えました。') from None
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

    def code(self, code, *, quiet=False, timeout=60, progress=None):
        # colab-cli can exit 0 even when the notebook raises an exception.
        # Only a marker printed AFTER executing the entire script proves success.
        marker = 'GLM_EXEC_OK_' + uuid.uuid4().hex
        path = STATE / 'execute.py'
        path.write_text(f'exec(compile({code!r}, "glm-setup", "exec"))\nprint({marker!r})\n')
        output = self.colab('exec', '-s', SESSION, '-f', str(path), '--timeout', str(max(1, timeout - 5)),
                            timeout=timeout, progress=progress,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = output.splitlines()
        completed = marker in lines
        output = '\n'.join(line for line in lines if line != marker)
        if output and not quiet:
            print(output, flush=True)
        if not completed:
            raise RuntimeError('Colab側のPython実行に失敗しました（実行完了を確認できません）。')
        return output

    def allocate(self):
        if not shutil.which('colab'):
            raise RuntimeError('colab is required.')
        status = self.colab('status', '-s', SESSION, stdout=subprocess.PIPE, text=True, timeout=30)
        if f"Session '{SESSION}' not found." not in status:
            raise RuntimeError(f'Session {SESSION} already exists. Use ./colab-glm stop first.')
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
        if 'GLM_DRIVE=true' not in output.splitlines():
            raise RuntimeError('Drive mount failed')

    def mount_drive(self):
        probe = "import os, json\nprint('GLM_DRIVE=' + json.dumps(os.path.ismount('/content/drive') and os.path.isdir('/content/drive/MyDrive')))\n"
        say('GDriveのマウント状態を確認します（最大20秒）。')
        output = self.code(probe, quiet=True, timeout=20)
        if 'GLM_DRIVE=true' in output.splitlines():
            self.drive_mounted = True
            return
        if not sys.stdin.isatty():
            raise RuntimeError('GDrive authorization needs a terminal. Run ./colab-glm start in your terminal.')
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
            records = [line.removeprefix('GLM_DRIVE_RESULT=') for line in output.splitlines()
                       if line.startswith('GLM_DRIVE_RESULT=')]
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

    def start(self):
        remote_port = manifest()['server_port']
        if not isinstance(remote_port, int) or not 1 <= remote_port <= 65535 or remote_port == 8080:
            raise RuntimeError('manifest must select a dedicated remote GLM port, not Colab port 8080.')
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
        self.code("from pathlib import Path\nimport os\nassert os.path.ismount('/content/drive') and Path('/content/drive/MyDrive').is_dir(), 'Drive mount failed'\nPath('/content/glm53-a100').mkdir(exist_ok=True)\n")
        for name in ('remote.py', 'manifest.json'):
            self.colab('upload', '-s', SESSION, str(HERE / name), 'content/glm53-a100/' + name, timeout=45)
        remote_args = ['python3', '-u', '/content/glm53-a100/remote.py', '--data-dir', self.args.data_dir,
                       '--context', str(self.args.context), '--variant', self.variant]
        if self.args.reverify:
            remote_args.append('--reverify')
        self.code('import subprocess\nfrom pathlib import Path\n'
                  "log=open('/content/glm53-a100/bootstrap.log','ab',buffering=0)\n"
                  f'p=subprocess.Popen({remote_args!r},stdout=log,stderr=log,start_new_session=True)\n'
                  "Path('/content/glm53-a100/bootstrap.pid').write_text(str(p.pid))\nprint('BOOTSTRAP_PID',p.pid)\n")
        proxy = shlex.join(['colab', 'ssh', '--proxy-mode', '-s', SESSION, '--identity', str(identity)])
        ssh_log = (STATE / 'ssh.log').open('ab', buffering=0)
        command = ['ssh', '-M', '-S', self.socket, '-N', '-i', str(identity),
                   '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60',
                   '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
                   '-o', 'ExitOnForwardFailure=yes', '-o', 'StrictHostKeyChecking=accept-new',
                   '-o', f'UserKnownHostsFile={STATE / "known_hosts"}',
                   '-o', 'HostKeyAlias=glm53-a100-' + uuid.uuid4().hex,
                   '-o', 'ProxyCommand=' + proxy,
                   '-L', f'127.0.0.1:{self.args.port}:127.0.0.1:{remote_port}', 'root@colab-runtime']
        self.master = subprocess.Popen(command, stdout=ssh_log, stderr=ssh_log, start_new_session=True)
        ssh_log.close()
        self.children.append(self.master)
        deadline = time.monotonic() + self.args.startup_minutes * 60
        last_tail = None
        while time.monotonic() < deadline:
            if self.master.poll() is not None:
                raise RuntimeError(f'SSH tunnel failed. See {STATE / "ssh.log"}')
            if health(self.args.port, self.profile['model']):
                say(f'準備完了: http://127.0.0.1:{self.args.port}/v1 ({self.variant})')
                return
            if Path(self.socket).exists():
                tail = self.ssh("tail -n 4 /content/glm53-a100/bootstrap.log; "
                                "pid=$(cat /content/glm53-a100/bootstrap.pid); state=$(ps -o stat= -p \"$pid\"); "
                                "case \"$state\" in ''|Z*) echo GLM_BOOTSTRAP_FAILED;; esac; "
                                "if test -f /content/glm53-a100/failed; then echo GLM_BOOTSTRAP_FAILED; fi")
                if tail != last_tail:
                    print(tail.rstrip(), flush=True)
                    last_tail = tail
                if 'GLM_BOOTSTRAP_FAILED' in tail:
                    full = self.ssh('tail -n 80 /content/glm53-a100/bootstrap.log')
                    (STATE / 'bootstrap-error.log').write_text(full)
                    raise RuntimeError(f'Bootstrap failed. See {STATE / "bootstrap-error.log"}')
            time.sleep(5)
        raise RuntimeError('Startup timed out; attempting Drive sync before stopping the runtime.')

    def flush_drive(self):
        minutes = getattr(self.args, 'flush_minutes', 10.0)
        seconds = max(1, int(minutes * 60))
        say(f'GDriveの未同期データを同期してから停止します（最大{minutes:g}分）。')
        helper = (HERE / 'shutdown_remote.py').read_text()
        code = (f'ns={{"__name__":"glm_shutdown"}}\nexec(compile({helper!r}, "glm-shutdown", "exec"), ns)\n'
                f'result=ns["shutdown"]({seconds})\nprint("GLM_DRIVE_SYNC=" + result)\n')
        record = {'time_unix': time.time(), 'timeout_seconds': seconds, 'confirmed': False}
        try:
            output = self.code(code, quiet=True, timeout=seconds + 30,
                               progress='GDriveの同期完了を待っています。同期中もruntimeは稼働しています。')
            if 'GLM_DRIVE_SYNC=flushed' not in output.splitlines():
                raise RuntimeError('Drive was not mounted; cloud sync could not be confirmed.')
            record['confirmed'] = True
            self.drive_mounted = False
            say('GDriveの同期完了を確認しました。')
        except Exception as exc:
            record['error_type'] = type(exc).__name__
            say('GDriveの同期完了を確認できませんでした。未同期ファイルは次回再取得が必要になる可能性があります。')
        path = STATE / 'drive-sync-status.json'
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        path.chmod(0o600)
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
            try:
                if self.drive_mounted:
                    sync_confirmed = self.flush_drive()
            finally:
                say('runtimeを停止します。')
                stop_runtime()
        return sync_confirmed


def stop_runtime():
    subprocess.run(['colab', 'stop', '-s', SESSION], check=True, timeout=120)
    result = subprocess.run(['colab', 'status', '-s', SESSION], check=True,
                            capture_output=True, text=True, timeout=60)
    if f"Session '{SESSION}' not found." not in result.stdout:
        raise RuntimeError('Runtime stop could not be confirmed. Run colab stop -s glm53-a100 manually.')
    say('runtime停止を確認しました。')


def stop():
    path = STATE / 'session.json'
    if path.is_file():
        state = json.loads(path.read_text())
        pid = state['pid']
        if proc_identity(pid) == state.get('identity') and pid != os.getpid():
            os.kill(pid, signal.SIGTERM)
            # The owner stops children/runtime before releasing its lock.
            deadline = time.monotonic() + state.get('flush_minutes', 10.0) * 60 + 240
            notice = time.monotonic() + 30
            say('起動元の終了処理とGDrive同期を待ちます。')
            while time.monotonic() < deadline:
                if proc_identity(pid) != state.get('identity') or not path.exists():
                    break
                if time.monotonic() >= notice:
                    say('起動元の同期・停止処理を待っています。')
                    notice = time.monotonic() + 30
                time.sleep(0.25)
            else:
                say('起動元の停止待ちが上限に達しました。同期未完了の可能性がありますがruntimeを停止します。')
    stop_runtime()  # Works after a crashed/killed supervisor, too.


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'stop', 'codex', 'client', 'status', 'doctor', 'check-drive'])
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--context', type=int, help='Context length; the options differ per variant (see manifest.json)')
    parser.add_argument('--data-dir', default='/content/drive/MyDrive/GLM53-data')
    parser.add_argument('--variant', help='Model variant: flash / flash-uncensored / full (default from manifest.json)')
    parser.add_argument('--max-hours', type=float, default=6.0, help='Hard runtime lifetime, including setup (default 6 hours)')
    parser.add_argument('--startup-minutes', type=float, default=240.0,
                        help='Maximum first-run setup: downloads plus the engine build (default 240 minutes)')
    parser.add_argument('--flush-minutes', type=float, default=10.0,
                        help='Maximum Drive flush wait during shutdown (default 10 minutes; extra to max-hours)')
    parser.add_argument('--reverify', action='store_true', help='Read and hash all cached model files again')
    parser.add_argument('--drive-auth', choices=['cli', 'browser'], default='cli',
                        help='Drive mounting via CLI or the browser UI of the existing runtime')
    args, extra = parser.parse_known_args()
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
        args.context = select_context(profile, args.context)
        say('Model variants: ' + ', '.join(sorted(manifest()['variants'])))
        say(f'Selected variant: {variant} ({profile["model"]}, engine {profile["engine"]}, context {args.context})')
        say('GGUF bytes: ' + str(sum(x['size'] for x in profile['files'])))
        catalog(args.context, profile)
        say('Codex model catalog generated; no runtime allocated.')
        return 0
    if args.command == 'stop':
        stop()
        return 0
    if args.command == 'status':
        subprocess.run(['colab', 'status', '-s', SESSION], check=True)
        say('API: ' + json.dumps(health(args.port), ensure_ascii=False))
        return 0
    if args.command == 'client':
        info = health(args.port)
        if not info:
            raise RuntimeError('Start ./colab-glm start in another terminal first.')
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
    args.context = select_context(session.profile, args.context)
    state_path.write_text(json.dumps({'pid': os.getpid(), 'identity': proc_identity(os.getpid()),
                                     'port': args.port, 'variant': session.variant,
                                     'context': args.context, 'flush_minutes': args.flush_minutes}))
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
            say('このターミナルは起動したまま使います。Ctrl+C または別ターミナルの ./colab-glm stop で終了。')
            say('Codex: ./colab-glm client -- [Codex arguments]')
            say(f'起動した変種: {session.variant} ({session.profile["model"]}, context {args.context})')
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
            say('自動停止を確認できません。colab stop -s glm53-a100 を実行してください。')
            raise
        else:
            state_path.unlink(missing_ok=True)
        finally:
            lock.close()
        if sync_confirmed is False and sys.exc_info()[0] is None:
            raise RuntimeError('runtimeは停止済みですが、GDriveの同期完了を確認できませんでした。drive-sync-status.jsonを確認してください。')
    return result


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        say(str(exc))
        sys.exit(1)
