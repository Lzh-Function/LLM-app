"""Local resource dashboard: machine telemetry and read-only inference metrics."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import collector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / 'static'
STATE = HERE / '.runtime'
MAX_BODY = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def validate_url(value):
    if not isinstance(value, str):
        raise ValueError('API URLを指定してください。')
    value = value.strip().rstrip('/')
    parsed = urllib.parse.urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError('ポート番号が不正です。') from exc
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/v1') or port is None):
        raise ValueError('http://127.0.0.1:ポート の形式で推論APIを指定してください。')
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, '', '', ''))


def get_api(base, path):
    try:
        try:
            response = OPENER.open(base + path, timeout=1.5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read(MAX_BODY + 1)
            if len(raw) > MAX_BODY:
                return None
            text = raw.decode('utf-8', errors='replace')
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            return {'code': response.code, 'data': data, 'text': text if data is None else None}
    except (OSError, urllib.error.URLError, ValueError):
        return None


def num(value):
    if isinstance(value, bool):
        return None
    return collector.number(value)


def ratio(part, whole):
    part, whole = num(part), num(whole)
    return max(0, min(100, 100 * part / whole)) if part is not None and whole and whole > 0 else None


def text(value, limit=512):
    return str(value)[:limit] if isinstance(value, (str, int, float)) else None


def prometheus(raw):
    values = {}
    for line in raw.splitlines():
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([^\s]+)', line)
        if match:
            value = num(match[2])
            if value is not None:
                values[match[1]] = values.get(match[1], 0) + value
    return values


def normalize_model(base, health, metrics, status, props):
    """Allowlist numeric statistics; no prompts, responses, traces, or raw request records."""
    health = health if isinstance(health, dict) else {}
    status = status if isinstance(status, dict) else {}
    props = props if isinstance(props, dict) else {}
    structured = isinstance(metrics, dict) and isinstance(metrics.get('live'), dict)
    engine = metrics.get('engine', {}) if structured else {}
    live = metrics.get('live', {}) if structured else {}
    totals = metrics.get('totals', {}) if structured else {}
    engine = engine if isinstance(engine, dict) else {}
    totals = totals if isinstance(totals, dict) else {}
    requests = metrics.get('requests') if structured else None
    last = requests[0] if isinstance(requests, list) and requests else {}
    last = last if isinstance(last, dict) else {}
    timings = status.get('last_timings') or {}
    timings = timings if isinstance(timings, dict) else {}
    settings = props.get('default_generation_settings') or {}
    settings = settings if isinstance(settings, dict) else {}
    model_path = text(props.get('model_path'))
    name = text(engine.get('model') or health.get('model') or status.get('model') or props.get('model_alias'))
    name = name or (Path(model_path).name if model_path else '推論サーバー')
    quant = re.search(r'(?:IQ|Q)\d[A-Z0-9_]*', name + ' ' + (model_path or ''))
    state = text(live.get('state')) or ('loading' if health.get('status') == 'loading model' else 'idle')
    loaded = health.get('loaded', status.get('loaded', state not in ('unloaded', 'loading')))
    result = {'name': name, 'api_url': base, 'engine': 'Strata' if structured else 'llama.cpp / OpenAI互換',
              'version': text(engine.get('version') or status.get('engine')),
              'state': state, 'loaded': bool(loaded),
              'context': num(engine.get('max_context') or health.get('max_context')
                             or status.get('cache_max_tokens') or settings.get('n_ctx')),
              'quantization': quant[0] if quant else None,
              'kv': text(engine.get('kv')),
              'mtp': text(engine.get('spec')) if (num(engine.get('spec')) or 0) > 0 else None,
              'paths': [{'role': 'model', 'path': model_path}] if model_path else [],
              'uptime_s': num(status.get('uptime_s')),
              'stats': {'tok_s': num(live.get('tok_s')), 'speed_kind': 'rolling' if structured else 'unavailable',
                        'mean_tok_s': num(live.get('tok_s_mean')), 'last_tok_s': num(last.get('decode_tok_s')
                                         or timings.get('predicted_per_second')),
                        'prefill_tok_s': num(live.get('prefill_tok_s_mean')),
                        'last_prefill_tok_s': num(timings.get('prompt_per_second')),
                        'prompt_tokens': num(live.get('prompt_tokens')), 'generated_tokens': num(live.get('generated')),
                        'elapsed_s': num(live.get('elapsed_s')), 'queued': num(live.get('queued')),
                        'requests_total': num(totals.get('requests')), 'input_tokens_total': num(totals.get('prompt_tokens')),
                        'output_tokens_total': num(totals.get('output_tokens')),
                        'cache_hit_percent': ratio(totals.get('reused'), totals.get('prompt_tokens')),
                        'draft_accept_percent': ratio(totals.get('drafts_accepted'), totals.get('drafts_offered')),
                        'prompt_ms': num(last.get('prompt_ms') or timings.get('prompt_ms')),
                        'decode_ms': num(last.get('decode_ms') or timings.get('predicted_ms'))}}
    if structured:
        if state == 'idle':
            result['stats']['tok_s'] = 0
        total_time = num(totals.get('decode_ms'))
        total_output = num(totals.get('output_tokens'))
        result['stats']['average_tok_s'] = total_output / (total_time / 1000) if total_time and total_output is not None else None
        prompt_time = num(totals.get('prompt_ms'))
        prompt_total = num(totals.get('prompt_tokens'))
        reused = num(totals.get('reused')) or 0
        result['stats']['average_prefill_tok_s'] = (prompt_total - reused) / (prompt_time / 1000) if prompt_time and prompt_total is not None else None
        if result['uptime_s'] is None and num(totals.get('since')) is not None:
            result['uptime_s'] = max(0, time.time() - num(totals['since']))
    elif isinstance(metrics, str):
        values = prometheus(metrics)
        result['stats'].update(speed_kind='average',
                               average_tok_s=values.get('llamacpp:predicted_tokens_seconds'),
                               average_prefill_tok_s=values.get('llamacpp:prompt_tokens_seconds'),
                               output_tokens_total=values.get('llamacpp:tokens_predicted_total'),
                               input_tokens_total=values.get('llamacpp:prompt_tokens_total'),
                               queued=values.get('llamacpp:requests_deferred'))
        running = values.get('llamacpp:requests_processing')
        result['state'] = 'generating' if running and running > 0 else 'idle'
    return result


def model_metrics(base):
    response = get_api(base, '/health')
    if not response or response['code'] not in (200, 503):
        return None
    health = response['data']
    if not isinstance(health, dict):
        return None
    responses = {path: get_api(base, path) for path in ('/metrics', '/v1/status', '/props')}
    metrics = responses['/metrics']
    metrics = (metrics['data'] if metrics['data'] is not None else metrics['text']) if metrics and metrics['code'] == 200 else None
    status = responses['/v1/status']
    props = responses['/props']
    result = normalize_model(base, health, metrics, status['data'] if status and status['code'] == 200 else None,
                             props['data'] if props and props['code'] == 200 else None)
    if response['code'] == 503:
        result.update(state='loading', loaded=False)
    return result


def runtime_port(folder, default):
    try:
        state = json.loads((ROOT / 'codex' / folder / '.runtime/session.json').read_text())
        value = state.get('port')
        return value if isinstance(value, int) and 1 <= value <= 65535 else default
    except (OSError, ValueError):
        return default


def local_urls():
    ports = {5071, 8080}
    for path in (ROOT / 'codex').glob('*-runtime.json'):
        try:
            data = json.loads(path.read_text())
            port = data.get('backend_port')
            if isinstance(port, int) and 1 <= port <= 65535:
                ports.add(port)
        except (OSError, ValueError):
            pass
    return [f'http://127.0.0.1:{port}' for port in sorted(ports)]


def ssh_probe(session, work):
    sockets = []
    for path in Path('/tmp').glob(f'{session}-{os.getuid()}-*.sock'):
        try:
            info = path.stat()
            if stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid():
                sockets.append(path)
        except OSError:
            pass
    if not sockets:
        raise RuntimeError('SSH接続待ちです。対応する colab コマンドでruntimeを起動してください。')
    source = (HERE / 'collector.py').read_text()
    probe = ("import json; scope={'__name__':'dashboard_probe'}; "
             f'exec({source!r},scope); print(json.dumps(scope["collect"]({work!r}),ensure_ascii=False,allow_nan=False))')
    # A dead multiplex socket must not fall back to creating a new SSH/Colab session.
    command = ['ssh', '-S', str(max(sockets, key=lambda p: p.stat().st_mtime)), '-o', 'BatchMode=yes',
               '-o', 'ControlMaster=no', '-o', 'ProxyCommand=false', '-o', 'ConnectTimeout=2',
               'root@colab-runtime', 'python3 -B -c ' + shlex.quote(probe)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=12)
        if result.returncode:
            raise RuntimeError('既存SSH接続に応答がありません。再接続を待っています。')
        return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise RuntimeError('Colabのリソース情報を取得できません。接続を再確認しています。') from exc


class Worker:
    def __init__(self, target, interval, stop_event):
        self.target, self.interval, self.stop_event = target, interval, stop_event
        self.lock = threading.Lock()
        self.history = collections.deque(maxlen=150)
        self.previous = None
        self.value = {**target, 'available': False, 'state': 'connecting', 'sampled_at': None,
                      'hardware': None, 'models': [], 'history': [], 'error': None}
        self.thread = threading.Thread(target=self.loop, name='monitor-' + target['id'], daemon=True)
        self.thread.start()

    def rates(self, hardware):
        previous = self.previous
        self.previous = hardware
        per_core = []
        hardware['cpu']['util_percent'] = None
        hardware['io_rates'] = {}
        if previous and hardware.get('boot_id') == previous.get('boot_id'):
            dt = hardware['time'] - previous['time']
            for name, (total, idle) in hardware['cpu'].pop('counters', {}).items():
                old = previous['cpu'].get('_counters', {}).get(name)
                value = None
                if old and total > old[0] and idle >= old[1]:
                    value = max(0, min(100, 100 * (1 - (idle - old[1]) / (total - old[0]))))
                if name == 'cpu':
                    hardware['cpu']['util_percent'] = value
                else:
                    per_core.append({'name': name, 'util_percent': value})
            if dt > 0:
                for key, value in hardware['io_counters'].items():
                    old = previous.get('io_counters', {}).get(key)
                    hardware['io_rates'][key.removesuffix('_bytes') + '_bps'] = (value - old) / dt if old is not None and value >= old else None
        else:
            hardware['cpu'].pop('counters', None)
        hardware['cpu']['per_core'] = per_core
        hardware['ram']['percent'] = ratio(hardware['ram'].get('used_bytes'), hardware['ram'].get('total_bytes'))
        for gpu in hardware['gpus']:
            gpu['memory_percent'] = ratio(gpu.get('used_bytes'), gpu.get('total_bytes'))
        for disk in hardware['disks']:
            disk['percent'] = ratio(disk.get('used_bytes'), disk.get('total_bytes'))

    def sample(self):
        target = self.target
        if target['kind'] == 'colab':
            hardware = ssh_probe(target['session'], target['work'])
            urls = [f'http://127.0.0.1:{runtime_port(target["id"], target["port"])}']
        else:
            hardware = collector.collect()
            urls = target.get('api_urls') or local_urls()
        counters = dict(hardware['cpu'].get('counters', {}))
        self.rates(hardware)
        hardware['cpu']['_counters'] = counters
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool:
            models = [model for model in pool.map(model_metrics, urls) if model]
        config = hardware.pop('model_config', {})
        if models and target['kind'] == 'colab':
            models[0]['paths'] = config.get('paths') or models[0]['paths']
            for src, dst in [('quantization', 'quantization'), ('kv', 'kv')]:
                if config.get(src):
                    models[0][dst] = config[src]
        primary = next((m for m in models if m['state'] in ('generating', 'reading')), models[0] if models else None)
        gpu = hardware['gpus'][0] if hardware['gpus'] else {}
        stats = primary['stats'] if primary else {}
        point = {'time': hardware['time'], 'cpu': hardware['cpu']['util_percent'],
                 'gpu': gpu.get('util_percent'), 'ram': hardware['ram']['percent'], 'vram': gpu.get('memory_percent'),
                 'tok_s': stats.get('tok_s') if stats.get('speed_kind') == 'rolling' else stats.get('average_tok_s'),
                 'prefill_tok_s': stats.get('prefill_tok_s') if stats.get('speed_kind') == 'rolling' else stats.get('average_prefill_tok_s'),
                 'gpu_temp': gpu.get('temperature_c'),
                 'models': {m['api_url']: {
                     'tok_s': m['stats'].get('tok_s') if m['stats'].get('speed_kind') == 'rolling' else m['stats'].get('average_tok_s'),
                     'prefill_tok_s': m['stats'].get('prefill_tok_s') if m['stats'].get('speed_kind') == 'rolling' else m['stats'].get('average_prefill_tok_s')}
                     for m in models}}
        self.history.append(point)
        # Internal CPU counters are kept in memory for deltas, never returned to the browser.
        public_hw = {**hardware, 'cpu': {k: v for k, v in hardware['cpu'].items() if not k.startswith('_')}}
        return {**target, 'available': True, 'state': 'online', 'sampled_at': time.time(),
                'hardware': public_hw, 'models': models, 'history': list(self.history), 'error': None}

    def loop(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                value = self.sample()
            except Exception as exc:
                # Keep the last sample visible with a timestamp; never disguise stale data as live.
                with self.lock:
                    value = {**self.value, 'available': False, 'state': 'offline',
                             'error': str(exc) if isinstance(exc, RuntimeError) else 'リソース情報の取得に失敗しました。'}
            with self.lock:
                self.value = value
            self.stop_event.wait(max(0.2, self.interval - (time.monotonic() - started)))

    def snapshot(self):
        with self.lock:
            return self.value


class Monitor:
    def __init__(self, interval=2, state_dir=STATE):
        self.interval, self.state_dir = interval, state_dir
        self.lock = threading.Lock()
        self.workers = {}
        self.events = {}
        self.custom = []
        self.catalog = []
        for path in sorted(ROOT.glob('*/model.toml')):
            try:
                model = tomllib.loads(path.read_text())
                self.catalog.append({k: model.get(k) for k in ('id', 'name', 'description', 'default_context_length', 'max_context_length')})
            except (OSError, ValueError):
                pass
        targets = [{'id': 'local', 'name': 'ローカル', 'kind': 'local', 'builtin': True},
                   {'id': 'colab-qwen', 'name': 'Colab · Qwen', 'kind': 'colab', 'builtin': True,
                    'session': 'strata-qwen', 'work': '/content/strata-qwen', 'port': 8085},
                   {'id': 'colab-glm', 'name': 'Colab · GLM', 'kind': 'colab', 'builtin': True,
                    'session': 'glm53-a100', 'work': '/content/glm53-a100', 'port': 8086}]
        try:
            stored = json.loads((state_dir / 'targets.json').read_text())
            for item in stored[:9] if isinstance(stored, list) else []:
                if (isinstance(item, dict) and isinstance(item.get('id'), str)
                        and re.fullmatch(r'custom-[a-f0-9]{12}', item['id'])):
                    url = validate_url(item['api_urls'][0])
                    self.custom.append({'id': item['id'], 'name': text(item.get('name'), 60) or 'ローカルAPI',
                                        'kind': 'local', 'builtin': False, 'api_urls': [url]})
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
        for target in targets + self.custom:
            self.start(target)

    def start(self, target):
        event = threading.Event()
        self.events[target['id']] = event
        self.workers[target['id']] = Worker(target, self.interval, event)

    def state(self):
        with self.lock:
            targets = [worker.snapshot() for worker in self.workers.values()]
        return {'time': time.time(), 'interval': self.interval, 'targets': targets, 'catalog': self.catalog}

    def save(self):
        self.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        temporary = self.state_dir / 'targets.json.tmp'
        temporary.write_text(json.dumps(self.custom, ensure_ascii=False, indent=2))
        temporary.chmod(0o600)
        temporary.replace(self.state_dir / 'targets.json')

    def add(self, name, url):
        url = validate_url(url)
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise ValueError('接続名を1〜60文字で入力してください。')
        with self.lock:
            if len(self.custom) >= 9:
                raise ValueError('追加できる接続は9件までです。')
            target = {'id': 'custom-' + uuid.uuid4().hex[:12], 'name': name.strip(),
                      'kind': 'local', 'builtin': False, 'api_urls': [url]}
            self.custom.append(target)
            try:
                self.save()
            except OSError:
                self.custom.remove(target)
                raise
            self.start(target)
            return target

    def delete(self, target_id):
        with self.lock:
            worker = self.workers.get(target_id)
            if not worker or worker.target.get('builtin'):
                raise ValueError('追加した接続だけを削除できます。')
            previous = list(self.custom)
            self.custom = [item for item in self.custom if item['id'] != target_id]
            try:
                self.save()
            except OSError:
                self.custom = previous
                raise
            self.events.pop(target_id).set()
            self.workers.pop(target_id)

    def close(self):
        for event in self.events.values():
            event.set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/api/state':
            self.reply(200, self.server.monitor.state())
        elif path in ('/', '/index.html', '/app.js', '/style.css', '/favicon.svg'):
            file = STATIC / ('index.html' if path == '/' else path.lstrip('/'))
            content_type = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml'}[file.suffix]
            self.reply(200, file.read_bytes(), content_type + '; charset=utf-8')
        else:
            self.reply(404, {'error': 'Not found'})

    def mutate(self, deleting=False):
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self.reply(403, {'error': '外部ページからの設定変更は拒否しました。'})
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            self.reply(415, {'error': 'application/json is required'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8192:
                raise ValueError('リクエストサイズが不正です。')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('設定形式が不正です。')
            if deleting:
                self.server.monitor.delete(body.get('id'))
                result = {'ok': True}
            else:
                result = self.server.monitor.add(body.get('name'), body.get('api_url'))
            self.reply(200, result)
        except (ValueError, TypeError) as exc:
            self.reply(400, {'error': str(exc)})
        except OSError:
            self.reply(500, {'error': 'ローカル設定を保存できません。'})

    def do_POST(self):
        if self.path == '/api/targets':
            self.mutate()
        else:
            self.reply(404, {'error': 'Not found'})

    def do_DELETE(self):
        if self.path == '/api/targets':
            self.mutate(deleting=True)
        else:
            self.reply(404, {'error': 'Not found'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5090)
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not math.isfinite(args.interval) or not 1 <= args.interval <= 60:
        parser.error('port must be valid; interval must be between 1 and 60 seconds')
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        parser.error(f'Cannot listen on {args.host}:{args.port}: {exc}')
    server.monitor = Monitor(args.interval)
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    print(f'LLM Resource Dashboard: http://{args.host}:{args.port}', flush=True)
    print('Read-only monitoring · Local / Colab Qwen / Colab GLM · Ctrl+C to stop dashboard', flush=True)
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.monitor.close()
        server.server_close()


if __name__ == '__main__':
    main()
