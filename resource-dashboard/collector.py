"""Read-only machine probe. Also executed over SSH, without installing remote files."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path


def read(path, default=''):
    try:
        return Path(path).read_text(errors='replace')
    except OSError:
        return default


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def cpu():
    info = read('/proc/cpuinfo')
    names = re.findall(r'^model name\s*:\s*(.+)$', info, re.M)
    pairs = set()
    for block in info.split('\n\n'):
        fields = dict(re.findall(r'^([^:\n]+?)\s*:\s*(.+)$', block, re.M))
        if 'physical id' in fields and 'core id' in fields:
            pairs.add((fields['physical id'], fields['core id']))
    counters = {}
    for line in read('/proc/stat').splitlines():
        parts = line.split()
        if not parts or not re.fullmatch(r'cpu\d*', parts[0]):
            continue
        # Guest time is already included in user/nice; do not count it twice.
        values = [int(v) for v in parts[1:9]]
        if len(values) >= 5:
            counters[parts[0]] = [sum(values), values[3] + values[4]]
    clocks = [float(v) for v in re.findall(r'^cpu MHz\s*:\s*([\d.]+)', info, re.M)]
    temps = []
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        kind = read(hwmon / 'name').strip()
        if kind not in ('coretemp', 'k10temp', 'zenpower', 'cpu_thermal', 'acpitz'):
            continue
        for sensor in hwmon.glob('temp*_input'):
            value = number(read(sensor))
            if value is not None and 0 < value < 150000:
                temps.append(value / 1000)
    quota = read('/sys/fs/cgroup/cpu.max').split()
    return {'name': names[0] if names else platform.processor() or platform.machine(),
            'threads': os.cpu_count(), 'cores': len(pairs) or None,
            'frequency_mhz': round(sum(clocks) / len(clocks)) if clocks else None,
            'temperature_c': max(temps) if temps else None, 'counters': counters,
            'quota_cores': number(quota[0]) / float(quota[1])
            if len(quota) == 2 and number(quota[0]) is not None and float(quota[1]) > 0 else None}


def memory():
    fields = {}
    for line in read('/proc/meminfo').splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key] = int(value.strip().split()[0]) * 1024
    total, available = fields.get('MemTotal'), fields.get('MemAvailable')
    limit = number(read('/sys/fs/cgroup/memory.max').strip())
    return {'total_bytes': total, 'available_bytes': available,
            'used_bytes': total - available if total is not None and available is not None else None,
            'cached_bytes': fields.get('Cached'), 'swap_total_bytes': fields.get('SwapTotal'),
            'swap_used_bytes': fields.get('SwapTotal', 0) - fields.get('SwapFree', 0),
            'container_limit_bytes': int(limit) if limit is not None else None,
            'container_used_bytes': number(read('/sys/fs/cgroup/memory.current').strip())}


def gpus():
    keys = ('index', 'uuid', 'name', 'utilization.gpu', 'utilization.memory', 'memory.used', 'memory.total',
            'temperature.gpu', 'power.draw', 'power.limit', 'fan.speed', 'clocks.current.graphics', 'driver_version')
    fields = ('index', 'uuid', 'name', 'util_percent', 'memory_util_percent', 'used_mib', 'total_mib',
              'temperature_c', 'power_w', 'power_limit_w', 'fan_percent', 'clock_mhz', 'driver')
    if shutil.which('nvidia-smi'):
        for count in (len(keys), 8):
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(keys[:count]),
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
                if result.returncode:
                    continue
                devices = []
                for row in csv.reader(result.stdout.splitlines()):
                    if len(row) != count:
                        continue
                    gpu = {k: (v.strip() if k in ('uuid', 'name', 'driver') else number(v.strip()))
                           for k, v in zip(fields[:count], row)}
                    for key in ('used', 'total'):
                        value = gpu.pop(key + '_mib', None)
                        gpu[key + '_bytes'] = int(value * 2**20) if value is not None else None
                    devices.append(gpu)
                if devices:
                    return devices
            except (OSError, subprocess.TimeoutExpired):
                break
    devices = []
    for card in sorted(Path('/sys/class/drm').glob('card[0-9]*')):
        if not re.fullmatch(r'card\d+', card.name):
            continue
        dev = card / 'device'
        if read(dev / 'vendor').strip() != '0x1002':
            continue
        gpu = {'index': len(devices), 'uuid': card.name,
               'name': read(dev / 'product_name').strip() or 'AMD GPU',
               'util_percent': number(read(dev / 'gpu_busy_percent')),
               'used_bytes': number(read(dev / 'mem_info_vram_used')),
               'total_bytes': number(read(dev / 'mem_info_vram_total'))}
        hwmon = next((dev / 'hwmon').glob('hwmon*'), None)
        if hwmon:
            for field, file, scale in [('temperature_c', 'temp1_input', 1000), ('power_w', 'power1_average', 1e6),
                                       ('power_limit_w', 'power1_cap', 1e6)]:
                value = number(read(hwmon / file))
                gpu[field] = value / scale if value is not None else None
        devices.append(gpu)
    return devices


def disks(work):
    paths = [('ワークスペース', os.getcwd())] if not work else [('runtime SSD', '/content')]
    if Path('/content/drive/MyDrive').is_dir():
        paths.append(('Google Drive', '/content/drive/MyDrive'))
    mounts = []
    for line in read('/proc/mounts').splitlines():
        parts = line.split()
        if len(parts) > 3:
            mounts.append((parts[1].replace('\\040', ' '), parts[0], parts[2]))
    result = []
    for label, path in paths:
        try:
            usage = shutil.disk_usage(path)
            matches = [m for m in mounts if path == m[0] or path.startswith(m[0].rstrip('/') + '/')]
            mount = max(matches, key=lambda m: len(m[0])) if matches else ('', '', '')
            result.append({'label': label, 'path': path, 'total_bytes': usage.total, 'used_bytes': usage.used,
                           'free_bytes': usage.free, 'filesystem': mount[2], 'device': mount[1],
                           'virtual': label == 'Google Drive'})
        except OSError:
            pass
    return result


def io_counters():
    disk_read = disk_write = 0
    devices = {p.name for p in Path('/sys/block').glob('*') if re.match(r'^(sd|vd|xvd|nvme)', p.name)}
    for line in read('/proc/diskstats').splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[2] in devices:
            disk_read += int(parts[5]) * 512
            disk_write += int(parts[9]) * 512
    rx = tx = 0
    for line in read('/proc/net/dev').splitlines():
        if ':' not in line:
            continue
        name, body = line.split(':', 1)
        parts = body.split()
        if name.strip() != 'lo' and not name.strip().startswith(('veth', 'docker', 'br-')) and len(parts) >= 9:
            rx += int(parts[0])
            tx += int(parts[8])
    return {'disk_read_bytes': disk_read, 'disk_write_bytes': disk_write, 'network_rx_bytes': rx, 'network_tx_bytes': tx}


def model_config(work):
    if not work:
        return {}
    path = next((p for p in (Path(work) / 'strata.json', Path(work) / 'project-maya/maya-maya-l.json') if p.is_file()), None)
    if path is None:
        return {}
    try:
        config = json.loads(path.read_text())
        args = config.get('args') or []
        paths = []
        for flag in ('--native', '--ple-gguf', '--pack', '--mtp'):
            if flag in args and args.index(flag) + 1 < len(args):
                value = args[args.index(flag) + 1]
                if isinstance(value, str):
                    paths.append({'role': flag.removeprefix('--'), 'path': value})
        ctx = args[args.index('--max-context') + 1] if '--max-context' in args else None
        kv = args[args.index('--kv') + 1] if '--kv' in args else None
        quant = next((m.group(0) for p in paths if (m := re.search(r'(?:IQ|Q)\d[A-Z0-9_]*', Path(p['path']).name))), None)
        return {'model': config.get('model_name'), 'paths': paths, 'max_context': number(ctx), 'kv': kv,
                'quantization': quant, 'mtp': '--mtp' in args, 'api_monitor': config.get('api_monitor')}
    except (OSError, ValueError, TypeError, IndexError):
        return {}


def collect(work=None):
    release = platform.release()
    boot_id = read('/proc/sys/kernel/random/boot_id').strip()
    uptime = number(read('/proc/uptime').split(' ')[0])
    scope = 'WSL2' if 'microsoft' in release.lower() else 'Linux'
    if Path('/.dockerenv').exists():
        scope += ' / Container'
    return {'time': time.time(), 'boot_id': boot_id, 'hostname': platform.node(), 'os': platform.system(),
            'kernel': release, 'scope': 'Colab runtime' if work else scope, 'uptime_s': uptime,
            'cpu': cpu(), 'ram': memory(), 'gpus': gpus(), 'disks': disks(work),
            'io_counters': io_counters(), 'model_config': model_config(work)}


if __name__ == '__main__':
    print(json.dumps(collect(), ensure_ascii=False, allow_nan=False))
