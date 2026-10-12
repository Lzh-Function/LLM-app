"""Read-only DriveFS diagnostics; report known indicators, never raw log contents."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


PATTERNS = {
    'root_folder_timeout': r'QueryManager timed out',
    'auth_error': r'UNAUTHENTICATED|invalid_grant|invalid.credentials|authentication.failed|unauthori[sz]ed|HTTP[/ ]\S*\s+401',
    'permission_error': r'PERMISSION_DENIED|permission.denied|access.denied|HTTP[/ ]\S*\s+403',
    'domain_policy': r'The domain policy has disabled Drive File Stream',
    'network_error': r'connection.refused|connection.timed.out|name.resolution|could.not.resolve|network.is.unreachable',
    'fuse_error': r'failed.to.mount|fuse:|fusermount:|Transport endpoint is not connected',
    'process_crash': r'SIGSEGV|segmentation.fault|Check failed:|FATAL|encountered a problem and has stopped',
    'memory_error': r'out.of.memory|Cannot allocate memory|bad_alloc',
    'quota_error': r'quota.exceeded|RESOURCE_EXHAUSTED|rate.limit.exceeded',
    'missing_library': r'error while loading shared libraries|cannot open shared object file',
}


def summarize(text):
    return {name: len(re.findall(pattern, text, re.IGNORECASE))
            for name, pattern in PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)}


def collect():
    result = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'mounted': os.path.ismount('/content/drive'),
        'mydrive_exists': Path('/content/drive/MyDrive').is_dir(),
        'fuse_device_exists': Path('/dev/fuse').exists(),
        'logs': [],
    }
    try:
        from google.colab import drive
        result['drivefs_binary_exists'] = (Path(drive._env().root_dir) / 'opt/google/drive/drive').exists()
        directory = Path(drive._logs_dir())
        for name in ('drive_fs.txt', 'drive_fs.txt.1', 'dpb.txt'):
            path = directory / name
            if not path.is_file():
                continue
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 2 * 1024 * 1024))
                content = stream.read().decode('utf-8', errors='replace')
            result['logs'].append({'name': name, 'bytes': path.stat().st_size,
                                   'indicators': summarize(content)})
    except Exception as exc:
        # Exception messages and arbitrary log lines can contain credentials.
        result['diagnostic_error_type'] = type(exc).__name__
    return result


if __name__ == '__main__':
    print('GLM_DRIVE_DIAGNOSTICS=' + json.dumps(collect()))
