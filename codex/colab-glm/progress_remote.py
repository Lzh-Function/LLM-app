"""Read-only snapshot of the existing bootstrap and its children. Never rents/stops a VM."""
import json
import os
import re
from pathlib import Path

WORK = Path('/content/glm53-a100')


def snapshot():
    pid = int((WORK / 'bootstrap.pid').read_text())
    manifest = json.loads((WORK / 'manifest.json').read_text())
    expected_sizes = {Path(item['destination']).name: item['size']
                      for profile in manifest['variants'].values() for item in profile['files']}
    processes = {}
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = stat.read_text()
            fields = text.rsplit(')', 1)[1].split()
            processes[int(stat.parent.name)] = (int(fields[1]), fields[0], text.split('(', 1)[1].rsplit(')', 1)[0])
        except (OSError, IndexError):
            pass
    wanted = {pid}
    for _ in range(4):
        wanted.update(child for child, fields in processes.items() if fields[0] in wanted)
    result = []
    for child in sorted(wanted):
        if child not in processes:
            continue
        _, state, name = processes[child]
        files = []
        try:
            for fd in Path(f'/proc/{child}/fd').iterdir():
                try:
                    path = Path(os.readlink(fd))
                    if not path.name.endswith(('.gguf', '.gguf.part')):
                        continue
                    info = Path(f'/proc/{child}/fdinfo/{fd.name}').read_text()
                    offset = int(re.search(r'^pos:\s+(\d+)', info, re.M)[1])
                    total = expected_sizes.get(path.name.removesuffix('.part'), path.stat().st_size)
                    files.append({'name': path.name, 'position': offset, 'size': total})
                except (OSError, TypeError):
                    pass
        except OSError:
            pass
        result.append({'pid': child, 'name': name, 'state': state, 'files': files})
    log = WORK / 'bootstrap.log'
    tail = log.read_text(errors='replace').splitlines()[-4:] if log.is_file() else []
    tail = [re.sub(r'https?://\S+', '[URL]', line) for line in tail]
    return {'bootstrap_pid': pid, 'processes': result, 'last_log': tail}


if __name__ == '__main__':
    info = snapshot()
    for process in info['processes']:
        print(f"PID {process['pid']} {process['name']} state={process['state']}")
        for item in process['files']:
            print(f"  {item['name']}: position {item['position'] / 1e9:.2f}/{item['size'] / 1e9:.2f} GB")
    print('Latest bootstrap log:')
    print('\n'.join(info['last_log']))
