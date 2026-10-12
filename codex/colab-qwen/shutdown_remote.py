"""Stop the owned Strata process group, then flush Drive before releasing its VM."""
import os
import signal
import time
from pathlib import Path

WORK = Path('/content/strata-qwen')


def stop_bootstrap():
    pidfile = WORK / 'bootstrap.pid'
    if not pidfile.is_file():
        return
    try:
        pid = int(pidfile.read_text())
        if pid <= 1 or os.getpgid(pid) != pid:
            return
        # A stale pid file must never kill a different process in this VM.
        if str(WORK).encode() not in Path(f'/proc/{pid}/cmdline').read_bytes():
            return
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, FileNotFoundError):
        return


def shutdown(flush_seconds):
    from google.colab import drive
    stop_bootstrap()
    if not os.path.ismount('/content/drive'):
        return 'not_mounted'
    drive.flush_and_unmount(timeout_ms=int(flush_seconds * 1000))
    return 'flushed'
