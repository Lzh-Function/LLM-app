"""Stop the owned GLM process group, then flush Drive before releasing its VM."""
import os
import json
import subprocess
import signal
import time
from pathlib import Path

WORK = Path('/content/glm53-a100')


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


def recover_drive(binary):
    """Restart the existing DriveFS cache using the runtime's prior authorization."""
    metadata = os.environ['TBE_EPHEM_CREDS_ADDR']
    uri = metadata if '://' in metadata else 'http://' + metadata
    folder = binary.parent
    command = [str(binary),
               '--features=crash_throttle_percentage:100,fuse_max_background:1000,max_read_qps:1000,'
               'max_write_qps:1000,max_operation_batch_size:15,max_parallel_push_task_instances:10,'
               'opendir_timeout_ms:120000,virtual_folders_omit_spaces:true,read_only_mode:false',
               '--metadata_server_auth_uri=' + uri.rstrip('/') + '/computeMetadata/v1',
               '--preferences=trusted_root_certs_file_path:' + str(folder / 'roots.pem') +
               ',feature_flag_restart_seconds:129600,mount_point_path:/content/drive']
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if os.path.ismount('/content/drive') and Path('/content/drive/MyDrive').is_dir():
            return
        if process.poll() is not None:
            raise RuntimeError('DriveFS cache recovery failed; runtime retained')
        time.sleep(1)
    raise RuntimeError('DriveFS cache recovery mount not confirmed; runtime retained')


def drive_daemon_running(binary):
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
            if argv[0] == str(binary).encode() and b'--push_changes_and_quit' not in argv:
                return True
        except OSError:
            continue
    return False


def shutdown(flush_seconds):
    from google.colab import drive
    stop_bootstrap()
    status = WORK / 'drive-sync-status.json'
    previous = json.loads(status.read_text()) if status.exists() else {}
    mounted = os.path.ismount('/content/drive')
    if not mounted and not previous:
        return 'not_mounted'
    if not mounted and previous.get('confirmed') is True:
        return 'flushed'
    # DriveFS can detach the mount before uploads finish. Persist only control
    # metadata, so a later stop can retry the pending flush after detachment.
    WORK.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({'confirmed': False, 'time_unix': time.time()}))
    if mounted:
        drive.flush_and_unmount(timeout_ms=int(flush_seconds * 1000))
    else:
        # Same controller/flags as google.colab.drive.flush_and_unmount, without
        # its early "not mounted" return. _env supplies the runtime-specific root.
        env = drive._env()
        binary = Path(env.root_dir) / 'opt/google/drive/drive'
        if not drive_daemon_running(binary):
            recover_drive(binary)
            drive.flush_and_unmount(timeout_ms=int(flush_seconds * 1000))
        else:
            subprocess.run([str(binary), '--push_changes_and_quit', '--single_process',
                            f'--timeout_sec={int(flush_seconds)}'],
                           check=True, timeout=flush_seconds + 5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    status.write_text(json.dumps({'confirmed': True, 'time_unix': time.time()}))
    return 'flushed'
