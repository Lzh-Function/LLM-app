"""Launch Colab inference with disk writes denied. Uploaded to both runtimes.

Preparation happens outside this process. No conversation is accepted until the
kernel restrictions are installed, inherited log FDs are closed, and exec runs.
Unsupported hosts fail closed; there is deliberately no unprotected fallback.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import resource
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long
TMPFS_MAGIC = 0x01021994
MEMORY_ROOT = Path('/dev/shm')
WRITE_FILE = 1 << 1
REMOVE_DIR = 1 << 4
REMOVE_FILE = 1 << 5
MAKE_DIR = 1 << 7
MAKE_REG = 1 << 8
MAKE_SOCK = 1 << 9
MAKE_FIFO = 1 << 10
MAKE_SYM = 1 << 12
REFER = 1 << 13
TRUNCATE = 1 << 14
IOCTL_DEV = 1 << 15
# Also handle MAKE_CHAR/MAKE_BLOCK, but never grant them, including in tmpfs.
FS_WRITES = WRITE_FILE | sum(1 << bit for bit in range(4, 15))
MEMORY_WRITES = (WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG |
                 MAKE_SOCK | MAKE_FIFO | MAKE_SYM | REFER | TRUNCATE)
CORE_GUARD_SOURCE = '''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <sys/prctl.h>
#include <unistd.h>
/* exec resets dumpability. Every dynamically linked server/child starts here
   before it can read a prompt, even when core_pattern uses a host-side pipe. */
__attribute__((constructor)) static void colab_no_core_dump(void) {
    if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 ||
        prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0) _exit(125);
}
/* Colab disallows a fresh proc mount. Its existing proc allows CUDA thread
   names, but magic links can refer to another process's mount view. For the trusted
   Python/C++ servers, reject persistent writes through libc as well. This
   complements the read-only mounts; it is not a hostile-code syscall sandbox. */
/* CUDA writes its own thread names through proc. These are kernel RAM
   metadata, not disk files. Admit only comm files belonging to this process. */
static int own_thread_name(const char *path) {
    char resolved[PATH_MAX], prefix[80];
    if (!realpath(path, resolved)) return 0;
    snprintf(prefix, sizeof(prefix), "/proc/%ld/task/", (long)getpid());
    if (strncmp(resolved, prefix, strlen(prefix))) return 0;
    const char *thread = resolved + strlen(prefix);
    char *end; unsigned long tid = strtoul(thread, &end, 10);
    if (!tid || end == thread || strcmp(end, "/comm")) return 0;
    struct statfs fs;
    return statfs(resolved, &fs) == 0 && (unsigned long)fs.f_type == 0x9fa0UL;
}
static int private_write_path(int dirfd, const char *path) {
    if (!getenv("COLAB_PRIVATE_WRITES")) return 1;
    if (!path || !*path) return 0;
    char full[PATH_MAX];
    int n = (path[0] == '/' || dirfd == AT_FDCWD)
        ? snprintf(full, sizeof(full), "%s", path)
        : snprintf(full, sizeof(full), "/proc/self/fd/%d/%s", dirfd, path);
    if (n < 0 || n >= (int)sizeof(full)) return 0;
    if (own_thread_name(full)) return 1;
    struct stat st;
    if (stat(full, &st) == 0) {
        if (S_ISBLK(st.st_mode)) return 0;
        if (S_ISCHR(st.st_mode))
            return !strcmp(full, "/dev/null") || !strcmp(full, "/dev/zero") ||
                   !strcmp(full, "/dev/random") || !strcmp(full, "/dev/urandom") ||
                   !strncmp(full, "/dev/nvidia", 11);
    } else {
        char *slash = strrchr(full, '/');
        if (!slash) strcpy(full, ".");
        else if (slash == full) full[1] = 0;
        else *slash = 0;
    }
    struct statfs fs;
    return statfs(full, &fs) == 0 &&
           ((unsigned long)fs.f_type == 0x01021994UL || (unsigned long)fs.f_type == 0x858458f6UL);
}
static void *next_symbol(const char *name) {
    void *symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) _exit(125);
    return symbol;
}
static int writing(int flags) {
    return (flags & O_ACCMODE) != O_RDONLY || (flags & (O_CREAT | O_TRUNC | O_APPEND));
}
#define WRAP_OPEN(name) \\
int name(const char *path, int flags, ...) { \\
    mode_t mode = 0; \\
    if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) { \\
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); \\
    } \\
    if (writing(flags) && !private_write_path(AT_FDCWD, path)) { errno = EROFS; return -1; } \\
    return ((int (*)(const char *, int, ...))next_symbol(#name))(path, flags, mode); \\
}
WRAP_OPEN(open)
WRAP_OPEN(open64)
#define WRAP_OPENAT(name) \\
int name(int dirfd, const char *path, int flags, ...) { \\
    mode_t mode = 0; \\
    if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) { \\
        va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); \\
    } \\
    if (writing(flags) && !private_write_path(dirfd, path)) { errno = EROFS; return -1; } \\
    return ((int (*)(int, const char *, int, ...))next_symbol(#name))(dirfd, path, flags, mode); \\
}
WRAP_OPENAT(openat)
WRAP_OPENAT(openat64)
#define WRAP_FOPEN(name) \\
FILE *name(const char *path, const char *mode) { \\
    if (mode && (mode[0] != 'r' || strchr(mode, '+')) && !private_write_path(AT_FDCWD, path)) { \\
        errno = EROFS; return NULL; \\
    } \\
    return ((FILE *(*)(const char *, const char *))next_symbol(#name))(path, mode); \\
}
WRAP_FOPEN(fopen)
WRAP_FOPEN(fopen64)
int creat(const char *path, mode_t mode) { return open(path, O_WRONLY | O_CREAT | O_TRUNC, mode); }
int creat64(const char *path, mode_t mode) { return open64(path, O_WRONLY | O_CREAT | O_TRUNC, mode); }
int truncate(const char *path, off_t length) {
    if (!private_write_path(AT_FDCWD, path)) { errno = EROFS; return -1; }
    return ((int (*)(const char *, off_t))next_symbol("truncate"))(path, length);
}
int truncate64(const char *path, off64_t length) {
    if (!private_write_path(AT_FDCWD, path)) { errno = EROFS; return -1; }
    return ((int (*)(const char *, off64_t))next_symbol("truncate64"))(path, length);
}
'''


class PrivacyError(RuntimeError):
    pass


class Ruleset(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]


def checked(result, operation):
    if result < 0:
        code = ctypes.get_errno()
        raise PrivacyError(f'{operation}: {os.strerror(code)}; refusing unprotected inference')
    return result


def landlock_abi():
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'aarch64'):
        raise PrivacyError('Privacy guard requires Linux x86_64/aarch64')
    abi = checked(LIBC.syscall(444, None, 0, 1), 'Landlock ABI query')
    if abi < 3:
        raise PrivacyError('Landlock ABI >= 3 is required to deny writes AND truncation')
    return abi


def require_tmpfs(path):
    # statfs starts with a native long f_type; the oversized buffer holds the
    # complete struct on both supported architectures, without a libc dependency.
    buffer = ctypes.create_string_buffer(256)
    checked(LIBC.statfs(os.fsencode(path), ctypes.byref(buffer)), 'statfs')
    if ctypes.c_long.from_buffer(buffer).value != TMPFS_MAGIC:
        raise PrivacyError(f'{path} must be tmpfs; disk-backed temporary storage is forbidden')


def disable_swap():
    def active():
        return bool(Path('/proc/swaps').read_text().splitlines()[1:])
    if active():
        subprocess.run(['swapoff', '-a'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if active():
        raise PrivacyError('Swap is still enabled; refusing inference with pageable disk backing')


def disable_core_dumps():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Colab's piped core_pattern is read-only, and ignores this limit. The
    # constructor below also sets PR_SET_DUMPABLE=0 at EVERY exec, preventing
    # that handler from receiving conversation memory without changing the VM.


def require_dynamic_executable(path):
    path = Path(path)
    with path.open('rb') as binary:
        header = binary.read(64)
        if header[:6] != b'\x7fELF\x02\x01':
            raise PrivacyError(f'Core protection requires a dynamically linked ELF64 executable: {path}')
        offset = struct.unpack_from('<Q', header, 32)[0]
        size, count = struct.unpack_from('<HH', header, 54)
        for index in range(count):
            binary.seek(offset + index * size)
            if binary.read(4) == struct.pack('<I', 3):  # PT_INTERP: a dynamic loader
                return
    raise PrivacyError(f'Static executable cannot inherit core dump protection: {path}')


def core_guard_library():
    signature = hashlib.sha256(CORE_GUARD_SOURCE.encode()).hexdigest()[:16]
    library = Path(__file__).resolve().with_name(f'privacy-core-{signature}.so')
    if library.is_file():
        return library
    compiler = shutil.which('cc')
    if not compiler:
        raise PrivacyError('A C compiler (cc) is required for exec-time core dump protection')
    # This is a constant code artifact prepared before conversation admission,
    # like the Python bootstrap or model engine. It contains no conversation.
    temporary = library.with_suffix('.so.tmp')
    subprocess.run([compiler, '-shared', '-fPIC', '-O2', '-x', 'c', '-', '-o', str(temporary), '-ldl'],
                   input=CORE_GUARD_SOURCE, text=True, check=True, timeout=30)
    temporary.replace(library)
    return library


def core_guard_environment(env):
    env = dict(env)
    env['LD_PRELOAD'] = str(core_guard_library())
    return env


def check_core_guard():
    require_dynamic_executable(sys.executable)
    code = 'import ctypes; assert ctypes.CDLL(None).prctl(3,0,0,0,0) == 0, "core protection was not loaded"'
    result = subprocess.run([sys.executable, '-B', '-c', code],
                            env=core_guard_environment(os.environ), capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise PrivacyError('Exec-time core dump protection failed: ' + result.stderr.strip())


def preflight():
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'aarch64'):
        raise PrivacyError('Privacy guard requires Linux x86_64/aarch64')
    try:
        abi = landlock_abi()
    except PrivacyError as exc:
        # Colab's kernel can return ENOSYS for Landlock. A private read-only
        # mount namespace is an alternative enforcement mechanism, not a
        # fallback to unprotected serving. Its probe must succeed below.
        print(f'Landlock unavailable ({exc}); checking bubblewrap read-only mounts', flush=True)
        abi = None
    require_tmpfs(MEMORY_ROOT)
    # A read-only/full shm mount is not a viable cache fallback; find out before
    # the expensive preparation, rather than after the model has been fetched.
    with tempfile.TemporaryFile(dir=MEMORY_ROOT) as probe:
        probe.write(b'colab privacy preflight\n')
        probe.flush()
    disable_swap()
    disable_core_dumps()
    check_core_guard()
    return abi


def bubblewrap_binary():
    binary = shutil.which('bwrap')
    if binary:
        return binary
    if os.geteuid() != 0 or not Path('/content').is_dir():
        raise PrivacyError('Landlock unavailable and bubblewrap is missing; install bubblewrap to use read-only mounts')
    # Install only in the disposable Colab VM, before accepting conversations.
    print('Installing bubblewrap in Colab for read-only mount protection', flush=True)
    for command in (['apt-get', 'update', '-qq'], ['apt-get', 'install', '-qq', '-y', '--no-install-recommends', 'bubblewrap']):
        subprocess.run(command, check=True, timeout=180)
    binary = shutil.which('bwrap')
    if not binary:
        raise PrivacyError('bubblewrap installation did not provide bwrap')
    return binary


def bubblewrap_command(command, cwd, *, check=False):
    # Colab disallows a fresh proc mount. Keep its existing proc, allowing
    # CUDA's own thread names (kernel RAM metadata). The native write guard
    # admits only those comm files and rejects disk writes through magic links. Keep PIDs consistent with that proc view (psutil and CUDA inspect
    # child PIDs), and retain the bootstrap process group for shutdown cleanup.
    # The bootstrap already creates a session without a controlling terminal;
    # the inner launcher discards all standard/extra FDs before inference.
    argv = [bubblewrap_binary(), '--die-with-parent']
    # Colab's root already has CAP_SYS_ADMIN in its current user namespace.
    # Keep the current user namespace as root;
    # bubblewrap and the inner launcher still drop all capabilities before exec.
    if os.geteuid() != 0:
        argv.append('--unshare-user-try')
    argv += ['--ro-bind', '/', '/', '--dev', '/dev',
            '--bind', '/proc', '/proc', '--tmpfs', '/tmp',
            '--bind', str(MEMORY_ROOT), str(MEMORY_ROOT)]
    for device in device_paths():
        if device.name.startswith('nvidia'):
            argv += ['--dev-bind', str(device), str(device)]
    argv += ['--cap-drop', 'ALL', '--chdir', str(cwd), '--', sys.executable, '-B',
             str(Path(__file__).resolve()), '--inside-bwrap',
             '--parent-mountns', os.readlink('/proc/self/ns/mnt')]
    if check:
        argv.append('--namespace-check')
    else:
        argv += ['--cwd', str(cwd), '--', *map(str, command)]
    return argv


def mount_records():
    def unescape(value):
        # mountinfo escapes spaces, tabs, newlines and backslashes as octal.
        for code, replacement in (('040', ' '), ('011', '\t'), ('012', '\n'), ('134', '\\')):
            value = value.replace('\\' + code, replacement)
        return value
    records = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        separator = fields.index('-')
        records.append((Path(unescape(fields[4])), set(fields[5].split(',')), fields[separator + 1]))
    return records


def verify_readonly_namespace(parent_mountns):
    if not parent_mountns or os.readlink('/proc/self/ns/mnt') == parent_mountns:
        raise PrivacyError('bubblewrap did not isolate the mount namespace')
    require_tmpfs(MEMORY_ROOT)
    devices = set(device_paths())
    root_readonly = False
    proc_available = False
    for path, options, filesystem in mount_records():
        if path == Path('/') and 'ro' in options:
            root_readonly = True
        if path == Path('/proc') and filesystem == 'proc':
            proc_available = True
            continue
        if 'ro' in options or filesystem in ('tmpfs', 'ramfs', 'devpts'):
            continue
        if path in devices and stat.S_ISCHR(path.stat().st_mode):
            continue
        raise PrivacyError(f'bubblewrap left a disk/pseudo-filesystem writable: {path}')
    if not root_readonly or not proc_available:
        raise PrivacyError('bubblewrap must provide a read-only root and existing /proc')


def check_bubblewrap():
    command = bubblewrap_command([], Path.cwd(), check=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=core_guard_environment(os.environ), timeout=30)
    if result.returncode:
        raise PrivacyError('Landlock unavailable and bubblewrap protection failed: ' + result.stdout.strip())
    print(result.stdout, end='', flush=True)


def check_gpu_driver():
    # CPU-only smoke tests need no driver. On a GPU runtime, reject a broken
    # protected CUDA initialization before downloading tens of gigabytes.
    if not any(Path('/dev').glob('nvidia[0-9]*')):
        return
    code = ('import ctypes; c=ctypes.CDLL("libcuda.so.1"); r=c.cuInit(0); '
            'assert r == 0, f"CUDA initialization under privacy protection failed: {r}"')
    env = core_guard_environment(os.environ)
    env.update(COLAB_PRIVATE_WRITES='1', CUDA_CACHE_DISABLE='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise PrivacyError('Protected GPU preflight failed: ' + result.stderr.strip())


def drop_capabilities():
    # Colab runs as root. Keep its uid for read access to Drive, but remove root
    # capabilities and their exec inheritance so mounts/swap cannot be restored.
    checked(LIBC.prctl(38, 1, 0, 0, 0), 'PR_SET_NO_NEW_PRIVS')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if os.geteuid() == 0 and int(status['CapBnd'].strip(), 16):
        if int(status['CapEff'].strip(), 16) & (1 << 8):  # CAP_SETPCAP
            securebits = checked(LIBC.prctl(27, 0, 0, 0, 0), 'Read securebits')
            checked(LIBC.prctl(28, securebits | 3, 0, 0, 0), 'Lock SECBIT_NOROOT')
            for cap in range(int(Path('/proc/sys/kernel/cap_last_cap').read_text()) + 1):
                checked(LIBC.prctl(24, cap, 0, 0, 0), 'Drop capability bounding set')
        elif int(status['CapPrm'].strip(), 16):
            raise PrivacyError('Cannot seal root capabilities without CAP_SETPCAP')
        # bubblewrap --cap-drop ALL already emptied effective/permitted caps.
        # no_new_privs above prevents exec from granting them again. Attempting
        # another privileged securebits operation at this point is invalid.
    checked(LIBC.prctl(47, 4, 0, 0, 0), 'Clear ambient capabilities')

    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]

    class Caps(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]

    header, caps = Header(0x20080522, 0), (Caps * 2)()
    checked(LIBC.capset(ctypes.byref(header), ctypes.byref(caps)), 'Clear process capabilities')


def device_paths():
    # Never grant /dev as a whole: block devices, /dev/kmsg and /dev/fd must not
    # become alternate routes to disk or host logs. GPU nodes are character-only.
    candidates = [Path('/dev/null'), Path('/dev/zero'), *Path('/dev').glob('nvidia*')]
    caps = Path('/dev/nvidia-caps')
    if caps.is_dir():
        candidates += list(caps.iterdir())
    return [path for path in candidates if not path.is_symlink() and stat.S_ISCHR(path.stat().st_mode)]


def restrict_writes(memory_root=MEMORY_ROOT, abi=None):
    abi = landlock_abi() if abi is None else abi
    require_tmpfs(memory_root)
    handled = FS_WRITES | (IOCTL_DEV if abi >= 5 else 0)
    attr = Ruleset(handled)
    fd = checked(LIBC.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0), 'Create Landlock ruleset')
    try:
        for path, rights in [(memory_root, MEMORY_WRITES), (Path('/proc'), WRITE_FILE), *[(p, WRITE_FILE | TRUNCATE |
                                (IOCTL_DEV if abi >= 5 else 0)) for p in device_paths()]]:
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, parent)
                checked(LIBC.syscall(445, fd, 1, ctypes.byref(rule), 0), 'Add Landlock rule')
            finally:
                os.close(parent)
        checked(LIBC.prctl(38, 1, 0, 0, 0), 'PR_SET_NO_NEW_PRIVS')
        # ABI 7 adds denial auditing: a rejected filename can itself contain
        # conversation text. Suppress our domain and nested-domain audit output;
        # do not enable auditing on exec either. Older ABIs have no such logging.
        flags = (1 | 4) if abi >= 7 else 0
        checked(LIBC.syscall(446, fd, flags), 'Enforce Landlock')
    finally:
        os.close(fd)


def recording_environment_key(key):
    return ((key.startswith('STRATA_') and
             (any(word in key for word in ('DEBUG', 'TRACE', 'DUMP', 'LOG')) or key == 'STRATA_REQUEST_LINES')) or
            key.startswith(('LLAMA_ARG_LOG', 'LLAMA_ARG_VERBOSE', 'LLAMA_ARG_SLOT_SAVE',
                            'LLAMA_ARG_PROMPT_CACHE', 'LLAMA_ARG_PROMPTS_LOG',
                            'CUDA_PROFILE', 'CUDA_INJECTION', 'NSYS_', 'NVPROF_')))


def private_environment(scratch, original=None):
    env = dict(os.environ if original is None else original)
    # Discard recording/instrumentation overrides, while preserving settings for
    # model placement, expert caching and SSD reads.
    for key in list(env):
        if recording_environment_key(key):
            del env[key]
    env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_CACHE_DISABLE='1',
               CUDA_CACHE_PATH=str(scratch / 'cache/cuda'),
               TMPDIR=str(scratch / 'tmp'), TMP=str(scratch / 'tmp'), TEMP=str(scratch / 'tmp'),
               XDG_CACHE_HOME=str(scratch / 'cache'), XDG_CONFIG_HOME=str(scratch / 'config'),
               XDG_DATA_HOME=str(scratch / 'data'), XDG_STATE_HOME=str(scratch / 'state'),
               XDG_RUNTIME_DIR=str(scratch / 'run'), HF_HOME=str(scratch / 'cache/huggingface'),
               HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TRANSFORMERS_OFFLINE='1')
    return env


def private_config(argv, scratch):
    argv = list(map(str, argv))
    if '--config' not in argv:
        return argv
    index = argv.index('--config') + 1
    settings = json.loads(Path(argv[index]).read_text())
    settings.update(log=str(scratch / 'inference.log'), api_monitor=False, open_browser=False)
    for key in ('slot_save_path', 'mcp_servers', 'mcp_config'):
        settings.pop(key, None)
    settings['env'] = {key: value for key, value in settings.get('env', {}).items()
                       if not recording_environment_key(key) and key != 'LD_PRELOAD'}
    if settings.get('exe'):
        require_dynamic_executable(settings['exe'])
    config = scratch / 'server.json'
    config.write_text(json.dumps(settings))
    argv[index] = str(config)
    return argv


def discard_inherited_output(diagnostic=None):
    # Landlock cannot revoke a previously opened log FD. Replace inherited FDs
    # with null input and optional RAM-only diagnostics before any server code.
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open('/dev/null', os.O_RDWR)
    try:
        os.dup2(fd, 0)
        if diagnostic is None:
            os.dup2(fd, 1)
            os.dup2(fd, 2)
        else:
            diagnostic = Path(diagnostic)
            require_tmpfs(diagnostic.parent)
            output = os.open(diagnostic, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            try:
                os.dup2(output, 1)
                os.dup2(output, 2)
            finally:
                if output > 2:
                    os.close(output)
    finally:
        if fd > 2:
            os.close(fd)
    for name in os.listdir('/proc/self/fd'):
        number = int(name)
        if number > 2:
            try:
                os.close(number)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise


def launch(argv, cwd):
    abi = preflight()                    # repeat the checks immediately before serving
    scratch = Path(tempfile.mkdtemp(prefix='colab-privacy-', dir=MEMORY_ROOT))
    for name in ('tmp', 'cache', 'config', 'data', 'state', 'run'):
        (scratch / name).mkdir(mode=0o700)
    argv = private_config(argv, scratch)
    require_dynamic_executable(argv[0])
    env = core_guard_environment(private_environment(scratch))
    env['COLAB_PRIVATE_WRITES'] = '1'
    diagnostic = scratch / 'inference.log'
    env['COLAB_PRIVACY_DIAGNOSTIC'] = str(diagnostic)
    print('COLAB_PRIVACY_DIAGNOSTIC=' + str(diagnostic), flush=True)
    if abi is None:
        command = bubblewrap_command(argv, cwd)
        # The inner launcher verifies the actual mount view before marking it
        # enforced, closing inherited FDs, or executing any inference code.
        # bubblewrap's setup needs to populate its temporary root. The inner
        # launcher enables the libc write guard only for the inference exec.
        env.pop('COLAB_PRIVATE_WRITES', None)
        os.execve(command[0], command, env)
        return
    os.chdir(cwd)
    drop_capabilities()
    restrict_writes(abi=abi)
    print(f'COLAB_PRIVACY_ENFORCED: Landlock ABI {abi}; disk writes denied; swap/core dumps disabled; '
          'inference diagnostics in RAM only', flush=True)
    discard_inherited_output(diagnostic)
    os.execve(argv[0], argv, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--cwd', type=Path)
    parser.add_argument('--inside-bwrap', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--namespace-check', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--parent-mountns', help=argparse.SUPPRESS)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.inside_bwrap:
        verify_readonly_namespace(args.parent_mountns)
        disable_swap()
        disable_core_dumps()
        if LIBC.prctl(3, 0, 0, 0, 0) != 0:
            raise PrivacyError('Exec-time core dump protection was not loaded inside bubblewrap')
        drop_capabilities()
        if args.namespace_check:
            code = ('from pathlib import Path; '
                    's=dict(x.split(":",1) for x in Path("/proc/self/status").read_text().splitlines() if ":" in x); '
                    'assert all(int(s[k].strip(),16)==0 for k in ("CapEff","CapPrm","CapAmb")); '
                    'assert s["NoNewPrivs"].strip()=="1"')
            subprocess.run([sys.executable, '-B', '-c', code], check=True, timeout=10)
            # Verify both sides: new disk files must fail, RAM scratch must work.
            probe = Path.cwd() / f'.colab-privacy-write-probe-{os.getpid()}'
            try:
                probe.write_text('privacy probe')
            except OSError as exc:
                if exc.errno != errno.EROFS:
                    raise PrivacyError('Read-only disk probe did not return EROFS') from exc
            else:
                probe.unlink()
                raise PrivacyError('Disk probe was writable inside bubblewrap')
            with tempfile.TemporaryFile(dir=MEMORY_ROOT) as memory:
                memory.write(b'privacy probe')
                memory.flush()
            check_gpu_driver()
            print('COLAB_PRIVACY_PREFLIGHT_OK: bubblewrap read-only mounts', flush=True)
            return
        if not command or args.cwd is None:
            parser.error('An inner launcher requires --cwd and a server command')
        print('COLAB_PRIVACY_ENFORCED: bubblewrap read-only mounts; disk writes denied; '
              'swap/core dumps disabled; inference diagnostics in RAM only', flush=True)
        discard_inherited_output(os.environ.get('COLAB_PRIVACY_DIAGNOSTIC'))
        os.chdir(args.cwd)
        env = os.environ.copy()
        env['COLAB_PRIVATE_WRITES'] = '1'
        os.execve(command[0], command, env)
        return
    if args.check:
        abi = preflight()
        if abi is None:
            check_bubblewrap()
            return
        # Exercise the actual kernel policy, including capability dropping, in
        # this disposable process before a large model download/build starts.
        drop_capabilities()
        restrict_writes(abi=abi)
        check_gpu_driver()
        print(f'COLAB_PRIVACY_PREFLIGHT_OK: Landlock ABI {abi}', flush=True)
        return
    if not command or args.cwd is None:
        parser.error('--cwd and a server command after -- are required')
    launch(command, args.cwd)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Colab privacy guard refused startup: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
