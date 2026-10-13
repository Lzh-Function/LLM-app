"""Privacy policy tests; all irreversible kernel restrictions run in subprocesses."""
import contextlib
import importlib.util
import io
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import colab_privacy as privacy


class ConfigurationTests(unittest.TestCase):
    def test_config_and_environment_disable_recording_without_losing_model_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            original = scratch / 'original.json'
            original.write_text(json.dumps({
                'log': '/content/drive/MyDrive/engine.log', 'api_monitor': True,
                'slot_save_path': '/content/sessions', 'mcp_servers': {'unsafe': {}},
                'args': ['--pack', '/content/drive/MyDrive/pack'], 'tokenizer': '/tokenizer',
                'env': {'STRATA_DEBUG': '1', 'STRATA_GLM_RAM_RESIDENT': '1', 'LD_PRELOAD': '/recording.so'},
            }))
            argv = privacy.private_config(['/python', 'server.py', '--config', str(original)], scratch)
            settings = json.loads(Path(argv[-1]).read_text())
            self.assertEqual(settings['log'], str(scratch / 'inference.log'))
            self.assertFalse(settings['api_monitor'])
            self.assertNotIn('slot_save_path', settings)
            self.assertNotIn('mcp_servers', settings)
            self.assertEqual(settings['env'], {'STRATA_GLM_RAM_RESIDENT': '1'})
            self.assertEqual(settings['args'], ['--pack', '/content/drive/MyDrive/pack'])
            self.assertTrue(json.loads(original.read_text())['api_monitor'])
            env = privacy.private_environment(scratch, {
                'PATH': '/usr/bin', 'HOME': '/root', 'STRATA_DEBUG': '1',
                'STRATA_DUMP_LOGITS': '/content/logits', 'LLAMA_ARG_LOG_FILE': '/content/log',
                'CUDA_PROFILE_LOG': '/content/profile', 'CUDA_CACHE_PATH': '/root/.nv',
                'STRATA_ARENA_MMAP': '1', 'STRATA_GLM_RAM_RESIDENT': '1', 'STRATA_ENGINE_READY_S': '10800',
                'LLAMA_ARG_MOE_CACHE_MIB': '32768',
            })
            self.assertEqual(env['HOME'], '/root')
            self.assertEqual(env['CUDA_CACHE_DISABLE'], '1')
            self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertEqual(env['TMPDIR'], str(scratch / 'tmp'))
            self.assertEqual(env['STRATA_ARENA_MMAP'], '1')
            self.assertEqual(env['STRATA_GLM_RAM_RESIDENT'], '1')
            self.assertEqual(env['STRATA_ENGINE_READY_S'], '10800')
            self.assertEqual(env['LLAMA_ARG_MOE_CACHE_MIB'], '32768')
            self.assertFalse(any(privacy.recording_environment_key(key) for key in env))

    def test_disk_backed_temporary_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(privacy.PrivacyError, 'tmpfs'):
                privacy.require_tmpfs(Path(directory))

    def test_old_or_unavailable_landlock_is_rejected(self):
        with patch.object(privacy.LIBC, 'syscall', return_value=2):
            with self.assertRaisesRegex(privacy.PrivacyError, 'ABI >= 3'):
                privacy.landlock_abi()
        with patch.object(privacy.LIBC, 'syscall', return_value=-1):
            with self.assertRaisesRegex(privacy.PrivacyError, 'refusing unprotected'):
                privacy.landlock_abi()

    def test_swap_is_disabled_and_failure_never_falls_back(self):
        header = 'Filename Type Size Used Priority\n'
        active = header + '/swap file 1024 0 -2\n'
        with patch.object(Path, 'read_text', side_effect=[active, header]), \
             patch.object(privacy.subprocess, 'run') as command:
            privacy.disable_swap()
            self.assertEqual(command.call_args.args[0], ['swapoff', '-a'])
        with patch.object(Path, 'read_text', return_value=active), patch.object(privacy.subprocess, 'run'):
            with self.assertRaisesRegex(privacy.PrivacyError, 'Swap is still enabled'):
                privacy.disable_swap()

    def test_gpu_preflight_detects_driver_failure_before_model_download(self):
        failure = subprocess.CompletedProcess([], 1, '', 'CUDA initialization under privacy protection failed: 304')
        with patch.object(Path, 'glob', return_value=iter([Path('/dev/nvidia0')])), \
             patch.object(privacy, 'core_guard_environment', side_effect=lambda env: dict(env)), \
             patch.object(privacy.subprocess, 'run', return_value=failure) as run:
            with self.assertRaisesRegex(privacy.PrivacyError, 'CUDA initialization'):
                privacy.check_gpu_driver()
            self.assertEqual(run.call_args.kwargs['env']['COLAB_PRIVATE_WRITES'], '1')

    def test_core_limit_is_zero_without_modifying_colabs_global_handler(self):
        with patch.object(privacy.resource, 'setrlimit') as limit, \
             patch.object(Path, 'write_text') as write:
            privacy.disable_core_dumps()
            limit.assert_called_once_with(resource.RLIMIT_CORE, (0, 0))
            write.assert_not_called()

    def test_launcher_stops_before_exec_if_policy_fails(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(privacy, 'preflight', return_value=7), \
             patch.object(privacy.tempfile, 'mkdtemp', return_value=directory), \
             patch.object(privacy.os, 'chdir'), patch.object(privacy, 'drop_capabilities'), \
             patch.object(privacy, 'require_dynamic_executable'), \
             patch.object(privacy, 'core_guard_environment', side_effect=lambda env: env), \
             patch.object(privacy, 'restrict_writes', side_effect=privacy.PrivacyError('blocked')), \
             patch.object(privacy, 'discard_inherited_output') as discard, \
             patch.object(privacy.os, 'execve') as execute:
            with self.assertRaisesRegex(privacy.PrivacyError, 'blocked'):
                privacy.launch(['/server'], Path('/source'))
            execute.assert_not_called()
            discard.assert_not_called()

    def test_root_capabilities_are_removed_without_clearing_existing_securebit_locks(self):
        def prctl(option, *args):
            return 0x30 if option == 27 else 0

        with patch.object(privacy.os, 'geteuid', return_value=0), \
             patch.object(Path, 'read_text', side_effect=['CapBnd:\t3\nCapEff:\t100\nCapPrm:\t100\n', '1']), \
             patch.object(privacy.LIBC, 'prctl', side_effect=prctl) as control, \
             patch.object(privacy.LIBC, 'capset', return_value=0) as clear:
            privacy.drop_capabilities()
        control.assert_any_call(28, 0x33, 0, 0, 0)
        control.assert_any_call(24, 0, 0, 0, 0)
        control.assert_any_call(24, 1, 0, 0, 0)
        clear.assert_called_once()

    def test_bubblewrap_already_dropped_root_caps_does_not_require_privileged_securebits(self):
        with patch.object(privacy.os, 'geteuid', return_value=0), \
             patch.object(Path, 'read_text', return_value='CapBnd:\tffff\nCapEff:\t0\nCapPrm:\t0\n'), \
             patch.object(privacy.LIBC, 'prctl', return_value=0) as control, \
             patch.object(privacy.LIBC, 'capset', return_value=0):
            privacy.drop_capabilities()
        self.assertNotIn(28, [call.args[0] for call in control.call_args_list])
        self.assertNotIn(24, [call.args[0] for call in control.call_args_list])
        control.assert_any_call(38, 1, 0, 0, 0)

    def test_landlock_unavailable_selects_namespace_without_skipping_memory_protection(self):
        with patch.object(privacy, 'landlock_abi', side_effect=privacy.PrivacyError('Function not implemented')), \
             patch.object(privacy, 'require_tmpfs') as memory, \
             patch.object(privacy.tempfile, 'TemporaryFile'), \
             patch.object(privacy, 'disable_swap') as swap, \
             patch.object(privacy, 'disable_core_dumps') as core, \
             patch.object(privacy, 'check_core_guard') as exec_guard, contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(privacy.preflight())
        memory.assert_called_once_with(privacy.MEMORY_ROOT)
        swap.assert_called_once()
        core.assert_called_once()
        exec_guard.assert_called_once()

    def test_namespace_launcher_does_not_drop_mount_privileges_before_bubblewrap(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(privacy, 'preflight', return_value=None), \
             patch.object(privacy, 'require_dynamic_executable'), \
             patch.object(privacy, 'core_guard_environment', side_effect=lambda env: env), \
             patch.object(privacy.tempfile, 'mkdtemp', return_value=directory), \
             patch.object(privacy, 'bubblewrap_command', return_value=['/bwrap', 'protected-command']) as wrap, \
             patch.object(privacy, 'drop_capabilities') as drop, \
             patch.object(privacy, 'restrict_writes') as landlock, \
             patch.object(privacy.os, 'execve') as execute:
            privacy.launch(['/server'], Path('/source'))
        wrap.assert_called_once_with(['/server'], Path('/source'))
        self.assertEqual(execute.call_args.args[:2], ('/bwrap', ['/bwrap', 'protected-command']))
        drop.assert_not_called()
        landlock.assert_not_called()


class NamespaceTests(unittest.TestCase):
    def test_colab_root_keeps_user_and_pid_namespaces_consistent_with_existing_proc(self):
        with patch.object(privacy.os, 'geteuid', return_value=0), \
             patch.object(privacy, 'bubblewrap_binary', return_value='/bwrap'), \
             patch.object(privacy, 'device_paths', return_value=[]):
            command = privacy.bubblewrap_command(['/engine'], Path('/content'))
        self.assertNotIn('--unshare-user-try', command)
        self.assertNotIn('--unshare-pid', command)
        self.assertNotIn('--new-session', command)
        self.assertIn('--cap-drop', command)

    def test_namespace_command_isolates_proc_and_devices_but_keeps_network_and_model_reads(self):
        with patch.object(privacy, 'bubblewrap_binary', return_value='/usr/bin/bwrap'), \
             patch.object(privacy, 'device_paths', return_value=[Path('/dev/null'), Path('/dev/nvidia0')]), \
             patch.object(privacy.os, 'readlink', return_value='pid:[old]'):
            command = privacy.bubblewrap_command(['/engine', '--model', '/content/drive/model.gguf'], Path('/content/source'))
        self.assertNotIn('--unshare-pid', command)
        self.assertNotIn('--unshare-net', command)
        self.assertEqual(command[command.index('--ro-bind') + 1:command.index('--ro-bind') + 3], ['/', '/'])
        self.assertIn('--bind', command)
        self.assertTrue(any(command[i:i+3] == ['--bind', '/proc', '/proc'] for i in range(len(command)-2)))
        self.assertEqual(command[command.index('--dev-bind') + 1:command.index('--dev-bind') + 3],
                         ['/dev/nvidia0', '/dev/nvidia0'])
        self.assertEqual(command[command.index('--cap-drop') + 1], 'ALL')
        self.assertEqual(command[-3:], ['/engine', '--model', '/content/drive/model.gguf'])
        self.assertIn('--inside-bwrap', command)
        self.assertIn('pid:[old]', command)

    def verify(self, records, pid='pid:[new]'):
        with patch.object(privacy.os, 'readlink', return_value=pid), \
             patch.object(privacy, 'require_tmpfs'), \
             patch.object(privacy, 'device_paths', return_value=[]), \
             patch.object(privacy, 'mount_records', return_value=records):
            privacy.verify_readonly_namespace('pid:[old]')

    def test_readonly_disk_and_writable_ram_mounts_are_accepted(self):
        self.verify([(Path('/'), {'ro'}, 'ext4'), (Path('/proc'), {'ro'}, 'proc'),
                     (Path('/dev/shm'), {'rw'}, 'tmpfs'), (Path('/content/drive'), {'ro'}, 'fuse')])

    def test_writable_disk_or_unisolated_proc_is_rejected(self):
        base = [(Path('/'), {'ro'}, 'ext4'), (Path('/proc'), {'ro'}, 'proc')]
        with self.assertRaisesRegex(privacy.PrivacyError, 'writable'):
            self.verify(base + [(Path('/content/drive'), {'rw'}, 'fuse')])
        with self.assertRaisesRegex(privacy.PrivacyError, 'isolate'):
            self.verify(base, pid='pid:[old]')
        self.verify([(Path('/'), {'ro'}, 'ext4'), (Path('/proc'), {'rw'}, 'proc')])

    def test_failed_namespace_probe_never_falls_back_to_bare_serving(self):
        failure = subprocess.CompletedProcess(['/bwrap'], 1, 'bwrap: Creating new namespace failed: Operation not permitted')
        with patch.object(privacy, 'bubblewrap_command', return_value=['/bwrap']), \
             patch.object(privacy, 'core_guard_environment', side_effect=lambda env: env), \
             patch.object(privacy.subprocess, 'run', return_value=failure):
            with self.assertRaisesRegex(privacy.PrivacyError, 'Operation not permitted'):
                privacy.check_bubblewrap()

    def test_mountinfo_paths_with_spaces_are_decoded(self):
        mountinfo = '10 1 0:5 / /content/model\\040cache ro,nosuid - ext4 /dev/vda ro\n'
        with patch.object(Path, 'read_text', return_value=mountinfo):
            self.assertEqual(privacy.mount_records(), [(Path('/content/model cache'), {'ro', 'nosuid'}, 'ext4')])


class CoreGuardTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'A C compiler is needed')
    def test_proc_exception_is_only_own_thread_name_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / 'probe.so'
            source = privacy.CORE_GUARD_SOURCE + '\nint allowed(const char *path) { return private_write_path(AT_FDCWD, path); }\n'
            subprocess.run(['cc', '-shared', '-fPIC', '-O2', '-x', 'c', '-', '-o', str(library), '-ldl'],
                           input=source, text=True, check=True, capture_output=True)
            code = """import ctypes, os
c=ctypes.CDLL(os.environ['LD_PRELOAD']); c.allowed.argtypes=[ctypes.c_char_p]
assert c.allowed(f'/proc/self/task/{os.getpid()}/comm'.encode()) == 1
for path in [b'/proc/self/mem', b'/proc/sys/kernel/core_pattern', b'/proc/self/root/tmp/conversation.txt', b'/proc/1/task/1/comm']:
 assert c.allowed(path)==0, path
"""
            subprocess.run([sys.executable, '-B', '-c', code], check=True, capture_output=True,
                           env=dict(os.environ, LD_PRELOAD=str(library), COLAB_PRIVATE_WRITES='1'))


    @unittest.skipUnless(shutil.which('cc'), 'A C compiler is needed for the native guard test')
    def test_native_guard_rejects_disk_writes_through_open_fopen_and_proc_magic_links(self):
        code = '''
import ctypes, errno, os, sys
from pathlib import Path
root = Path(sys.argv[1])
assert (root / 'model').read_text() == 'weights'
for path in [root / 'conversation.json', root / 'model', Path('/proc/self/root') / str(root).lstrip('/') / 'conversation.json']:
    try: path.write_text('private input')
    except OSError as e: assert e.errno == errno.EROFS, e
    else: raise AssertionError('native guard allowed a disk write')
libc = ctypes.CDLL(None)
libc.fopen.restype = ctypes.c_void_p
assert not libc.fopen(os.fsencode(root / 'conversation.json'), b'w')
fd = os.memfd_create('private-memory', os.MFD_CLOEXEC)
copy = os.open('/proc/self/fd/' + str(fd), os.O_RDWR)
os.write(copy, b'private memory')
os.lseek(fd, 0, 0)
assert os.read(fd, 100) == b'private memory'
os.close(copy)
os.close(fd)
'''
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(privacy, '__file__', str(Path(directory) / 'colab_privacy.py')):
            root = Path(directory)
            (root / 'model').write_text('weights')
            env = privacy.core_guard_environment(os.environ)
            env['COLAB_PRIVATE_WRITES'] = '1'
            result = subprocess.run([sys.executable, '-B', '-c', code, str(root)],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'conversation.json').exists())
            self.assertEqual((root / 'model').read_text(), 'weights')

    @unittest.skipUnless(shutil.which('cc'), 'A C compiler is needed for the native guard test')
    def test_nondumpable_constructor_is_loaded_on_parent_and_child_exec(self):
        privacy.require_dynamic_executable(sys.executable)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(privacy, '__file__', str(Path(directory) / 'colab_privacy.py')):
            library = privacy.core_guard_library()
            self.assertTrue(library.is_file())
            self.assertEqual(privacy.core_guard_library(), library)
            privacy.check_core_guard()
            check = 'import ctypes; assert ctypes.CDLL(None).prctl(3,0,0,0,0) == 0'
            code = ('import subprocess, sys; ' + check + '; '
                    f'subprocess.run([sys.executable, "-B", "-c", {check!r}], check=True)')
            result = subprocess.run([sys.executable, '-B', '-c', code],
                                    env=privacy.core_guard_environment(os.environ), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_static_executable_is_rejected_before_conversation_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'static'
            binary.write_bytes(b'\x7fELF\x02\x01' + b'\0' * 58)
            with self.assertRaisesRegex(privacy.PrivacyError, 'Static executable'):
                privacy.require_dynamic_executable(binary)

    def test_native_guard_load_failure_is_not_ignored(self):
        failure = subprocess.CompletedProcess([], 1, '', 'core protection was not loaded')
        with patch.object(privacy, 'require_dynamic_executable'), \
             patch.object(privacy, 'core_guard_environment', side_effect=lambda env: env), \
             patch.object(privacy.subprocess, 'run', return_value=failure):
            with self.assertRaisesRegex(privacy.PrivacyError, 'core protection was not loaded'):
                privacy.check_core_guard()


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            privacy.landlock_abi()
            privacy.require_tmpfs(privacy.MEMORY_ROOT)
        except privacy.PrivacyError as exc:
            raise unittest.SkipTest(str(exc))

    def child(self, code, *args, **kwargs):
        return subprocess.run([sys.executable, '-B', '-c', code, *map(str, args)],
                              cwd=Path(__file__).parent, capture_output=True, text=True, timeout=15, **kwargs)

    def test_kernel_blocks_disk_writes_truncation_rename_delete_and_child_exec(self):
        code = '''
import errno, mmap, os, subprocess, sys
from pathlib import Path
import colab_privacy as p
root = Path(sys.argv[1])
p.drop_capabilities()
p.restrict_writes()
def denied(fn):
    try: fn()
    except OSError as e:
        assert e.errno in (errno.EACCES, errno.EPERM), e
    else: raise AssertionError('disk operation was allowed')
assert (root / 'model').read_text() == 'weights'
with (root / 'model').open('rb') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as model:
        assert model[:] == b'weights'  # SSD-backed model offload remains usable.
denied(lambda: (root / 'conversation.json').write_text('private input'))
denied(lambda: (root / 'model').open('a'))
denied(lambda: os.open(root / 'model', os.O_RDONLY | os.O_TRUNC))
denied(lambda: os.truncate(root / 'model', 0))
denied(lambda: (root / 'model').rename(root / 'renamed'))
denied(lambda: (root / 'model').unlink())
denied(lambda: (root / 'newdir').mkdir())
denied(lambda: (root / 'alias').write_text('private input'))
assert subprocess.run([sys.executable, '-B', '-c',
    'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("private output")',
    str(root / 'child.json')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
# Memory-only file descriptors remain usable.
fd = os.memfd_create('private-memory', os.MFD_CLOEXEC)
os.write(fd, b'private reasoning')
os.lseek(fd, 0, 0)
assert os.read(fd, 100) == b'private reasoning'
os.close(fd)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model').write_text('weights')
            (root / 'alias').symlink_to(root / 'model')
            result = self.child(code, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / 'model').read_text(), 'weights')
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['alias', 'model'])

    def test_inherited_log_fds_and_stdout_stderr_are_removed_before_exec(self):
        code = '''
import os, sys
import colab_privacy as p
p.drop_capabilities()
p.restrict_writes()
p.discard_inherited_output()
next_code = """
import errno, os, sys
assert os.readlink('/proc/self/fd/1') == '/dev/null'
assert os.readlink('/proc/self/fd/2') == '/dev/null'
try: os.write(int(sys.argv[1]), b'private input')
except OSError as e: assert e.errno == errno.EBADF
else: raise AssertionError('inherited log survived')
print('private output')
print('private reasoning', file=sys.stderr)
"""
os.execv(sys.executable, [sys.executable, '-B', '-c', next_code, sys.argv[1]])
'''
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'bootstrap.log'
            with log.open('w') as stream:
                result = self.child(code, stream.fileno(), pass_fds=(stream.fileno(),))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertEqual(result.stderr, '')
            self.assertEqual(log.read_text(), '')


class RemoteIntegrationTests(unittest.TestCase):
    def load_remote(self, name):
        path = Path(__file__).parent / name / 'remote.py'
        spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_both_launchers_exec_only_through_the_uploaded_guard(self):
        for name in ('colab-qwen', 'colab-glm'):
            with self.subTest(name=name):
                remote = self.load_remote(name)
                with patch.object(remote.os, 'execv') as execute:
                    remote.serve_private(['/engine', '--port', 18080], Path('/model-source'))
                binary, argv = execute.call_args.args
                self.assertEqual(binary, sys.executable)
                self.assertEqual(Path(argv[1]).name, 'colab_privacy.py')
                self.assertEqual(argv[argv.index('--cwd') + 1], '/model-source')
                self.assertEqual(argv[argv.index('--') + 1:], ['/engine', '--port', '18080'])

    def test_both_preflights_run_the_guard_before_model_preparation(self):
        for name in ('colab-qwen', 'colab-glm'):
            with self.subTest(name=name):
                remote = self.load_remote(name)
                result = subprocess.CompletedProcess([], 0, 'COLAB_PRIVACY_PREFLIGHT_OK\n')
                with patch.object(remote.subprocess, 'run', return_value=result) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    remote.check_privacy()
                self.assertEqual(run.call_args.args[0][-1], '--check')
                self.assertEqual(Path(run.call_args.args[0][1]).name, 'colab_privacy.py')

    def test_actual_guard_error_survives_into_both_bootstrap_failure_messages(self):
        reason = 'Colab privacy guard refused startup: Landlock and bubblewrap unavailable'
        for name in ('colab-qwen', 'colab-glm'):
            with self.subTest(name=name):
                remote = self.load_remote(name)
                result = subprocess.CompletedProcess([], 1, 'diagnostic preamble\n' + reason + '\n')
                with patch.object(remote.subprocess, 'run', return_value=result):
                    with self.assertRaisesRegex(RuntimeError, 'Landlock and bubblewrap unavailable'):
                        remote.check_privacy()

    def test_privacy_failure_in_both_main_paths_prevents_download_or_build(self):
        for name in ('colab-qwen', 'colab-glm'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                remote = self.load_remote(name)
                work = Path(directory)
                manifest = Path(remote.__file__).with_name('manifest.json').read_text()
                (work / 'manifest.json').write_text(manifest)
                with patch.object(remote, 'WORK', work), \
                     patch.object(remote, 'ensure_drive', return_value=work), \
                     patch.object(remote, 'check_server_port'), \
                     patch.object(remote, 'check_privacy', side_effect=RuntimeError('privacy unavailable')), \
                     patch.object(remote, 'fetch_verified') as fetch, \
                     patch.object(remote, 'source_tree') as build, \
                     patch.object(remote, 'serve_private') as serve, \
                     patch.object(sys, 'argv', ['remote.py']), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'privacy unavailable'):
                        remote.main()
                fetch.assert_not_called()
                build.assert_not_called()
                serve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
