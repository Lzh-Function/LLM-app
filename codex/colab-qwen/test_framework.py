"""Exercise storage integrity and billable-runtime cleanup without allocating a VM."""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import tomllib
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


remote = module('remote', 'remote.py')
session = module('session', 'session.py')
diagnostics = module('drive_diagnostics', 'drive_diagnostics.py')
mount_helper = module('mount_helper', 'mount_helper.py')
shutdown_remote = module('shutdown_remote', 'shutdown_remote.py')
PAYLOAD = b'GGUF-test-data' * 100


class Handler(http.server.BaseHTTPRequestHandler):
    ranges = []
    honor_range = True

    def do_GET(self):
        requested = self.headers.get('Range')
        self.ranges.append(requested)
        offset = int(requested.split('=')[1].split('-')[0]) if requested and self.honor_range else 0
        self.send_response(206 if offset else 200)
        if offset:
            self.send_header('Content-Range', f'bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}')
        self.send_header('Content-Length', str(len(PAYLOAD) - offset))
        self.end_headers()
        self.wfile.write(PAYLOAD[offset:])

    def log_message(self, *args):
        pass


class ContextTests(unittest.TestCase):
    def test_doctor_uses_256k_by_default_and_accepts_128k_override(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session, 'catalog') as catalog, \
             contextlib.redirect_stdout(io.StringIO()):
            for arguments, expected in ((['doctor'], 262144),
                                        (['doctor', '--context', '131072'], 131072)):
                with self.subTest(arguments=arguments), patch.object(sys, 'argv', ['colab-qwen', *arguments]):
                    self.assertEqual(session.main(), 0)
                    self.assertEqual(catalog.call_args.args[0], expected)


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/model.gguf'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'model.gguf'
        self.partial = self.path.with_name('model.gguf.part')
        Handler.ranges.clear()
        Handler.honor_range = True

    def fetch(self, digest=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            remote.fetch_verified(self.url, self.path, len(PAYLOAD),
                                  digest or hashlib.sha256(PAYLOAD).hexdigest(), **kwargs)

    def test_resume_verify_publish_and_cache(self):
        self.partial.write_bytes(PAYLOAD[:81])
        self.fetch()
        self.assertEqual(Handler.ranges, ['bytes=81-'])
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertFalse(self.partial.exists())
        self.fetch()
        self.assertEqual(len(Handler.ranges), 1)

    def test_checksum_failure_never_publishes(self):
        with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
            self.fetch('0' * 64)
        self.assertFalse(self.path.exists())
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.path.with_name('model.gguf.verified.json').exists())

    def test_server_ignores_range_preserves_partial(self):
        self.partial.write_bytes(PAYLOAD[:81])
        Handler.honor_range = False
        with self.assertRaisesRegex(RuntimeError, 'honor resume Range'):
            self.fetch()
        self.assertEqual(self.partial.read_bytes(), PAYLOAD[:81])

    def test_modified_cache_detected(self):
        self.fetch()
        self.path.write_bytes(b'x' * len(PAYLOAD))
        with self.assertRaisesRegex(RuntimeError, 'Existing file fails verification'):
            self.fetch(skip_cache_mtime=False)

    def test_default_reuses_verified_cache_without_hashing_or_rewriting_record(self):
        self.fetch()
        marker = self.path.with_name('model.gguf.verified.json')
        previous = marker.read_text()
        timestamp = self.path.stat().st_mtime_ns + 180_000_000_000
        session.os.utime(self.path, ns=(timestamp, timestamp))
        with patch.object(remote, 'digest') as digest:
            self.fetch()
            digest.assert_not_called()
        self.assertEqual(marker.read_text(), previous)
        self.assertEqual(len(Handler.ranges), 1)

    def test_reverify_overrides_skip_mtime_and_detects_same_size_changes(self):
        self.fetch()
        self.path.write_bytes(b'x' * len(PAYLOAD))
        with patch.object(remote, 'digest', wraps=remote.digest) as digest:
            with self.assertRaisesRegex(RuntimeError, 'Existing file fails verification'):
                self.fetch(reverify=True)
            digest.assert_called_once_with(self.path)

    def test_skip_mtime_still_requires_matching_size_and_sha_record(self):
        self.fetch()
        marker = self.path.with_name('model.gguf.verified.json')
        invalid = [None, '{', '[]',
                   json.dumps({'size': len(PAYLOAD), 'sha256': '0' * 64}),
                   json.dumps({'size': len(PAYLOAD) + 1, 'sha256': hashlib.sha256(PAYLOAD).hexdigest()})]
        for record in invalid:
            with self.subTest(record=record):
                if record is None:
                    marker.unlink()
                else:
                    marker.write_text(record)
                with patch.object(remote, 'digest', wraps=remote.digest) as digest:
                    self.fetch(skip_cache_mtime=True)
                    digest.assert_called_once_with(self.path)
        self.assertEqual(len(Handler.ranges), 1)

    def test_skip_mtime_rejects_a_truncated_file(self):
        self.fetch()
        self.path.write_bytes(PAYLOAD[:-1])
        with self.assertRaisesRegex(RuntimeError, 'Existing file fails verification'):
            self.fetch(skip_cache_mtime=True)

    def test_skip_mtime_preserves_initial_download_hash_verification(self):
        with patch.object(remote, 'digest', wraps=remote.digest) as digest:
            self.fetch(skip_cache_mtime=True)
            digest.assert_called_once_with(self.partial)
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_drive_sync_timestamp_change_rechecks_content_without_download(self):
        self.fetch()
        timestamp = self.path.stat().st_mtime_ns + 180_000_000_000
        session.os.utime(self.path, ns=(timestamp, timestamp))
        with patch.object(remote, 'digest', wraps=remote.digest) as digest:
            self.fetch(skip_cache_mtime=False)
            digest.assert_called_once_with(self.path)
        self.assertEqual(len(Handler.ranges), 1)

    def test_cached_model_revalidation_hashes_during_one_drive_read(self):
        self.fetch()
        timestamp = self.path.stat().st_mtime_ns + 180_000_000_000
        session.os.utime(self.path, ns=(timestamp, timestamp))
        destination = Path(self.tmp.name) / 'ssd/model.gguf'
        original_open = Path.open
        reads = []

        def tracked_open(path, mode='r', *args, **kwargs):
            if path == self.path and mode == 'rb':
                reads.append(path)
            return original_open(path, mode, *args, **kwargs)

        with patch.object(remote, 'SSD_RESERVE', 0), \
             patch.object(Path, 'open', tracked_open), \
             patch.object(remote, 'digest') as digest:
            self.fetch(runtime_path=destination, skip_cache_mtime=False)
            digest.assert_not_called()
        self.assertEqual(reads, [self.path])
        self.assertEqual(destination.read_bytes(), PAYLOAD)
        self.assertEqual(len(Handler.ranges), 1)

    def test_initial_download_verifies_while_copying_to_ssd(self):
        destination = Path(self.tmp.name) / 'ssd/model.gguf'
        with patch.object(remote, 'SSD_RESERVE', 0), patch.object(remote, 'digest') as digest:
            self.fetch(runtime_path=destination)
            digest.assert_not_called()
        self.assertEqual(destination.read_bytes(), PAYLOAD)
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertFalse(self.partial.exists())

    def test_reverify_rejects_corruption_before_publishing_ssd_copy(self):
        self.fetch()
        self.path.write_bytes(b'x' * len(PAYLOAD))
        destination = Path(self.tmp.name) / 'ssd/model.gguf'
        with patch.object(remote, 'SSD_RESERVE', 0):
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                self.fetch(runtime_path=destination, reverify=True, skip_cache_mtime=True)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_name('model.gguf.part').exists())

    def test_default_copies_trusted_cache_without_extra_hash_pass(self):
        self.fetch()
        timestamp = self.path.stat().st_mtime_ns + 180_000_000_000
        session.os.utime(self.path, ns=(timestamp, timestamp))
        destination = Path(self.tmp.name) / 'ssd/model.gguf'
        with patch.object(remote, 'SSD_RESERVE', 0), \
             patch.object(remote, 'copy_model_file', wraps=remote.copy_model_file) as copy:
            self.fetch(runtime_path=destination)
            self.assertNotIn('sha256', copy.call_args.kwargs)
        self.assertEqual(destination.read_bytes(), PAYLOAD)

    def test_unmounted_drive_refused_before_directory_creation(self):
        destination = Path(self.tmp.name) / 'weights'
        with patch.object(remote, 'DRIVE', Path(self.tmp.name) / 'drive'):
            with self.assertRaisesRegex(RuntimeError, 'not mounted'):
                remote.ensure_drive(destination)
        self.assertFalse(destination.exists())

    def test_ssd_destination_refused_even_when_drive_is_mounted(self):
        drive = Path(self.tmp.name) / 'drive'
        (drive / 'MyDrive').mkdir(parents=True)
        with patch.object(remote, 'DRIVE', drive), patch.object(Path, 'is_mount', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'SSD fallback'):
                remote.ensure_drive(Path(self.tmp.name) / 'ssd')


class CacheOptionTests(unittest.TestCase):
    def test_local_cli_defaults_and_explicit_overrides(self):
        for flags, skip, reverify in [([], True, False), (['--skip-cache-mtime'], True, False),
                                     (['--no-skip-cache-mtime'], False, False), (['--reverify'], True, True)]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as work, \
                 patch.object(session, 'STATE', Path(work)), \
                 patch.object(sys, 'argv', ['session.py', 'stop', *flags]), \
                 patch.object(session, 'stop') as stop:
                self.assertEqual(session.main(), 0)
                args = stop.call_args.args[0]
                self.assertEqual(args.skip_cache_mtime, skip)
                self.assertEqual(args.reverify, reverify)


class BootstrapLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.log = self.work / 'bootstrap.log'
        self.log.touch()
        (self.work / 'bootstrap.pid').write_text(str(os.getpid()))
        self.instance = session.Session(None)
        self.ssh = patch.object(self.instance, 'ssh', side_effect=self.local_reader)
        self.ssh.start()
        self.addCleanup(self.ssh.stop)

    def local_reader(self, command):
        args = shlex.split(command)
        args[0], args[3] = sys.executable, str(self.work)
        args[2] = args[2].replace('/dev/shm/colab-privacy-', str(self.work) + '/colab-privacy-')
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout

    def test_appended_lines_appear_once_without_losing_fast_output(self):
        self.log.write_text('start\nSHA 0 GB\n')
        self.assertEqual(self.instance.bootstrap_output(), 'start\nSHA 0 GB\n')
        self.assertEqual(self.instance.bootstrap_output(), '')
        added = ''.join(f'SHA {i} GB\n' for i in range(1, 20)) + 'same\nsame\n'
        with self.log.open('a') as stream:
            stream.write(added)
        self.assertEqual(self.instance.bootstrap_output(), added)
        self.assertEqual(self.instance.bootstrap_output(), '')

    def test_partial_utf8_line_waits_until_complete_and_truncation_resets_cursor(self):
        line = '準備中\n'.encode()
        self.log.write_bytes(line[:2])
        self.assertEqual(self.instance.bootstrap_output(), '')
        with self.log.open('ab') as stream:
            stream.write(line[2:])
        self.assertEqual(self.instance.bootstrap_output(), '準備中\n')
        self.log.write_text('new\n')
        self.assertEqual(self.instance.bootstrap_output(), 'new\n')

    def test_exit_without_completion_reports_failure(self):
        (self.work / 'bootstrap.pid').write_text('0')
        self.log.write_text('engine crashed')
        output = self.instance.bootstrap_output()
        self.assertIn('engine crashed', output)
        self.assertIn('STRATA_BOOTSTRAP_FAILED', output)

    def test_failed_file_detected_even_when_bootstrap_is_alive(self):
        (self.work / 'failed').touch()
        self.assertIn('STRATA_BOOTSTRAP_FAILED', self.instance.bootstrap_output())

    def test_ram_startup_log_is_forwarded_once_without_writing_bootstrap(self):
        ram = self.work / 'colab-privacy-test' / 'inference.log'
        ram.parent.mkdir()
        ram.write_text('loading weights\n')
        metadata = 'COLAB_PRIVACY_DIAGNOSTIC=' + str(ram) + '\n'
        self.log.write_text(metadata)
        self.assertEqual(self.instance.bootstrap_output(), metadata + 'loading weights\n')
        self.assertEqual(self.instance.bootstrap_output(), '')
        with ram.open('a') as stream:
            stream.write('filling GPU cache\n')
        self.assertEqual(self.instance.bootstrap_output(), 'filling GPU cache\n')
        self.assertEqual(self.log.read_text(), metadata)


class LifecycleTests(unittest.TestCase):
    def test_idempotent_setup_reconnects_after_a_client_timeout(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session.uuid, 'uuid4', return_value=SimpleNamespace(hex='fixed')), \
             patch.object(instance, 'colab', side_effect=[session.CommandTimeout('connection wait'),
                                                        'prepared\nSTRATA_EXEC_OK_fixed\n']) as colab, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(instance.code('pass', timeout=120, retry_safe=True,
                                           stage='runtime再接続・準備', quiet=True), 'prepared')
            self.assertEqual(colab.call_count, 2)
            self.assertTrue(all(call.kwargs['timeout'] == 120 for call in colab.call_args_list))

    def test_reconnect_attempts_are_bounded_and_preserve_the_failed_stage(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'colab', side_effect=session.CommandTimeout('connection wait')) as colab, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(session.CommandTimeout, 'runtime再接続・準備'):
                instance.code('pass', timeout=120, retry_safe=True, stage='runtime再接続・準備')
            self.assertEqual(colab.call_count, 3)
            record = json.loads((Path(directory) / 'setup-error.json').read_text())
            self.assertEqual(record['stage'], 'runtime再接続・準備')
            self.assertEqual(record['attempt'], 3)
            self.assertEqual(record['timeout_seconds'], 120)
            self.assertEqual(record['error_type'], 'CommandTimeout')

    def test_bootstrap_and_flush_are_not_blindly_retried(self):
        instance = session.Session(None)
        for stage in ['モデル準備プロセス起動', 'GDrive同期']:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory, \
                 patch.object(session, 'STATE', Path(directory)), \
                 patch.object(instance, 'colab', side_effect=session.CommandTimeout('connection wait')) as colab:
                with self.assertRaises(session.CommandTimeout):
                    instance.code('pass', stage=stage)
                colab.assert_called_once()

    def test_remote_python_failure_is_not_retried_as_a_connection_timeout(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'colab', return_value='Traceback\nValueError: failed\n') as colab, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Python'):
                instance.code('pass', retry_safe=True)
            colab.assert_called_once()
            self.assertEqual(json.loads((Path(directory) / 'setup-error.json').read_text())['error_type'],
                             'IncompleteExecution')


    def test_start_tunnel_and_remote_config_use_dedicated_port(self):
        manifest = json.loads((session.HERE / 'manifest.json').read_text())
        args = SimpleNamespace(port=8085, context=131072, data_dir='/content/drive/MyDrive/Strata-data',
                               reverify=False, startup_minutes=1, variant='uncensored', skip_cache_mtime=True)
        instance = session.Session(args)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'id_ed25519').touch()
            child = Mock()
            child.poll.return_value = None
            with patch.object(session, 'STATE', Path(directory)), \
                 patch.object(session, 'catalog'), patch.object(session, 'available_port'), \
                 patch.object(session.shutil, 'which', return_value='/bin/tool'), \
                 patch.object(instance, 'allocate'), patch.object(instance, 'mount_drive'), \
                 patch.object(instance, 'code') as code, patch.object(instance, 'colab') as colab, \
                 patch.object(session.subprocess, 'Popen', return_value=child) as popen, \
                 patch.object(session, 'health', return_value={'loaded': True}), \
                 contextlib.redirect_stdout(io.StringIO()):
                instance.start()
            profile = manifest['variants']['uncensored']
            config = remote.engine_config(Path('/source'), Path('/engine'), Path('/data'), profile, 131072, 8,
                                          manifest['server_port'])
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index('-L') + 1], f'127.0.0.1:8085:127.0.0.1:{config["port"]}')
            self.assertEqual(config['host'], '127.0.0.1')
            self.assertEqual(config['log'], '/dev/null')
            self.assertFalse(config['api_monitor'])
            self.assertNotEqual(config['port'], 8080)
            launched = [call.args[0] for call in code.call_args_list]
            bootstrap = next(text for text in launched if 'bootstrap.pid' in text)
            preparation = next(call for call in code.call_args_list if 'mkdir(exist_ok=True)' in call.args[0])
            self.assertEqual(preparation.kwargs['timeout'], 120)
            self.assertTrue(preparation.kwargs['retry_safe'])
            launch = next(call for call in code.call_args_list if 'bootstrap.pid' in call.args[0])
            self.assertFalse(launch.kwargs.get('retry_safe', False))
            self.assertIn("'--variant', 'uncensored'", bootstrap)
            self.assertIn("'--skip-cache-mtime'", bootstrap)
            self.assertIn("'--engine-ready-seconds', '60'", bootstrap)
            self.assertEqual(config['model_name'], profile['model'])
            uploaded = [call.args[4] for call in colab.call_args_list]
            self.assertIn('content/strata-qwen/colab_privacy.py', uploaded)

    def test_close_flushes_before_stopping_owned_runtime(self):
        instance = session.Session(None)
        instance.owned = instance.drive_mounted = True
        actions = []
        with patch.object(instance, 'flush_drive', side_effect=lambda: (actions.append('flush') or True)), \
             patch.object(session, 'stop_runtime', side_effect=lambda: actions.append('stop')), \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            instance.close()
        self.assertEqual(actions, ['flush', 'stop'])

    def test_close_preserves_runtime_if_flush_crashes(self):
        instance = session.Session(None)
        instance.owned = instance.drive_mounted = True
        with patch.object(instance, 'flush_drive', side_effect=OSError('disk full')), \
             patch.object(session, 'stop_runtime') as stop, \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                instance.close()
            stop.assert_not_called()

    def test_orphan_stop_preserves_runtime_if_flush_fails(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session.subprocess, 'run', return_value=SimpleNamespace(stdout='active')), \
             patch.object(session.Session, 'flush_drive', return_value=False), \
             patch.object(session, 'stop_runtime') as stop:
            with self.assertRaisesRegex(RuntimeError, '停止していません'):
                session.stop()
            stop.assert_not_called()

    def test_stop_timeout_preserves_active_owner_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'session.json').write_text(json.dumps({'pid': 1234, 'identity': 'owner'}))
            with patch.object(session, 'STATE', Path(directory)), \
                 patch.object(session, 'proc_identity', return_value='owner'), \
                 patch.object(session.os, 'kill'), \
                 patch.object(session.time, 'monotonic', side_effect=[0, 0, 100000]), \
                 patch.object(session, 'stop_runtime') as stop:
                with self.assertRaisesRegex(RuntimeError, 'runtimeを残しています'):
                    session.stop()
                stop.assert_not_called()

    def test_force_stop_is_the_explicit_sync_bypass(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session.Session, 'flush_drive') as flush, \
             patch.object(session, 'stop_runtime') as stop:
            session.stop(SimpleNamespace(force=True))
            flush.assert_not_called()
            stop.assert_called_once()

    def test_default_flush_uses_colabs_24_hour_wait(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'code', return_value='STRATA_DRIVE_SYNC=flushed') as code, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(instance.flush_drive())
            self.assertEqual(code.call_args.kwargs['timeout'], 86400 + 180)

    def test_flush_success_records_confirmation(self):
        instance = session.Session(SimpleNamespace(flush_minutes=2))
        instance.drive_mounted = True
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'code', return_value='STRATA_DRIVE_SYNC=flushed') as code, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(instance.flush_drive())
            self.assertEqual(code.call_args.kwargs['timeout'], 300)
            self.assertTrue(json.loads((Path(directory) / 'drive-sync-status.json').read_text())['confirmed'])
            self.assertFalse(instance.drive_mounted)

    def test_flush_timeout_preserves_unsynced_runtime(self):
        instance = session.Session(SimpleNamespace(flush_minutes=2))
        instance.owned = instance.drive_mounted = True
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'code', side_effect=RuntimeError('timeout')), \
             patch.object(session, 'stop_runtime') as stop, \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(instance.close())
            stop.assert_not_called()
            self.assertFalse(json.loads((Path(directory) / 'drive-sync-status.json').read_text())['confirmed'])

    def test_sync_failure_keeps_session_state_for_retry(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(sys, 'argv', ['session.py', 'check-drive']), \
             patch.object(session.Session, 'check_drive'), \
             patch.object(session.Session, 'close', return_value=False), \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'):
            with self.assertRaisesRegex(RuntimeError, 'runtimeを残しています'):
                session.main()
            self.assertTrue((Path(directory) / 'session.json').exists())

    def test_stop_waits_for_owner_sync_before_calling_runtime_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            path.write_text(json.dumps({'pid': 1234, 'identity': 'owner', 'flush_minutes': 2}))
            actions = []
            ticks = iter(range(1000))
            with patch.object(session, 'STATE', Path(directory)), \
                 patch.object(session, 'proc_identity', side_effect=['owner'] * 50 + ['gone']), \
                 patch.object(session.os, 'kill', side_effect=lambda *a: actions.append('signal')), \
                 patch.object(session.time, 'monotonic', side_effect=lambda: next(ticks)), \
                 patch.object(session.time, 'sleep') as sleep, \
                 patch.object(session, 'stop_runtime', side_effect=lambda: actions.append('stop')), \
                 patch.object(session.subprocess, 'run', return_value=SimpleNamespace(stdout='active')), \
                 patch.object(session.Session, 'flush_drive', return_value=True), \
                 contextlib.redirect_stdout(io.StringIO()):
                session.stop()
            self.assertEqual(actions, ['signal', 'stop'])
            self.assertGreater(sleep.call_count, 40)  # The former 10s wait would kill an ongoing flush.

    def test_allocation_503_skips_drive_and_cleans_up(self):
        error = subprocess.CalledProcessError(1, ['colab', 'new'],
                                             output='ColabRequestError: POST /assign: Service Unavailable')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(sys, 'argv', ['session.py', 'check-drive', '--drive-auth', 'browser']), \
             patch.object(session.shutil, 'which', return_value='/bin/colab'), \
             patch.object(session.Session, 'colab', side_effect=["Session 'strata-qwen' not found.", error]) as colab, \
             patch.object(session.Session, 'mount_drive') as mount, \
             patch.object(session, 'stop_runtime') as stop, \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, '503 Service Unavailable'):
                session.main()
            mount.assert_not_called()
            stop.assert_called_once()
            self.assertEqual([call.args[0] for call in colab.call_args_list], ['status', 'new'])
            self.assertEqual(colab.call_args_list[-1].args,
                             ('new', '-s', 'strata-qwen', '--gpu', 'A100', '--high-mem'))
            record = json.loads((Path(directory) / 'allocation-error.json').read_text())
            self.assertEqual(record['reason'], 'service_unavailable')
            self.assertFalse((Path(directory) / 'session.json').exists())

    def test_other_allocation_error_is_not_misclassified_as_503(self):
        error = subprocess.CalledProcessError(1, ['colab', 'new'], output='Unauthorized')
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session.shutil, 'which', return_value='/bin/colab'), \
             patch.object(instance, 'colab', side_effect=["Session 'strata-qwen' not found.", error]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                instance.allocate()
            record = json.loads((Path(directory) / 'allocation-error.json').read_text())
            self.assertEqual(record['reason'], 'unknown')

    def test_kernel_error_with_zero_cli_exit_is_rejected(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'colab', return_value='Traceback\nValueError: mount failed\n'), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Python'):
                instance.code("raise ValueError('mount failed')")

    def test_marker_echo_in_traceback_is_not_success(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session.uuid, 'uuid4', return_value=SimpleNamespace(hex='fixed')), \
             patch.object(instance, 'colab', return_value="Traceback\n  print('STRATA_EXEC_OK_fixed')\n"), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                instance.code('broken()')

    def test_completed_execution_returns_payload(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session.uuid, 'uuid4', return_value=SimpleNamespace(hex='fixed')), \
             patch.object(instance, 'colab', return_value='payload\nSTRATA_EXEC_OK_fixed\n'):
            self.assertEqual(instance.code("print('payload')", quiet=True), 'payload')

    def test_mount_failure_stops_before_upload_and_saves_diagnostics(self):
        instance = session.Session(SimpleNamespace(port=8085, context=131072))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'id_ed25519').touch()
            with patch.object(session, 'STATE', Path(directory)), \
                 patch.object(session.shutil, 'which', return_value='/bin/tool'), \
                 patch.object(session, 'available_port'), patch.object(session, 'catalog'), \
                 patch.object(sys.stdin, 'isatty', return_value=True), \
                 patch.object(instance, 'allocate'), \
                 patch.object(instance, 'colab') as colab, \
                 patch.object(instance, 'run', return_value='STRATA_DRIVE_RESULT={"mounted": false, "logs": []}'), \
                 patch.object(instance, 'code', return_value='STRATA_DRIVE=false') as code, \
                 patch.object(session, 'colab_interpreter', return_value=[sys.executable]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'GDrive'):
                    instance.start()
                colab.assert_not_called()
                code.assert_called_once()  # No exec after the failed mount.
                record = json.loads((Path(directory) / 'drive-mount-error.json').read_text())
                self.assertFalse(record['mounted'])

    def test_diagnostic_failure_preserves_mount_failure(self):
        instance = session.Session(None)
        with patch.object(sys.stdin, 'isatty', return_value=True), \
             patch.object(instance, 'run', return_value='diagnostics unavailable'), \
             patch.object(instance, 'code', return_value='STRATA_DRIVE=false') as code, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'GDrive'):
                instance.mount_drive()
            code.assert_called_once()

    def test_process_timeout_terminates_child(self):
        instance = session.Session(None)
        with self.assertRaisesRegex(RuntimeError, '待機時間上限'):
            instance.run([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=0.15)
        self.assertIsNotNone(instance.children[-1].poll())

    def test_stream_timeout_terminates_child_and_preserves_prompt(self):
        instance = session.Session(None)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, '待機時間上限'):
                instance.run([sys.executable, '-c',
                              'import time; print("authorize", end="", flush=True); time.sleep(30)'],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             stream_output=True, timeout=0.15)
        self.assertEqual(output.getvalue(), 'authorize')
        self.assertIsNotNone(instance.children[-1].poll())

    def test_stream_success_captures_result(self):
        instance = session.Session(None)
        with contextlib.redirect_stdout(io.StringIO()):
            result = instance.run([sys.executable, '-c', 'print("日本語")'],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  stream_output=True, timeout=3)
        self.assertEqual(result, '日本語\n')

    def test_mount_helper_reuses_kernel_and_saves_callback_ids(self):
        from unittest.mock import Mock
        current = SimpleNamespace(kernel_id='existing-kernel', session_id='existing-session')
        state = SimpleNamespace(get_session=Mock(return_value=current), store=Mock())
        factory = Mock()
        mount_helper.attached_runtime(factory, state, 'url', 'token', session_name='strata-qwen')
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs['kernel_id'], 'existing-kernel')
        self.assertEqual(kwargs['session_id'], 'existing-session')
        kwargs['on_kernel_started']('replacement')
        self.assertEqual(current.kernel_id, 'replacement')
        state.store.add.assert_called_once_with(current)

    def test_mount_cell_captures_failure_and_diagnostics(self):
        import types
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        drive = types.ModuleType('google.colab.drive')
        def mount(path):
            raise ValueError('mount failed')
        drive.mount = mount
        colab.drive = drive
        google.colab = colab
        output = io.StringIO()
        with patch.dict(sys.modules, {'google': google, 'google.colab': colab, 'google.colab.drive': drive}), \
             contextlib.redirect_stdout(output):
            exec(mount_helper.mount_code(), {})
        line = next(line for line in output.getvalue().splitlines() if line.startswith('STRATA_DRIVE_RESULT='))
        result = json.loads(line.removeprefix('STRATA_DRIVE_RESULT='))
        self.assertEqual(result['mount_error_type'], 'ValueError')
        self.assertIn('logs', result)

    def test_browser_mode_opens_existing_runtime_without_cli_auth(self):
        instance = session.Session(SimpleNamespace(drive_auth='browser'))
        with patch.object(sys.stdin, 'isatty', return_value=True), \
             patch.object(instance, 'code', side_effect=['STRATA_DRIVE=false', 'STRATA_DRIVE=true']), \
             patch.object(instance, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            instance.mount_drive()
        self.assertEqual(run.call_args_list[0].args[0], ['colab', 'url', '-s', 'strata-qwen'])
        self.assertEqual(run.call_count, 2)  # URL and bounded human wait only.

    def test_check_drive_success_stops_runtime_without_starting_model(self):
        def allocate(instance):
            instance.owned = True

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(sys, 'argv', ['session.py', 'check-drive']), \
             patch.object(session.Session, 'allocate', allocate), \
             patch.object(session.Session, 'mount_drive'), \
             patch.object(session.Session, 'start') as start, \
             patch.object(session, 'stop_runtime') as stop, \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(session.main(), 0)
            start.assert_not_called()
            stop.assert_called_once()

    def test_diagnostics_do_not_return_log_credentials(self):
        text = 'UNAUTHENTICATED https://host/?token=secret\nQueryManager timed out email@example.com'
        info = diagnostics.summarize(text)
        self.assertEqual(info, {'root_folder_timeout': 1, 'auth_error': 1})
        self.assertNotIn('secret', json.dumps(info))

    def test_setup_failure_still_closes_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(session, 'STATE', Path(directory)), \
                 patch.object(sys, 'argv', ['session.py', 'start']), \
                 patch.object(session.Session, 'start', side_effect=RuntimeError('mount failed')), \
                 patch.object(session.Session, 'close') as close, \
                 patch.object(signal, 'signal'), patch.object(signal, 'alarm'):
                with self.assertRaisesRegex(RuntimeError, 'mount failed'):
                    session.main()
                close.assert_called_once()
                self.assertFalse((Path(directory) / 'session.json').exists())

    def test_close_always_stops_owned_runtime(self):
        instance = session.Session(None)
        instance.owned = True
        with patch.object(session, 'stop_runtime') as stop, \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            instance.close()
            stop.assert_called_once()

    def test_unowned_runtime_is_preserved(self):
        instance = session.Session(None)
        with patch.object(session, 'stop_runtime') as stop, \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'):
            instance.close()
            stop.assert_not_called()

    def test_stop_failure_propagates(self):
        with patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['colab', 'stop'])):
            with self.assertRaises(subprocess.CalledProcessError):
                session.stop_runtime()

    def test_stop_success_requires_server_confirmation(self):
        status = subprocess.CompletedProcess(['colab', 'status'], 0, stdout='session still alive')
        with patch.object(subprocess, 'run', return_value=status):
            with self.assertRaisesRegex(RuntimeError, 'could not be confirmed'):
                session.stop_runtime()

    def test_codex_settings_are_valid_toml(self):
        profile = session.manifest()['variants']['standard']
        command = session.codex_command(8085, 131072, ['exec', 'test'], profile)
        values = [command[i + 1] for i, value in enumerate(command) if value == '-c']
        settings = tomllib.loads('\n'.join(values))
        self.assertEqual(settings['model'], profile['model'])
        provider = settings['model_providers']['strata_colab']
        self.assertEqual(provider['wire_api'], 'responses')
        self.assertEqual(provider['base_url'], 'http://127.0.0.1:8085/v1')
        self.assertEqual(command[-2:], ['exec', 'test'])

    def test_stale_pid_does_not_signal_other_process(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'session.json').write_text(json.dumps({'pid': 1234, 'identity': 'old'}))
            with patch.object(session, 'STATE', Path(directory)), \
                 patch.object(session, 'proc_identity', return_value='new'), \
                 patch.object(session.os, 'kill') as kill, \
                 patch.object(session, 'stop_runtime') as stop, \
                 patch.object(session.subprocess, 'run', return_value=SimpleNamespace(stdout='active')), \
                 patch.object(session.Session, 'flush_drive', return_value=True):
                session.stop()
                kill.assert_not_called()
                stop.assert_called_once()


class RuntimeCopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = patch.object(remote, 'SSD_RESERVE', 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_copy_error_preserves_previous_model_and_removes_partial(self):
        source, destination = self.root / 'drive.gguf', self.root / 'ssd.gguf'
        source.write_bytes(PAYLOAD)
        destination.write_bytes(b'previous model')
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                remote.copy_model_file(source, destination, len(PAYLOAD), sha256='0' * 64)
        self.assertEqual(destination.read_bytes(), b'previous model')
        self.assertEqual(source.read_bytes(), PAYLOAD)
        self.assertFalse((self.root / 'ssd.gguf.part').exists())

    def test_short_or_growing_copy_is_not_published(self):
        source, destination = self.root / 'source', self.root / 'destination'
        source.write_bytes(PAYLOAD)
        for size in [len(PAYLOAD) - 1, len(PAYLOAD) + 1]:
            with self.subTest(size=size), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
                    remote.copy_model_file(source, destination, size)
                self.assertFalse(destination.exists())
                self.assertFalse((self.root / 'destination.part').exists())

    def test_pack_copy_preserves_shard_names_and_publishes_completion_last(self):
        source, destination = self.root / 'drive-pack', self.root / 'ssd-pack'
        source.mkdir()
        (source / 'native_experts.txt').write_text('# first.gguf\n0 1 2 3 4 5 6 7 second.gguf\n')
        (source / 'dense.bin').write_bytes(b'weights')
        (source / '.complete.json').write_text('{}')
        with patch.object(remote, 'copy_model_file', wraps=remote.copy_model_file) as copy, \
             contextlib.redirect_stdout(io.StringIO()):
            remote.copy_model_tree(source, destination)
        self.assertEqual(copy.call_args_list[-1].args[0].name, '.complete.json')
        self.assertEqual((destination / 'native_experts.txt').read_bytes(), (source / 'native_experts.txt').read_bytes())
        self.assertEqual((destination / 'dense.bin').read_bytes(), b'weights')
        self.assertFalse(any(p.is_symlink() for p in destination.rglob('*')))

    def test_pack_symlink_into_drive_is_rejected(self):
        source = self.root / 'pack'
        source.mkdir()
        (self.root / 'model').write_bytes(PAYLOAD)
        (source / 'dense.bin').symlink_to(self.root / 'model')
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            remote.copy_model_tree(source, self.root / 'ssd')
        self.assertFalse((self.root / 'ssd').exists())

    def test_pack_copy_excludes_logs_and_unrelated_files(self):
        source, destination = self.root / 'pack', self.root / 'ssd'
        (source / 'tokenizer').mkdir(parents=True)
        (source / 'dense.bin').write_bytes(PAYLOAD)
        (source / 'inference.log').write_text('private text')
        (source / 'conversation.json').write_text('private text')
        (source / 'dense.bin.tmp').write_text('incomplete')
        (source / 'tokenizer' / 'vocab.json').write_text('{}')
        (source / 'tokenizer' / 'session.json').write_text('private text')
        with contextlib.redirect_stdout(io.StringIO()):
            remote.copy_model_tree(source, destination)
        self.assertEqual({str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()},
                         {'dense.bin', 'tokenizer/vocab.json'})

    def test_insufficient_runtime_space_fails_before_copy_or_download(self):
        profile = {'model': 'test', 'pack': 'pack', 'files': [{'destination': 'model.gguf', 'size': 10000}]}
        manifest = {'default_variant': 'test', 'server_port': 18080, 'variants': {'test': profile}}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        with patch.object(remote, 'WORK', self.root), \
             patch.object(sys, 'argv', ['remote.py']), \
             patch.object(remote, 'ensure_drive', return_value=self.root / 'drive'), \
             patch.object(remote, 'check_server_port'), patch.object(remote, 'check_privacy'), \
             patch.object(remote, 'check_drive_space'), \
             patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=100)), \
             patch.object(remote, 'fetch_verified') as fetch, patch.object(remote, 'source_tree') as build, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'runtime SSD space'):
                remote.main()
            fetch.assert_not_called()
            build.assert_not_called()

    def test_all_runtime_model_paths_use_ssd_for_both_variants(self):
        root = Path('/content/strata-qwen/model-data')
        for profile in session.manifest()['variants'].values():
            settings = remote.engine_config(Path('/content/strata-qwen/Strata'), Path('/engine'), root,
                                            profile, 131072, 11, 18080)
            self.assertNotIn('/content/drive', json.dumps(settings))
            for argument in ['--pack', '--native', '--ple-gguf', '--mtp']:
                path = Path(settings['args'][settings['args'].index(argument) + 1])
                self.assertTrue(path.is_relative_to(root))
            self.assertTrue(Path(settings['tokenizer']).is_relative_to(root))


class PackAndReadyTests(unittest.TestCase):
    def test_iq_pack_cache_uses_index_not_mtp_dense_text(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory)
            identity = {'strata': 'commit', 'gguf': 'sha'}
            (pack / '.complete.json').write_text(json.dumps(identity))
            for name in ('dense.bin', 'index.txt', 'native_experts.txt', 'tokenizer/vocab.json',
                         'tokenizer/chat_template.jinja'):
                path = pack / name
                path.parent.mkdir(exist_ok=True)
                path.touch()
            self.assertTrue(remote.pack_complete(pack, identity))
            (pack / 'index.txt').unlink()
            self.assertFalse(remote.pack_complete(pack, identity))

    def test_ready_timeout_is_exported_before_server_exec(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            profile = {'pack': 'pack', 'model': 'test', 'files': [
                {'destination': name, 'sha256': sha, 'repo': 'test/model', 'revision': 'rev',
                 'source': name, 'size': 1}
                for name, sha in [('first.gguf', 'a'), ('second.gguf', 'b')]]}
            manifest = {'strata_commit': 'commit', 'llama_commit': 'llama', 'server_port': 18080,
                        'default_variant': 'test', 'variants': {'test': profile}}
            (work / 'manifest.json').write_text(json.dumps(manifest))
            source = work / 'Strata'
            (source / '.venv/bin').mkdir(parents=True)
            (source / '.venv/bin/python').touch()
            (source / 'engine-cuda12').mkdir()
            (source / 'engine-cuda12/strata').write_text('engine')
            pack = work / 'data/pack'
            for name in ('dense.bin', 'index.txt', 'native_experts.txt', 'tokenizer/vocab.json',
                         'tokenizer/chat_template.jinja'):
                path = pack / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('pack')
            (pack / '.complete.json').write_text(json.dumps({'strata': 'commit', 'gguf': 'a'}))
            read_text = Path.read_text

            def read(path, *args, **kwargs):
                if str(path) == '/proc/meminfo':
                    return 'MemTotal: 167772160 kB\n'
                return read_text(path, *args, **kwargs)

            with patch.dict(os.environ), \
                 patch.object(Path, 'read_text', read), \
                 patch.object(remote, 'WORK', work), \
                 patch.object(sys, 'argv', ['remote.py', '--engine-ready-seconds', '2400']), \
                 patch.object(remote, 'ensure_drive', return_value=work / 'data'), \
                 patch.object(remote, 'check_privacy'), patch.object(remote, 'check_drive_space'), \
                 patch.object(remote, 'source_tree'), patch.object(remote, 'fetch_verified') as fetch, \
                 patch.object(remote, 'SSD_RESERVE', 0), \
                 patch.object(remote, 'run'), \
                 patch.object(remote.shutil, 'which', return_value='/bin/nvcc'), \
                 patch.object(remote.subprocess, 'check_output', side_effect=['A100, 81920 MiB, 8.0', 'release 12.6,']), \
                 patch.object(remote, 'serve_private') as serve, \
                 contextlib.redirect_stdout(io.StringIO()):
                remote.main()
                self.assertEqual(os.environ['STRATA_ENGINE_READY_S'], '2400.0')
                self.assertTrue(all(call.kwargs['skip_cache_mtime'] for call in fetch.call_args_list))
                settings = json.loads((work / 'strata.json').read_text())
                self.assertNotIn(str(work / 'data'), json.dumps(settings))
                self.assertEqual(settings['tokenizer'], str(work / 'model-data/pack/tokenizer'))
                serve.assert_called_once()


class PreflightTests(unittest.TestCase):
    def test_busy_port_refused_before_build_and_download(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as busy:
            busy.bind(('127.0.0.1', 0))
            manifest = {'server_port': busy.getsockname()[1], 'default_variant': 'standard',
                        'variants': {'standard': {'files': [], 'pack': 'packs/iq3_s'}}}
            (Path(directory) / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(remote, 'WORK', Path(directory)), \
                 patch.object(remote, 'ensure_drive', return_value=Path(directory)), \
                 patch.object(sys, 'argv', ['remote.py']), \
                 patch.object(remote, 'source_tree') as build, \
                 patch.object(remote, 'fetch_verified') as download:
                with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                    remote.main()
                build.assert_not_called()
                download.assert_not_called()

    def test_reserved_colab_port_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'Colab'):
            remote.check_server_port(8080)

    def test_insufficient_drive_space_rejected(self):
        manifest = {'pack': 'packs/iq3_s', 'files': [{'destination': 'models/model.gguf', 'size': 10_000_000_000}]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000)), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient Drive'):
                remote.check_drive_space(Path(directory), manifest)

    def test_drive_space_counts_existing_partial(self):
        manifest = {'pack': 'packs/iq3_s', 'files': [{'destination': 'model.gguf', 'size': 100}]}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'model.gguf.part').write_bytes(b'x' * 90)
            with patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=5_000_000_010)), \
                 contextlib.redirect_stdout(io.StringIO()):
                remote.check_drive_space(Path(directory), manifest)

    def test_unknown_variant_refused_before_download(self):
        options = {'standard': {'files': []}, 'uncensored': {'files': []}}
        self.assertEqual(remote.select_variant({'default_variant': 'standard', 'variants': options}, None)[0],
                         'standard')
        self.assertEqual(remote.select_variant({'default_variant': 'standard', 'variants': options},
                                               'uncensored')[0], 'uncensored')
        for requested in ('jailbroken', None):  # No usable default when the manifest omits one.
            with self.assertRaisesRegex(RuntimeError, 'choose from'):
                remote.select_variant({'variants': options}, requested)

    def test_uncensored_variant_keeps_its_own_shards_and_pack(self):
        manifest = session.manifest()
        standard, uncensored = manifest['variants']['standard'], manifest['variants']['uncensored']
        self.assertEqual(manifest['default_variant'], 'standard')
        self.assertNotEqual(standard['model'], uncensored['model'])
        self.assertNotEqual(standard['pack'], uncensored['pack'])
        # iq_pack.py finds -0000N-of-0000M siblings beside --gguf, so shards of two
        # models must never share a directory even when one shard is byte-identical.
        for profile in (standard, uncensored):
            shards = profile['files'][:2]
            self.assertEqual(len({item['destination'].rsplit('/', 1)[0] for item in shards}), 1)
        standard_dir = standard['files'][0]['destination'].rsplit('/', 1)[0]
        uncensored_dir = uncensored['files'][0]['destination'].rsplit('/', 1)[0]
        self.assertNotEqual(standard_dir, uncensored_dir)
        self.assertEqual(uncensored['files'][0]['repo'], 'SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF')
        self.assertNotEqual(standard['files'][0]['sha256'], uncensored['files'][0]['sha256'])
        # Both variants run the largest tier Strata supports, and share the MTP runtime files.
        self.assertEqual([item['destination'] for item in standard['files'][2:]],
                         [item['destination'] for item in uncensored['files'][2:]])
        self.assertTrue(standard['files'][0]['source'].startswith('IQ3_S/'))
        self.assertTrue(uncensored['files'][0]['source'].startswith('IQ3_S/'))

    def test_engine_config_selects_variant_paths_and_model(self):
        manifest = session.manifest()
        for name in ('standard', 'uncensored'):
            profile = manifest['variants'][name]
            config = remote.engine_config(Path('/source'), Path('/engine'), Path('/data'), profile, 65536, 4, 18080)
            self.assertEqual(config['model_name'], profile['model'])
            self.assertIn(str(Path('/data') / profile['pack']), config['args'])
            self.assertEqual(config['args'][config['args'].index('--native') + 1],
                             str(Path('/data') / profile['files'][0]['destination']))
            self.assertEqual(config['args'][config['args'].index('--ple-gguf') + 1],
                             str(Path('/data') / profile['files'][1]['destination']))

    def test_health_accepts_either_variant_model(self):
        models = [profile['model'] for profile in session.manifest()['variants'].values()]
        for served in models + ['some-other-model', None]:
            body = json.dumps({'model': served, 'loaded': True, 'max_context': 131072}).encode()
            for expected in [None] + models:
                with patch.object(session.OPENER, 'open', return_value=io.BytesIO(body)):
                    info = session.health(8085, expected)
                accepted = served == expected if expected else served in models
                if accepted:
                    self.assertEqual(info['model'], served)
                else:
                    self.assertIsNone(info)

    def test_variant_selection_and_model_lookup(self):
        default, standard = session.select_variant(None)
        self.assertEqual(default, session.manifest()['default_variant'])
        name, uncensored = session.select_variant('uncensored')
        self.assertEqual(name, 'uncensored')
        self.assertNotEqual(standard['model'], uncensored['model'])
        self.assertEqual(session.profile_for_model(uncensored['model'])[0], 'uncensored')
        self.assertEqual(session.profile_for_model(standard['model'])[0], 'standard')
        self.assertEqual(session.profile_for_model('unknown'), (None, None))
        with self.assertRaisesRegex(RuntimeError, 'variant'):
            session.select_variant('nope')

    def test_uncensored_codex_settings_are_valid_toml(self):
        profile = session.manifest()['variants']['uncensored']
        command = session.codex_command(8085, 131072, ['exec', 'test'], profile)
        values = [command[i + 1] for i, value in enumerate(command) if value == '-c']
        settings = tomllib.loads('\n'.join(values))
        self.assertEqual(settings['model'], profile['model'])
        self.assertEqual(settings['model_providers']['strata_colab']['name'], profile['provider_name'])

    def test_remote_shutdown_stops_model_before_drive_flush(self):
        import types
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        actions = []
        colab.drive = SimpleNamespace(flush_and_unmount=lambda **kwargs: actions.append(('flush', kwargs)))
        google.colab = colab
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(shutdown_remote, 'WORK', Path(directory)), \
             patch.dict(sys.modules, {'google': google, 'google.colab': colab}), \
             patch.object(shutdown_remote, 'stop_bootstrap', side_effect=lambda: actions.append(('stop', {}))), \
             patch.object(shutdown_remote.os.path, 'ismount', return_value=True):
            self.assertEqual(shutdown_remote.shutdown(45), 'flushed')
        self.assertEqual(actions, [('stop', {}), ('flush', {'timeout_ms': 45000})])

    def test_remote_shutdown_does_not_report_unmounted_as_synced(self):
        import types
        colab = types.ModuleType('google.colab')
        colab.drive = Mock()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(shutdown_remote, 'WORK', Path(directory)), \
             patch.dict(sys.modules, {'google.colab': colab}), \
             patch.object(shutdown_remote, 'stop_bootstrap'), \
             patch.object(shutdown_remote.os.path, 'ismount', return_value=False):
            self.assertEqual(shutdown_remote.shutdown(45), 'not_mounted')
        colab.drive.flush_and_unmount.assert_not_called()

    def test_detached_failed_flush_restarts_cached_drive_before_retry(self):
        import types
        colab = types.ModuleType('google.colab')
        colab.drive = Mock()
        colab.drive._env.return_value = SimpleNamespace(root_dir='/')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / 'drive-sync-status.json').write_text('{"confirmed": false}')
            with patch.dict(sys.modules, {'google.colab': colab}), \
                 patch.object(shutdown_remote, 'WORK', work), \
                 patch.object(shutdown_remote, 'stop_bootstrap'), \
                 patch.object(shutdown_remote.os.path, 'ismount', return_value=False), \
                 patch.object(shutdown_remote, 'drive_daemon_running', return_value=False), \
                 patch.object(shutdown_remote, 'recover_drive') as recover:
                self.assertEqual(shutdown_remote.shutdown(90), 'flushed')
                recover.assert_called_once()
                colab.drive.flush_and_unmount.assert_called_once_with(timeout_ms=90000)
                self.assertTrue(json.loads((work / 'drive-sync-status.json').read_text())['confirmed'])

    def test_failed_flush_leaves_pending_marker_for_retry(self):
        import types
        colab = types.ModuleType('google.colab')
        colab.drive = Mock()
        colab.drive.flush_and_unmount.side_effect = ValueError('flush failed')
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {'google.colab': colab}), \
             patch.object(shutdown_remote, 'WORK', Path(directory)), \
             patch.object(shutdown_remote, 'stop_bootstrap'), \
             patch.object(shutdown_remote.os.path, 'ismount', return_value=True):
            with self.assertRaises(ValueError):
                shutdown_remote.shutdown(90)
            self.assertFalse(json.loads((Path(directory) / 'drive-sync-status.json').read_text())['confirmed'])

    def test_stale_bootstrap_pid_cannot_kill_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'bootstrap.pid').write_text('1234')
            with patch.object(shutdown_remote, 'WORK', Path(directory)), \
                 patch.object(shutdown_remote.os, 'getpgid', return_value=1234), \
                 patch.object(Path, 'read_bytes', return_value=b'other-job'), \
                 patch.object(shutdown_remote.os, 'killpg') as kill:
                shutdown_remote.stop_bootstrap()
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
