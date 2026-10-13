"""Exercise storage integrity, engine caching and billable-runtime cleanup without allocating a VM."""
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
    authorizations = []
    reject_token = False

    def do_GET(self):
        self.authorizations.append(self.headers.get('Authorization'))
        if self.reject_token and self.headers.get('Authorization'):
            self.send_response(401)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
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

    def test_drive_sync_timestamp_change_rechecks_content_without_download(self):
        self.fetch()
        timestamp = self.path.stat().st_mtime_ns + 180_000_000_000
        session.os.utime(self.path, ns=(timestamp, timestamp))
        with patch.object(remote, 'digest', wraps=remote.digest) as verify:
            self.fetch(skip_cache_mtime=False)
            verify.assert_called_once_with(self.path)
        self.assertEqual(len(Handler.ranges), 1)
        marker = self.path.with_name('model.gguf.verified.json')
        self.assertEqual(json.loads(marker.read_text())['mtime_ns'], self.path.stat().st_mtime_ns)

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
            with self.assertRaisesRegex(RuntimeError, 'Persistent model destination'):
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


class ManifestTests(unittest.TestCase):
    def test_variants_are_declared_once_each_with_their_own_shard_folder(self):
        manifest = session.manifest()
        self.assertEqual(set(manifest['variants']), {'flash', 'flash-uncensored', 'full'})
        self.assertEqual(manifest['default_variant'], 'flash')
        folders = set()
        for name, profile in manifest['variants'].items():
            self.assertIn(profile['engine'], ('maya', 'llama'))
            self.assertIn(profile['default_context'], profile['contexts'])
            destinations = [item['destination'] for item in profile['files']]
            self.assertEqual(len(destinations), len(set(destinations)))
            # iq_pack.py and llama-server both resolve sibling shards by name in the same folder,
            # so shards of different models must never share a directory.
            folders.add(destinations[0].rsplit('/', 1)[0])
            self.assertEqual({item['destination'].rsplit('/', 1)[0] for item in profile['files']},
                             {destinations[0].rsplit('/', 1)[0]}, name)
            for item in profile['files']:
                self.assertGreater(item['size'], 1024 ** 3)
                self.assertEqual(len(item['sha256']), 64)
                self.assertTrue(item['revision'])
        self.assertEqual(len(folders), 3)

    def test_llama_variants_point_at_a_downloaded_first_shard(self):
        for name, profile in session.manifest()['variants'].items():
            if profile['engine'] != 'llama':
                continue
            first = profile['llama']['first_shard']
            self.assertIn(first, [item['destination'] for item in profile['files']])
            # llama-server loads a split model from its first shard and finds the rest by name.
            self.assertTrue(first.endswith(f'-00001-of-{len(profile["files"]):05d}.gguf'), first)
            for flag in ('-cmoe', '-ngl'):
                self.assertIn(flag, profile['llama']['args'], f'{name} needs {flag}: the weights exceed 80GB VRAM')
            self.assertIn('--jinja', profile['llama']['args'])

    def test_maya_variant_uses_the_models_dir_layout_maya_expects(self):
        profile = session.manifest()['variants']['flash']
        self.assertEqual(profile['engine'], 'maya')
        # Maya looks in <models-dir>/<model>/ for its own shard names (download_dir).
        model = profile['maya']['model']
        for item in profile['files']:
            self.assertTrue(item['destination'].startswith(f'models/{model}/'), item['destination'])
            self.assertEqual(item['source'].rsplit('/', 1)[-1], item['destination'].rsplit('/', 1)[-1])
        self.assertEqual(profile['maya']['env']['STRATA_GLM_RAM_RESIDENT'], '1')
        self.assertGreaterEqual(int(profile['maya']['env']['STRATA_GLM_RAM_HEADROOM_GB']), 16)

    def test_engines_are_pinned_to_commits(self):
        manifest = session.manifest()
        for key in ('maya_commit', 'llama_commit', 'maya_llama_commit'):
            self.assertEqual(len(manifest[key]), 40)
        # Maya builds against its own pinned llama.cpp, not the one the llama variants use.
        self.assertNotEqual(manifest['maya_llama_commit'], manifest['llama_commit'])

    def test_variant_selection_and_model_lookup(self):
        default, flash = session.select_variant(None)
        self.assertEqual(default, session.manifest()['default_variant'])
        name, full = session.select_variant('full')
        self.assertEqual(name, 'full')
        self.assertNotEqual(flash['model'], full['model'])
        self.assertEqual(session.profile_for_model(full['model'])[0], 'full')
        self.assertEqual(session.profile_for_model('unknown'), (None, None))
        with self.assertRaisesRegex(RuntimeError, 'variant'):
            session.select_variant('nope')

    def test_context_options_are_per_variant(self):
        _, flash = session.select_variant('flash')
        _, full = session.select_variant('full')
        self.assertEqual(session.select_context(flash, None), 131072)
        self.assertEqual(session.select_context(full, None), 65536)
        self.assertEqual(session.select_context(full, 32768), 32768)
        # 131072 fits the Flash KV budget but is not offered for the full model.
        with self.assertRaisesRegex(RuntimeError, 'context'):
            session.select_context(full, 131072)


class PreflightTests(unittest.TestCase):
    def test_busy_port_refused_before_build_and_download(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as busy:
            busy.bind(('127.0.0.1', 0))
            manifest = json.loads((session.HERE / 'manifest.json').read_text())
            manifest['server_port'] = busy.getsockname()[1]
            (Path(directory) / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(remote, 'WORK', Path(directory)), \
                 patch.object(remote, 'ensure_drive', return_value=Path(directory)), \
                 patch.object(sys, 'argv', ['remote.py']), \
                 patch.object(remote, 'check_accelerator'), patch.object(remote, 'check_ram'), \
                 patch.object(remote, 'source_tree') as build, \
                 patch.object(remote, 'fetch_verified') as download:
                with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                    remote.main()
                build.assert_not_called()
                download.assert_not_called()

    def test_reserved_colab_port_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'Colab'):
            remote.check_server_port(8080)

    def test_missing_weights_are_counted_before_download(self):
        profile = {'files': [{'destination': 'models/model.gguf', 'size': 10_000_000_000}]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000)), \
             patch.object(remote, 'drive_free_bytes', return_value=1000), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, 'less free space'):
                remote.check_drive_space(Path(directory), profile)
            self.assertIn('0.00/10.00 GB', output.getvalue())

    def test_drive_space_counts_existing_partial(self):
        profile = {'files': [{'destination': 'model.gguf', 'size': 100}]}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'model.gguf.part').write_bytes(b'x' * 90)
            with patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=5_000_000_010)), \
                 patch.object(remote, 'drive_free_bytes', return_value=10), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                remote.check_drive_space(Path(directory), profile)
            self.assertIn('missing weights: 0.00 GB', output.getvalue())

    def test_small_runtime_cache_is_not_misclassified_as_cloud_quota(self):
        profile = {'files': [{'destination': 'model.gguf', 'size': 134_000_000_000}]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=122_000_000_000)), \
             patch.object(remote, 'drive_free_bytes', return_value=5_000_000_000_000), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            remote.check_drive_space(Path(directory), profile)
        self.assertIn('Runtime DriveFS cache disk free: 122.00 GB', output.getvalue())
        self.assertIn('Drive API cloud free: 5000.00 GB', output.getvalue())

    def test_unknown_cloud_quota_does_not_fall_back_to_cache_quota(self):
        profile = {'files': [{'destination': 'model.gguf', 'size': 134_000_000_000}]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000)), \
             patch.object(remote, 'drive_free_bytes', return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            remote.check_drive_space(Path(directory), profile)

    def test_quota_uses_cloud_api_and_normalizes_metadata_address(self):
        metadata = io.StringIO('{"access_token":"credential-must-stay-private"}')
        quota = io.StringIO('{"storageQuota":{"limit":"5000000000000","usage":"400000000000"}}')
        local = Mock()
        local.open.return_value = metadata
        with patch.dict(remote.os.environ, {'TBE_EPHEM_CREDS_ADDR': 'metadata.internal:8181'}), \
             patch.object(remote.urllib.request, 'build_opener', return_value=local), \
             patch.object(remote.urllib.request, 'urlopen', return_value=quota) as cloud, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(remote.drive_free_bytes(), 4_600_000_000_000)
        self.assertTrue(local.open.call_args.args[0].full_url.startswith('http://metadata.internal:8181/'))
        self.assertEqual(cloud.call_args.args[0].full_url,
                         'https://www.googleapis.com/drive/v3/about?fields=storageQuota')
        self.assertNotIn('credential-must-stay-private', output.getvalue())

    def test_quota_failure_does_not_expose_credential_errors(self):
        with patch.dict(remote.os.environ, {'TBE_EPHEM_CREDS_ADDR': 'metadata.internal:8181'}), \
             patch.object(remote.urllib.request, 'build_opener', side_effect=RuntimeError('private-token-value')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(remote.drive_free_bytes())
        self.assertNotIn('private-token-value', output.getvalue())

    def test_unknown_variant_and_context_refused_before_download(self):
        options = {'flash': {'files': [], 'contexts': [32768], 'default_context': 32768}}
        self.assertEqual(remote.select_variant({'default_variant': 'flash', 'variants': options}, None)[0], 'flash')
        for requested in ('full-uncensored', None):  # No usable default when the manifest omits one.
            with self.assertRaisesRegex(RuntimeError, 'choose from'):
                remote.select_variant({'variants': options}, requested)
        with self.assertRaisesRegex(RuntimeError, 'choose from'):
            remote.select_context(options['flash'], 131072)

    def test_ram_below_the_expert_budget_refuses_to_start(self):
        for total, message in ((140 * 1024 * 1024, 'high-mem RAM'), (170 * 1024 * 1024, None)):
            text = f'MemTotal:       {total} kB\nMemFree: 1 kB\n'
            with patch.object(remote.Path, 'read_text', return_value=text), \
                 contextlib.redirect_stdout(io.StringIO()):
                if message:
                    with self.assertRaisesRegex(RuntimeError, message):
                        remote.check_ram()
                else:
                    remote.check_ram()

    def test_engine_signature_separates_engines_and_toolchains(self):
        manifest = session.manifest()
        flash = manifest['variants']['flash']
        full = manifest['variants']['full']
        with patch.object(remote.Path, 'read_text', return_value='flags\t: avx2\n'):
            maya = remote.engine_signature(manifest, flash, '/usr/local/cuda/bin/nvcc', 'Cuda V12.2')
            llama = remote.engine_signature(manifest, full, '/usr/local/cuda/bin/nvcc', 'Cuda V12.2')
            other = remote.engine_signature(manifest, full, '/usr/local/cuda-11.8/bin/nvcc', 'Cuda V11.8')
        self.assertNotEqual(maya, llama)   # different engine and commit
        self.assertNotEqual(llama, other)  # same engine, different toolkit
        self.assertEqual(len(maya), 20)


class EngineCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / 'drive'
        self.data.mkdir()
        self.manifest = session.manifest()

    def test_cached_llama_server_skips_the_build(self):
        cached = self.data / 'engines' / 'llama-sig' / 'llama-server'
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b'server')
        with tempfile.TemporaryDirectory() as work:
            with patch.object(remote, 'WORK', Path(work)), \
                 patch.object(remote, 'source_tree') as tree, \
                 patch.object(remote, 'run') as run:
                binary = remote.llama_server(self.manifest, self.data, 'sig', '/usr/bin/nvcc')
            self.assertEqual(run.call_count, 0)  # no venv, no cmake, no compile
            self.assertTrue(binary.is_file())
            self.assertEqual(binary.read_bytes(), b'server')
            self.assertTrue(binary.stat().st_mode & 0o111)
            tree.assert_called_once()

    def test_llama_build_targets_sm80_and_caches_the_binary(self):
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / 'llama.cpp'
            (source / 'build' / 'bin').mkdir(parents=True)
            (source / 'build' / 'bin' / 'llama-server').write_bytes(b'server')
            with patch.object(remote, 'WORK', Path(work)), \
                 patch.object(remote, 'source_tree', side_effect=lambda repo, commit, dest: dest.mkdir(parents=True, exist_ok=True)), \
                 patch.object(remote, 'ensure_venv', return_value=Path('/venv/bin/python')), \
                 patch.object(remote, 'run') as run:
                binary = remote.llama_server(self.manifest, self.data, 'sig', '/usr/bin/nvcc')
            flags = [arg for call in run.call_args_list for arg in call.args]
            self.assertIn('-DGGML_CUDA=ON', ' '.join(map(str, flags)))
            self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=80', ' '.join(map(str, flags)))
            # Pinned llama.cpp nests tools/server under COMMON && TOOLS && SERVER.
            self.assertIn('-DLLAMA_BUILD_COMMON=ON', flags)
            self.assertIn('-DLLAMA_BUILD_TOOLS=ON', flags)
            self.assertIn('-DLLAMA_BUILD_SERVER=ON', flags)
            self.assertNotIn('-DLLAMA_BUILD_TOOLS=OFF', flags)
            self.assertIn('--target', flags)
            self.assertEqual(flags[flags.index('--target') + 1], 'llama-server')
            self.assertEqual(binary, source / 'build' / 'bin' / 'llama-server')
            cached = self.data / 'engines' / 'llama-sig' / 'llama-server'
            self.assertEqual(cached.read_bytes(), b'server')

    def test_maya_engine_is_restored_and_saved_around_the_setup(self):
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / 'project-maya'
            (source / 'build').mkdir(parents=True)
            (source / 'build' / 'strata').write_bytes(b'engine')
            (source / 'build' / 'MAYA-BUILD.json').write_text('{"archs": [80]}')
            (source / 'maya-maya-l.json').write_text(json.dumps({'model_name': 'glm-5.3-flash'}))
            cached = self.data / 'engines' / 'maya-sig'
            with patch.object(remote, 'WORK', Path(work)), \
                 patch.object(remote, 'source_tree', side_effect=lambda repo, commit, dest: dest.mkdir(parents=True, exist_ok=True)), \
                 patch.object(remote, 'ensure_venv', return_value=Path('/venv/bin/python')), \
                 patch.object(remote, 'run') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                python, root, config = remote.maya_setup(self.manifest, self.manifest['variants']['flash'],
                                                         self.data, self.data.parent / 'model-data',
                                                         131072, 18081, 'sig')
            self.assertEqual((cached / 'strata').read_bytes(), b'engine')
            command = [str(part) for part in run.call_args_list[-1].args]
            self.assertIn('--yes', command)
            self.assertIn('--no-start', command)
            self.assertIn('--no-vision', command)
            self.assertIn('--model', command)
            self.assertEqual(command[command.index('--model') + 1], 'Maya-L')
            self.assertEqual(command[command.index('--models-dir') + 1],
                             str(self.data.parent / 'model-data' / 'models'))
            self.assertEqual(command[command.index('--context') + 1], '131072')
            self.assertIn('--env', command)
            self.assertEqual(command[command.index('--env') + 1], 'STRATA_GLM_RAM_RESIDENT=1')
            # Maya reports its own MODEL_NAME; the config carries ours so /props matches the catalog.
            self.assertEqual(json.loads(config.read_text())['model_name'], 'glm-5.3-flash-maya-l')

    def test_cached_maya_engine_is_restored_before_the_setup_runs(self):
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / 'project-maya'
            source.mkdir()
            (source / 'maya-maya-l.json').write_text(json.dumps({'model_name': 'glm-5.3-flash'}))
            cached = self.data / 'engines' / 'maya-sig'
            cached.mkdir(parents=True)
            (cached / 'strata').write_bytes(b'engine')
            (cached / 'MAYA-BUILD.json').write_text('{"archs": [80]}')
            with patch.object(remote, 'WORK', Path(work)), \
                 patch.object(remote, 'source_tree', side_effect=lambda repo, commit, dest: dest.mkdir(parents=True, exist_ok=True)), \
                 patch.object(remote, 'ensure_venv', return_value=Path('/venv/bin/python')), \
                 patch.object(remote, 'run') as run, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                remote.maya_setup(self.manifest, self.manifest['variants']['flash'], self.data,
                                   self.data.parent / 'model-data', 65536, 18081, 'sig')
            self.assertIn('Cached A100 Maya engine:', output.getvalue())
            self.assertEqual((source / 'build' / 'strata').read_bytes(), b'engine')
            self.assertIn('Cached A100 Maya engine for the next run:', output.getvalue())

    def run_main(self, variant, work, extra=()):
        calls = []
        chdirs = []

        def execv(path, argv):
            calls.append(argv)
            raise SystemExit(0)

        def serve_private(argv, cwd):
            chdirs.append(cwd)
            execv(argv[0], list(map(str, argv)))

        (Path(work) / 'manifest.json').write_text(json.dumps(self.manifest))
        data = Path(work) / 'drive'
        config = Path(work) / 'maya-maya-l.json'
        pack = data / 'models' / 'Maya-L' / 'pack'
        config.write_text(json.dumps({'args': ['--glm-pack', str(pack)],
                                      'tokenizer': str(pack / 'tokenizer'),
                                      'installer': {'models_dir': str(data / 'models')}}))
        with patch.object(remote, 'WORK', Path(work)), \
             patch.object(remote, 'ensure_drive', return_value=Path(work) / 'drive'), \
             patch.multiple(remote, check_server_port=Mock(), check_drive_space=Mock(), check_privacy=Mock(),
                            check_accelerator=Mock(), check_ram=Mock(), check_runtime_space=Mock(),
                            persist_maya_pack=Mock(), engine_signature=Mock(return_value='sig'),
                            hf_token=Mock(return_value=None)), \
             patch.object(remote.os, 'chdir', side_effect=lambda path: chdirs.append(path)), \
             patch.object(remote.shutil, 'which', return_value='/usr/bin/nvcc'), \
             patch.object(remote.subprocess, 'check_output', return_value='nvcc'), \
             patch.object(remote, 'fetch_verified') as fetch, \
             patch.object(remote, 'fetch_hf_to_ssd') as direct, \
             patch.object(remote, 'llama_server', return_value=Path('/content/glm53-a100/llama.cpp/build/bin/llama-server')), \
             patch.object(remote, 'maya_setup', return_value=(Path('/venv/bin/python'), Path('/maya'), config)), \
             patch.object(remote, 'serve_private', side_effect=serve_private), \
             patch.object(sys, 'argv', ['remote.py', '--variant', variant, *extra]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                remote.main()
        return calls[0], direct, chdirs

    def test_llama_variant_starts_llama_server_with_its_own_shards(self):
        with tempfile.TemporaryDirectory() as work:
            argv, direct, chdirs = self.run_main('full', work)
        self.assertEqual(argv[0], '/content/glm53-a100/llama.cpp/build/bin/llama-server')
        self.assertEqual(argv[argv.index('-m') + 1],
                         str(Path(work) / 'model-data' / 'models/slim-e192/GLM-5.3-Q2-DGX-Spark-00001-of-00004.gguf'))
        self.assertEqual(argv[argv.index('-c') + 1], '65536')
        self.assertEqual(argv[argv.index('--alias') + 1], 'glm-5.3-slim-e192')
        self.assertEqual(argv[argv.index('--port') + 1], '18081')
        self.assertIn('--log-disable', argv)
        self.assertIn('--metrics', argv)
        self.assertEqual(len(direct.call_args_list), 4)

    def test_maya_variant_starts_from_the_generated_config(self):
        with tempfile.TemporaryDirectory() as work:
            argv, direct, chdirs = self.run_main('flash', work)
        self.assertEqual(argv[0], '/venv/bin/python')
        self.assertEqual(argv[1], '/maya/serve/server.py')
        self.assertEqual(argv[argv.index('--engine') + 1], 'strata')
        self.assertTrue(argv[argv.index('--config') + 1].endswith('/maya-maya-l.json'))
        self.assertEqual(argv[argv.index('--port') + 1], '18081')
        # Maya resolves its own relative paths, so the server starts from its own tree.
        self.assertEqual(chdirs, [Path('/maya')])
        self.assertEqual(len(direct.call_args_list), 4)

    def test_llama_variant_reads_the_shards_downloaded_onto_the_ssd(self):
        with tempfile.TemporaryDirectory() as work:
            argv, direct, _ = self.run_main('full', work)
            self.assertEqual(argv[argv.index('-m') + 1],
                             str(Path(work) / 'model-data' / self.manifest['variants']['full']['llama']['first_shard']))
            for call in direct.call_args_list:
                self.assertTrue(call.args[1].is_relative_to(Path(work) / 'model-data'))

    def test_maya_variant_keeps_the_pack_path_maya_chose(self):
        with tempfile.TemporaryDirectory() as work:
            argv, _, _ = self.run_main('flash', work)
            config = json.loads(Path(argv[argv.index('--config') + 1]).read_text())
            self.assertEqual(config['args'][config['args'].index('--glm-pack') + 1],
                             str(Path(work) / 'drive' / 'models' / 'Maya-L' / 'pack'))

    def run_prepare(self, argv, work):
        (Path(work) / 'manifest.json').write_text(json.dumps(self.manifest))
        with patch.object(remote, 'WORK', Path(work)), \
             patch.object(remote, 'ensure_drive', return_value=Path(work) / 'drive'), \
             patch.multiple(remote, check_server_port=Mock(), check_drive_space=Mock(), check_accelerator=Mock(),
                            check_ram=Mock(), check_runtime_space=Mock(), persist_maya_pack=Mock(),
                            release_model_bodies=Mock(), engine_signature=Mock(return_value='sig'),
                            hf_token=Mock(return_value=None)), \
             patch.object(remote.shutil, 'which', return_value='/usr/bin/nvcc'), \
             patch.object(remote.subprocess, 'check_output', return_value='nvcc'), \
             patch.object(remote, 'fetch_verified') as fetch, \
             patch.object(remote, 'fetch_hf_to_ssd') as direct, \
             patch.object(remote, 'llama_server', return_value=Path('/engine/llama-server')) as llama, \
             patch.object(remote, 'maya_setup', return_value=(Path('/venv/bin/python'), Path('/maya'), Path('/maya/c.json'))) as maya, \
             patch.object(remote.os, 'execv') as execv, \
             patch.object(sys, 'argv', ['remote.py', *argv]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            remote.main()
        return direct, llama, maya, execv, output.getvalue()

    def test_serving_reverifies_every_downloaded_body(self):
        with tempfile.TemporaryDirectory() as work:
            _, direct, _ = self.run_main('full', work, extra=('--reverify',))
        self.assertTrue(direct.call_args_list)
        self.assertTrue(all(call.kwargs['reverify'] for call in direct.call_args_list))

    def test_prepare_passes_reverify_to_every_variant(self):
        with tempfile.TemporaryDirectory() as work:
            direct, _, _, _, _ = self.run_prepare(['--prepare', '--reverify'], work)
        self.assertTrue(direct.call_args_list)
        self.assertTrue(all(call.kwargs['reverify'] for call in direct.call_args_list))

    def test_prepare_without_variant_fetches_all_variants_and_starts_nothing(self):
        with tempfile.TemporaryDirectory() as work:
            direct, llama, maya, execv, output = self.run_prepare(['--prepare'], work)
        self.assertEqual(execv.call_count, 0)          # prepare never becomes a server
        self.assertEqual(maya.call_count, 1)           # Maya's setup also writes the pack
        self.assertEqual(llama.call_count, 2)          # the second llama variant reuses the cached binary
        self.assertEqual(len(direct.call_args_list), sum(len(v['files']) for v in self.manifest['variants'].values()))
        self.assertIn('GLM_PREPARE_COMPLETE ["flash", "flash-uncensored", "full"]', output)

    def test_prepare_of_one_variant_leaves_the_others_untouched(self):
        with tempfile.TemporaryDirectory() as work:
            direct, llama, maya, execv, output = self.run_prepare(['--prepare', '--variant', 'full', '--context', '32768'], work)
        self.assertEqual((maya.call_count, llama.call_count, len(direct.call_args_list)), (0, 1, 4))
        self.assertIn('GLM_PREPARE_COMPLETE ["full"]', output)


class MayaPackCacheTests(unittest.TestCase):
    """The generated Maya pack keeps using Drive; the shards around it no longer do."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / 'runtime'
        self.data = self.work / 'drive'
        self.runtime_data = self.work / 'model-data'
        self.data.mkdir(parents=True)
        self.runtime_data.mkdir(parents=True)
        self.profile = {'maya': {'model': 'Maya-L'}}
        self.relative = 'models/Maya-L/pack'
        self.config = self.work / 'maya-maya-l.json'

    def write_pack(self, folder):
        for name in remote.MAYA_PACK_FILES:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())

    def test_built_pack_is_published_to_drive_with_an_index(self):
        pack = self.runtime_data / self.relative
        self.write_pack(pack)
        self.config.write_text(json.dumps({'args': ['--glm-pack', str(pack)]}))
        with patch.object(remote, 'WORK', self.work), patch.object(remote, 'SSD_RESERVE', 1), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            remote.persist_maya_pack(self.data, self.runtime_data, self.config, self.profile, 131072)
        self.write_pack(self.data / self.relative)
        self.assertEqual((self.data / self.relative / 'index.txt').read_bytes(), b'index.txt')
        index = json.loads((self.data / remote.MAYA_PACK_INDEX).read_text())
        self.assertEqual(index['Maya-L|131072'], self.relative)
        self.assertIn('Cached the Maya pack on Drive', output.getvalue())

    def test_cached_pack_is_seeded_back_onto_the_ssd(self):
        self.write_pack(self.data / self.relative)
        (self.data / remote.MAYA_PACK_INDEX).parent.mkdir(parents=True, exist_ok=True)
        (self.data / remote.MAYA_PACK_INDEX).write_text(json.dumps({'Maya-L|131072': self.relative}))
        with patch.object(remote, 'WORK', self.work), patch.object(remote, 'SSD_RESERVE', 1), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            remote.seed_maya_pack(self.data, self.runtime_data, self.profile, 131072)
        self.assertEqual((self.runtime_data / self.relative / 'index.txt').read_bytes(), b'index.txt')
        self.assertIn('Seeding the cached Maya pack', output.getvalue())

    def test_a_different_context_does_not_seed_the_other_pack(self):
        self.write_pack(self.data / self.relative)
        (self.data / remote.MAYA_PACK_INDEX).parent.mkdir(parents=True, exist_ok=True)
        (self.data / remote.MAYA_PACK_INDEX).write_text(json.dumps({'Maya-L|131072': self.relative}))
        with patch.object(remote, 'WORK', self.work), patch.object(remote, 'SSD_RESERVE', 1), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            remote.seed_maya_pack(self.data, self.runtime_data, self.profile, 32768)
        self.assertFalse((self.runtime_data / self.relative).exists())
        self.assertEqual(output.getvalue(), '')

    def test_a_full_disk_still_starts_with_a_rebuilt_pack(self):
        pack = self.runtime_data / self.relative
        self.write_pack(pack)
        self.config.write_text(json.dumps({'args': ['--glm-pack', str(pack)]}))
        with patch.object(remote, 'WORK', self.work), patch.object(remote, 'SSD_RESERVE', 1), \
             patch.object(remote, 'copy_model_file', side_effect=remote.InsufficientRuntimeSpace('full')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            remote.persist_maya_pack(self.data, self.runtime_data, self.config, self.profile, 131072)
        self.assertIn('the next runtime rebuilds it', output.getvalue())
        self.assertFalse((self.data / remote.MAYA_PACK_INDEX).exists())

    def test_an_incomplete_built_pack_stops_the_start(self):
        pack = self.runtime_data / self.relative
        (pack / 'index.txt').parent.mkdir(parents=True)
        (pack / 'index.txt').write_bytes(b'index.txt')
        self.config.write_text(json.dumps({'args': ['--glm-pack', str(pack)]}))
        with patch.object(remote, 'WORK', self.work), patch.object(remote, 'SSD_RESERVE', 1), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Maya pack is missing'):
                remote.persist_maya_pack(self.data, self.runtime_data, self.config, self.profile, 131072)

    def test_prepared_variants_release_their_bodies_for_the_next_one(self):
        profile = {'files': [{'destination': 'models/a.gguf', 'size': 5, 'fetch': 'huggingface'},
                             {'destination': 'mtp/rt/dense.bin', 'size': 5, 'fetch': 'drive'}]}
        for relative in ('models/a.gguf', 'mtp/rt/dense.bin'):
            path = self.runtime_data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'bytes')
            path.with_name(path.name + '.verified.json').write_text('{}')
        with contextlib.redirect_stdout(io.StringIO()):
            remote.release_model_bodies(profile, self.runtime_data)
        self.assertFalse((self.runtime_data / 'models/a.gguf').exists())
        self.assertFalse((self.runtime_data / 'models/a.gguf.verified.json').exists())
        self.assertTrue((self.runtime_data / 'mtp/rt/dense.bin').exists())


class RuntimeModelDownloadTests(unittest.TestCase):
    """Model bodies arrive from HuggingFace on the runtime SSD; Drive stays out of that path."""

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
        self.root = Path(self.tmp.name)
        self.path = self.root / 'model-data' / 'models' / 'model.gguf'
        self.partial = self.path.with_name('model.gguf.part')
        self.marker = self.path.with_name('model.gguf.verified.json')
        Handler.ranges.clear()
        Handler.authorizations.clear()
        Handler.honor_range = True
        Handler.reject_token = False

    def fetch(self, token=None, digest=None, **kwargs):
        with patch.object(remote, 'WORK', self.root), contextlib.redirect_stdout(io.StringIO()):
            remote.fetch_hf_to_ssd(self.url, self.path, len(PAYLOAD),
                                   digest or hashlib.sha256(PAYLOAD).hexdigest(), token, **kwargs)

    def test_publishes_the_ssd_copy_and_skips_the_next_start(self):
        self.fetch()
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertFalse(self.partial.exists())
        self.assertEqual(json.loads(self.marker.read_text())['size'], len(PAYLOAD))
        self.fetch()
        self.assertEqual(len(Handler.ranges), 1)   # a restart on the same runtime transfers nothing

    def test_resume_hashes_the_prefix_it_already_had(self):
        self.path.parent.mkdir(parents=True)
        self.partial.write_bytes(PAYLOAD[:81])
        self.fetch()
        self.assertEqual(Handler.ranges, ['bytes=81-'])
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_checksum_mismatch_discards_the_partial(self):
        with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
            self.fetch(digest='0' * 64)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.partial.exists())    # a corrupt body must not be resumed forever

    def test_refused_token_falls_back_to_anonymous(self):
        Handler.reject_token = True
        self.fetch(token='hf_secretvalue')
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertEqual(Handler.authorizations, ['Bearer hf_secretvalue', None])

    def test_corrupt_runtime_copy_is_rehashed_and_rejected(self):
        self.fetch()
        self.path.write_bytes(PAYLOAD)
        self.marker.unlink()
        with patch.object(remote, 'digest', return_value='0' * 64), \
             self.assertRaisesRegex(RuntimeError, 'fails verification'):
            self.fetch()

    def test_token_value_never_reaches_the_log(self):
        (self.root / '.hf_token').write_text('hf_secretvalue\n')
        with patch.object(remote, 'WORK', self.root), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(remote.hf_token(), 'hf_secretvalue')
        self.assertNotIn('hf_secretvalue', output.getvalue())

    def test_missing_token_downloads_anonymously(self):
        with patch.object(remote, 'WORK', self.root), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(remote.hf_token())
        self.assertIn('anonymously', output.getvalue())

    def test_every_model_body_in_the_shipped_manifest_bypasses_drive(self):
        for profile in session.manifest()['variants'].values():
            self.assertTrue(all(item.get('fetch') == 'huggingface' for item in profile['files']), profile)

    def test_fetch_variant_sends_bodies_to_the_ssd(self):
        profile = {'files': [{'destination': 'models/big.gguf', 'size': 1, 'sha256': 'a',
                              'repo': 'r/m', 'revision': 'v', 'source': 'big.gguf', 'fetch': 'huggingface'},
                             {'destination': 'mtp/rt/dense.bin', 'size': 1, 'sha256': 'b',
                              'repo': 'r/m', 'revision': 'v', 'source': 'dense.bin', 'fetch': 'drive'}]}
        with patch.object(remote, 'fetch_hf_to_ssd') as direct, \
             patch.object(remote, 'fetch_verified') as cached:
            remote.fetch_variant(profile, Path('/drive'), Path('/ssd/model-data'), None, False)
        self.assertEqual(direct.call_args_list[0].args[1].parts[-3:], ('model-data', 'models', 'big.gguf'))
        self.assertEqual(cached.call_args_list[0].args[1].parts[1:4], ('drive', 'mtp', 'rt'))

    def test_drive_space_check_ignores_model_bodies(self):
        profile = {'engine': 'llama', 'files': [
            {'destination': 'models/big.gguf', 'size': 160_000_000_000, 'fetch': 'huggingface'}]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000)), \
             patch.object(remote, 'drive_free_bytes', return_value=1000), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            remote.check_drive_space(Path(directory), profile)
        self.assertIn('missing weights: 0.00 GB', output.getvalue())

    def test_runtime_capacity_is_checked_before_the_download(self):
        profile = {'engine': 'llama', 'files': [{'destination': 'models/big.gguf', 'size': 160_000_000_000}]}
        with patch.object(remote, 'WORK', self.root), \
             patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=150_000_000_000)), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Not enough runtime SSD space'):
                remote.check_runtime_space(profile, self.root / 'model-data')

    def test_full_variant_fits_once_the_drive_copy_is_gone(self):
        profile = session.manifest()['variants']['full']
        with patch.object(remote, 'WORK', self.root), \
             patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=253_000_000_000)), \
             contextlib.redirect_stdout(io.StringIO()):
            remote.check_runtime_space(profile, self.root / 'model-data')


class HealthTests(unittest.TestCase):
    """Maya and llama.cpp answer /health differently; both answer /props."""

    def probe(self, health_body, props_body):
        bodies = [io.BytesIO(json.dumps(health_body).encode()), io.BytesIO(json.dumps(props_body).encode())]
        with patch.object(session.OPENER, 'open', side_effect=bodies):
            return session.health(8086)

    def test_maya_reports_model_and_context_in_health(self):
        info = self.probe({'status': 'ok', 'max_context': 131072, 'model': 'glm-5.3-flash-maya-l', 'images': False},
                          {'default_generation_settings': {'n_ctx': 131072}, 'model_alias': 'glm-5.3-flash-maya-l'})
        self.assertEqual(info['model'], 'glm-5.3-flash-maya-l')
        self.assertEqual(info['max_context'], 131072)

    def test_llama_health_needs_props_for_identity(self):
        info = self.probe({'status': 'ok'},
                          {'default_generation_settings': {'n_ctx': 65536}, 'model_alias': 'glm-5.3-slim-e192'})
        self.assertEqual(info['model'], 'glm-5.3-slim-e192')
        self.assertEqual(info['max_context'], 65536)

    def test_loading_engine_is_not_ready(self):
        self.assertIsNone(self.probe({'status': 'ok', 'max_context': 0, 'model': 'glm-5.3-flash-maya-l'},
                                     {'default_generation_settings': {'n_ctx': 0}}))
        self.assertIsNone(self.probe({'status': 'reloading'}, {}))

    def test_other_model_is_not_ours(self):
        self.assertIsNone(self.probe({'status': 'ok'}, {'model_alias': 'qwen3.8-flash-next',
                                                        'default_generation_settings': {'n_ctx': 131072}}))

    def test_expected_variant_must_match_the_served_one(self):
        bodies = [io.BytesIO(json.dumps({'status': 'ok'}).encode()),
                  io.BytesIO(json.dumps({'model_alias': 'glm-5.3-slim-e192',
                                         'default_generation_settings': {'n_ctx': 65536}}).encode())]
        with patch.object(session.OPENER, 'open', side_effect=bodies):
            self.assertIsNone(session.health(8086, 'glm-5.3-flash-uncensored-iq3_xxs'))
        bodies = [io.BytesIO(json.dumps({'status': 'ok'}).encode()),
                  io.BytesIO(json.dumps({'model_alias': 'glm-5.3-slim-e192',
                                         'default_generation_settings': {'n_ctx': 65536}}).encode())]
        with patch.object(session.OPENER, 'open', side_effect=bodies):
            self.assertIsNotNone(session.health(8086, 'glm-5.3-slim-e192'))

    def test_unreachable_server_is_not_ready(self):
        with patch.object(session.OPENER, 'open', side_effect=OSError('refused')):
            self.assertIsNone(session.health(8086))


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

    def test_normal_prepare_exit_drains_logs_without_a_failure_marker(self):
        (self.work / 'bootstrap.pid').write_text('0')
        self.log.write_text(('build output\n' * 90000) + 'GLM_PREPARE_COMPLETE []\n')
        first = self.instance.bootstrap_output()
        self.assertNotIn('GLM_BOOTSTRAP_FAILED', first)
        second = self.instance.bootstrap_output()
        self.assertIn('GLM_PREPARE_COMPLETE', second)
        self.assertNotIn('GLM_BOOTSTRAP_FAILED', second)
        self.assertEqual((first + second).count('build output'), 90000)
        self.assertEqual(self.instance.bootstrap_output(), '')

    def test_exit_without_completion_reports_failure(self):
        (self.work / 'bootstrap.pid').write_text('0')
        self.log.write_text('engine crashed')
        output = self.instance.bootstrap_output()
        self.assertIn('engine crashed', output)
        self.assertIn('GLM_BOOTSTRAP_FAILED', output)

    def test_explicit_failure_is_not_hidden_by_a_completion_marker(self):
        (self.work / 'failed').touch()
        self.log.write_text('GLM_PREPARE_COMPLETE []\n')
        self.assertIn('GLM_BOOTSTRAP_FAILED', self.instance.bootstrap_output())

    def test_failed_file_detected_even_when_bootstrap_is_alive(self):
        (self.work / 'failed').touch()
        self.assertIn('GLM_BOOTSTRAP_FAILED', self.instance.bootstrap_output())


class LifecycleTests(unittest.TestCase):
    def test_idempotent_setup_reconnects_after_a_client_timeout(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session.uuid, 'uuid4', return_value=SimpleNamespace(hex='fixed')), \
             patch.object(instance, 'colab', side_effect=[session.CommandTimeout('connection wait'),
                                                        'prepared\nGLM_EXEC_OK_fixed\n']) as colab, \
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


    def test_prepare_rejects_explicit_failure_even_with_completion_in_same_poll(self):
        instance = session.Session(SimpleNamespace(startup_minutes=1))
        instance.master = Mock()
        instance.master.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(instance, 'bootstrap_output',
                          return_value='GLM_PREPARE_COMPLETE []\nGLM_BOOTSTRAP_FAILED\n'), \
             patch.object(instance, 'save_bootstrap_error', side_effect=RuntimeError('real failure')), \
             contextlib.redirect_stdout(io.StringIO()):
            instance.socket = str(Path(directory) / 'socket')
            Path(instance.socket).touch()
            with self.assertRaisesRegex(RuntimeError, 'real failure'):
                instance.wait_prepare()

    def test_start_tunnel_and_bootstrap_use_dedicated_port_and_variant(self):
        manifest = session.manifest()
        args = SimpleNamespace(port=8086, context=65536, data_dir='/content/drive/MyDrive/GLM53-data',
                               reverify=False, startup_minutes=1, variant='full', skip_cache_mtime=True)
        instance = session.Session(args)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'id_ed25519').touch()
            child = Mock()
            child.poll.return_value = None
            with patch.object(session, 'STATE', Path(directory)), \
                 patch.object(session, 'catalog'), patch.object(session, 'available_port'), \
                 patch.object(session.shutil, 'which', return_value='/bin/tool'), \
                 patch.object(instance, 'allocate'), patch.object(instance, 'mount_drive'), \
                 patch.object(instance, 'code') as code, \
                 patch.object(instance, 'colab') as colab, \
                 patch.object(session.subprocess, 'Popen', return_value=child) as popen, \
                 patch.object(session, 'health', return_value={'model': 'glm-5.3-slim-e192', 'max_context': 65536}), \
                 contextlib.redirect_stdout(io.StringIO()):
                instance.start()
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index('-L') + 1], f'127.0.0.1:8086:127.0.0.1:{manifest["server_port"]}')
            self.assertNotEqual(manifest['server_port'], 8080)
            launched = [call.args[0] for call in code.call_args_list]
            bootstrap = next(text for text in launched if 'bootstrap.pid' in text)
            self.assertIn("'--variant', 'full'", bootstrap)
            self.assertIn("'--context', '65536'", bootstrap)
            self.assertIn('/content/glm53-a100/remote.py', bootstrap)
            uploaded = [call.args[4] for call in colab.call_args_list]
            self.assertIn('content/glm53-a100/manifest.json', uploaded)
            self.assertIn('content/glm53-a100/remote.py', uploaded)
            self.assertIn('content/glm53-a100/colab_privacy.py', uploaded)

    def prepare_start(self, args):
        """Run the prepare path against a fake runtime and hand back what it asked for."""
        instance = session.Session(args)
        socket_path = Path(instance.socket)
        socket_path.touch()
        self.addCleanup(socket_path.unlink, missing_ok=True)
        child = Mock()
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(session, 'catalog') as catalog, \
             patch.object(session, 'available_port') as available, \
             patch.object(session, 'health') as health, \
             patch.object(session.shutil, 'which', return_value='/bin/tool'), \
             patch.object(instance, 'allocate'), patch.object(instance, 'mount_drive'), \
             patch.object(instance, 'code') as code, \
             patch.object(instance, 'colab') as colab, \
             patch.object(instance, 'ssh', return_value=json.dumps({
                 'offset': 32, 'text': 'GLM_PREPARE_COMPLETE ["full"]\n', 'failed': False})), \
             patch.object(session.subprocess, 'Popen', return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            (Path(directory) / 'id_ed25519').touch()  # the runtime key already exists; no ssh-keygen run
            instance.start(prepare=True)
        preparation = next(call for call in code.call_args_list if 'mkdir(exist_ok=True)' in call.args[0])
        self.assertEqual(preparation.kwargs['timeout'], 120)
        self.assertTrue(preparation.kwargs['retry_safe'])
        launched = [call.args[0] for call in code.call_args_list]
        return {'command': popen.call_args.args[0],
                'bootstrap': next(text for text in launched if 'bootstrap.pid' in text),
                'uploaded': [call.args[4] for call in colab.call_args_list],
                'serving_checks': (catalog.called, health.called, available.called)}

    def test_prepare_skips_the_port_forward_and_the_codex_checks(self):
        args = SimpleNamespace(port=8086, context=65536, data_dir='/content/drive/MyDrive/GLM53-data',
                               reverify=False, startup_minutes=1, variant=None)
        started = self.prepare_start(args)
        self.assertEqual(started['serving_checks'], (False, False, False))
        self.assertNotIn('-L', started['command'])
        self.assertIn("'--prepare'", started['bootstrap'])
        self.assertNotIn("'--variant'", started['bootstrap'])  # every variant, each with its own default context
        self.assertIn('content/glm53-a100/remote.py', started['uploaded'])

    def test_prepare_of_one_variant_passes_variant_and_context(self):
        args = SimpleNamespace(port=8086, context=32768, data_dir='/content/drive/MyDrive/GLM53-data',
                               reverify=True, startup_minutes=1, variant='full', skip_cache_mtime=True)
        started = self.prepare_start(args)
        self.assertIn("'--prepare'", started['bootstrap'])
        self.assertIn("'--variant', 'full'", started['bootstrap'])
        self.assertIn("'--context', '32768'", started['bootstrap'])
        self.assertIn("'--reverify'", started['bootstrap'])
        self.assertIn("'--skip-cache-mtime'", started['bootstrap'])

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
             patch.object(instance, 'code', return_value='GLM_DRIVE_SYNC=flushed') as code, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(instance.flush_drive())
            self.assertEqual(code.call_args.kwargs['timeout'], 86400 + 180)

    def test_flush_success_records_confirmation(self):
        instance = session.Session(SimpleNamespace(flush_minutes=2))
        instance.drive_mounted = True
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'code', return_value='GLM_DRIVE_SYNC=flushed') as code, \
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
            self.assertGreater(sleep.call_count, 40)

    def test_allocation_503_skips_drive_and_cleans_up(self):
        error = subprocess.CalledProcessError(1, ['colab', 'new'],
                                             output='ColabRequestError: POST /assign: Service Unavailable')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(sys, 'argv', ['session.py', 'check-drive', '--drive-auth', 'browser']), \
             patch.object(session.shutil, 'which', return_value='/bin/colab'), \
             patch.object(session.Session, 'colab', side_effect=["Session 'glm53-a100' not found.", error]) as colab, \
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
                             ('new', '-s', 'glm53-a100', '--gpu', 'A100', '--high-mem'))
            record = json.loads((Path(directory) / 'allocation-error.json').read_text())
            self.assertEqual(record['reason'], 'service_unavailable')
            self.assertFalse((Path(directory) / 'session.json').exists())

    def test_kernel_error_with_zero_cli_exit_is_rejected(self):
        instance = session.Session(None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'colab', return_value='Traceback\nValueError: mount failed\n'), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Python'):
                instance.code("raise ValueError('mount failed')")

    def test_mount_failure_stops_before_upload_and_saves_diagnostics(self):
        instance = session.Session(SimpleNamespace(port=8086, context=65536))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'id_ed25519').touch()
            with patch.object(session, 'STATE', Path(directory)), \
                 patch.object(session.shutil, 'which', return_value='/bin/tool'), \
                 patch.object(session, 'available_port'), patch.object(session, 'catalog'), \
                 patch.object(sys.stdin, 'isatty', return_value=True), \
                 patch.object(instance, 'allocate'), \
                 patch.object(instance, 'colab') as colab, \
                 patch.object(instance, 'run', return_value='GLM_DRIVE_RESULT={"mounted": false, "logs": []}'), \
                 patch.object(instance, 'code', return_value='GLM_DRIVE=false') as code, \
                 patch.object(session, 'colab_interpreter', return_value=[sys.executable]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'GDrive'):
                    instance.start()
                colab.assert_not_called()
                code.assert_called_once()  # No exec after the failed mount.
                record = json.loads((Path(directory) / 'drive-mount-error.json').read_text())
                self.assertFalse(record['mounted'])

    def test_process_timeout_terminates_child(self):
        instance = session.Session(None)
        with self.assertRaisesRegex(RuntimeError, '待機時間上限'):
            instance.run([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=0.15)
        self.assertIsNotNone(instance.children[-1].poll())

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

    def test_stop_success_requires_server_confirmation(self):
        status = subprocess.CompletedProcess(['colab', 'status'], 0, stdout='session still alive')
        with patch.object(subprocess, 'run', return_value=status):
            with self.assertRaisesRegex(RuntimeError, 'could not be confirmed'):
                session.stop_runtime()

    def test_codex_settings_are_valid_toml_for_every_variant(self):
        for name, profile in session.manifest()['variants'].items():
            command = session.codex_command(8086, profile['default_context'], ['exec', 'test'], profile)
            values = [command[i + 1] for i, value in enumerate(command) if value == '-c']
            settings = tomllib.loads('\n'.join(values))
            self.assertEqual(settings['model'], profile['model'], name)
            provider = settings['model_providers']['glm53_colab']
            self.assertEqual(provider['wire_api'], 'responses', name)
            self.assertEqual(provider['base_url'], 'http://127.0.0.1:8086/v1', name)
            self.assertEqual(provider['name'], profile['provider_name'], name)
            self.assertEqual(settings['model_auto_compact_token_limit'], profile['default_context'] * 3 // 4, name)
            self.assertEqual(command[-2:], ['exec', 'test'])

    def test_launcher_runs_this_session_module(self):
        launcher = session.HERE.parents[1] / 'llm'
        self.assertTrue(launcher.is_file() and launcher.stat().st_mode & 0o111)
        self.assertIn('codex/colab-glm/session.py', launcher.read_text())


class DriveHelperTests(unittest.TestCase):
    def test_diagnostics_do_not_return_log_credentials(self):
        text = 'UNAUTHENTICATED https://host/?token=secret\nQueryManager timed out email@example.com'
        info = diagnostics.summarize(text)
        self.assertEqual(info, {'root_folder_timeout': 1, 'auth_error': 1})
        self.assertNotIn('secret', json.dumps(info))

    def test_mount_helper_reuses_kernel_and_saves_callback_ids(self):
        current = SimpleNamespace(kernel_id='existing-kernel', session_id='existing-session')
        state = SimpleNamespace(get_session=Mock(return_value=current), store=Mock())
        factory = Mock()
        mount_helper.attached_runtime(factory, state, 'url', 'token', session_name='glm53-a100')
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
        line = next(line for line in output.getvalue().splitlines() if line.startswith('GLM_DRIVE_RESULT='))
        result = json.loads(line.removeprefix('GLM_DRIVE_RESULT='))
        self.assertEqual(result['mount_error_type'], 'ValueError')
        self.assertIn('logs', result)

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

    def test_served_process_group_is_stopped_before_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (Path(directory) / 'bootstrap.pid').write_text('4321')
            with patch.object(shutdown_remote, 'WORK', work), \
                 patch.object(shutdown_remote.os, 'getpgid', return_value=4321), \
                 patch.object(Path, 'read_bytes', return_value=f'{work}/remote.py\0'.encode()), \
                 patch.object(shutdown_remote.os, 'killpg') as kill:
                shutdown_remote.stop_bootstrap()
            self.assertEqual(kill.call_args_list[0].args[1], signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
