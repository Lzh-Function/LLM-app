"""Exercise storage integrity, engine caching and billable-runtime cleanup without allocating a VM."""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
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
            self.fetch()

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
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, 'less free space'):
                remote.check_drive_space(Path(directory), profile)
            self.assertIn('0.00/10.00 GB', output.getvalue())

    def test_drive_space_counts_existing_partial(self):
        profile = {'files': [{'destination': 'model.gguf', 'size': 100}]}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'model.gguf.part').write_bytes(b'x' * 90)
            with patch.object(remote.shutil, 'disk_usage', return_value=SimpleNamespace(free=5_000_000_010)), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                remote.check_drive_space(Path(directory), profile)
            self.assertIn('missing weights: 0.00 GB', output.getvalue())

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
                                                         self.data, 131072, 18081, 'sig')
            self.assertEqual((cached / 'strata').read_bytes(), b'engine')
            command = [str(part) for part in run.call_args_list[-1].args]
            self.assertIn('--yes', command)
            self.assertIn('--no-start', command)
            self.assertIn('--no-vision', command)
            self.assertIn('--model', command)
            self.assertEqual(command[command.index('--model') + 1], 'Maya-L')
            self.assertEqual(command[command.index('--models-dir') + 1], str(self.data / 'models'))
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
                remote.maya_setup(self.manifest, self.manifest['variants']['flash'], self.data, 65536, 18081, 'sig')
            self.assertIn('Cached A100 Maya engine:', output.getvalue())
            self.assertEqual((source / 'build' / 'strata').read_bytes(), b'engine')
            self.assertIn('Cached A100 Maya engine for the next run:', output.getvalue())

    def run_main(self, variant, work):
        calls = []
        chdirs = []

        def execv(path, argv):
            calls.append(argv)
            raise SystemExit(0)

        (Path(work) / 'manifest.json').write_text(json.dumps(self.manifest))
        with patch.object(remote, 'WORK', Path(work)), \
             patch.object(remote, 'ensure_drive', return_value=Path(work) / 'drive'), \
             patch.object(remote, 'check_server_port'), patch.object(remote, 'check_drive_space'), \
             patch.object(remote, 'check_accelerator'), patch.object(remote, 'check_ram'), \
             patch.object(remote.os, 'chdir', side_effect=lambda path: chdirs.append(path)), \
             patch.object(remote.shutil, 'which', return_value='/usr/bin/nvcc'), \
             patch.object(remote.subprocess, 'check_output', return_value='nvcc'), \
             patch.object(remote, 'engine_signature', return_value='sig'), \
             patch.object(remote, 'fetch_verified') as fetch, \
             patch.object(remote, 'llama_server', return_value=Path('/content/glm53-a100/llama.cpp/build/bin/llama-server')), \
             patch.object(remote, 'maya_setup', return_value=(Path('/venv/bin/python'), Path('/maya'), Path('/maya/maya-maya-l.json'))), \
             patch.object(remote.os, 'execv', side_effect=execv), \
             patch.object(sys, 'argv', ['remote.py', '--variant', variant]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                remote.main()
        return calls[0], fetch, chdirs

    def test_llama_variant_starts_llama_server_with_its_own_shards(self):
        with tempfile.TemporaryDirectory() as work:
            argv, fetch, chdirs = self.run_main('full', work)
        self.assertEqual(argv[0], '/content/glm53-a100/llama.cpp/build/bin/llama-server')
        self.assertEqual(argv[argv.index('-m') + 1],
                         str(Path(work) / 'drive' / 'models/slim-e192/GLM-5.3-Q2-DGX-Spark-00001-of-00004.gguf'))
        self.assertEqual(argv[argv.index('-c') + 1], 65536)
        self.assertEqual(argv[argv.index('--alias') + 1], 'glm-5.3-slim-e192')
        self.assertEqual(argv[argv.index('--port') + 1], 18081)
        self.assertEqual(len(fetch.call_args_list), 4)

    def test_maya_variant_starts_from_the_generated_config(self):
        with tempfile.TemporaryDirectory() as work:
            argv, fetch, chdirs = self.run_main('flash', work)
        self.assertEqual(argv[0], '/venv/bin/python')
        self.assertEqual(argv[1], '/maya/maya.py')
        self.assertEqual(argv[argv.index('--config') + 1], '/maya/maya-maya-l.json')
        self.assertEqual(argv[argv.index('--port') + 1], '18081')
        # Maya resolves its own relative paths, so the server starts from its own tree.
        self.assertEqual(chdirs, [Path('/maya')])
        self.assertEqual(len(fetch.call_args_list), 4)

    def run_prepare(self, argv, work):
        (Path(work) / 'manifest.json').write_text(json.dumps(self.manifest))
        with patch.object(remote, 'WORK', Path(work)), \
             patch.object(remote, 'ensure_drive', return_value=Path(work) / 'drive'), \
             patch.object(remote, 'check_server_port'), patch.object(remote, 'check_drive_space'), \
             patch.object(remote, 'check_accelerator'), patch.object(remote, 'check_ram'), \
             patch.object(remote.shutil, 'which', return_value='/usr/bin/nvcc'), \
             patch.object(remote.subprocess, 'check_output', return_value='nvcc'), \
             patch.object(remote, 'engine_signature', return_value='sig'), \
             patch.object(remote, 'fetch_verified') as fetch, \
             patch.object(remote, 'llama_server', return_value=Path('/engine/llama-server')) as llama, \
             patch.object(remote, 'maya_setup', return_value=(Path('/venv/bin/python'), Path('/maya'), Path('/maya/c.json'))) as maya, \
             patch.object(remote.os, 'execv') as execv, \
             patch.object(sys, 'argv', ['remote.py', *argv]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            remote.main()
        return fetch, llama, maya, execv, output.getvalue()

    def test_prepare_without_variant_fetches_all_variants_and_starts_nothing(self):
        with tempfile.TemporaryDirectory() as work:
            fetch, llama, maya, execv, output = self.run_prepare(['--prepare'], work)
        self.assertEqual(execv.call_count, 0)          # prepare never becomes a server
        self.assertEqual(maya.call_count, 1)           # Maya's setup also writes the pack
        self.assertEqual(llama.call_count, 2)          # the second llama variant reuses the cached binary
        self.assertEqual(len(fetch.call_args_list), sum(len(v['files']) for v in self.manifest['variants'].values()))
        self.assertIn('GLM_PREPARE_COMPLETE ["flash", "flash-uncensored", "full"]', output)

    def test_prepare_of_one_variant_leaves_the_others_untouched(self):
        with tempfile.TemporaryDirectory() as work:
            fetch, llama, maya, execv, output = self.run_prepare(['--prepare', '--variant', 'full', '--context', '32768'], work)
        self.assertEqual((maya.call_count, llama.call_count, len(fetch.call_args_list)), (0, 1, 4))
        self.assertIn('GLM_PREPARE_COMPLETE ["full"]', output)


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


class LifecycleTests(unittest.TestCase):
    def test_start_tunnel_and_bootstrap_use_dedicated_port_and_variant(self):
        manifest = session.manifest()
        args = SimpleNamespace(port=8086, context=65536, data_dir='/content/drive/MyDrive/GLM53-data',
                               reverify=False, startup_minutes=1, variant='full')
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
             patch.object(instance, 'ssh', return_value='GLM_PREPARE_COMPLETE ["full"]'), \
             patch.object(session.subprocess, 'Popen', return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            (Path(directory) / 'id_ed25519').touch()  # the runtime key already exists; no ssh-keygen run
            instance.start(prepare=True)
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
                               reverify=True, startup_minutes=1, variant='full')
        started = self.prepare_start(args)
        self.assertIn("'--prepare'", started['bootstrap'])
        self.assertIn("'--variant', 'full'", started['bootstrap'])
        self.assertIn("'--context', '32768'", started['bootstrap'])
        self.assertIn("'--reverify'", started['bootstrap'])

    def test_close_flushes_before_stopping_owned_runtime(self):
        instance = session.Session(None)
        instance.owned = instance.drive_mounted = True
        actions = []
        with patch.object(instance, 'flush_drive', side_effect=lambda: actions.append('flush')), \
             patch.object(session, 'stop_runtime', side_effect=lambda: actions.append('stop')), \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            instance.close()
        self.assertEqual(actions, ['flush', 'stop'])

    def test_close_stops_runtime_even_if_flush_crashes(self):
        instance = session.Session(None)
        instance.owned = instance.drive_mounted = True
        with patch.object(instance, 'flush_drive', side_effect=OSError('disk full')), \
             patch.object(session, 'stop_runtime') as stop, \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                instance.close()
            stop.assert_called_once()

    def test_flush_success_records_confirmation(self):
        instance = session.Session(SimpleNamespace(flush_minutes=2))
        instance.drive_mounted = True
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'code', return_value='GLM_DRIVE_SYNC=flushed') as code, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(instance.flush_drive())
            self.assertEqual(code.call_args.kwargs['timeout'], 150)
            self.assertTrue(json.loads((Path(directory) / 'drive-sync-status.json').read_text())['confirmed'])
            self.assertFalse(instance.drive_mounted)

    def test_flush_timeout_records_failure_and_still_stops(self):
        instance = session.Session(SimpleNamespace(flush_minutes=2))
        instance.owned = instance.drive_mounted = True
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(session, 'STATE', Path(directory)), \
             patch.object(instance, 'code', side_effect=RuntimeError('timeout')), \
             patch.object(session, 'stop_runtime') as stop, \
             patch.object(signal, 'signal'), patch.object(signal, 'alarm'), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(instance.close())
            stop.assert_called_once()
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
        launcher = session.HERE.parents[1] / 'colab-glm'
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
        with patch.dict(sys.modules, {'google': google, 'google.colab': colab}), \
             patch.object(shutdown_remote, 'stop_bootstrap', side_effect=lambda: actions.append(('stop', {}))), \
             patch.object(shutdown_remote.os.path, 'ismount', return_value=True):
            self.assertEqual(shutdown_remote.shutdown(45), 'flushed')
        self.assertEqual(actions, [('stop', {}), ('flush', {'timeout_ms': 45000})])

    def test_remote_shutdown_does_not_report_unmounted_as_synced(self):
        import types
        colab = types.ModuleType('google.colab')
        colab.drive = Mock()
        with patch.dict(sys.modules, {'google.colab': colab}), \
             patch.object(shutdown_remote, 'stop_bootstrap'), \
             patch.object(shutdown_remote.os.path, 'ismount', return_value=False):
            self.assertEqual(shutdown_remote.shutdown(45), 'not_mounted')
        colab.drive.flush_and_unmount.assert_not_called()

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
