"""Verify metric semantics, content filtering, HTTP configuration, and SSH isolation."""
import copy
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

import collector
import server


class ModelTests(unittest.TestCase):
    def test_strata_metrics_keep_rates_but_exclude_conversation_content(self):
        metrics = {'engine': {'model': 'qwen-test', 'max_context': 131072, 'version': '0.1.43', 'kv': 'int8'},
                   'live': {'state': 'generating', 'tok_s': 123.4, 'tok_s_mean': 120,
                            'prefill_tok_s_mean': 2500, 'prompt_tokens': 1000, 'generated': 100, 'queued': 2,
                            'input': 'SECRET PROMPT', 'output': 'SECRET OUTPUT'},
                   'totals': {'requests': 3, 'prompt_tokens': 2000, 'reused': 1000,
                              'output_tokens': 500, 'decode_ms': 5000, 'prompt_ms': 1000},
                   'requests': [{'decode_tok_s': 111, 'input': 'SECRET PROMPT', 'reasoning': 'SECRET THOUGHT'}],
                   'conversation_cache': {'text': 'SECRET CACHE'}}
        result = server.normalize_model('http://127.0.0.1:8085', {'loaded': True}, metrics, {}, {})
        self.assertEqual(result['stats']['tok_s'], 123.4)
        self.assertEqual(result['stats']['average_tok_s'], 100)
        self.assertEqual(result['stats']['average_prefill_tok_s'], 1000)
        self.assertEqual(result['stats']['cache_hit_percent'], 50)
        self.assertEqual(result['context'], 131072)
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertNotIn('requests', result)

    def test_prometheus_average_is_not_misrepresented_as_live_speed(self):
        metrics = '# HELP model metrics\nllamacpp:predicted_tokens_seconds 135.2\nllamacpp:prompt_tokens_seconds 2050\nllamacpp:requests_processing 1\nllamacpp:requests_deferred 2\nllamacpp:tokens_predicted_total 8000\n'
        result = server.normalize_model('http://127.0.0.1:8082', {'status': 'ok'}, metrics, {},
                                        {'model_path': '/models/model-Q5_K_XL.gguf', 'default_generation_settings': {'n_ctx': 32768}})
        self.assertEqual(result['stats']['speed_kind'], 'average')
        self.assertIsNone(result['stats']['tok_s'])
        self.assertEqual(result['stats']['average_tok_s'], 135.2)
        self.assertEqual(result['context'], 32768)
        self.assertEqual(result['quantization'], 'Q5_K_XL')
        self.assertEqual(result['state'], 'generating')

    def test_absent_and_nonfinite_metrics_remain_unknown(self):
        result = server.normalize_model('http://localhost:5071', {'status': 'ok'}, None, {}, {})
        self.assertEqual(result['stats']['speed_kind'], 'unavailable')
        self.assertIsNone(result['stats']['tok_s'])
        self.assertEqual(server.prometheus('bad NaN\nbad2 +Inf\ngood 12\n'), {'good': 12.0})
        self.assertIsNone(server.num(float('nan')))
        self.assertIsNone(server.num(True))

    def test_idle_and_reading_do_not_reuse_completed_generation_speed(self):
        for state, expected in [('idle', 0), ('reading', None)]:
            with self.subTest(state=state):
                result = server.normalize_model('http://localhost:5071', {},
                    {'engine': {}, 'live': {'state': state}, 'requests': [{'decode_tok_s': 99}], 'totals': {}}, {}, {})
                self.assertEqual(result['stats']['tok_s'], expected)
                self.assertEqual(result['stats']['last_tok_s'], 99)

    def test_loading_api_is_visible_even_without_metrics(self):
        health = {'code': 503, 'data': {'status': 'loading model'}, 'text': None}
        with patch.object(server, 'get_api', side_effect=[health, None, None, None]):
            result = server.model_metrics('http://localhost:5071')
        self.assertEqual(result['state'], 'loading')
        self.assertFalse(result['loaded'])

    def test_disabled_mtp_is_not_displayed_as_enabled(self):
        for spec, expected in [(0, None), (4, '4')]:
            result = server.normalize_model('http://localhost:5071', {},
                                           {'engine': {'spec': spec}, 'live': {}}, {}, {})
            self.assertEqual(result['mtp'], expected)


class CounterTests(unittest.TestCase):
    def test_cpu_io_deltas_and_reboot_reset(self):
        worker = server.Worker.__new__(server.Worker)
        worker.previous = None
        sample = {'time': 10, 'boot_id': 'a', 'cpu': {'counters': {'cpu': [1000, 700], 'cpu0': [500, 350]}},
                  'ram': {'used_bytes': 25, 'total_bytes': 100}, 'gpus': [{'used_bytes': 10, 'total_bytes': 40}],
                  'disks': [], 'io_counters': {'network_rx_bytes': 100}}
        counters = copy.deepcopy(sample['cpu']['counters'])
        worker.rates(sample)
        sample['cpu']['_counters'] = counters
        self.assertIsNone(sample['cpu']['util_percent'])
        next_sample = {'time': 12, 'boot_id': 'a', 'cpu': {'counters': {'cpu': [1100, 740], 'cpu0': [550, 370]}},
                       'ram': {'used_bytes': 25, 'total_bytes': 100}, 'gpus': [], 'disks': [],
                       'io_counters': {'network_rx_bytes': 300}}
        worker.rates(next_sample)
        self.assertAlmostEqual(next_sample['cpu']['util_percent'], 60)
        self.assertEqual(next_sample['io_rates']['network_rx_bps'], 100)
        self.assertEqual(next_sample['ram']['percent'], 25)
        rebooted = copy.deepcopy(next_sample)
        rebooted.update(boot_id='b', time=13)
        rebooted['cpu']['counters'] = {'cpu': [10, 5]}
        worker.rates(rebooted)
        self.assertIsNone(rebooted['cpu']['util_percent'])
        self.assertEqual(rebooted['io_rates'], {})

    def test_failed_sample_preserves_data_without_claiming_live_connection(self):
        worker = server.Worker.__new__(server.Worker)
        worker.lock = threading.Lock()
        worker.stop_event = Mock()
        worker.stop_event.is_set.side_effect = [False, True]
        worker.interval = 2
        worker.value = {'available': True, 'state': 'online', 'sampled_at': 123,
                        'hardware': {'cpu': {'util_percent': 12}}}
        worker.sample = Mock(side_effect=RuntimeError('Disconnected'))
        worker.loop()
        self.assertFalse(worker.value['available'])
        self.assertEqual(worker.value['sampled_at'], 123)
        self.assertEqual(worker.value['hardware']['cpu']['util_percent'], 12)


class ConnectionTests(unittest.TestCase):
    def test_api_urls_are_loopback_only_and_do_not_contain_credentials(self):
        for value in ('http://127.0.0.1:5071', 'http://localhost:8082/v1/', 'http://[::1]:8084'):
            self.assertNotIn('/v1', server.validate_url(value))
        for value in ('http://169.254.169.254:80', 'https://example.com:443', 'file:///etc/passwd',
                      'http://name:password@localhost:5071', 'http://localhost:5071/private',
                      'http://localhost:5071?token=secret', 'http://localhost:999999', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                server.validate_url(value)

    def test_redirects_are_not_followed(self):
        self.assertIsNone(server.NoRedirect().redirect_request(None, None, 302, '', {}, 'http://example.com'))

    def test_custom_connections_persist_locally_and_builtins_cannot_be_removed(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(server, 'Worker', side_effect=lambda target, *_: Mock(target=target)):
            monitor = server.Monitor(state_dir=Path(directory))
            self.addCleanup(monitor.close)
            target = monitor.add('Test API', 'http://127.0.0.1:63999')
            self.assertTrue((Path(directory) / 'targets.json').exists())
            with self.assertRaises(ValueError):
                monitor.delete('colab-qwen')
            restored = server.Monitor(state_dir=Path(directory))
            self.addCleanup(restored.close)
            self.assertEqual(restored.custom[0]['id'], target['id'])
            monitor.delete(target['id'])
            self.assertEqual(json.loads((Path(directory) / 'targets.json').read_text()), [])

    def test_remote_probe_uses_existing_socket_and_never_installs_a_remote_file(self):
        socket = Mock(spec=Path)
        socket.__str__ = Mock(return_value='/tmp/strata-qwen-1000-test.sock')
        socket.stat.return_value = Mock(st_mode=0o140600, st_uid=server.os.getuid(), st_mtime=1)
        result = Mock(returncode=0, stdout='{"cpu": {}}')
        with patch.object(Path, 'glob', return_value=[socket]), patch.object(server.subprocess, 'run', return_value=result) as run:
            self.assertEqual(server.ssh_probe('strata-qwen', '/content/strata-qwen'), {'cpu': {}})
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], 'ssh')
        self.assertIn('ProxyCommand=false', argv)
        self.assertTrue(argv[-1].startswith('python3 -B -c '))
        self.assertNotIn('scp', argv)
        self.assertNotIn('colab', argv)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        cls.httpd.monitor = Mock()
        cls.httpd.monitor.state.return_value = {'targets': []}
        cls.httpd.monitor.add.return_value = {'id': 'test'}
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.httpd.server_port}'
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()

    def test_static_and_snapshot_are_served_without_external_assets(self):
        for path in ('/', '/app.js', '/style.css', '/favicon.svg', '/api/state'):
            with self.opener.open(self.base + path) as response:
                self.assertEqual(response.code, 200)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.opener.open(self.base + '/../../server.py')
        self.assertEqual(error.exception.code, 404)

    def test_cross_site_configuration_and_oversized_body_are_rejected(self):
        for headers, body, code in [({'Content-Type': 'application/json', 'Sec-Fetch-Site': 'cross-site'}, b'{}', 403),
                                    ({'Content-Type': 'text/plain'}, b'{}', 415),
                                    ({'Content-Type': 'application/json'}, b'x' * 8193, 400)]:
            request = urllib.request.Request(self.base + '/api/targets', data=body, headers=headers)
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.opener.open(request)
            self.assertEqual(error.exception.code, code)


if __name__ == '__main__':
    unittest.main()
