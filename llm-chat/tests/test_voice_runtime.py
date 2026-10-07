import asyncio
import shlex
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import Request, Response

from llm_chat import voice, voice_runtime


FAKE_ENGINE = '''
import argparse, json, signal
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser()
p.add_argument('--host'); p.add_argument('--port', type=int)
a = p.parse_args()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps([{'name': 'ずんだもん', 'styles': [{'id': 3, 'name': 'ノーマル'}]}]).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(data)
    def log_message(self, *args): pass
HTTPServer((a.host, a.port), Handler).serve_forever()
'''


def unused_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class VoiceRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='speech runtime ')
        self.root = Path(self.temp.name)
        self.port = unused_port()
        self.urls = {'voicevox': f'http://127.0.0.1:{self.port}'}
        self.runtime = voice_runtime.VoiceRuntime(self.root, self.urls)

    async def asyncTearDown(self):
        await self.runtime.stop()
        self.temp.cleanup()

    def install_fake(self, body=FAKE_ENGINE):
        folder = self.root / 'voicevox'
        (folder / 'linux-cpu-x64').mkdir(parents=True)
        (folder / 'linux-cpu-x64/run').touch()
        script = folder / 'engine.py'
        script.write_text(body)
        (folder / 'serve.sh').write_text(
            'exec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(script)) + ' "$@"\n')
        return script

    async def wait_listening(self, expected=True):
        async with asyncio.timeout(3):
            while await voice_runtime.port_in_use('127.0.0.1', self.port) != expected:
                await asyncio.sleep(0.02)

    async def test_installed_engine_starts_on_configured_port_and_stops(self):
        self.install_fake()
        self.runtime.start()
        self.assertTrue(self.runtime.starting)
        await asyncio.wait_for(asyncio.gather(*self.runtime.tasks), timeout=3)
        self.assertFalse(self.runtime.starting)
        with patch.dict(voice.ENGINE_URLS, self.urls, clear=True):
            result = await voice.voices()
        self.assertEqual(result[0]['key'], 'voicevox:ずんだもん')
        proc = self.runtime.processes['voicevox']
        await self.runtime.stop()
        self.assertIsNotNone(proc.returncode)
        await self.wait_listening(False)

    async def test_running_engine_is_reused_and_survives_chat_shutdown(self):
        script = self.install_fake()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script), '--host', '127.0.0.1', '--port', str(self.port))
        try:
            await self.wait_listening()
            self.runtime.start()
            await asyncio.gather(*self.runtime.tasks)
            self.assertEqual(self.runtime.processes, {})
            await self.runtime.stop()
            self.assertIsNone(proc.returncode)
            self.assertTrue(await voice_runtime.port_in_use('127.0.0.1', self.port))
        finally:
            proc.terminate()
            await proc.wait()

    async def test_missing_engine_and_remote_endpoint_do_not_spawn(self):
        self.runtime.urls['aivis'] = 'http://speech.example:10101'
        self.runtime.start()
        await asyncio.gather(*self.runtime.tasks)
        self.assertEqual(self.runtime.processes, {})

    async def test_default_off_and_repeated_on_off_on(self):
        self.install_fake()
        self.assertFalse(self.runtime.enabled)
        self.assertEqual(self.runtime.tasks, [])
        self.assertFalse(await voice_runtime.port_in_use('127.0.0.1', self.port))
        await asyncio.gather(*(self.runtime.set_enabled(True) for _ in range(3)))
        await asyncio.gather(*self.runtime.tasks)
        first = self.runtime.processes['voicevox']
        self.assertEqual(len(self.runtime.tasks), 1)
        await self.runtime.set_enabled(False)
        await self.wait_listening(False)
        await self.runtime.set_enabled(True)
        await asyncio.gather(*self.runtime.tasks)
        self.assertNotEqual(first.pid, self.runtime.processes['voicevox'].pid)

    async def test_api_default_off_and_toggles_leave_loaded_llm_alone(self):
        from llm_chat import server
        self.install_fake()
        with (patch.object(server, 'LLM_ROOT', self.root),
              patch.dict(voice.ENGINE_URLS, self.urls, clear=True),
              patch.object(server.manager, 'unload', new_callable=AsyncMock) as unload,
              patch.object(server.manager, 'snapshot', return_value={'active': 'test-model', 'status': 'ready'})):
            async with server.lifespan(server.app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                    initial = (await client.get('/api/status')).json()
                    self.assertFalse(initial['voice']['enabled'])
                    with patch.object(voice, '_speakers', new_callable=AsyncMock) as speakers:
                        result = await client.get('/api/voice/voices')
                        self.assertEqual(result.status_code, 200)
                        self.assertEqual(result.json(), [])
                        audio = await client.post('/api/voice/synthesize', json={'text': 'テスト', 'style_id': 3})
                        self.assertEqual(audio.status_code, 409)
                        speakers.assert_not_called()
                    for enabled in (True, False, True, False):
                        result = await client.put('/api/voice/runtime', json={'enabled': enabled})
                        self.assertEqual(result.status_code, 200)
                        self.assertEqual(result.json()['enabled'], enabled)
                        if enabled:
                            await asyncio.gather(*server.app.state.speech_runtime.tasks)
                            self.assertEqual(len((await client.get('/api/voice/voices')).json()), 1)
                        await self.wait_listening(enabled)
                        self.assertEqual((await client.get('/api/status')).json()['active'], 'test-model')
                    unload.assert_not_awaited()
            unload.assert_awaited_once()

    async def test_concurrent_on_off_finishes_off(self):
        self.install_fake('import time\ntime.sleep(60)\n')
        await asyncio.gather(self.runtime.set_enabled(True), self.runtime.set_enabled(False))
        self.assertFalse(self.runtime.enabled)
        self.assertEqual(self.runtime.processes, {})
        self.assertEqual(self.runtime.tasks, [])
        await self.wait_listening(False)

    async def test_shutdown_during_startup_stops_child(self):
        self.install_fake('import time\ntime.sleep(60)\n')
        self.runtime.start()
        async with asyncio.timeout(3):
            while not self.runtime.processes:
                await asyncio.sleep(0.02)
        proc = self.runtime.processes['voicevox']
        await self.runtime.stop()
        self.assertIsNotNone(proc.returncode)
        self.assertEqual(self.runtime.tasks, [])

    async def test_exited_launcher_does_not_leave_its_worker(self):
        worker = self.install_fake()
        worker.write_text(FAKE_ENGINE.replace(
            'HTTPServer((a.host, a.port)', 'signal.signal(signal.SIGTERM, lambda *_: None)\nHTTPServer((a.host, a.port)'))
        launcher = worker.with_name('launcher.py')
        launcher.write_text('import subprocess,sys\nsubprocess.Popen([sys.executable, ' +
                            repr(str(worker)) + ', *sys.argv[1:]])\n')
        (worker.parent / 'serve.sh').write_text(
            'exec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(launcher)) + ' "$@"\n')
        self.runtime.start()
        await asyncio.gather(*self.runtime.tasks)
        await self.wait_listening()
        await self.runtime.stop()
        await self.wait_listening(False)

    async def test_partial_voice_list_marks_remaining_engine_as_starting(self):
        from fastapi import FastAPI
        app = FastAPI()
        app.state.speech_runtime = self.runtime
        self.runtime.enabled = True
        task = asyncio.create_task(asyncio.sleep(60))
        self.runtime.tasks = [task]

        async def speakers(engine):
            return [{'name': 'ずんだもん', 'styles': [{'id': 3, 'name': 'ノーマル'}]}]

        request = Request({'type': 'http', 'app': app})
        response = Response()
        with patch.dict(voice.ENGINE_URLS, self.urls, clear=True), patch.object(voice, '_speakers', speakers):
            result = await voice.voices(response, request)
        self.assertEqual(result[0]['engine'], 'voicevox')
        self.assertEqual(response.headers['X-Voice-Starting'], '1')


class LocalEndpointTest(unittest.TestCase):
    def test_external_or_proxy_endpoints_are_not_started_locally(self):
        for url in ('https://localhost:10101', 'http://speech.example:10101',
                    'http://localhost:10101/proxy', 'http://user:secret@localhost:10101',
                    'http://localhost:invalid', 'http://localhost:0'):
            self.assertIsNone(voice_runtime.local_address(url))
        self.assertEqual(voice_runtime.local_address('http://localhost:10102'), ('localhost', 10102))


if __name__ == '__main__':
    unittest.main()
