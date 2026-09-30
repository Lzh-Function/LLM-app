import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image
from pydantic import ValidationError

from llm_chat import server
from llm_chat.images import ChatMessage, ImageAttachment


class EventStream(httpx.AsyncByteStream):
    def __init__(self, text):
        self.data = text.encode()

    async def __aiter__(self):
        yield self.data


def attachment(fmt='PNG'):
    out = io.BytesIO()
    Image.new('RGB', (16, 16), 'red').save(out, format=fmt)
    mime = {'PNG': 'png', 'JPEG': 'jpeg', 'WEBP': 'webp'}[fmt]
    return {'name': 'test.png', 'data_url': f'data:image/{mime};base64,' + base64.b64encode(out.getvalue()).decode()}


class ImageValidationTest(unittest.TestCase):
    def test_supported_formats(self):
        for fmt in ('PNG', 'JPEG', 'WEBP'):
            ImageAttachment(**attachment(fmt))

    def test_invalid_data_and_remote_urls_rejected(self):
        for url in ('https://example.com/a.png', 'data:image/svg+xml;base64,PHN2Zz4=',
                    'data:image/png;base64,bm90LWltYWdl', 'data:image/png;base64,==='):
            with self.subTest(url=url), self.assertRaises(ValidationError):
                ImageAttachment(data_url=url)
        wrong = attachment(); wrong['data_url'] = wrong['data_url'].replace('image/png', 'image/jpeg')
        with self.assertRaises(ValidationError):
            ImageAttachment(**wrong)

    def test_image_limits_and_roles(self):
        with self.assertRaises(ValidationError):
            ChatMessage(role='user', content='', images=[attachment()] * 5)
        with self.assertRaises(ValidationError):
            ChatMessage(role='assistant', content='', images=[attachment()])
        with patch('llm_chat.images.MAX_IMAGE_BYTES', 1), self.assertRaises(ValidationError):
            ImageAttachment(**attachment())
        with patch('llm_chat.images.MAX_IMAGE_PIXELS', 1), self.assertRaises(ValidationError):
            ImageAttachment(**attachment())


class ImageAPITest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / 'projector.gguf').touch()
        self.manager = server.ModelManager()
        self.manager.models = {
            'vision': server.ModelInfo('vision', 'Vision', '', 1, root, mmproj='projector.gguf'),
            'text': server.ModelInfo('text', 'Text', '', 2, root),
        }
        self.manager.active = 'vision'; self.manager.status = 'ready'
        self.patches = [patch.object(server, 'manager', self.manager), patch.object(server, 'HISTORY_DIR', root / 'history')]
        for p in self.patches: p.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    async def test_history_round_trip_images_and_old_text(self):
        conv = {'id': 'image-test', 'created': '2026-09-29', 'updated': '2026-09-29',
                'messages': [{'role': 'user', 'content': '', 'images': [attachment()]},
                             {'role': 'assistant', 'content': 'red'}]}
        saved = await self.client.put('/api/history/image-test', json=conv)
        self.assertEqual(saved.status_code, 200)
        loaded = (await self.client.get('/api/history/image-test')).json()
        self.assertEqual(loaded['messages'][0]['images'], conv['messages'][0]['images'])
        self.assertIn(attachment()['data_url'], (server.HISTORY_DIR / 'image-test.md').read_text())
        # A saved legacy conversation without images still loads.
        conv['messages'] = [{'role': 'user', 'content': 'hello'}]
        (server.HISTORY_DIR / 'image-test.json').write_text(json.dumps(conv))
        self.assertEqual((await self.client.get('/api/history/image-test')).status_code, 200)

    async def test_images_relay_search_and_voice_preserve_payload(self):
        real_client = httpx.AsyncClient
        for search in (False, True):
            sent = []
            def backend(request):
                sent.append(json.loads(request.content))
                data = {'choices': [{'delta': {'content': '[[VOICE_STYLE:normal]]red'}}]}
                return httpx.Response(200, stream=EventStream('data: ' + json.dumps(data) + '\n\ndata: [DONE]\n\n'))
            def backend_client(**kwargs):
                return real_client(transport=httpx.MockTransport(backend), **kwargs)
            async def styles(_): return [{'id': 1, 'name': 'normal'}]
            body = {'model': 'vision', 'web_search': search, 'voice_speaker': 'test:speaker',
                    'messages': [{'role': 'system', 'content': 'hello'},
                                 {'role': 'user', 'content': '', 'images': [attachment(), attachment('JPEG')]}]}
            with patch.object(server.httpx, 'AsyncClient', backend_client), patch.object(server.voice, 'styles_for_speaker', styles):
                response = await self.client.post('/api/chat', json=body)
            self.assertEqual(response.status_code, 200)
            self.assertIn('red', response.text)
            self.assertIn('voice_style', response.text)
            self.assertEqual(sent[0]['messages'][-1]['content'][0]['image_url']['url'], attachment()['data_url'])
            self.assertEqual(len(sent[0]['messages'][-1]['content']), 3)
            self.assertEqual(sent[0]['messages'][-1]['content'][-1]['type'], 'text')
            self.assertIsInstance(sent[0]['messages'][0]['content'], str)

    async def test_dolphin_and_missing_projector_rejected(self):
        self.manager.active = 'text'
        body = {'model': 'text', 'messages': [{'role': 'user', 'content': 'look', 'images': [attachment()]}]}
        self.assertEqual((await self.client.post('/api/chat', json=body)).status_code, 400)
        self.manager.active = 'vision'
        self.manager.models['vision'].mmproj = 'missing.gguf'
        body['model'] = 'vision'
        self.assertEqual((await self.client.post('/api/chat', json=body)).status_code, 409)

    async def test_text_relay_unchanged(self):
        real_client = httpx.AsyncClient
        sent = []
        def backend(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, stream=EventStream('data: [DONE]\n\n'))
        with patch.object(server.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(backend), **kw)):
            response = await self.client.post('/api/chat', json={'model': 'vision', 'messages': [{'role': 'user', 'content': 'hello'}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(sent[0]['messages'], [{'role': 'user', 'content': 'hello'}])

    async def test_validation_returns_422(self):
        body = {'model': 'vision', 'messages': [{'role': 'user', 'content': '', 'images': [{'data_url': 'bad'}]}]}
        self.assertEqual((await self.client.post('/api/chat', json=body)).status_code, 422)


if __name__ == '__main__':
    unittest.main()
