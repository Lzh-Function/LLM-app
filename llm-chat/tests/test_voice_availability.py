import asyncio
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from llm_chat import voice


class VoiceAvailabilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_engine_checks_are_parallel(self):
        entered = set()
        both_entered = asyncio.Event()

        async def speakers(engine):
            entered.add(engine)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=0.5)
            return [{'name': engine, 'styles': [{'id': 1, 'name': 'normal'}]}]

        with patch.object(voice, '_speakers', speakers):
            result = await voice.voices()
        self.assertEqual([item['key'] for item in result], ['aivis:aivis', 'voicevox:voicevox'])

    async def test_unavailable_engine_does_not_hide_other_voices(self):
        async def speakers(engine):
            if engine == 'aivis':
                raise HTTPException(503, 'aivis unavailable')
            return [{'name': 'ずんだもん', 'styles': [{'id': 3, 'name': 'ノーマル'}]}]

        with patch.object(voice, '_speakers', speakers):
            result = await voice.voices()
        self.assertEqual(result[0]['key'], 'voicevox:ずんだもん')

    async def test_both_engines_unavailable_return_503(self):
        async def speakers(engine):
            raise HTTPException(503, f'{engine} unavailable')

        with patch.object(voice, '_speakers', speakers), self.assertRaises(HTTPException) as caught:
            await voice.voices()
        self.assertEqual(caught.exception.status_code, 503)
        self.assertIn('aivis', caught.exception.detail)
        self.assertIn('voicevox', caught.exception.detail)

    async def test_stalled_response_has_total_timeout(self):
        real_client = httpx.AsyncClient
        cancelled = asyncio.Event()

        async def stalled(request):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        def client(**kwargs):
            return real_client(transport=httpx.MockTransport(stalled), **kwargs)

        with patch.object(voice, 'SPEAKERS_TIMEOUT_S', 0.02), patch.object(voice.httpx, 'AsyncClient', client):
            with self.assertRaises(HTTPException) as caught:
                await asyncio.wait_for(voice._speakers('aivis'), timeout=0.5)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertTrue(cancelled.is_set())


if __name__ == '__main__':
    unittest.main()
