import asyncio
import json
import unittest

from llm_chat.voice_control import separate_style, sse

STYLES = [{"id": 1, "name": "ノーマル"}, {"id": 2, "name": "あまあま"}]


async def streamed(parts, width=5):
    data = b"".join(parts)
    for offset in range(0, len(data), width):
        yield data[offset : offset + width]


async def collect(parts):
    return [part async for part in separate_style(streamed(parts), STYLES)]


def events(parts):
    return [
        json.loads(part[6:-2]) if part != b"data: [DONE]\n\n" else "[DONE]"
        for part in parts
    ]


class VoiceControlTest(unittest.TestCase):
    def test_style_is_separate_even_when_tag_crosses_chunks(self):
        parts = [
            sse({"choices": [{"delta": {"content": "[[VOICE_"}}]}),
            sse(
                {"choices": [{"delta": {"content": "STYLE:あまあま]]\nこんにちは。"}}]}
            ),
            b"data: [DONE]\n\n",
        ]
        result = events(asyncio.run(collect(parts)))
        self.assertEqual(result[0], {"voice_style": {"id": 2, "name": "あまあま"}})
        self.assertEqual(result[1]["choices"][0]["delta"]["content"], "こんにちは。")
        self.assertEqual(result[2], "[DONE]")

    def test_unknown_style_is_removed_without_event(self):
        parts = [
            sse(
                {"choices": [{"delta": {"content": "[[VOICE_STYLE:知らない]]本文。"}}]}
            ),
            b"data: [DONE]\n\n",
        ]
        result = events(asyncio.run(collect(parts)))
        self.assertEqual(result[0]["choices"][0]["delta"]["content"], "本文。")
        self.assertEqual(result[1], "[DONE]")

    def test_tool_event_and_plain_content_survive(self):
        tool = {"tool_event": {"id": "1", "status": "done"}}
        parts = [
            sse(tool),
            sse({"choices": [{"delta": {"content": "普通の回答。"}}]}),
            b"data: [DONE]\n\n",
        ]
        result = events(asyncio.run(collect(parts)))
        self.assertEqual(result[0], tool)
        self.assertEqual(result[1]["choices"][0]["delta"]["content"], "普通の回答。")


if __name__ == "__main__":
    unittest.main()
