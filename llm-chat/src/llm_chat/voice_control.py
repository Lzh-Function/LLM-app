"""Turn an LLM voice-style directive into a separate SSE event."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

MARKER = "[[VOICE_STYLE:"


def prompt(speaker: str, styles: list[dict]) -> str:
    names = " / ".join(style["name"] for style in styles)
    return (
        "音声読み上げ用の制御指示です。"
        f"話者「{speaker}」の声色を回答の雰囲気に合わせて、次の候補から必ず1つ選んでください: {names}。"
        "最終回答の先頭に [[VOICE_STYLE:選んだ声色名]] を1回だけ出力し、"
        "直後から通常の回答本文を書いてください。タグをコードブロックに入れず、説明もしないでください。"
        "ツールを使う場合はツール呼び出しを済ませ、最終回答の先頭でのみタグを出力してください。"
        "回答本文には声色の制御情報を含めないでください。"
    )


class StyleFilter:
    def __init__(self, styles: list[dict]):
        self.allowed = {style["name"]: style for style in styles}
        self.pending = ""
        self.selected = False
        self.strip_leading = False

    def feed(self, fragment: str) -> tuple[str, dict | None]:
        self.pending += fragment
        visible = ""
        chosen = None
        while self.pending:
            start = self.pending.find(MARKER)
            if start >= 0:
                visible += self.pending[:start]
                self.pending = self.pending[start:]
                end = self.pending.find("]]", len(MARKER))
                if end < 0:
                    if len(self.pending) > 96:
                        self.pending = ""
                    break
                name = self.pending[len(MARKER) : end].strip()
                if not self.selected and name in self.allowed:
                    chosen = self.allowed[name]
                    self.selected = True
                    self.strip_leading = True
                self.pending = (
                    self.pending[end + 2 :].removeprefix("\r\n").removeprefix("\n")
                )
                continue

            hold = max(
                (n for n in range(1, len(MARKER)) if self.pending.endswith(MARKER[:n])),
                default=0,
            )
            visible += self.pending[:-hold] if hold else self.pending
            self.pending = self.pending[-hold:] if hold else ""
            break
        if self.strip_leading:
            visible = visible.lstrip()
            if visible:
                self.strip_leading = False
        return visible, chosen

    def finish(self) -> str:
        rest = "" if self.pending.startswith(MARKER) else self.pending
        self.pending = ""
        return rest


def sse(data: dict) -> bytes:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def separate_style(
    source: AsyncIterator[bytes], styles: list[dict]
) -> AsyncIterator[bytes]:
    """Filter complete SSE events, preserving normal chat and tool events."""
    style_filter = StyleFilter(styles)
    buffer = b""
    async for chunk in source:
        buffer += chunk
        while b"\n\n" in buffer:
            block, buffer = buffer.split(b"\n\n", 1)
            lines = [
                line[5:].strip()
                for line in block.splitlines()
                if line.startswith(b"data:")
            ]
            if not lines:
                continue
            data = b"\n".join(lines)
            if data == b"[DONE]":
                rest = style_filter.finish()
                if rest:
                    yield sse({"choices": [{"delta": {"content": rest}}]})
                yield b"data: [DONE]\n\n"
                continue
            try:
                event = json.loads(data)
            except ValueError:
                yield block + b"\n\n"
                continue
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                visible, selected = style_filter.feed(content)
                if selected:
                    yield sse(
                        {
                            "voice_style": {
                                "id": selected["id"],
                                "name": selected["name"],
                            }
                        }
                    )
                if visible:
                    delta["content"] = visible
                    yield sse(event)
                elif any(
                    value for key, value in delta.items() if key != "content"
                ) or event.get("timings"):
                    delta["content"] = ""
                    yield sse(event)
            else:
                yield sse(event)
    if buffer:
        # Upstream ended without a complete final event. Preserve non-directive text.
        rest = style_filter.finish()
        if rest:
            yield sse({"choices": [{"delta": {"content": rest}}]})
