"""ウェブ検索ツール付きチャットのエージェントループ。

llama-server の OpenAI 互換 API に tools を渡してストリーミングし、
モデルが tool_calls を返したらサーバー側で実行して結果を messages に積み、
最終回答が出るまで繰り返す。

クライアントへは llama-server のチャンク (content / reasoning_content / timings)
をそのまま中継し、加えて独自イベント {"tool_event": {...}} を SSE で送る。
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from .search import SearchResult, fetch_page, get_provider

MAX_ROUNDS = int(os.environ.get("LLM_SEARCH_MAX_ROUNDS", "6"))
MAX_RESULTS = int(os.environ.get("LLM_SEARCH_MAX_RESULTS", "5"))
FETCH_MAX_CHARS = int(os.environ.get("LLM_FETCH_MAX_CHARS", "5000"))
# 検索結果の整合に悩んで思考がループするのを防ぐため、1 ラウンドの思考量を制限する
THINKING_BUDGET = int(os.environ.get("LLM_SEARCH_THINKING_BUDGET", "2048"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "ウェブを検索し、上位の検索結果 (番号・タイトル・URL・概要) を返す。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "検索語"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "URL のウェブページを取得し、本文テキストを返す。"
            "検索結果の概要だけでは情報が足りないときに使う。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "取得する URL"}
                },
                "required": ["url"],
            },
        },
    },
]


def system_prompt() -> str:
    now = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M (JST)")
    return (
        f"あなたはウェブ検索ツールを使えるアシスタントです。現在日時: {now}\n"
        "- 最新情報・時事・事実確認・具体的な数値や仕様が必要な質問では web_search を使う。"
        "一般的な知識や雑談で足りる場合は検索しない。\n"
        "- 概要だけで不十分なら fetch_page で本文を読む。必要なら検索語を変えて再検索する。\n"
        "- 回答では根拠にした情報源を [1] のように検索結果の番号で引用する。\n"
        "- 検索で確認できなかったことは推測で断定しない。"
    )


def sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


class SourceRegistry:
    """1 回の回答内で参照した URL に通し番号を振る。"""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def number(self, url: str, title: str = "") -> int:
        for s in self.items:
            if s["url"] == url:
                if title and not s["title"]:
                    s["title"] = title
                return s["n"]
        n = len(self.items) + 1
        self.items.append({"n": n, "url": url, "title": title})
        return n


async def execute_tool(
    name: str, args: dict, sources: SourceRegistry
) -> tuple[str, list[dict]]:
    """ツールを実行し (モデルに返すテキスト, UI 用の出典リスト) を返す。"""
    if name == "web_search":
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query が空です")
        results: list[SearchResult] = await get_provider().search(query, MAX_RESULTS)
        if not results:
            return "検索結果はありませんでした。", []
        lines, used = [], []
        for r in results:
            n = sources.number(r.url, r.title)
            lines.append(f"[{n}] {r.title}\nURL: {r.url}\n概要: {r.snippet}")
            used.append({"n": n, "url": r.url, "title": r.title})
        return "\n\n".join(lines), used
    if name == "fetch_page":
        url = str(args.get("url", "")).strip()
        text = await fetch_page(url, FETCH_MAX_CHARS)
        n = sources.number(url)
        return f"[{n}] {url} の本文:\n\n{text}", [{"n": n, "url": url, "title": ""}]
    raise ValueError(f"unknown tool: {name}")


async def run(backend_url: str, payload: dict) -> AsyncIterator[bytes]:
    messages = list(payload["messages"])
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            "role": "system",
            "content": system_prompt() + "\n\n" + messages[0]["content"],
        }
    else:
        messages.insert(0, {"role": "system", "content": system_prompt()})
    sources = SourceRegistry()

    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5)) as client:
        for round_no in range(MAX_ROUNDS + 1):
            body = {
                **payload,
                "messages": messages,
                "thinking_budget_tokens": THINKING_BUDGET,
            }
            if round_no < MAX_ROUNDS:  # 最終ラウンドはツールなしで回答させる
                body["tools"] = TOOLS
            content, reasoning = "", ""
            calls: dict[int, dict] = {}
            try:
                async with client.stream(
                    "POST", f"{backend_url}/v1/chat/completions", json=body
                ) as r:
                    if r.status_code != 200:
                        err = (await r.aread()).decode(errors="replace")
                        yield sse({"error": err})
                        return
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        chunk = json.loads(data)
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        content += delta.get("content") or ""
                        reasoning += delta.get("reasoning_content") or ""
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(
                                tc.get("index", 0),
                                {"id": "", "name": "", "arguments": ""},
                            )
                            fn = tc.get("function") or {}
                            slot["id"] += tc.get("id") or ""
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
                        if (
                            delta.get("content")
                            or delta.get("reasoning_content")
                            or chunk.get("timings")
                        ):
                            yield sse(chunk)
            except httpx.HTTPError as e:
                yield sse({"error": str(e)})
                return

            if not calls:
                yield sse({"sources": sources.items})
                yield b"data: [DONE]\n\n"
                return

            tool_calls = [
                {
                    "id": c["id"] or f"call_{round_no}_{i}",
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": c["arguments"] or "{}",
                    },
                }
                for i, c in sorted(calls.items())
            ]
            assistant: dict = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            if reasoning:
                assistant["reasoning_content"] = reasoning
            messages.append(assistant)

            for tc in tool_calls:
                name = tc["function"]["name"]
                event = {"id": tc["id"], "name": name, "args": {}, "status": "running"}
                try:
                    event["args"] = json.loads(tc["function"]["arguments"])
                    yield sse({"tool_event": event})
                    result, used = await execute_tool(name, event["args"], sources)
                    event.update(status="done", sources=used)
                except Exception as e:  # noqa: BLE001 - ツールの失敗はモデルに伝えて続行
                    result = f"エラー: {e}"
                    event.update(status="error", error=str(e))
                yield sse({"tool_event": event})
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result}
                )
