from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from .base import SearchProvider, SearchResult


class DuckDuckGoProvider:
    """DuckDuckGo (ddgs ライブラリ)。API キー不要。"""

    name = "duckduckgo"

    def __init__(self) -> None:
        self.region = os.environ.get("LLM_SEARCH_REGION", "jp-jp")

    def _search_sync(self, query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS

        rows = DDGS().text(query, region=self.region, max_results=max_results)
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            )
            for r in rows
            if r.get("href")
        ]

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return await asyncio.to_thread(self._search_sync, query, max_results)


# API 型プロバイダを追加するときはここに登録する。例:
#   class TavilyProvider:
#       name = "tavily"
#       def __init__(self): self.key = os.environ["TAVILY_API_KEY"]
#       async def search(self, query, max_results): ...  # httpx で API を呼ぶ
#   PROVIDERS["tavily"] = TavilyProvider
PROVIDERS: dict[str, Callable[[], SearchProvider]] = {
    "duckduckgo": DuckDuckGoProvider,
}


def get_provider(name: str | None = None) -> SearchProvider:
    name = name or os.environ.get("LLM_SEARCH_PROVIDER", "duckduckgo")
    try:
        return PROVIDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown search provider: {name} (available: {list(PROVIDERS)})"
        )
