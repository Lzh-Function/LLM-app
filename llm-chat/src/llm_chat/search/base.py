from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


class SearchProvider(Protocol):
    """検索エンジンの共通インターフェース。"""

    name: str

    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...
