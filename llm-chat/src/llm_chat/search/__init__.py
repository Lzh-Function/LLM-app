"""ウェブ検索パッケージ。

検索エンジンは SearchProvider プロトコルで抽象化しており、環境変数
LLM_SEARCH_PROVIDER で切り替える (既定: duckduckgo)。
API 型の検索 (Tavily / Brave 等) を追加する場合は providers.py に
SearchProvider を実装したクラスを書き、PROVIDERS に登録するだけでよい。
"""

from .base import SearchProvider, SearchResult
from .fetch import fetch_page
from .providers import get_provider

__all__ = ["SearchProvider", "SearchResult", "fetch_page", "get_provider"]
