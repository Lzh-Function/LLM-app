"""Check the MCP protocol, input validation and search-provider error handling."""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.exceptions import ToolError

import server
from llm_chat.search import SearchResult


class SearchToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_initialization_tools_and_validation(self):
        params = StdioServerParameters(
            command=sys.executable, args=[str(Path(__file__).with_name("server.py"))]
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                info = await session.initialize()
                self.assertIn("fetch_page", info.instructions)
                tools = (await session.list_tools()).tools
                self.assertEqual({t.name for t in tools}, {"web_search", "fetch_page"})
                self.assertTrue(all(t.annotations.readOnlyHint for t in tools))
                self.assertTrue(all(t.annotations.openWorldHint for t in tools))
                for name, arguments in (
                    ("web_search", {"query": "x", "max_results": 100}),
                    ("web_search", {"query": "  "}),
                    ("fetch_page", {"url": "file:///etc/passwd"}),
                    ("fetch_page", {"url": "https://example.com", "max_chars": 999999}),
                ):
                    result = await session.call_tool(name, arguments)
                    self.assertTrue(result.isError, (name, result))

    async def test_search_returns_source_urls(self):
        provider = AsyncMock()
        provider.search.return_value = [SearchResult("Python", "https://python.org/", "Docs")]
        with patch.object(server, "get_provider", return_value=provider):
            result = await server.web_search("  Python docs  ", 3)
        provider.search.assert_awaited_once_with("Python docs", 3)
        self.assertEqual(result["results"][0]["url"], "https://python.org/")

    async def test_provider_failure_is_a_tool_error(self):
        provider = AsyncMock()
        provider.search.side_effect = RuntimeError("search backend unavailable")
        with patch.object(server, "get_provider", return_value=provider):
            with self.assertRaisesRegex(ToolError, "search backend unavailable"):
                await server.mcp.call_tool("web_search", {"query": "Python"})

    async def test_timeout_is_not_reported_as_success(self):
        provider = AsyncMock()
        provider.search.side_effect = asyncio.TimeoutError
        with patch.object(server, "get_provider", return_value=provider):
            with self.assertRaisesRegex(ToolError, "timed out"):
                await server.mcp.call_tool("web_search", {"query": "Python"})


if __name__ == "__main__":
    unittest.main()
