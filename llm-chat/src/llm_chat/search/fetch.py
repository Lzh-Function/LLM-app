from __future__ import annotations

import asyncio

import httpx

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)


async def fetch_page(url: str, max_chars: int) -> str:
    """URL を取得して本文テキストを抽出し、max_chars で切り詰めて返す。"""
    if not url.startswith(("http://", "https://")):
        raise ValueError("http(s) の URL のみ取得できます")
    async with httpx.AsyncClient(
        timeout=15, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        html = r.text

    def extract() -> str:
        import trafilatura

        return trafilatura.extract(html, include_links=False, include_tables=True) or ""

    text = (await asyncio.to_thread(extract)).strip()
    if not text:
        raise ValueError("本文を抽出できませんでした")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(以下省略)"
    return text
