"""Search providers. A provider takes a query and returns results; it knows nothing about turns or policy."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, Protocol
from urllib.parse import parse_qs, urlparse

import httpx

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PC-Companion/1.0"
MAX_SNIPPET_CHARS = 300
RATE_LIMIT_STATUSES = frozenset({202, 429})


class SearchFailed(Exception):
    """The provider could not return results (network error, bad response)."""


class RateLimited(SearchFailed):
    """The provider asked us to slow down. The caller may retry later."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int, timeout_s: float) -> list[SearchResult]:
        """Return up to max_results results. Raise RateLimited or SearchFailed on errors."""
        ...


class _ResultParser(HTMLParser):
    """Collects title, link, and snippet for each result on the DuckDuckGo HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._field: str | None = None
        self._field_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if "result__a" in classes:
            self.results.append({"title": "", "url": _clean_url(values.get("href") or ""), "snippet": ""})
            self._field, self._field_tag = "title", tag
        elif "result__snippet" in classes and self.results:
            self._field, self._field_tag = "snippet", tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._field_tag:
            self._field = self._field_tag = None

    def handle_data(self, data: str) -> None:
        if self._field and self.results:
            self.results[-1][self._field] += data


def _clean_url(href: str) -> str:
    """DuckDuckGo wraps links in a redirect. Return the real destination when it is present."""
    if href.startswith("//"):
        href = "https:" + href
    target = parse_qs(urlparse(href).query).get("uddg")
    return target[0] if target else href


def parse_results(html: str, max_results: int) -> list[SearchResult]:
    parser = _ResultParser()
    parser.feed(html)
    cleaned: list[SearchResult] = []
    for item in parser.results:
        title = " ".join(item["title"].split())
        if not title or not item["url"]:
            continue
        snippet = " ".join(item["snippet"].split())[:MAX_SNIPPET_CHARS]
        cleaned.append(SearchResult(title=title, url=item["url"], snippet=snippet))
        if len(cleaned) >= max_results:
            break
    return cleaned


def _fetch_html(query: str, timeout_s: float) -> str:
    """Download the DuckDuckGo results page for query."""
    response = httpx.get(
        SEARCH_URL,
        params={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout_s,
        follow_redirects=True,
    )
    if response.status_code in RATE_LIMIT_STATUSES:
        raise RateLimited(f"DuckDuckGo is limiting requests (HTTP {response.status_code})")
    response.raise_for_status()
    return response.text


class DuckDuckGoProvider:
    """Scrapes DuckDuckGo's HTML page. No API key needed."""

    name = "duckduckgo"

    def __init__(self, fetch: Callable[[str, float], str] = _fetch_html) -> None:
        self._fetch = fetch

    def search(self, query: str, max_results: int, timeout_s: float) -> list[SearchResult]:
        try:
            html = self._fetch(query, timeout_s)
        except RateLimited:
            raise
        except httpx.HTTPError as exc:
            raise SearchFailed(str(exc) or exc.__class__.__name__) from exc
        return parse_results(html, max_results)
