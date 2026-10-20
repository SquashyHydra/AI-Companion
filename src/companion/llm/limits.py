"""Find the loaded context window of the model served by LM Studio."""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

Fetch = Callable[[str, dict[str, str], float], bytes]


def _default_fetch(url: str, headers: dict[str, str], timeout_s: float) -> bytes:
    # A proxy must never be used for localhost, so the handler is empty.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=timeout_s) as response:
        return response.read()


def native_root(base_url: str) -> str:
    """LM Studio's native API lives next to the OpenAI-compatible /v1 path."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def detect_context_length(
    base_url: str,
    model: str,
    api_key: str,
    *,
    timeout_s: float = 2.0,
    fetch: Fetch | None = None,
) -> int | None:
    """Return the loaded context length for model, or None if it cannot be read.

    Uses LM Studio's /api/v0/models endpoint, which reports loaded_context_length. The standard
    /v1/models endpoint does not include it. Any failure returns None, and the caller falls back.
    """
    url = f"{native_root(base_url)}/api/v0/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        body = (fetch or _default_fetch)(url, headers, timeout_s)
        payload: Any = json.loads(body.decode("utf-8"))
    except Exception as exc:
        logger.debug("Could not read the context length from %s: %s", url, exc)
        return None

    entries = payload.get("data", []) if isinstance(payload, dict) else payload
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("id") != model:
            continue
        length = entry.get("loaded_context_length")
        if isinstance(length, int) and not isinstance(length, bool) and length > 0:
            return length
    return None
