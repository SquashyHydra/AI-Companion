"""Heuristic check for text that looks like a secret. Used before anything is saved to memory."""

from __future__ import annotations

import re

_KEYWORDS_RE = re.compile(
    r"\b(password|passwd|passcode|pin code|api[ _-]?key|secret|token|private key|credential|"
    r"auth code|one-time code|otp)\b",
    re.IGNORECASE,
)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_LONG_RANDOM_RE = re.compile(r"(?=\S*[A-Za-z])(?=\S*\d)[A-Za-z0-9_\-+/=]{24,}")


def secret_reason(text: str) -> str | None:
    """Return why text looks like a secret, or None if it does not look like one."""
    if _PRIVATE_KEY_RE.search(text):
        return "it looks like a private key"
    if _KEYWORDS_RE.search(text):
        return "it mentions a password, key, or token"
    if _LONG_RANDOM_RE.search(text):
        return "it looks like a long random string"
    return None
