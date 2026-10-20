"""Checks applied before the companion saves a fact on its own.

Automatic saves are stricter than manual ones: the user asked for a manual save, but nobody
asked for an automatic one, so anything sensitive is skipped.
"""

from __future__ import annotations

import re

from companion.safety.secrets import secret_reason

_SENSITIVE = re.compile(
    r"\b(password|passcode|passphrase|pin code|api key|token|secret|social security|ssn|"
    r"credit card|debit card|card number|bank account|routing number|iban|sort code|passport|"
    r"driver'?s licen[cs]e|diagnos\w*|diabetes|cancer|hiv|depress\w*|anxiety|therapy|"
    r"medication|prescription|pregnan\w*|sexual\w*|divorc\w*|salary|debt|bankrupt\w*)\b",
    re.IGNORECASE,
)
_LONG_NUMBER = re.compile(r"\d{7,}")


def sensitive_reason(text: str) -> str | None:
    """Why this fact must not be saved automatically, or None if it may be."""
    secret = secret_reason(text)
    if secret:
        return secret
    match = _SENSITIVE.search(text)
    if match:
        return f"it mentions a sensitive topic ({match.group(0).lower()})"
    if _LONG_NUMBER.search(re.sub(r"[\s-]", "", text)):
        return "it contains a long number"
    return None


def normalize(text: str) -> str:
    """Lowercase words only, for comparing two memories."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())
