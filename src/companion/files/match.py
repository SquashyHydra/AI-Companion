"""Name normalization and similarity scores for file search.

Names are folded to lowercase ASCII-ish text, punctuation becomes spaces, and a leading track
number such as "01 - " is dropped from file names. That lets "hey jude" match "01 - Hey Jude.mp3".
"""

from __future__ import annotations

import difflib
import os
import re
import unicodedata

FUZZY_MIN = 0.72
"""Lowest similarity (0 to 1) that counts as a fuzzy match. Below this nothing is suggested."""

_TRACK_RE = re.compile(r"^\d{1,3}\s*[-._)\]]\s*(?=\S)")
_SPLIT_RE = re.compile(r"[\W_]+")

try:  # Optional faster scorer. Not required, and not listed in requirements.txt.
    from rapidfuzz.fuzz import ratio as _rapid_ratio
except ImportError:  # pragma: no cover - depends on the environment
    _rapid_ratio = None


def backend_name() -> str:
    """Which similarity scorer is in use. Shown by diagnose_files."""
    return "rapidfuzz" if _rapid_ratio is not None else "difflib"


def fold(text: str) -> str:
    """Lowercase and remove accents."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def normalize(text: str) -> str:
    """Folded text with punctuation turned into single spaces."""
    return " ".join(_SPLIT_RE.split(fold(text))).strip()


def leaf_key(name: str, is_folder: bool) -> str:
    """Normalized name used for exact and fuzzy matching. Drops the extension and a track number."""
    base = name if is_folder else os.path.splitext(name)[0]
    base = _TRACK_RE.sub("", fold(base).strip(), count=1)
    return normalize(base)


def similarity(query: str, candidate: str) -> float:
    """Score from 0 to 1 for two normalized strings."""
    if not query or not candidate:
        return 0.0
    if _rapid_ratio is not None:
        return _rapid_ratio(query, candidate) / 100.0
    return difflib.SequenceMatcher(None, query, candidate).ratio()
