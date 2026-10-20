"""Paralinguistic voice tags such as [laugh], for engines that can render them.

The set of tags an engine renders is read from the model at runtime (its tokenizer's added
tokens, see tags_from_vocabulary), not hard-coded. Only tags in that set are kept. Other
bracketed text is never spoken. Piper has no tags, so its set is empty.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

MAX_TAG_CHARS = 24
TAG_PATTERN = re.compile(r"\[([^\[\]\n]{1,%d})\]" % MAX_TAG_CHARS)

# A model's tag token looks like "[laugh]" or "[clear throat]".
_TAG_TOKEN = re.compile(r"\[([a-z][a-z ]*[a-z])\]")

# Variants the model might write. They map to a tag name, but only if that name is in the set.
_ALIASES: dict[str, str] = {
    "laughs": "laugh",
    "laughing": "laugh",
    "chuckles": "chuckle",
    "chuckling": "chuckle",
    "coughs": "cough",
    "coughing": "cough",
}


def tags_from_vocabulary(tokens: Iterable[str]) -> frozenset[str]:
    """Return the tag names among a model's added tokens: "[clear throat]" -> "clear throat".

    Tokens that are not bracketed tags (ordinary words, special tokens) are ignored.
    """
    names: set[str] = set()
    for token in tokens:
        match = _TAG_TOKEN.fullmatch(token.strip())
        if match:
            names.add(" ".join(match.group(1).split()))
    return frozenset(names)


def normalize_tag(raw: str, tags: frozenset[str]) -> str | None:
    """Return the canonical name for text inside brackets, or None if the engine does not render it."""
    name = " ".join(raw.strip().lower().split())
    name = _ALIASES.get(name, name)
    return name if name in tags else None


class TagDisplayFilter:
    """Streams text to the screen and hides supported tags when show is False.

    Text is held from an opening bracket until its closing bracket arrives, so a tag split across
    streamed chunks is still hidden. Call flush() at the end of a reply.
    """

    def __init__(self, show: bool, tags: frozenset[str] = frozenset()) -> None:
        self._show = show
        self._tags = frozenset(tags)
        self._pending = ""

    def set_tags(self, tags: frozenset[str]) -> None:
        """Set the tags the voice renders, once the voice has loaded."""
        self._tags = frozenset(tags)

    def feed(self, text: str) -> str:
        if self._show:
            return text
        out: list[str] = []
        for char in text:
            if not self._pending:
                if char == "[":
                    self._pending = char
                else:
                    out.append(char)
                continue
            if char == "[":
                out.append(self._pending)
                self._pending = char
                continue
            self._pending += char
            if char == "]":
                if normalize_tag(self._pending[1:-1], self._tags) is None:
                    out.append(self._pending)
                self._pending = ""
            elif char == "\n" or len(self._pending) > MAX_TAG_CHARS + 2:
                out.append(self._pending)
                self._pending = ""
        return "".join(out)

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return text
