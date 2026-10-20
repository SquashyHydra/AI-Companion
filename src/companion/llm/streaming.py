"""Streaming helpers for the LLM client."""

from __future__ import annotations

import re

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"

_TERMINATOR = re.compile(r"[.!?]+[\"')\]]*")
_WORD = re.compile(r"\S+")
_CLAUSE_END = re.compile(r"[,;:\u2014](?=\s)")
MIN_CLAUSE_WORDS = 3
_ABBREVIATION = re.compile(r"([A-Za-z][A-Za-z.]*)$")
ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "e.g", "i.e"})
_SOFT_BREAKS = (", ", "; ", ": ")


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of text that is the start of tag (and not the whole tag)."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class ThinkFilter:
    """Removes <think>...</think> blocks from a stream of text chunks.

    Tags can be split across chunks, so text that might be the start of a tag is held back.
    """

    def __init__(self) -> None:
        self._inside = False
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Return the visible part of the text received so far."""
        self._buffer += chunk
        visible: list[str] = []
        while True:
            tag = CLOSE_TAG if self._inside else OPEN_TAG
            index = self._buffer.find(tag)
            if index != -1:
                if not self._inside:
                    visible.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(tag) :]
                self._inside = not self._inside
                continue
            keep = _partial_suffix(self._buffer, tag)
            if not self._inside:
                visible.append(self._buffer[: len(self._buffer) - keep])
            self._buffer = self._buffer[len(self._buffer) - keep :]
            return "".join(visible)

    def flush(self) -> str:
        """Call at the end of the stream. Returns any held-back visible text."""
        rest = "" if self._inside else self._buffer
        self._buffer = ""
        self._inside = False
        return rest


class SentenceSplitter:
    """Cuts streamed reply text into sentences for speech.

    A sentence ends at ., ! or ? followed by whitespace, or at a newline. A terminator at the very
    end of the buffer waits for more text, so "3.5" and "example.com" are not split. Common
    abbreviations (Dr., e.g.) are not split either. A run with no full stop longer than max_chars
    is cut at a comma or, failing that, at a space, so speech can start sooner.

    first_words > 0 also cuts the first piece of the stream early, so speech starts sooner. It is
    cut at the first clause boundary (, ; : or an em dash) that comes after at least three words
    and within first_words words, or else after the first_words-th word. Only the first piece is
    affected. Brackets such as [clear throat] are never cut.
    """

    def __init__(self, max_chars: int = 200, soft_min: int = 60, first_words: int = 0) -> None:
        self._max_chars = max_chars
        self._soft_min = soft_min
        self._first_words = first_words
        self._emitted = False
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add text and return the sentences that are now complete."""
        self._buffer += text
        sentences: list[str] = []
        while True:
            cut = self._next_cut()
            if cut is None:
                return sentences
            piece = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:]
            self._emitted = True
            if piece:
                sentences.append(piece)

    def flush(self) -> list[str]:
        """Call at the end of a reply. Returns whatever is left as a final sentence."""
        piece = self._buffer.strip()
        self._buffer = ""
        return [piece] if piece else []

    def _next_cut(self) -> int | None:
        sentence_cut = self._sentence_cut()
        early_cut = self._early_cut()
        if early_cut is not None and (sentence_cut is None or early_cut < sentence_cut):
            return early_cut
        if sentence_cut is not None:
            return sentence_cut
        return self._length_cut()

    def _sentence_cut(self) -> int | None:
        buffer = self._buffer
        newline = buffer.find("\n")
        for match in _TERMINATOR.finditer(buffer):
            if newline != -1 and newline < match.start():
                break
            end = match.end()
            if end >= len(buffer):
                break
            if not buffer[end].isspace():
                continue
            if match.group().startswith(".") and _is_abbreviation(buffer[: match.start()]):
                continue
            return end
        if newline != -1:
            return newline + 1
        return None

    def _early_cut(self) -> int | None:
        """Cut the first piece at a clause boundary or after first_words words, if still open."""
        if self._first_words <= 0 or self._emitted:
            return None
        buffer = self._buffer
        # A word is complete only when whitespace follows it in the buffer.
        ends = [match.end() for match in _WORD.finditer(buffer) if match.end() < len(buffer)]
        min_words = min(MIN_CLAUSE_WORDS, self._first_words)
        for clause in _CLAUSE_END.finditer(buffer):
            count = sum(1 for end in ends if end <= clause.end())
            if min_words <= count <= self._first_words and _brackets_closed(buffer, clause.end()):
                return clause.end()
        if len(ends) >= self._first_words:
            cut = ends[self._first_words - 1]
            if _brackets_closed(buffer, cut):
                return cut
        return None

    def _length_cut(self) -> int | None:
        buffer = self._buffer
        if len(buffer) >= self._max_chars:
            for mark in _SOFT_BREAKS:
                position = buffer.rfind(mark, self._soft_min, self._max_chars)
                if position != -1:
                    return position + 1
            position = buffer.rfind(" ", self._soft_min, self._max_chars)
            if position != -1:
                return position + 1
        return None


def _brackets_closed(buffer: str, cut: int) -> bool:
    """False when the cut would fall inside a bracketed tag such as [clear throat]."""
    head = buffer[:cut]
    return head.count("[") == head.count("]")


def _is_abbreviation(before: str) -> bool:
    match = _ABBREVIATION.search(before)
    return match is not None and match.group(1).lower() in ABBREVIATIONS
