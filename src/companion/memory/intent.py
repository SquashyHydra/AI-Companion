"""Plain-text memory commands, handled before the LLM runs.

Saves and deletes done here never depend on the model, so the companion cannot claim something
was saved or forgotten when it was not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from companion.memory.guards import normalize

REMEMBER_PENDING = frozenset(
    {
        "remember",
        "remember this",
        "remember that",
        "remember it",
        "remember this please",
        "remember that please",
        "please remember this",
        "please remember that",
        "remember this for me",
        "remember that for me",
    }
)
REMEMBER_PREVIOUS = frozenset(
    {
        "remember what i just said",
        "remember what i said",
        "remember what i just told you",
        "remember what i said earlier",
    }
)
FORGET_THAT = frozenset(
    {
        "forget that",
        "forget it",
        "forget this",
        "forget that last one",
        "forget the last one",
        "forget that memory",
        "forget what you just saved",
    }
)
CANCEL = frozenset(
    {
        "never mind",
        "nevermind",
        "cancel",
        "cancel that",
        "forget it",
        "no thanks",
        "nothing",
        "don't save it",
        "do not save it",
        "don't remember that",
        "never mind that",
    }
)
YES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "confirm",
        "do it",
        "go ahead",
        "yes please",
        "please do",
        "forget them",
        "forget them all",
        "delete them",
        "yes forget them",
    }
)
NO = frozenset(
    {
        "no",
        "nope",
        "nah",
        "cancel",
        "never mind",
        "don't",
        "do not",
        "keep them",
        "no thanks",
        "stop",
        "no keep them",
    }
)
QUESTION_WORDS = frozenset(
    {"what", "when", "where", "who", "whom", "whose", "why", "how", "is", "are", "do", "does",
     "did", "can", "could", "will", "would", "should"}
)  # fmt: skip

_REMEMBER_RE = re.compile(
    r"^(?:please\s+)?remember\b(?:\s+(?:that|this|it))?(?:\s+please)?\s*[:,]?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_POLITE = r"^(?:(?:please\s+)?(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
_FORGET_RE = re.compile(
    _POLITE + r"(?P<verb>forget|remove|delete)\b(?:\s+(?:about|that|this|it))?"
    r"\s*[:,]?\s*(?P<target>.*)$",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    _POLITE
    + r"(?:list|show|read(?:\s+out)?|tell\s+me)\s+(?:me\s+)?(?:all\s+)?(?:of\s+)?"
    + r"(?:my\s+|your\s+|the\s+)?(?:saved\s+|stored\s+)?memor(?:y|ies)"
    + r"(?:\s+(?:right\s+now|now|please|for\s+me|to\s+me))*\s*$"
    + r"|^what\s+(?:do\s+you\s+remember|memories\s+do\s+you\s+have|are\s+my\s+memories"
    + r"|have\s+you\s+saved)(?:\s+about\s+me)?(?:\s+right\s+now)?\s*$",
    re.IGNORECASE,
)
_MEMORY_HINT = re.compile(r"memor|\bnumbers?\b|#|\bduplicates?\b|\bentr(?:y|ies)\b", re.IGNORECASE)

MAX_FORGET_IDS = 200
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}  # fmt: skip
_UNITS = frozenset(word for word, value in _WORD_NUMBERS.items() if 1 <= value <= 9)
_TENS = frozenset({"twenty", "thirty", "forty", "fifty"})
_RANGE_WORDS = frozenset({"to", "through", "thru", "-"})
_ID_FILLER = frozenset(
    {"memory", "memories", "number", "numbers", "entry", "entries", "item", "items", "id",
     "ids", "the", "and", "or", "please"}
)  # fmt: skip
_MAX_RANGE = 10_000


@dataclass(frozen=True)
class Intent:
    """kind is one of remember_text, remember_pending, remember_previous, forget_that,
    forget_target, forget_ids, or list_memories."""

    kind: str
    text: str = ""
    ids: tuple[int, ...] = ()


def parse_id_spec(text: str) -> list[int] | None:
    """Memory numbers named in text: '3', 'memory 3', '#3', 'number three', '3-5', '3 to 5',
    '2, 4 and 6', 'twenty three'. Returns None when the text names anything else."""
    tokens = re.findall(r"\d+|[a-z]+|-", text.lower())
    items: list[int | str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token.isdigit():
            items.append(int(token))
        elif token in _WORD_NUMBERS:
            value = _WORD_NUMBERS[token]
            if token in _TENS and index < len(tokens) and tokens[index] in _UNITS:
                value += _WORD_NUMBERS[tokens[index]]
                index += 1
            items.append(value)
        elif token in _RANGE_WORDS:
            items.append("-")
        elif token not in _ID_FILLER:
            return None
    ids: list[int] = []
    position = 0
    while position < len(items):
        current = items[position]
        if not isinstance(current, int):
            return None
        if position + 1 < len(items) and items[position + 1] == "-":
            if position + 2 >= len(items) or not isinstance(items[position + 2], int):
                return None
            end = items[position + 2]
            if end < current or end - current > _MAX_RANGE:
                return None
            ids.extend(range(current, end + 1))
            position += 3
        else:
            ids.append(current)
            position += 1
    return list(dict.fromkeys(ids)) or None


def _names_memory(target: str) -> bool:
    return bool(_MEMORY_HINT.search(target)) or parse_id_spec(target) is not None


def parse_intent(text: str) -> Intent | None:
    """Recognize a memory command in one user message. Returns None for ordinary messages."""
    original = " ".join(text.split())
    key = normalize(original)
    if not key:
        return None
    if key in REMEMBER_PENDING:
        return Intent("remember_pending")
    if key in REMEMBER_PREVIOUS:
        return Intent("remember_previous")
    if key in FORGET_THAT:
        return Intent("forget_that")
    if _LIST_RE.match(original.strip(" ?!.")):
        return Intent("list_memories")

    remember = _REMEMBER_RE.match(original)
    if remember:
        content = remember.group("rest").strip(" .!?")
        if not content:
            return Intent("remember_pending")
        return Intent("remember_text", content)

    forget = _FORGET_RE.match(original)
    if forget:
        verb = forget.group("verb").lower()
        target = forget.group("target").strip(" .!?")
        # remove and delete only count as memory commands when they name a memory.
        if verb != "forget" and not _names_memory(target):
            return None
        if not target:
            return Intent("forget_that")
        ids = parse_id_spec(target)
        if ids is not None:
            return Intent("forget_ids", ids=tuple(ids))
        return Intent("forget_target", target)
    return None


def is_cancel(text: str) -> bool:
    return normalize(text) in CANCEL


def is_yes(text: str) -> bool | None:
    """True for yes, False for no, None for anything else."""
    key = normalize(text)
    if key in YES:
        return True
    if key in NO:
        return False
    return None


def looks_like_question(text: str) -> bool:
    """A new question while a save is pending means the user moved on."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    words = normalize(stripped).split()
    return bool(words) and words[0] in QUESTION_WORDS


def is_ambiguous(text: str) -> bool:
    """A note that needs its earlier message to make sense, such as 'it is at 3 pm'."""
    if re.match(r"^(it|that|this|they|them|he|she|those|these|there)\b", text.strip(), re.I):
        return True
    return len(text.split()) < 3
