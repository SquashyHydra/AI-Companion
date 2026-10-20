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
_FORGET_RE = re.compile(
    r"^forget\b(?:\s+(?:about|that|this|it))?\s*[:,]?\s*(?P<target>.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Intent:
    """kind is one of remember_text, remember_pending, remember_previous, forget_that, forget_target."""

    kind: str
    text: str = ""


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

    remember = _REMEMBER_RE.match(original)
    if remember:
        content = remember.group("rest").strip(" .!?")
        if not content:
            return Intent("remember_pending")
        return Intent("remember_text", content)

    forget = _FORGET_RE.match(original)
    if forget:
        target = forget.group("target").strip(" .!?")
        if not target:
            return Intent("forget_that")
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
