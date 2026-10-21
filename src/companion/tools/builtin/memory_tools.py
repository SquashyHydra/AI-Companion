"""Memory tools: remember, recall, and forget.

Listing memories is done in code (see the conversation module), not by a tool, so the model never
prints numbers or lists it made up.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from companion.config import get_config
from companion.memory.store import Memory, MemoryStore, clean_text, memory_line
from companion.safety.secrets import secret_reason
from companion.tools.registry import tool

_store: MemoryStore | None = None
_forget_stager: Callable[[str], str] | None = None
_lock = threading.Lock()
DISABLED = "Error: memory is turned off in config.yaml."


def set_store(store: MemoryStore | None) -> None:
    """Use a specific store (tests, or when the loop wants to share one instance)."""
    global _store
    with _lock:
        _store = store


def set_forget_stager(stager: Callable[[str], str] | None) -> None:
    """Where forget requests go. The conversation reads the matches back and waits for yes or no."""
    global _forget_stager
    with _lock:
        _forget_stager = stager


def active_store() -> MemoryStore | None:
    """The shared store, or None when memory is disabled in config."""
    global _store
    settings = get_config().memory
    if not settings.enabled:
        return None
    with _lock:
        if _store is None:
            _store = MemoryStore(settings.db_path)
        return _store


def _line(memory: Memory) -> str:
    return memory_line(memory)


@tool
def remember(fact: str) -> str:
    """Save a fact about the user or their preferences for future conversations.

    Use this only when the user asks you to remember something. Never save passwords, keys,
    tokens, or other secrets.

    Args:
        fact: The fact to save, in one short sentence.
    """
    store = active_store()
    if store is None:
        return DISABLED
    text = clean_text(fact)
    if not text:
        return "Error: nothing to save. Say what to remember."
    limit = get_config().memory.max_chars
    if len(text) > limit:
        return f"Error: that is too long to save ({len(text)} characters, limit {limit}). Shorten it."
    reason = secret_reason(text)
    if reason:
        return f"Error: not saved because {reason}. Memory is for facts, not secrets."
    existing = store.existing_for(text)
    if existing is not None:
        return f"Already saved as memory {existing.id}: {existing.text}"
    saved = store.add(text)
    return f"Saved memory {saved.id}: {saved.text}"


@tool
def recall(query: str, limit: int = 5) -> str:
    """Look up saved memories that relate to the words in query.

    Args:
        query: Words to look for, such as a topic or a name.
        limit: Maximum number of memories to return (1 to 20).
    """
    store = active_store()
    if store is None:
        return DISABLED
    limit = max(1, min(int(limit), 20))
    matches = store.search(query, limit)
    if not matches:
        return f"No saved memories match '{clean_text(query)}'."
    return "\n".join(_line(memory) for memory in matches)


@tool
def forget(target: str) -> str:
    """Forget saved memories. Nothing is deleted until the user answers yes.

    This asks the user to confirm. It does not delete anything by itself.

    Args:
        target: A memory number such as 3 or 2-4, the word 'all' to forget everything, or a few
            words that appear in the memories to forget.
    """
    with _lock:
        stager = _forget_stager
    if stager is None:
        return "Error: forgetting is not available in this session."
    wanted = clean_text(target)
    if not wanted:
        return "Error: say which memory to forget: a number, a few words, or 'all'."
    return stager(wanted)
