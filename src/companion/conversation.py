"""Conversation services shared by the text and voice loops.

This ties together the parts of Phase 5 that sit around a turn: the persona in the system prompt,
recalled memories for each request, background summaries of older turns, the conversation log,
and the status text. The loops only call into this class, so they stay free of these details.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from companion.conversation_log import ConversationLog
from companion.llm.client import LLMClient, Messages, TurnOutcome
from companion.llm.context import HistoryCompactor, messages_tokens
from companion.memory.extract import (
    build_extraction_messages,
    build_rewrite_messages,
    clean_rewrite,
    parse_extraction,
    Fact,
)
from companion.memory.guards import normalize, sensitive_reason
from companion.memory.intent import (
    Intent,
    is_ambiguous,
    is_cancel,
    is_yes,
    looks_like_question,
    parse_intent,
)
from companion.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)

PENDING_TIMEOUT_S = 120.0
"""How long a pending "what should I remember?" or yes-or-no question waits for an answer."""
AUTO_RECALL_SIZE = 5
"""Existing memories shown to the extractor so it can skip duplicates and update contradictions."""

Announce = Callable[[str, str | None], None]


def last_reply(messages: Messages) -> str:
    """The text of the last assistant message that is not a tool call."""
    for message in reversed(messages):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return str(message.get("content") or "").strip()
    return ""


def last_user_text(messages: Messages) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def forget_question(texts: list[str]) -> str:
    """Read-back for a forget request: at most three memories are named, the rest are counted."""

    def short(text: str) -> str:
        return text if len(text) <= 60 else text[:57] + "..."

    count = len(texts)
    shown = ", ".join(f'"{short(text)}"' for text in texts[:3])
    if count == 1:
        return f"Forget this memory: {shown}? Say yes or no."
    more = f", and {count - 3} more" if count > 3 else ""
    return f"Forget these {count} memories: {shown}{more}? Say yes or no."


@dataclass
class _Pending:
    """A question the companion asked and is waiting for the user to answer."""

    kind: str
    """remember: waiting for the text to save. forget: waiting for yes or no."""
    started: float
    context: str = ""
    """The user message just before the remember command, used to resolve vague notes."""
    ids: tuple[int, ...] = ()


class Conversation:
    """One companion session. Thread-safe enough for the loops: calls come from one main thread
    or from one reply worker at a time, and the log and compactor guard their own state.
    Memory commands and automatic saves are handled here, so every loop gets the same behavior.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        system_prompt: Callable[[], str],
        memory: MemoryStore | None = None,
        memory_limit: int = 5,
        memory_chars: int = 300,
        compactor: HistoryCompactor | None = None,
        log: ConversationLog | None = None,
        model: str = "",
        auto_save: bool = False,
        announce_auto: bool = False,
        announce: Announce | None = None,
        clock: Callable[[], float] = time.monotonic,
        pending_timeout_s: float = PENDING_TIMEOUT_S,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._memory = memory
        self._memory_limit = memory_limit
        self._memory_chars = memory_chars
        self._compactor = compactor
        self._log = log
        self._model = model
        self._auto_save = auto_save
        self._announce_auto = announce_auto
        self._announce = announce
        self._clock = clock
        self._pending_timeout_s = pending_timeout_s
        self._pending: _Pending | None = None
        self._pending_lock = threading.Lock()
        self._saves: list[threading.Thread] = []

    def set_announce(self, announce: Announce | None) -> None:
        """Where "[noted: ...]" lines go. The voice loop routes them through its event queue."""
        self._announce = announce

    # ----- history -----

    def new_history(self) -> Messages:
        """Start a new conversation with the persona as it is in the file right now."""
        if self._compactor is not None:
            self._compactor.reset()
        if self._log is not None:
            self._log.start_new()
        with self._pending_lock:
            self._pending = None
        return [{"role": "system", "content": self._system_prompt()}]

    def apply(self, history: Messages) -> Messages:
        """Swap in a finished summary before the next turn, if one is ready."""
        if self._compactor is None:
            return history
        return self._compactor.apply(history)

    def after_turn(self, history: Messages) -> None:
        """Start a background summary if the history has grown past the budget."""
        if self._compactor is not None:
            self._compactor.maybe_start(history)

    # ----- one turn -----

    def stream(
        self,
        history: Messages,
        text: str,
        cancel: threading.Event,
        on_delta: Callable[[str], None],
        *,
        confirm_note: str | None = None,
        low_confidence: bool = False,
    ) -> TurnOutcome:
        """Run one user message. Memory commands are answered here without the model. Otherwise
        the model answers, and a finished turn may save facts in the background.
        """
        reply = self._command_reply(history, text)
        if reply is not None:
            on_delta(reply)
            messages = [
                *history,
                {"role": "user", "content": text},
                {"role": "assistant", "content": reply},
            ]
            if self._log is not None:
                self._log.write_turn(text, reply)
            return TurnOutcome(messages=messages, reply=reply, interrupted=False)

        outcome = self._llm.stream_turn(
            history,
            text,
            cancel=cancel,
            on_delta=on_delta,
            extra_system=self._memory_block(text),
            confirm_note=confirm_note,
        )
        if not outcome.interrupted and self._log is not None:
            self._log.write_turn(text, last_reply(outcome.messages))
        if not outcome.interrupted:
            self._maybe_auto_save(history, text, outcome.messages, low_confidence)
        return outcome

    # ----- memory commands -----

    def stage_forget(self, target: str) -> str:
        """Find memories matching target and ask for yes or no. Nothing is deleted yet.

        Also the callback for the forget tool, so the model cannot delete without this answer.
        """
        if self._memory is None:
            return "Memory is turned off, so there is nothing to forget."
        matches = self._memory.find_ids(target)
        if not matches:
            return f'I found no saved memory matching "{target}".'
        with self._pending_lock:
            self._pending = _Pending(
                kind="forget",
                started=self._clock(),
                ids=tuple(match.id for match in matches),
            )
        return forget_question([match.text for match in matches])

    def _command_reply(self, history: Messages, text: str) -> str | None:
        """The reply for a memory command or a pending answer. None means the model should answer."""
        pending = self._take_pending()
        if pending is not None:
            if pending.kind == "forget":
                reply = self._answer_forget(pending, text)
                if reply is not None:
                    return reply
            elif pending.kind == "remember":
                reply = self._answer_remember(pending, text)
                if reply is not None:
                    return reply

        intent = parse_intent(text)
        if intent is None:
            return None
        return self._run_intent(intent, history)

    def _take_pending(self) -> _Pending | None:
        """Remove and return the pending question. Each new message resolves it, and it expires."""
        with self._pending_lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return None
        if self._clock() - pending.started > self._pending_timeout_s:
            return None
        return pending

    def _answer_forget(self, pending: _Pending, text: str) -> str | None:
        answer = is_yes(text)
        if answer is None:
            return None  # not an answer, so the message is handled as a new one
        if not answer:
            return "Okay, I kept them."
        if self._memory is None:
            return "Memory is turned off, so there is nothing to forget."
        deleted = self._memory.delete(list(pending.ids))
        noun = "memory" if deleted == 1 else "memories"
        return f"Forgot {deleted} {noun}."

    def _answer_remember(self, pending: _Pending, text: str) -> str | None:
        if is_cancel(text):
            return "Okay, I won't save anything."
        if parse_intent(text) is not None or looks_like_question(text):
            return None  # the user moved on, so the message is handled as a new one
        if not text.strip():
            return None
        return self._save(text.strip(), previous=pending.context)

    def _run_intent(self, intent: Intent, history: Messages) -> str:
        previous = last_user_text(history)
        if intent.kind == "remember_text":
            return self._save(intent.text, previous=previous)
        if intent.kind == "remember_pending":
            with self._pending_lock:
                self._pending = _Pending(
                    kind="remember", started=self._clock(), context=previous
                )
            return "What should I remember?"
        if intent.kind == "remember_previous":
            if not previous:
                return "There is nothing earlier in this conversation to remember."
            return self._save(previous, previous="")
        if intent.kind == "forget_that":
            return self._forget_latest_auto()
        return self.stage_forget(intent.text)

    def _save(self, content: str, *, previous: str) -> str:
        """Save a note the user asked for. Only the user's own words are stored, plus a rewrite
        for vague notes like "it is at 3" when the earlier message is available.
        """
        if self._memory is None:
            return "Memory is turned off, so I can't save that."
        reason = sensitive_reason(content)
        if reason:
            return f"I did not save that because {reason}."
        if len(content) > self._memory_chars:
            return (
                f"That is too long to save ({len(content)} characters, "
                f"limit {self._memory_chars}). Try a shorter note."
            )
        note = content
        if previous and is_ambiguous(content):
            note = self._rewrite(content, previous)
        memory = self._memory.add(note, source="manual")
        return f"Saved: {memory.text}"

    def _rewrite(self, note: str, previous: str) -> str:
        try:
            raw = self._llm.complete(build_rewrite_messages(note, previous), max_tokens=80)
        except Exception:
            logger.warning("could not rewrite the note, keeping the user's words", exc_info=True)
            return note
        return clean_rewrite(raw, note, self._memory_chars)

    def _forget_latest_auto(self) -> str:
        if self._memory is None:
            return "Memory is turned off, so there is nothing to forget."
        latest = self._memory.latest("auto")
        if latest is None:
            return "I have no automatically saved memory to forget."
        self._memory.delete([latest.id])
        return f"Forgot: {latest.text}"

    # ----- automatic saves -----

    def _maybe_auto_save(
        self, history: Messages, text: str, messages: Messages, low_confidence: bool
    ) -> None:
        if not self._auto_save or self._memory is None or low_confidence:
            return
        turn = messages[len(history) :]
        if any(message.get("role") == "tool" or message.get("tool_calls") for message in turn):
            return  # tool results are untrusted, so nothing is saved from a tool-using turn
        reply = last_reply(messages)
        if not text.strip() or not reply or parse_intent(text) is not None:
            return
        thread = threading.Thread(
            target=self._auto_save_turn, args=(text, reply), name="memory-auto-save", daemon=True
        )
        with self._pending_lock:
            self._saves = [item for item in self._saves if item.is_alive()]
            self._saves.append(thread)
        thread.start()

    def wait_background(self, timeout: float | None = None) -> None:
        """Wait for background saves to finish. Used by tests and at shutdown."""
        with self._pending_lock:
            threads = list(self._saves)
        for thread in threads:
            thread.join(timeout)

    def _auto_save_turn(self, text: str, reply: str) -> None:
        if self._memory is None:
            return
        try:
            existing = self._memory.search(text, limit=AUTO_RECALL_SIZE)
            known = {memory.id: memory.text for memory in existing}
            raw = self._llm.complete(
                build_extraction_messages(
                    text, reply, [(memory.id, memory.text) for memory in existing]
                ),
                max_tokens=300,
            )
            facts = parse_extraction(raw, set(known), self._memory_chars)
        except Exception:
            logger.warning("automatic memory extraction failed", exc_info=True)
            return
        for fact in facts:
            try:
                self._apply_fact(fact, known)
            except Exception:
                logger.warning("could not save an automatic memory", exc_info=True)

    def _apply_fact(self, fact: Fact, known: dict[int, str]) -> None:
        if self._memory is None:
            return
        reason = sensitive_reason(fact.text)
        if reason:
            logger.info("skipped an automatic memory: %s", reason)
            return
        if normalize(fact.text) in {normalize(value) for value in known.values()}:
            return  # already stored
        stored: Memory | None
        if fact.action == "update" and fact.memory_id in known:
            stored = self._memory.update(fact.memory_id, fact.text, source="auto")
        else:
            stored = self._memory.add(fact.text, source="auto")
        if stored is None:
            return
        self._notice_saved(stored)

    def _notice_saved(self, memory: Memory) -> None:
        line = f"[noted: {memory.text}]"
        spoken = f"Noted: {memory.text}" if self._announce_auto else None
        if self._announce is not None:
            self._announce(line, spoken)
        else:
            print(line)

    def _memory_block(self, text: str) -> str | None:
        if self._memory is None:
            return None
        try:
            return self._memory.block_for(
                text, limit=self._memory_limit, max_chars=self._memory_chars
            )
        except Exception:
            logger.warning("could not read saved memories for this turn", exc_info=True)
            return None

    # ----- status -----

    def status(self, history: Messages) -> str:
        used = messages_tokens(history)
        try:
            window = self._llm.context_window()
        except Exception:
            window = 0
        lines = []
        if self._model:
            lines.append(f"Model: {self._model}")
        if window:
            lines.append(f"Context: about {used} of {window} tokens used by this conversation.")
        if self._memory is not None:
            try:
                count = len(self._memory.all())
            except Exception:
                count = -1
            if count >= 0:
                lines.append(f"Saved memories: {count}.")
        if self._log is not None:
            path = self._log.path
            lines.append(f"Conversation log: {path if path else 'written after the first reply'}")
        return "\n".join(lines) or "No status available."
