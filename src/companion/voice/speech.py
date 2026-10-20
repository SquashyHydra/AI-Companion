"""Speaking replies: text cleanup, the synthesis worker, and interruption.

SpeechOut is driven from the main thread (feed, finish, interrupt). One worker thread synthesizes
sentences in order and hands the audio to the SpeakerPlayer. Interrupting bumps a generation
counter, so sentences still being synthesized are dropped instead of played late.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from companion.audio.errors import SpeechError, SupersededError
from companion.audio.playback import SpeakerPlayer
from companion.audio.tts import Synthesizer
from companion.llm.streaming import SentenceSplitter
from companion.voice.tags import normalize_tag

logger = logging.getLogger("companion.speech")

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_STAGE = re.compile(r"\[[^\]\n]*\]")
# Supported tags are wrapped in control characters while the text is cleaned, so the symbol
# stripping below cannot remove their brackets. They are restored (or removed) at the end.
_MARKED_TAG = re.compile(r"\x01([^\x01\x02]*)\x02")
_TAG_ONLY = re.compile(r"(?:\[[a-z ]+\]\s*)+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_STRIKE = re.compile(r"~~(.+?)~~")
_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SYMBOLS = re.compile(r"[<>{}\[\]|\\^~*#_`]")
_SPACES = re.compile(r"\s+")
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
_DROP_CATEGORIES = frozenset({"So", "Sk", "Cs", "Co", "Cf"})


def _spoken_date(match: re.Match[str]) -> str:
    year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return match.group(0)
    return f"{_MONTHS[month - 1]} {day}, {year}"


def _speakable(char: str) -> bool:
    if char in "\ufe0e\ufe0f":
        return False
    return unicodedata.category(char) not in _DROP_CATEGORIES


def _stage_direction(match: re.Match[str], tags: frozenset[str]) -> str:
    name = normalize_tag(match.group(0)[1:-1], tags)
    return f"\x01{name}\x02" if name else " "


def sanitize_for_speech(text: str, *, tags: frozenset[str] = frozenset()) -> str:
    """Turn reply text into something a TTS voice reads well.

    Removes markdown markers, code fences, URLs, bracketed stage directions, emoji and symbols.
    Keeps link text. Speaks ISO dates as "Month D, YYYY".

    Bracketed text that names a tag in `tags` (the set the voice renders, such as {"laugh"}) is
    kept as "[laugh]" so the engine can render it. Any other bracketed text is removed.
    """
    text = _CODE_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _STAGE.sub(lambda match: _stage_direction(match, tags), text)
    text = _URL.sub(" ", text)
    text = _BOLD.sub(r"\2", text)
    text = _STRIKE.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _ISO_DATE.sub(_spoken_date, text)
    text = text.replace("\u00b0", " degrees ")
    text = "".join(char if _speakable(char) else " " for char in text)
    text = _SYMBOLS.sub(" ", text)
    text = _MARKED_TAG.sub(lambda match: f"[{match.group(1)}]", text)
    return _SPACES.sub(" ", text).strip()


@dataclass(frozen=True)
class _Job:
    gen: int
    text: str


class SpeechOut:
    """Turns streamed reply text into audio, sentence by sentence."""

    def __init__(
        self,
        synth: Synthesizer,
        player: SpeakerPlayer,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_error: Callable[[str], None] | None = None,
        first_chunk_words: int = 0,
    ) -> None:
        self._synth = synth
        self._player = player
        self._clock = clock
        self._on_error = on_error or (lambda message: logger.warning(message))
        self._first_chunk_words = first_chunk_words
        self._splitter = self._new_splitter()
        self._carry = ""
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._lock = threading.Lock()
        self._gen = 0
        self._queued = 0
        self._reply_started: float | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Open the speaker output and start the synthesis worker."""
        self._player.start()
        self._thread = threading.Thread(target=self._work, name="tts-worker", daemon=True)
        self._thread.start()

    def begin_reply(self) -> None:
        """Call when a new reply starts. Resets the sentence splitter and starts the latency timer."""
        self._splitter = self._new_splitter()
        self._carry = ""
        self._reply_started = self._clock()

    @property
    def tags(self) -> frozenset[str]:
        """Voice tags the engine renders, such as {"laugh"}. Empty for Piper."""
        return frozenset(getattr(self._synth, "tags", frozenset()))

    def feed(self, text: str) -> None:
        for sentence in self._splitter.feed(text):
            self._submit(sentence)

    def finish(self) -> None:
        """Call when the reply ended normally. Speaks any unfinished last sentence."""
        for sentence in self._splitter.flush():
            self._submit(sentence)
        if self._carry:
            carried, self._carry = self._carry, ""
            self._enqueue(carried)

    def interrupt(self) -> tuple[str, bool]:
        """Stop speaking now.

        Returns (the text of sentences that finished playing since the last interrupt, whether
        anything was still queued or sounding).
        """
        with self._lock:
            was_busy = self._queued > 0 or self._player.busy()
            self._gen += 1
            generation = self._gen
            self._queued = 0
            self._drop_jobs()
            spoken = self._player.flush()
        self._splitter = self._new_splitter()
        self._carry = ""
        self._cancel_remote(generation)
        return " ".join(str(item) for item in spoken), was_busy

    def speaking(self, tail_s: float = 0.0) -> bool:
        """True while speech is queued or sounding, plus tail_s after the last sample."""
        with self._lock:
            return self._queued > 0 or self._player.busy(tail_s)

    def close(self) -> None:
        self.interrupt()
        self._jobs.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._player.stop()
        close_synth = getattr(self._synth, "close", None)
        if callable(close_synth):
            close_synth()

    def _submit(self, sentence: str) -> None:
        tags = self.tags
        spoken = sanitize_for_speech(sentence, tags=tags)
        if self._carry:
            spoken = f"{self._carry} {spoken}".strip()
            self._carry = ""
        if not any(char.isalnum() for char in spoken):
            return
        if tags and _TAG_ONLY.fullmatch(spoken):
            # A tag with no words (e.g. a laugh after a sentence) waits for the next sentence,
            # so it is rendered with its words instead of alone.
            self._carry = spoken
            return
        self._enqueue(spoken)

    def _new_splitter(self) -> SentenceSplitter:
        return SentenceSplitter(first_words=self._first_chunk_words)

    def _cancel_remote(self, generation: int) -> None:
        """Tell an engine that runs out of process to drop requests from older generations."""
        cancel = getattr(self._synth, "cancel", None)
        if not callable(cancel):
            return
        try:
            cancel(generation)
        except SpeechError as exc:
            logger.warning("could not cancel queued speech: %s", exc)

    def _enqueue(self, spoken: str) -> None:
        with self._lock:
            self._queued += 1
            self._jobs.put(_Job(self._gen, spoken))

    def _drop_jobs(self) -> None:
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                return

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            if not self._is_current(job.gen):
                continue
            try:
                samples = self._synth.synthesize(job.text, generation=job.gen)
            except SupersededError:
                # A barge-in cancelled this sentence on the engine side. Nothing to report.
                continue
            except Exception as exc:
                logger.exception("speech synthesis failed")
                self._release(job.gen)
                self._on_error(f"Could not speak that sentence: {exc}")
                continue
            with self._lock:
                if job.gen != self._gen:
                    continue
                self._player.write(samples, mark=job.text)
                self._queued -= 1
                if self._reply_started is not None:
                    logger.info("time to first audio: %.2f s", self._clock() - self._reply_started)
                    self._reply_started = None

    def _is_current(self, gen: int) -> bool:
        with self._lock:
            return gen == self._gen

    def _release(self, gen: int) -> None:
        with self._lock:
            if gen == self._gen:
                self._queued -= 1
