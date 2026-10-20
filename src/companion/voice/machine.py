"""Voice controller state machine. Pure logic: events in, actions out. No threads, no audio, no printing."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from companion.audio.segmenter import Noise, SpeechOngoing, Utterance
from companion.llm.client import TurnOutcome

DOUBLE_CANCEL_S = 2.0
"""Two Ctrl+C presses this close together quit the app, even while a confirmation is pending."""

MIC_RETRY_S = 10.0
"""While the microphone is failing, try to reopen it this often."""

SPOKEN_FAILURE = "Sorry, I could not get an answer just now. Please try again."
"""What the companion says aloud when a reply fails."""

RESET_PHRASES = frozenset(
    {
        "new conversation",
        "start a new conversation",
        "start new conversation",
        "clear conversation",
        "reset conversation",
    }
)
"""Spoken phrases that start a new conversation, the same as /new."""


class State(Enum):
    PAUSED = "paused"
    LISTENING = "listening"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"


# ---------- Events ----------


@dataclass(frozen=True)
class Toggle:
    """Enter on an empty line: pause or resume listening. Pausing also stops a reply."""


@dataclass(frozen=True)
class Quit:
    """Ctrl+C or end of input."""


@dataclass(frozen=True)
class Answer:
    """A y or n key press. Only meaningful while a confirmation is pending."""

    answer: bool


@dataclass(frozen=True)
class Cancel:
    """Ctrl+C. While a confirmation is pending it declines it. With nothing pending it quits.

    A second Ctrl+C within DOUBLE_CANCEL_S quits even if a confirmation is pending.
    """


@dataclass(frozen=True)
class Typed:
    text: str


@dataclass(frozen=True)
class Command:
    text: str


@dataclass(frozen=True)
class MicFailed:
    message: str


@dataclass(frozen=True)
class MicReady:
    """The microphone started. Clears a failure and lets listening resume."""


@dataclass(frozen=True)
class Tick:
    """Sent regularly by the loop. Used to retry the microphone while it is failing."""


@dataclass(frozen=True)
class Transcript:
    text: str = ""
    low_confidence: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ReplyDelta:
    turn: int
    text: str


@dataclass(frozen=True, eq=False)
class ReplyDone:
    turn: int
    outcome: TurnOutcome


@dataclass(frozen=True)
class ReplyFailed:
    turn: int
    message: str


@dataclass(frozen=True)
class ConfirmRequest:
    question: str
    timeout_s: float | None = None


@dataclass(frozen=True)
class ConfirmTimedOut:
    pass


@dataclass(frozen=True)
class Notice:
    """A message from another thread (for example, a speech error) to show on screen."""

    text: str
    spoken: str | None = None
    """If set, also say this aloud."""


Event = (
    Toggle
    | Quit
    | Answer
    | Cancel
    | Typed
    | Command
    | MicFailed
    | SpeechOngoing
    | Utterance
    | Noise
    | Transcript
    | ReplyDelta
    | ReplyDone
    | ReplyFailed
    | ConfirmRequest
    | ConfirmTimedOut
    | Notice
)


# ---------- Actions ----------


@dataclass(frozen=True)
class StartMic:
    pass


@dataclass(frozen=True)
class StopMic:
    pass


@dataclass(frozen=True, eq=False)
class Transcribe:
    audio: np.ndarray


@dataclass(frozen=True)
class StartReply:
    turn: int
    text: str
    low_confidence: bool = False
    """Set only when the confirm option is on, for tool confirmations."""
    unsure: bool = False
    """The transcript was low confidence, so automatic memory saves are skipped."""


@dataclass(frozen=True)
class CancelReply:
    pass


@dataclass(frozen=True)
class StopSpeech:
    """Stop speaking now. The reply keeps only the sentences that finished playing."""


@dataclass(frozen=True)
class Stream:
    text: str


@dataclass(frozen=True, eq=False)
class ApplyOutcome:
    outcome: TurnOutcome


@dataclass(frozen=True)
class ResetHistory:
    pass


@dataclass(frozen=True)
class RunCommand:
    text: str


@dataclass(frozen=True)
class Print:
    text: str
    spoken: str | None = None
    """If set, also say this aloud (used for reply failures)."""


@dataclass(frozen=True)
class ShowStatus:
    pass


@dataclass(frozen=True)
class ResolveConfirmation:
    answer: bool


@dataclass(frozen=True)
class Shutdown:
    pass


Action = (
    StartMic
    | StopMic
    | Transcribe
    | StartReply
    | CancelReply
    | StopSpeech
    | Stream
    | ApplyOutcome
    | ResetHistory
    | RunCommand
    | Print
    | ShowStatus
    | ResolveConfirmation
    | Shutdown
)


# ---------- Yes / no ----------

_YES = frozenset({"y", "yes", "yeah", "yep", "confirm"})
_NO = frozenset({"n", "no", "nope", "cancel"})


def parse_yes_no(text: str) -> bool | None:
    """Read an answer. Only the exact words in _YES or _NO count. Anything else returns None.

    Punctuation and case are ignored. "Yes, please" is not a yes.
    """
    normalized = _phrase(text)
    if normalized in _YES:
        return True
    if normalized in _NO:
        return False
    return None


def _phrase(text: str) -> str:
    """Lowercase words only, with punctuation removed and spaces collapsed."""
    return " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())


# ---------- Machine ----------


class ListenMachine:
    """Tracks what the voice loop is doing and decides what to do about each event.

    Only one transcription and one reply run at a time. A reply that is cancelled keeps its
    partial text in history, and the next reply waits until the cancelled one has finished.
    """

    def __init__(
        self,
        barge_in_ms: int,
        speech_busy: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        *,
        confirm_low_confidence: bool = False,
        mic_retry_s: float = MIC_RETRY_S,
    ) -> None:
        self._barge_in_ms = barge_in_ms
        self._speech_busy = speech_busy or (lambda: False)
        self._clock = clock
        self._confirm_low = confirm_low_confidence
        self._mic_retry_s = mic_retry_s
        self._last_cancel: float | None = None
        self.paused = False
        self.speaking = False
        self.mic_failed = False
        self._next_mic_retry = 0.0
        self._transcribing = False
        self._pending: deque[np.ndarray] = deque()
        self._reply_turn: int | None = None
        self._reply_cancelled = False
        self._deferred_text: str | None = None
        self._deferred_low = False
        self._deferred_reset = False
        self._awaiting_confirm = False
        self._turns = 0

    @property
    def state(self) -> State:
        if self.paused:
            return State.PAUSED
        if self._reply_turn is not None:
            return State.THINKING
        if self._transcribing:
            return State.TRANSCRIBING
        if self.speaking:
            return State.RECORDING
        return State.LISTENING

    @property
    def reply_running(self) -> bool:
        return self._reply_turn is not None

    @property
    def awaiting_confirm(self) -> bool:
        return self._awaiting_confirm

    def start(self) -> list[Action]:
        return [StartMic(), Print("Listening. Speak anytime. Press Enter to pause. Ctrl+C to quit.")]

    def handle(self, event: Event) -> list[Action]:
        match event:
            case Toggle():
                return self._toggle()
            case Quit():
                actions: list[Action] = self._cancel_reply() if self.reply_running else []
                return [*actions, StopMic(), Shutdown()]
            case Answer(answer=answer):
                if not self._awaiting_confirm:
                    return []
                self._awaiting_confirm = False
                return [Print("Allowed." if answer else "Declined."), ResolveConfirmation(answer)]
            case Cancel():
                now = self._clock()
                repeated = self._last_cancel is not None and now - self._last_cancel <= DOUBLE_CANCEL_S
                if not self._awaiting_confirm or repeated:
                    return self.handle(Quit())
                self._awaiting_confirm = False
                self._last_cancel = now
                return [
                    Print("Confirmation cancelled. Declined. Press Ctrl+C again within 2 seconds to quit."),
                    ResolveConfirmation(False),
                ]
            case Typed(text=text):
                return self._user_text(text)
            case Command(text=text):
                return self._command(text)
            case MicFailed(message=message):
                self.paused = True
                self.speaking = False
                self._pending.clear()
                self._next_mic_retry = self._clock() + self._mic_retry_s
                if self.mic_failed:
                    return [StopMic()]
                self.mic_failed = True
                return [
                    StopMic(),
                    Print(
                        f"Microphone error: {message}. Typing still works. "
                        f"Retrying every {self._mic_retry_s:g} seconds. "
                        "Press Enter to try the microphone now."
                    ),
                ]
            case MicReady():
                if not self.mic_failed:
                    return []
                self.mic_failed = False
                self.paused = False
                return [Print("Microphone is back. Listening again.")]
            case Tick():
                if self.mic_failed and self._clock() >= self._next_mic_retry:
                    self._next_mic_retry = self._clock() + self._mic_retry_s
                    return [StartMic()]
                return []
            case SpeechOngoing(voiced_ms=voiced_ms):
                return self._speech_ongoing(voiced_ms)
            case Noise():
                self.speaking = False
                return []
            case Utterance(audio=audio, cut=cut):
                return self._utterance(audio, cut)
            case Transcript():
                return self._transcript(event)
            case ReplyDelta(turn=turn, text=text):
                if turn == self._reply_turn and not self._reply_cancelled:
                    return [Stream(text)]
                return []
            case ReplyDone(turn=turn, outcome=outcome):
                return self._reply_finished(turn, [ApplyOutcome(outcome)])
            case ReplyFailed(turn=turn, message=message):
                return self._reply_finished(
                    turn, [Print(f"Error: {message}", spoken=SPOKEN_FAILURE)]
                )
            case ConfirmRequest(question=question, timeout_s=timeout_s):
                self._awaiting_confirm = True
                timeout_note = (
                    f" No answer within {timeout_s:g} seconds counts as no." if timeout_s else ""
                )
                return [
                    Print(
                        f"Confirm: {question}\nSay yes or no, press y or n, or Ctrl+C to cancel."
                        + timeout_note
                    )
                ]
            case ConfirmTimedOut():
                self._awaiting_confirm = False
                return [Print("No answer, so the action was cancelled.")]
            case Notice(text=text, spoken=spoken):
                return [Print(text, spoken=spoken)]
        raise TypeError(f"unknown event: {event!r}")

    # ----- handlers -----

    def _toggle(self) -> list[Action]:
        if self._awaiting_confirm:
            # Enter never answers a confirmation and never pauses while one is pending.
            return [Print("Answer yes or no first.")]
        if self.paused:
            self.paused = False
            self.mic_failed = False
            return [StartMic(), Print("Listening again.")]
        self.paused = True
        self.speaking = False
        self._pending.clear()
        self._deferred_text = None
        actions: list[Action] = [StopMic(), Print("Paused. Press Enter to listen again.")]
        if self.reply_running:
            actions += self._cancel_reply()
        if self._speech_busy():
            actions.append(StopSpeech())
        return actions

    def _command(self, text: str) -> list[Action]:
        command = text.strip().lower()
        if command in ("/exit", "/quit"):
            return self.handle(Quit())
        if self._awaiting_confirm:
            return [Print("Please answer yes or no.")]
        if command == "/reset" or command == "/new":
            if self.reply_running:
                self._deferred_reset = True
                self._deferred_text = None
                return self._cancel_reply()
            return [ResetHistory()]
        if command == "/status":
            return [ShowStatus()]
        return [RunCommand(text)]

    def _user_text(self, text: str, low_confidence: bool = False) -> list[Action]:
        if not text.strip():
            return []
        if self._awaiting_confirm:
            answer = parse_yes_no(text)
            if answer is None:
                return [Print("Please answer yes or no.")]
            self._awaiting_confirm = False
            return [ResolveConfirmation(answer)]
        if _phrase(text) in RESET_PHRASES:
            return self._command("/new")
        if self.reply_running:
            actions: list[Action] = [] if self._reply_cancelled else self._cancel_reply()
            self._deferred_text = text
            self._deferred_low = low_confidence
            return actions
        return self._start_reply(text, low_confidence)

    def _speech_ongoing(self, voiced_ms: int) -> list[Action]:
        if self.paused:
            return []
        self.speaking = True
        if self._awaiting_confirm or voiced_ms < self._barge_in_ms:
            return []
        reply_active = self.reply_running and not self._reply_cancelled
        talking = self._speech_busy()
        if not reply_active and not talking:
            return []
        actions: list[Action] = [StopSpeech()] if talking else []
        if reply_active:
            actions += self._cancel_reply()
        return actions

    def _utterance(self, audio: np.ndarray, cut: bool) -> list[Action]:
        self.speaking = False
        if self.paused:
            return []
        actions: list[Action] = []
        if cut:
            actions.append(Print("(A long stretch of speech was split into two parts.)"))
        if self._transcribing:
            self._pending.append(audio)
        else:
            self._transcribing = True
            actions.append(Transcribe(audio))
        return actions

    def _transcript(self, event: Transcript) -> list[Action]:
        self._transcribing = False
        if self.paused:
            return []
        actions: list[Action] = []
        if event.error is not None:
            actions.append(Print(f"Transcription error: {event.error}"))
        elif not event.text.strip():
            if not self._awaiting_confirm:
                actions.append(Print("Could not make out any words."))
            else:
                actions.append(Print("Please answer yes or no."))
        elif self._awaiting_confirm:
            actions += self._voice_answer(event)
        else:
            flag = " (low confidence)" if event.low_confidence else ""
            actions.append(Print(f'Heard{flag}: "{event.text}"'))
            actions += self._user_text(event.text, event.low_confidence)
        if self._pending and not self.paused:
            self._transcribing = True
            actions.append(Transcribe(self._pending.popleft()))
        return actions

    def _voice_answer(self, event: Transcript) -> list[Action]:
        """A spoken answer to a confirmation. Only a clear, confident yes or no counts.

        Ambiguous or low-confidence speech declines the action rather than guessing.
        """
        self._awaiting_confirm = False
        answer = parse_yes_no(event.text)
        if answer is None:
            return [
                Print(f'"{event.text}" is not a clear yes or no, so the action was declined.'),
                ResolveConfirmation(False),
            ]
        if event.low_confidence:
            return [
                Print(f'Heard "{event.text}" with low confidence, so the action was declined.'),
                ResolveConfirmation(False),
            ]
        return [Print(f'Heard: "{event.text}"'), ResolveConfirmation(answer)]

    def _reply_finished(self, turn: int, actions: list[Action]) -> list[Action]:
        if turn != self._reply_turn:
            return []
        self._reply_turn = None
        self._reply_cancelled = False
        if self._deferred_reset:
            self._deferred_reset = False
            actions = [*actions, ResetHistory()]
        if self._deferred_text is not None:
            text, self._deferred_text = self._deferred_text, None
            low, self._deferred_low = self._deferred_low, False
            if not self.paused:
                actions = [*actions, *self._start_reply(text, low)]
        return actions

    def _start_reply(self, text: str, low_confidence: bool = False) -> list[Action]:
        self._turns += 1
        self._reply_turn = self._turns
        self._reply_cancelled = False
        return [
            StartReply(
                self._reply_turn,
                text,
                low_confidence=self._confirm_low and low_confidence,
                unsure=low_confidence,
            )
        ]

    def _cancel_reply(self) -> list[Action]:
        self._reply_cancelled = True
        return [CancelReply()]
