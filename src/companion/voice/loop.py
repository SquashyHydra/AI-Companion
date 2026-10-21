"""The continuous voice loop.

The main thread owns the screen, the keyboard, and the state machine. Worker threads (the
listening pipeline, transcription, and LLM replies) only post events to one queue. The main
thread drains that queue, so no two threads write to the screen at the same time.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from companion.audio.capture import SAMPLE_RATE, FrameStream, resolve_input_device
from companion.audio.errors import SpeechError
from companion.audio.pipeline import ListeningPipeline
from companion.audio.playback import SpeakerPlayer, resolve_output_device
from companion.audio.records import TranscriptLog, save_utterance_wav
from companion.audio.segmenter import Segmenter
from companion.audio.stt import Transcription, make_transcriber
from companion.audio.tts import load_voice
from companion.audio.vad import SileroFrameDetector
from companion.audio.wakeword import FrameDetect, WakeGate, WakeWordDetector
from companion.config import PROJECT_ROOT, AppConfig, AudioConfig
from companion.conversation import Conversation
from companion.llm.client import INTERRUPTED_MARK, LOW_CONFIDENCE_NOTE, LLMClient, LLMError, Messages, TurnOutcome
from companion.notices import notices
from companion.tools.registry import ToolRegistry
from companion.voice.confirm import EventConfirmer
from companion.voice.keys import KeyReader, LineEditor, NoKeys
from companion.voice.machine import (
    ApplyOutcome,
    CancelReply,
    Event,
    ListenMachine,
    MicFailed,
    MicReady,
    Notice,
    Print,
    Quit,
    ReplyDelta,
    ReplyDone,
    ReplyFailed,
    ResetHistory,
    ResolveConfirmation,
    RunCommand,
    Shutdown,
    ShowStatus,
    StartMic,
    StartReply,
    Stream,
    StopMic,
    StopSpeech,
    Tick,
    Toggle,
    Transcribe,
    Transcript,
    WakeHeard,
)
from companion.voice.speech import SpeechOut
from companion.voice.tags import TagDisplayFilter

logger = logging.getLogger("companion.voice")

TICK_S = 0.05
"""How long the main loop waits for an event before polling the keyboard again."""
JOIN_S = 1.0
"""How long shutdown waits for a worker thread. Workers are daemon threads, so this is only a courtesy."""

StreamFn = Callable[..., TurnOutcome]
"""(history, text, cancel, on_delta, *, confirm_note=None) -> TurnOutcome."""
TranscribeFn = Callable[[np.ndarray], Transcription]


def write_stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


class VoiceLoop:
    """Runs until Ctrl+C, /exit, or /quit. All collaborators are injectable for tests."""

    def __init__(
        self,
        *,
        machine: ListenMachine,
        pipeline: ListeningPipeline,
        transcribe: TranscribeFn,
        stream: StreamFn,
        run_command: Callable[[str], str],
        new_history: Callable[[], Messages],
        keys: KeyReader,
        editor: LineEditor,
        events: queue.Queue[Event],
        answers: queue.Queue[bool],
        low_confidence_logprob: float,
        transcript_log: TranscriptLog | None = None,
        utterance_dir: Path | None = None,
        write: Callable[[str], None] = write_stdout,
        tick_s: float = TICK_S,
        stop: threading.Event | None = None,
        speech: SpeechOut | None = None,
        show_voice_tags: bool = True,
        conversation: Conversation | None = None,
        follow_up: Callable[[], None] | None = None,
    ) -> None:
        self._machine = machine
        self._pipeline = pipeline
        self._transcribe = transcribe
        self._stream = stream
        self._run_command = run_command
        self._new_history = new_history
        self._conversation = conversation
        self._keys = keys
        self._editor = editor
        self._events = events
        self._answers = answers
        self._low_confidence_logprob = low_confidence_logprob
        self._transcript_log = transcript_log
        self._utterance_dir = utterance_dir
        self._write = write
        self._tick_s = tick_s
        self._stop = stop or threading.Event()
        self._speech = speech
        engine_tags = getattr(speech, "tags", frozenset()) if speech is not None else frozenset()
        self._tags = TagDisplayFilter(show_voice_tags, engine_tags)

        self._history: Messages = new_history()
        self._cancel: threading.Event | None = None
        self._workers: list[threading.Thread] = []
        self._line_open = False
        self._streamed = False
        self._done = False
        # Speech state for the current reply. _cut is the text spoken before speech stopped.
        self._cut: str | None = None
        self._reply_in_history = False
        self._listening_listener: Callable[[bool], None] | None = None
        self._reported_listening: bool | None = None
        # Called once each time a reply finishes playing, so the wake word can be skipped for a while.
        self._follow_up = follow_up
        self._playing = False

    @property
    def done(self) -> bool:
        return self._done

    def request_quit(self) -> None:
        """Ask the loop to quit. Safe to call from any thread, such as the tray icon."""
        self._events.put(Quit())

    def request_toggle(self) -> None:
        """Pause or resume listening, as Enter does. Safe to call from any thread, such as the tray icon."""
        self._events.put(Toggle())

    def set_listening_listener(self, listener: Callable[[bool], None] | None) -> None:
        """Be told (on the main thread) whether listening is on, each time that changes."""
        self._listening_listener = listener

    def run(self) -> int:
        self._dispatch(self._machine.start())
        self._editor.redraw()
        try:
            while not self._done:
                self._watch_playback()
                for event in self._keys.poll(confirming=self._machine.awaiting_confirm):
                    self._handle(event)
                    if self._done:
                        break
                if self._done:
                    break
                try:
                    event = self._events.get(timeout=self._tick_s)
                except queue.Empty:
                    self._handle(Tick())
                    self._show_notices()
                    continue
                self._handle(event)
        except KeyboardInterrupt:
            self._handle(Quit())
        finally:
            self._shutdown()
        return 0

    # ----- dispatch -----

    def _watch_playback(self) -> None:
        """Open the follow-up window when a reply stops playing (and listening is on)."""
        playing = self._machine.reply_running or (self._speech is not None and self._speech.speaking())
        if self._playing and not playing and self._follow_up is not None and not self._machine.paused:
            self._follow_up()
        self._playing = playing

    # ----- dispatch -----

    def _handle(self, event: Event) -> None:
        try:
            self._dispatch(self._machine.handle(event))
        except Exception:
            logger.exception("voice loop error while handling %r", event)
            self._print("Something went wrong just now. The details are in the log. Carrying on.")

    def _dispatch(self, actions: list) -> None:
        for action in actions:
            self._perform(action)
        self._report_listening()

    def _report_listening(self) -> None:
        listening = not self._machine.paused
        if listening == self._reported_listening or self._listening_listener is None:
            return
        self._reported_listening = listening
        try:
            self._listening_listener(listening)
        except Exception:
            logger.exception("listening listener failed")

    def _perform(self, action: object) -> None:
        match action:
            case StartMic():
                self._start_mic()
            case StopMic():
                self._pipeline.stop()
            case Transcribe(audio=audio):
                self._spawn(self._transcribe_worker, audio)
            case StartReply(turn=turn, text=text, low_confidence=low, unsure=unsure):
                self._start_reply(turn, text, low, unsure)
            case CancelReply():
                if self._cancel is not None:
                    self._cancel.set()
                self._stop_speech()
            case StopSpeech():
                self._stop_speech()
            case Notice(text=text, spoken=spoken):
                self._print(text)
                if spoken and self._speech is not None:
                    self._speech.feed(spoken)
            case Stream(text=text):
                self._write(self._tags.feed(text))
                self._line_open = True
                self._streamed = True
                if self._speech is not None:
                    self._speech.feed(text)
            case ApplyOutcome(outcome=outcome):
                self._apply(outcome)
            case ResetHistory():
                self._history = self._new_history()
                self._print("Conversation cleared.")
            case ShowStatus():
                if self._conversation is not None:
                    self._print(self._conversation.status(self._history))
                else:
                    self._print("No status available.")
            case RunCommand(text=text):
                output = self._run_command(text)
                if output:
                    self._print(output)
            case Print(text=text, spoken=spoken):
                self._print(text)
                if spoken and self._speech is not None:
                    self._speech.feed(spoken)
                    self._speech.finish()
            case ResolveConfirmation(answer=answer):
                self._answers.put(answer)
            case Shutdown():
                self._done = True
            case _:
                raise TypeError(f"unknown action: {action!r}")

    # ----- actions -----

    def _start_mic(self) -> None:
        try:
            self._pipeline.start()
        except SpeechError as exc:
            self._events.put(MicFailed(str(exc)))
            return
        except Exception as exc:
            logger.exception("microphone failed to start")
            self._events.put(MicFailed(str(exc) or type(exc).__name__))
            return
        self._events.put(MicReady())

    def _start_reply(
        self, turn: int, text: str, low_confidence: bool = False, unsure: bool = False
    ) -> None:
        self._stop_speech()
        self._cut = None
        self._reply_in_history = False
        self._tags.flush()
        if self._speech is not None:
            self._speech.begin_reply()
        if self._conversation is not None:
            self._history = self._conversation.apply(self._history)
        cancel = threading.Event()
        self._cancel = cancel
        self._streamed = False
        self._editor.clear_line()
        self._write("Companion> ")
        self._line_open = True
        note = LOW_CONFIDENCE_NOTE if low_confidence else None
        self._spawn(self._reply_worker, turn, text, self._history, cancel, note, unsure)

    def _apply(self, outcome: TurnOutcome) -> None:
        self._history = outcome.messages
        pending_tag_text = self._tags.flush()
        if pending_tag_text:
            self._write(pending_tag_text)
        if outcome.interrupted:
            self._write(" (interrupted)\n")
            if self._speech is not None:
                self._stop_speech()
                self._mark_interrupted(self._cut or "")
        else:
            if self._speech is not None:
                self._speech.finish()
                self._reply_in_history = True
            if not self._streamed:
                self._write("(no reply)\n")
            else:
                self._write("\n")
            if self._conversation is not None:
                self._conversation.after_turn(self._history)
        self._line_open = False
        self._cancel = None
        self._editor.redraw()

    def _stop_speech(self) -> None:
        """Stop speaking once per reply. Keeps only the sentences that finished playing."""
        if self._speech is None or self._cut is not None:
            return
        spoken, was_busy = self._speech.interrupt()
        self._cut = spoken
        if was_busy and self._reply_in_history:
            self._mark_interrupted(spoken)
            self._print("(stopped speaking)")

    def _mark_interrupted(self, spoken: str) -> None:
        """Make the history match what was actually said, so the model does not assume I heard the rest."""
        if self._history and self._history[-1].get("role") == "assistant":
            self._history[-1]["content"] = f"{spoken} {INTERRUPTED_MARK}".strip()
        self._reply_in_history = False

    def _print(self, text: str) -> None:
        if self._line_open:
            self._write("\n")
            self._line_open = False
        self._editor.show(text)

    def _show_notices(self) -> None:
        """Show timer and reminder messages. Speak them only when no reply is in progress."""
        for text in notices.drain():
            self._print(text)
            if self._speech is not None and not self._machine.reply_running:
                self._speech.feed(text)
                self._speech.finish()

    # ----- worker threads -----

    def _spawn(self, target: Callable[..., None], *args: object) -> None:
        worker = threading.Thread(target=target, args=args, name="voice-worker", daemon=True)
        self._workers = [item for item in self._workers if item.is_alive()]
        self._workers.append(worker)
        worker.start()

    def _transcribe_worker(self, audio: np.ndarray) -> None:
        try:
            result = self._transcribe(audio)
        except Exception as exc:
            if not isinstance(exc, SpeechError):
                logger.exception("transcription failed")
            self._events.put(Transcript(error=str(exc) or type(exc).__name__))
            return
        seconds = audio.size / SAMPLE_RATE
        low_confidence = result.low_confidence(self._low_confidence_logprob)
        self._record(audio, result, seconds, low_confidence)
        self._events.put(Transcript(text=result.text, low_confidence=low_confidence))

    def _record(self, audio: np.ndarray, result: Transcription, seconds: float, low: bool) -> None:
        try:
            if self._transcript_log is not None:
                self._transcript_log.append(transcription=result, seconds=seconds, low_confidence=low)
            if self._utterance_dir is not None:
                save_utterance_wav(audio, self._utterance_dir)
        except Exception:
            logger.warning("could not save the debug record", exc_info=True)

    def _reply_worker(
        self,
        turn: int,
        text: str,
        history: Messages,
        cancel: threading.Event,
        confirm_note: str | None = None,
        unsure: bool = False,
    ) -> None:
        def on_delta(chunk: str) -> None:
            self._events.put(ReplyDelta(turn, chunk))

        # Only pass optional keywords when set, so simple stream functions keep working.
        extra: dict[str, object] = {"confirm_note": confirm_note} if confirm_note else {}
        if unsure:
            extra["low_confidence"] = True
        try:
            outcome = self._stream(history, text, cancel, on_delta, **extra)
        except LLMError as exc:
            self._events.put(ReplyFailed(turn, str(exc)))
            return
        except Exception as exc:
            logger.exception("reply failed")
            self._events.put(ReplyFailed(turn, f"unexpected error: {exc}"))
            return
        self._events.put(ReplyDone(turn, outcome))

    def _shutdown(self) -> None:
        # Set first so a pending confirmation returns at once instead of waiting out its timeout.
        self._stop.set()
        self._pipeline.stop()
        if self._cancel is not None:
            self._cancel.set()
        for worker in self._workers:
            worker.join(timeout=JOIN_S)
        if self._speech is not None:
            self._speech.close()


def build_voice_loop(
    config: AppConfig,
    llm: LLMClient,
    registry: ToolRegistry,
    run_command: Callable[[str], str],
    new_history: Callable[[], Messages],
    notify: Callable[[str], None] = print,
    conversation: Conversation | None = None,
    keyboard: bool = True,
) -> VoiceLoop:
    """Load the speech models and build the loop.

    Raises SpeechError if the models or the microphone cannot be set up, so the caller can
    fall back to typing. Also points the tool registry at the loop's confirmation prompt.
    keyboard=False (background mode) reads no keys; the tray icon is then the only control.
    """
    audio = config.audio
    events: queue.Queue[Event] = queue.Queue()
    answers: queue.Queue[bool] = queue.Queue()
    stop = threading.Event()

    transcriber = make_transcriber(audio, config.tools.apps_file)
    detector = SileroFrameDetector()
    wake_cfg = config.input.wake_word
    wake = WakeWordDetector(wake_cfg.model, wake_cfg.threshold) if wake_cfg.enabled else None
    notify("Loading speech models (the first run downloads Whisper)...")
    try:
        detector.load()
        transcriber.load()
        if wake is not None:
            wake.load()
    except SpeechError:
        raise
    except Exception as exc:
        raise SpeechError(f"could not load voice detection: {exc}") from exc
    notify(f"Speech models ready (Whisper {audio.whisper_model} on {transcriber.device}).")

    speech = _build_speech(config, events, notify) if audio.speak_replies else None
    speakers = audio.output_mode == "speakers"
    tail_s = audio.speakers_tail_ms / 1000 if speakers else 0.0
    # Speakers can be heard by the mic, so audio is gated while it plays and for a short tail.
    # Voice barge-in is therefore off on speakers; the wake word still interrupts.
    gate = (lambda: speech.speaking(tail_s)) if speech is not None and speakers else None

    listen_detect: FrameDetect = detector
    reset_listen = detector.reset
    wake_gate: WakeGate | None = None
    if wake is not None:
        wake_gate = WakeGate(
            detector,
            wake,
            window_s=wake_cfg.window_s,
            reset_detect=detector.reset,
            # Headphones: keep the gate open while talking, so voice barge-in can hear you.
            hold=(lambda: speech.speaking()) if speech is not None and not speakers else None,
            speech_level=audio.vad_threshold,
            on_wake=lambda: events.put(WakeHeard()),
        )
        listen_detect = wake_gate
        reset_listen = wake_gate.reset
        notify(
            f"Wake word on: say '{wake_cfg.model.replace('_', ' ')}' first, then your request "
            f"(listening for {wake_cfg.window_s:g} s after each wake word)."
        )

    muted_frame: Callable[[np.ndarray], None] | None = None
    if wake_gate is not None and speakers:
        muted_frame = wake_gate.muted
    follow_up: Callable[[], None] | None = None
    if wake_gate is not None and wake_cfg.follow_up_s > 0:
        follow_up = lambda: wake_gate.open_for(wake_cfg.follow_up_s)

    try:
        stream = FrameStream(device=resolve_input_device(audio.input_device))
        segmenter = Segmenter(
            vad_threshold=audio.vad_threshold,
            pre_roll_ms=audio.pre_roll_ms,
            min_speech_ms=audio.min_speech_ms,
            end_silence_ms=audio.end_silence_ms,
            max_utterance_s=audio.max_utterance_s,
        )
        pipeline = ListeningPipeline(
            source=stream,
            detect=listen_detect,
            segmenter=segmenter,
            on_event=events.put,
            on_error=lambda message: events.put(MicFailed(message)),
            reset_detector=reset_listen,
            gate=gate,
            muted_frame=muted_frame,
        )

        registry.set_confirmer(
            EventConfirmer(
                events.put,
                answers,
                timeout_s=config.input.confirm_timeout_s,
                stop=stop,
            )
        )
    except BaseException:
        if speech is not None:
            speech.close()
        raise
    editor = LineEditor(write_stdout)

    def stream_turn(
        history: Messages,
        text: str,
        cancel: threading.Event,
        on_delta: Callable[[str], None],
        *,
        confirm_note: str | None = None,
        low_confidence: bool = False,
    ) -> TurnOutcome:
        if conversation is not None:
            return conversation.stream(
                history,
                text,
                cancel,
                on_delta,
                confirm_note=confirm_note,
                low_confidence=low_confidence,
            )
        return llm.stream_turn(
            history, text, cancel=cancel, on_delta=on_delta, confirm_note=confirm_note
        )

    if conversation is not None:
        conversation.set_announce(lambda text, spoken: events.put(Notice(text, spoken=spoken)))

    return VoiceLoop(
        machine=ListenMachine(
            barge_in_ms=audio.barge_in_ms,
            speech_busy=(lambda: speech.speaking()) if speech is not None else None,
            confirm_low_confidence=config.input.confirm_low_confidence_tools,
            enter_pauses=keyboard,
            # Voice barge-in needs headphones. On speakers the mic would hear the reply itself.
            vad_barge_in=not speakers,
        ),
        pipeline=pipeline,
        transcribe=transcriber.transcribe_detailed,
        stream=stream_turn,
        run_command=run_command,
        new_history=new_history,
        keys=KeyReader(editor) if keyboard else NoKeys(),
        editor=editor,
        events=events,
        answers=answers,
        low_confidence_logprob=audio.low_confidence_logprob,
        transcript_log=TranscriptLog(PROJECT_ROOT / "logs" / "transcripts.jsonl"),
        utterance_dir=audio.utterance_dir if audio.save_utterances else None,
        stop=stop,
        speech=speech,
        show_voice_tags=config.display.show_voice_tags,
        conversation=conversation,
        follow_up=follow_up,
    )


def _build_speech(
    config: AppConfig, events: queue.Queue[Event], notify: Callable[[str], None]
) -> SpeechOut:
    """Load the configured voice (Chatterbox, or Piper as fallback) and open the speaker output.

    Raises SpeechError if no voice can be loaded.
    """
    audio = config.audio
    synth = load_voice(
        config.tts,
        audio,
        notify=notify,
        notice=lambda text, spoken=None: events.put(Notice(text, spoken=spoken)),
    )
    player = SpeakerPlayer(synth.sample_rate, device=resolve_output_device(audio.output_device))
    speech = SpeechOut(
        synth,
        player,
        on_error=lambda message: events.put(Notice(message)),
        first_chunk_words=config.tts.first_chunk_words,
    )
    speech.start()
    notify(f"Output: {audio.output_mode}.")
    return speech

