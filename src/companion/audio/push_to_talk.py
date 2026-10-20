"""Push-to-talk: Enter starts recording, Enter stops it, then trim silence and transcribe."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from companion.audio.capture import SAMPLE_RATE, Recorder, resolve_input_device
from companion.audio.errors import SpeechError
from companion.audio.stt import Transcriber, make_transcriber
from companion.audio.vad import SpeechTrimmer
from companion.config import AudioConfig


class PushToTalk:
    """Records one utterance per call to listen() and returns its transcript.

    Returns an empty string when nothing usable was captured, and says why.
    All collaborators are injectable so the flow can be tested without hardware.
    """

    def __init__(
        self,
        recorder: Recorder,
        trimmer: SpeechTrimmer,
        transcriber: Transcriber,
        min_seconds: float = 0.3,
        wait_for_enter: Callable[[str], str] = input,
        notify: Callable[[str], None] = print,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._recorder = recorder
        self._trimmer = trimmer
        self._transcriber = transcriber
        self._min_seconds = min_seconds
        self._wait_for_enter = wait_for_enter
        self._notify = notify
        self._sample_rate = sample_rate

    def listen(self) -> str:
        self._notify("Listening... press Enter to stop.")
        try:
            self._recorder.start()
        except SpeechError as exc:
            self._notify(f"Microphone error: {exc}")
            return ""

        try:
            self._wait_for_enter("")
        except (EOFError, KeyboardInterrupt):
            self._discard()
            self._notify("Recording cancelled.")
            return ""

        try:
            audio = self._recorder.stop()
        except SpeechError as exc:
            self._notify(f"Microphone error: {exc}")
            return ""

        seconds = audio.size / self._sample_rate
        if seconds < self._min_seconds:
            self._notify(f"Recording too short ({seconds:.1f}s). Try again.")
            return ""

        try:
            speech = self._trimmer.trim(audio)
        except Exception as exc:
            self._notify(f"Voice detection error: {exc}")
            return ""
        if speech.size == 0:
            self._notify("No speech detected.")
            return ""

        try:
            text = self._transcriber.transcribe(speech)
        except SpeechError as exc:
            self._notify(f"Transcription error: {exc}")
            return ""
        if not text:
            self._notify("Could not make out any words.")
        return text

    def _discard(self) -> None:
        if self._recorder.recording:
            try:
                self._recorder.stop()
            except SpeechError:
                pass


def build_push_to_talk(
    config: AudioConfig,
    notify: Callable[[str], None] = print,
    apps_file: Path | None = None,
) -> PushToTalk:
    """Build the speech pipeline and load its models now.

    Raises SpeechError if the models cannot be loaded, so the caller can stay in text mode.
    """
    recorder = Recorder(device=resolve_input_device(config.input_device))
    trimmer = SpeechTrimmer(threshold=config.vad_threshold)
    transcriber = make_transcriber(config, apps_file)
    notify("Loading speech models (the first run downloads Whisper)...")
    try:
        trimmer.load()
        transcriber.load()
    except SpeechError:
        raise
    except Exception as exc:
        raise SpeechError(f"could not load voice detection: {exc}") from exc
    notify(f"Speech models ready (Whisper {config.whisper_model} on {transcriber.device}).")
    return PushToTalk(
        recorder,
        trimmer,
        transcriber,
        min_seconds=config.min_seconds,
        notify=notify,
    )
