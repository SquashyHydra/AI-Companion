"""Errors raised by the speech input pipeline."""


class SpeechError(RuntimeError):
    """Raised when microphone capture, VAD, or transcription cannot run."""


class SupersededError(SpeechError):
    """A synthesis request was dropped because a barge-in cancelled it. Not a failure."""
