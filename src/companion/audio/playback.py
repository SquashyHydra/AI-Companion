"""Speaker output: WASAPI device selection and a gap-free audio player.

The output stream stays open for the whole session. Its callback pulls queued samples, so
consecutive sentences play back to back with no gap. A mark queued after a sentence is handed
back by take_completed() or flush() once the callback has reached it.

The stream factory and clock are injectable so tests can run without a speaker.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from companion.audio.capture import _devices, _host_apis, _wasapi_index
from companion.audio.errors import SpeechError

logger = logging.getLogger("companion.audio")

OutputStreamFactory = Callable[..., Any]
Clock = Callable[[], float]


@dataclass(frozen=True)
class OutputDeviceInfo:
    """Details of one WASAPI speaker, as shown by the diagnostic."""

    index: int
    name: str
    default_rate: int
    max_channels: int


def describe_output_devices() -> list[OutputDeviceInfo]:
    """Return WASAPI output devices only, in PortAudio index order."""
    api = _wasapi_index()
    if api is None:
        return []
    return [
        OutputDeviceInfo(
            index=index,
            name=str(device["name"]).strip(),
            default_rate=int(round(float(device["default_samplerate"]))),
            max_channels=int(device["max_output_channels"]),
        )
        for index, device in enumerate(_devices())
        if device["hostapi"] == api and device["max_output_channels"] > 0
    ]


def list_output_devices() -> list[tuple[int, str]]:
    return [(device.index, device.name) for device in describe_output_devices()]


def resolve_output_device(value: int | str | None) -> int:
    """Map a config value (None, a device number, or a name fragment) to a WASAPI output index."""
    devices = list_output_devices()
    if not devices:
        raise SpeechError("no WASAPI speakers found")
    if value is None:
        api = _wasapi_index()
        default = _host_apis()[api]["default_output_device"] if api is not None else -1
        return default if any(index == default for index, _ in devices) else devices[0][0]
    if isinstance(value, int):
        if any(index == value for index, _ in devices):
            return value
        raise SpeechError(
            f"device {value} is not a WASAPI output; run python -m companion.diagnose_tts to list them"
        )
    needle = value.lower()
    matches = [index for index, name in devices if needle in name.lower()]
    if not matches:
        raise SpeechError(f"no WASAPI speaker matches '{value}'; run python -m companion.diagnose_tts")
    if len(matches) > 1:
        raise SpeechError(f"'{value}' matches {len(matches)} speakers; use a number instead")
    return matches[0]


def _default_output_factory(**kwargs: Any) -> Any:
    import sounddevice as sd

    device = kwargs.get("device")
    if isinstance(device, int) and _devices()[device]["hostapi"] == _wasapi_index():
        # WASAPI shared mode can only run at the device's mix rate unless Windows converts it.
        kwargs["extra_settings"] = sd.WasapiSettings(auto_convert=True)
    return sd.OutputStream(**kwargs)


@dataclass(frozen=True)
class _Mark:
    payload: object


class SpeakerPlayer:
    """Plays mono float32 audio in the order it was written.

    write() is called from the synthesis thread. The audio callback runs on the device's thread
    and only moves samples and marks. Everything shared is guarded by one lock.
    """

    def __init__(
        self,
        sample_rate: int,
        device: int | None = None,
        stream_factory: OutputStreamFactory | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._sample_rate = sample_rate
        self._device = device
        self._stream_factory = stream_factory or _default_output_factory
        self._clock = clock
        self._stream: Any = None
        self._lock = threading.Lock()
        self._items: deque[np.ndarray | _Mark] = deque()
        self._completed: list[object] = []
        self._audible_until = 0.0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            stream = self._stream_factory(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
                latency="low",
                callback=self._on_audio,
            )
            stream.start()
        except Exception as exc:
            raise SpeechError(f"could not open the speaker output: {describe_output_error(exc)}") from exc
        self._stream = stream

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.warning("error while closing the speaker output", exc_info=True)
        with self._lock:
            self._items.clear()

    def write(self, samples: np.ndarray, mark: object = None) -> None:
        """Queue audio after everything already queued. mark is reported once its audio has played."""
        audio = np.asarray(samples, dtype=np.float32).ravel()
        with self._lock:
            if audio.size:
                self._items.append(audio)
            if mark is not None:
                self._items.append(_Mark(mark))

    def flush(self) -> list[object]:
        """Drop everything still queued. Returns the marks of audio that already played, in order."""
        with self._lock:
            self._items.clear()
            self._audible_until = min(self._audible_until, self._clock())
            completed, self._completed = self._completed, []
        return completed

    def take_completed(self) -> list[object]:
        with self._lock:
            completed, self._completed = self._completed, []
        return completed

    def busy(self, tail_s: float = 0.0) -> bool:
        """True while audio is queued or still sounding, plus tail_s after the last sample."""
        with self._lock:
            if self._items:
                return True
            return self._clock() < self._audible_until + tail_s

    def _on_audio(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        # Runs on the audio thread. Keep it short and never raise.
        if status:
            logger.debug("speaker stream status: %s", status)
        filled = 0
        with self._lock:
            while filled < frames and self._items:
                head = self._items[0]
                if isinstance(head, _Mark):
                    self._items.popleft()
                    self._completed.append(head.payload)
                    continue
                take = min(frames - filled, head.size)
                outdata[filled : filled + take, 0] = head[:take]
                if take == head.size:
                    self._items.popleft()
                else:
                    self._items[0] = head[take:]
                filled += take
            if filled:
                self._audible_until = self._clock() + filled / self._sample_rate
        outdata[filled:, 0] = 0.0


def describe_output_error(exc: Exception) -> str:
    text = str(exc)
    if "-9997" in text:
        return "the device rejected the sample rate or format; check that it is a WASAPI output"
    if "-9985" in text or "-9996" in text:
        return "the speaker is unavailable or was disconnected"
    return text or type(exc).__name__
