"""Microphone capture for push-to-talk.

The stream factory is injectable so tests can run without a microphone.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from companion.audio.errors import SpeechError

logger = logging.getLogger("companion.audio")

SAMPLE_RATE = 16000
"""Silero VAD and Whisper both expect 16 kHz mono audio."""

StreamFactory = Callable[..., Any]


WASAPI_HOST_API = "Windows WASAPI"
"""Other host APIs (MME, DirectSound, WDM-KS) list the same microphones again."""


def _host_apis() -> list[dict[str, Any]]:
    import sounddevice as sd

    return list(sd.query_hostapis())


def _devices() -> list[dict[str, Any]]:
    import sounddevice as sd

    return list(sd.query_devices())


def _wasapi_index() -> int | None:
    for index, api in enumerate(_host_apis()):
        if api["name"] == WASAPI_HOST_API:
            return index
    return None


def list_input_devices() -> list[tuple[int, str]]:
    """Return (index, name) for WASAPI microphones only."""
    api = _wasapi_index()
    if api is None:
        return []
    return [
        (index, str(device["name"]).strip())
        for index, device in enumerate(_devices())
        if device["hostapi"] == api and device["max_input_channels"] > 0
    ]


@dataclass(frozen=True)
class InputDeviceInfo:
    """Details of one WASAPI microphone, as shown by the diagnostic."""

    index: int
    name: str
    host_api: str
    default_rate: int
    max_channels: int


def describe_input_devices() -> list[InputDeviceInfo]:
    """Return details for WASAPI microphones only, in PortAudio index order."""
    api = _wasapi_index()
    if api is None:
        return []
    return [
        InputDeviceInfo(
            index=index,
            name=str(device["name"]).strip(),
            host_api=str(_host_apis()[api]["name"]),
            default_rate=int(round(float(device["default_samplerate"]))),
            max_channels=int(device["max_input_channels"]),
        )
        for index, device in enumerate(_devices())
        if device["hostapi"] == api and device["max_input_channels"] > 0
    ]


def _default_wasapi_input(devices: list[tuple[int, str]]) -> int | None:
    api = _wasapi_index()
    if api is None:
        return None
    index = _host_apis()[api]["default_input_device"]
    return index if any(index == device for device, _ in devices) else None


def resolve_input_device(value: int | str | None) -> int:
    """Map a config value (None, a /mics number, or a name fragment) to a WASAPI input index."""
    devices = list_input_devices()
    if not devices:
        raise SpeechError("no WASAPI microphones found")
    if value is None:
        default = _default_wasapi_input(devices)
        return default if default is not None else devices[0][0]
    if isinstance(value, int):
        if any(index == value for index, _ in devices):
            return value
        raise SpeechError(f"device {value} is not a WASAPI microphone; pick a number from /mics")
    needle = value.lower()
    matches = [index for index, name in devices if needle in name.lower()]
    if not matches:
        raise SpeechError(f"no WASAPI microphone matches '{value}'; see /mics")
    if len(matches) > 1:
        raise SpeechError(f"'{value}' matches {len(matches)} microphones; use a number from /mics")
    return matches[0]


def _default_stream_factory(**kwargs: Any) -> Any:
    import sounddevice as sd

    device = kwargs.get("device")
    if isinstance(device, int) and _devices()[device]["hostapi"] == _wasapi_index():
        # WASAPI shared mode rejects 16 kHz unless Windows converts the stream.
        kwargs["extra_settings"] = sd.WasapiSettings(auto_convert=True)
    return sd.InputStream(**kwargs)


class Recorder:
    """Records mono float32 audio between start() and stop()."""

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._stream_factory = stream_factory or _default_stream_factory
        self._stream: Any = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            raise SpeechError("already recording")
        with self._lock:
            self._chunks = []
        try:
            stream = self._stream_factory(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
                callback=self._on_audio,
            )
            stream.start()
        except Exception as exc:
            raise SpeechError(f"could not open the microphone: {exc}") from exc
        self._stream = stream

    def stop(self) -> np.ndarray:
        """Stop recording and return everything captured as a 1-D float32 array."""
        if self._stream is None:
            raise SpeechError("not recording")
        stream, self._stream = self._stream, None
        try:
            stream.stop()
            stream.close()
        finally:
            with self._lock:
                chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _on_audio(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        # Runs on the audio thread. Keep it short and never raise.
        if status:
            logger.warning("audio stream status: %s", status)
        with self._lock:
            self._chunks.append(np.array(indata[:, 0], dtype=np.float32, copy=True))


class FrameStream:
    """Streams fixed-size mono float32 frames from the microphone until stop() is called.

    read() is called from the listening thread. The audio callback only queues frames.
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        blocksize: int = 512,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._stream_factory = stream_factory or _default_stream_factory
        self._stream: Any = None
        self._frames: queue.Queue[np.ndarray] = queue.Queue()

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        self._drain()
        try:
            stream = self._stream_factory(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
                blocksize=self._blocksize,
                callback=self._on_audio,
            )
            stream.start()
        except SpeechError:
            raise
        except Exception as exc:
            raise SpeechError(describe_stream_error(exc)) from exc
        self._stream = stream

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            logger.warning("error while closing the microphone stream", exc_info=True)
        self._drain()

    def read(self, timeout: float) -> np.ndarray | None:
        """Return the next frame, or None if none arrived within timeout seconds."""
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def _drain(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _on_audio(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        # Runs on the audio thread. Keep it short and never raise.
        if status:
            logger.warning("audio stream status: %s", status)
        self._frames.put(np.array(indata[:, 0], dtype=np.float32, copy=True))


def describe_stream_error(exc: Exception) -> str:
    """Turn a PortAudio error into a short message the user can act on."""
    text = str(exc)
    if "-9997" in text:
        return "the device rejected the requested sample rate or format; check that it is a WASAPI input"
    if "-9985" in text or "-9996" in text:
        return "the microphone is unavailable or was disconnected"
    return f"could not open the microphone: {text}"
