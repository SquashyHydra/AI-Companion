"""Client for the Chatterbox server (companion.audio.chatterbox_server).

The server runs in .venv-tts and serves Chatterbox-Turbo or Chatterbox-Nano. This client can start
it, wait until the model is ready, send sentences to it over localhost, cancel stale sentences on
barge-in, and stop it again. Every failure is raised as SpeechError so the caller can fall back to
Piper.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, IO

import numpy as np

from companion.audio.errors import SpeechError, SupersededError

SERVER_MODULE = "companion.audio.chatterbox_server"
POLL_INTERVAL_S = 0.5
HEALTH_TIMEOUT_S = 2.0
ENGINES = ("chatterbox_turbo", "chatterbox_nano")


class ChatterboxSynthesizer:
    """Synthesizes sentences with a Chatterbox engine through the local server.

    tags is read from the model after load (the voice tags it renders, such as laugh). Nothing
    is hard-coded here, so a different model reports its own tags.
    """

    def __init__(
        self,
        *,
        python: Path,
        project_root: Path,
        engine: str = "chatterbox_turbo",
        host: str = "127.0.0.1",
        port: int = 8765,
        device: str = "cuda",
        offline: bool = True,
        voice_reference: Path | None = None,
        autostart: bool = True,
        startup_timeout_s: float = 300.0,
        request_timeout_s: float = 60.0,
        log_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        if engine not in ENGINES:
            raise SpeechError(f"unknown Chatterbox engine: {engine}")
        self._python = Path(python)
        self._project_root = Path(project_root)
        self._engine = engine
        self._base = f"http://{host}:{port}"
        self._device = device
        self._offline = offline
        self._voice_reference = voice_reference
        self._autostart = autostart
        self._startup_timeout_s = startup_timeout_s
        self._request_timeout_s = request_timeout_s
        self._log_path = log_path or self._project_root / "logs" / "chatterbox_server.log"
        self._clock = clock
        self._sleep = sleep
        self._popen = popen
        self._process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None
        # Bypass system proxy settings: this is always a localhost connection.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.sample_rate = 0
        self.tags: frozenset[str] = frozenset()
        self.gpu_peak_mb: float | None = None

    @property
    def loaded(self) -> bool:
        return self.sample_rate > 0

    def running(self) -> bool:
        """True when a server already answers at the configured address."""
        return self._health() is not None

    def refresh_gpu_peak(self) -> None:
        """Re-read the server's peak GPU memory, which grows with each sentence it generates."""
        health = self._health()
        if health is not None:
            self.gpu_peak_mb = health.get("gpu_peak_mb")

    def load(self) -> None:
        """Start the server if needed and wait until its model is ready."""
        if self._health() is None:
            if not self._autostart:
                raise SpeechError(
                    f"no Chatterbox server at {self._base}. Start it, or set "
                    "tts.chatterbox.autostart to true"
                )
            self._start_process()
        deadline = self._clock() + self._startup_timeout_s
        while True:
            health = self._health()
            if health is not None:
                if health.get("status") == "ready":
                    self.sample_rate = int(health.get("sample_rate", 0))
                    if self.sample_rate <= 0:
                        raise SpeechError("Chatterbox server reported no sample rate")
                    self.tags = frozenset(str(tag) for tag in health.get("tags") or [])
                    self.gpu_peak_mb = health.get("gpu_peak_mb")
                    return
                if health.get("status") == "error":
                    raise SpeechError(f"Chatterbox model failed to load: {health.get('error', '')}")
            if self._process is not None and self._process.poll() is not None:
                raise SpeechError(
                    f"Chatterbox server exited with code {self._process.returncode}. "
                    f"See {self._log_path}"
                )
            if self._clock() > deadline:
                raise SpeechError(
                    f"Chatterbox model did not load within {self._startup_timeout_s:.0f} s. "
                    f"See {self._log_path}"
                )
            self._sleep(POLL_INTERVAL_S)

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        """Return the utterance as mono float32 samples at sample_rate.

        Raises SupersededError if a later cancel(generation) arrived first.
        """
        if not self.loaded:
            raise SpeechError("the Chatterbox voice is not loaded")
        body = json.dumps({"text": text, "generation": generation}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base}/synthesize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self._request_timeout_s) as response:
                data = response.read()
                rate = int(response.headers.get("X-Sample-Rate", "0") or 0)
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                raise SupersededError("sentence dropped after a barge-in") from exc
            raise SpeechError(f"Chatterbox server error: {_error_text(exc)}") from exc
        except OSError as exc:
            raise SpeechError(f"Chatterbox server did not answer: {exc}") from exc
        if rate > 0:
            self.sample_rate = rate
        return np.frombuffer(data, dtype="<f4").astype(np.float32)

    def cancel(self, generation: int) -> None:
        """Tell the server to drop every request below this generation (called on barge-in)."""
        body = json.dumps({"generation": generation}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base}/cancel",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=HEALTH_TIMEOUT_S) as response:
                response.read()
        except OSError as exc:
            raise SpeechError(f"could not cancel Chatterbox requests: {exc}") from exc

    def close(self) -> None:
        """Stop the server if this client started it. Safe to call more than once."""
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()  # the server exits when its stdin closes
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        if self._log is not None:
            self._log.close()
            self._log = None

    def _health(self) -> dict[str, Any] | None:
        try:
            with self._opener.open(f"{self._base}/health", timeout=HEALTH_TIMEOUT_S) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _start_process(self) -> None:
        if not self._python.is_file():
            raise SpeechError(
                f"Chatterbox Python not found at {self._python}. Create .venv-tts as described in "
                "the README"
            )
        command = [
            str(self._python),
            "-u",
            "-m",
            SERVER_MODULE,
            "--host",
            "127.0.0.1",
            "--port",
            str(_port_of(self._base)),
            "--engine",
            self._engine,
            "--device",
            self._device,
            "--watch-stdin",
        ]
        if self._voice_reference is not None:
            command += ["--reference", str(self._voice_reference)]
        env = {**os.environ, "PYTHONPATH": str(self._project_root / "src")}
        if self._offline:
            # Use only the local model cache. Loading fails fast if the model was never downloaded.
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env.pop("HF_HUB_OFFLINE", None)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self._log_path.open("ab")
        self._process = self._popen(
            command,
            stdin=subprocess.PIPE,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            cwd=str(self._project_root),
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        atexit.register(self.close)


def _port_of(base: str) -> int:
    return int(base.rsplit(":", 1)[1])


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        return str(payload.get("error", exc.reason))
    except (OSError, ValueError):
        return str(exc.reason)
