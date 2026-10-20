"""Chatterbox text to speech server (Turbo or Nano).

This module runs in its own virtual environment (.venv-tts), not in the main app venv, because
chatterbox-tts pins an older torch than the app uses. The companion talks to it over localhost.

    .venv-tts\\Scripts\\python.exe -m companion.audio.chatterbox_server --engine chatterbox_nano

Endpoints:
    GET  /health      JSON: {"status": "loading" | "ready" | "error", "engine", "device",
                      "sample_rate": int, "tags": [str], "gpu_peak_mb": float | null, "error": str}
    POST /synthesize  JSON body: {"text": str, "generation": int}. Response: raw little-endian
                      float32 mono samples, with the sample rate in the X-Sample-Rate header.
                      409 if a later /cancel superseded this generation.
    POST /cancel      JSON body: {"generation": int}. Drops queued and future requests with a lower
                      generation. The companion calls it on barge-in.

The model loads in the background so /health can answer while it loads. Requests are served one at
a time because the model is not thread-safe. With --watch-stdin the server exits when its parent
closes stdin, so it does not outlive the companion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

# The companion package is importable here (PYTHONPATH=src). Only pure-Python modules are used.
from companion.voice.tags import tags_from_vocabulary

logger = logging.getLogger("companion.chatterbox_server")

ENGINES = ("chatterbox_turbo", "chatterbox_nano")
MAX_TEXT_CHARS = 2000
MAX_BODY_BYTES = 64 * 1024
WARMUP_TEXT = "Hello."


class Superseded(Exception):
    """A request was dropped because a barge-in cancelled its generation."""


class Engine:
    """Holds the loaded Chatterbox model and reports its state.

    Generations: the companion numbers each reply's audio. On barge-in it calls cancel(n) with
    its new generation. Any request with a lower generation is dropped, whether it is still
    waiting for the model or finished generating after the cancel.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        reference: Path | None = None,
        *,
        engine: str = "chatterbox_turbo",
        device: str = "cuda",
    ) -> None:
        self._model_factory = model_factory
        self._reference = reference
        self._model: Any = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cancelled = 0
        self.engine = engine
        self.device = device
        self.status = "loading"
        self.error = ""
        self.sample_rate = 0
        self.tags: list[str] = []

    def load(self) -> None:
        """Load the model, set the reference voice, and warm up. Sets status to ready or error."""
        try:
            model = self._model_factory()
            if self._reference is not None:
                model.prepare_conditionals(str(self._reference))
            if getattr(model, "conds", None) is None:
                raise RuntimeError(
                    "the model has no built-in voice; set tts.chatterbox.voice_reference to a "
                    "clip of at least 5 seconds"
                )
            model.generate(WARMUP_TEXT)
            self.sample_rate = int(model.sr)
            self.tags = sorted(_tags_of(model))
            self._model = model
            self.status = "ready"
            logger.info("model ready at %d Hz with %d voice tags", self.sample_rate, len(self.tags))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            logger.exception("model failed to load")

    def cancel(self, generation: int) -> None:
        """Drop every request with a generation lower than this one."""
        with self._state_lock:
            self._cancelled = max(self._cancelled, generation)

    def _check_current(self, generation: int) -> None:
        with self._state_lock:
            cancelled = self._cancelled
        if generation < cancelled:
            raise Superseded(f"generation {generation} was cancelled by {cancelled}")

    def synthesize(self, text: str, generation: int = 0) -> np.ndarray:
        if self.status != "ready":
            raise RuntimeError(f"model is not ready ({self.status})")
        self._check_current(generation)
        with self._lock:
            # A barge-in may have arrived while this request waited for the model.
            self._check_current(generation)
            wav = self._model.generate(text)
        # Or while the model was generating. Drop the audio rather than play stale speech.
        self._check_current(generation)
        samples = wav.squeeze().detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        return np.asarray(samples, dtype="<f4").reshape(-1)

    def gpu_peak_mb(self) -> float | None:
        """Peak GPU memory allocated by this process, in MB. None when running on CPU."""
        if self.device != "cuda":
            return None
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    def health(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "engine": self.engine,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "tags": self.tags,
            "gpu_peak_mb": self.gpu_peak_mb(),
            "error": self.error,
        }


def _tags_of(model: Any) -> set[str]:
    """Voice tags the model renders, read from its tokenizer's added tokens."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "get_added_vocab"):
        logger.warning("model exposes no tokenizer vocabulary; voice tags are disabled")
        return set()
    return set(tags_from_vocabulary(tokenizer.get_added_vocab()))


def _generation_of(payload: dict[str, Any], *, required: bool) -> int | None:
    """Return the generation number from a request body, or None if it is invalid."""
    if "generation" not in payload:
        return None if required else 0
    value = payload["generation"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def make_handler(engine: Engine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ChatterboxServer/2"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base class name
            logger.debug("%s - %s", self.address_string(), format % args)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any] | None:
            """Parse the JSON object body. Sends the error response and returns None if invalid."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "bad Content-Length"})
                return None
            if not 0 < length <= MAX_BODY_BYTES:
                self._send_json(400, {"error": "request body missing or too large"})
                return None
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "body must be JSON"})
                return None
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "body must be a JSON object"})
                return None
            return payload

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(200, engine.health())

        def do_POST(self) -> None:
            if self.path not in ("/synthesize", "/cancel"):
                self._send_json(404, {"error": "not found"})
                return
            payload = self._read_json()
            if payload is None:
                return
            if self.path == "/cancel":
                generation = _generation_of(payload, required=True)
                if generation is None:
                    self._send_json(400, {"error": "generation must be a whole number of 0 or more"})
                    return
                engine.cancel(generation)
                self._send_json(200, {"cancelled_below": generation})
                return
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                self._send_json(400, {"error": "text must be a non-empty string"})
                return
            if len(text) > MAX_TEXT_CHARS:
                self._send_json(400, {"error": f"text is longer than {MAX_TEXT_CHARS} characters"})
                return
            generation = _generation_of(payload, required=False)
            if generation is None:
                self._send_json(400, {"error": "generation must be a whole number of 0 or more"})
                return
            if engine.status != "ready":
                self._send_json(503, {"error": f"model is not ready ({engine.status})"})
                return
            try:
                samples = engine.synthesize(text, generation)
            except Superseded as exc:
                self._send_json(409, {"error": str(exc)})
                return
            except Exception as exc:
                logger.exception("synthesis failed")
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            body = samples.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Sample-Rate", str(engine.sample_rate))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def make_server(engine: Engine, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    server.daemon_threads = True
    return server


def _watch_stdin() -> None:
    """Exit when the parent process closes our stdin (it has exited or was killed).

    Reads the raw file descriptor on purpose. A blocking read through sys.stdin.buffer holds
    that stream's lock, which stalls other threads that touch sys.stdin, including the model load.
    """
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return
    try:
        while os.read(fd, 4096):
            pass
    except OSError:
        pass
    os._exit(0)


def _default_model_factory(device: str, engine: str = "chatterbox_turbo") -> Callable[[], Any]:
    def factory() -> Any:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        return ChatterboxTurboTTS.from_pretrained(device=device, nano=engine == "chatterbox_nano")

    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chatterbox TTS server for the companion.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--engine", default="chatterbox_turbo", choices=ENGINES)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--reference", type=Path, default=None, help="voice clip of at least 5 s")
    parser.add_argument("--watch-stdin", action="store_true", help="exit when stdin closes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.host not in ("127.0.0.1", "localhost"):
        print("Refusing to bind to a non-local address.", file=sys.stderr)
        return 2
    engine = Engine(
        _default_model_factory(args.device, args.engine),
        reference=args.reference,
        engine=args.engine,
        device=args.device,
    )
    server = make_server(engine, args.host, args.port)
    print(
        f"{args.engine} server ({args.device}) listening on http://{args.host}:{args.port}",
        flush=True,
    )

    def load_then_watch() -> None:
        # The stdin watcher starts after loading: a blocked stdin reader thread running during the
        # model import stalled startup in testing. If the parent died while loading, stdin is
        # already at EOF, so the watcher exits the process as soon as loading finishes.
        engine.load()
        if args.watch_stdin:
            threading.Thread(target=_watch_stdin, daemon=True).start()

    threading.Thread(target=load_then_watch, name="model-load", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
