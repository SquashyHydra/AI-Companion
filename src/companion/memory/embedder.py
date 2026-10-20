"""Sentence embeddings on CPU for synonym recall.

The model runs through fastembed (ONNX Runtime, no GPU). It is optional: when it is turned off or
cannot load, memory recall falls back to keywords and the caller says so once.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class EmbedderError(RuntimeError):
    """The embedding model is unavailable or failed."""


class Embedder(Protocol):
    """What the memory store needs. Vectors come back L2-normalized, so cosine is a dot product."""

    def embed(self, texts: list[str]) -> list[np.ndarray]: ...


class FastEmbedder:
    """A fastembed text model. The first load downloads the model into cache_dir."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self._model_name = model_name
        self._cache_dir = Path(cache_dir)
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            from fastembed import TextEmbedding

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=self._model_name, cache_dir=str(self._cache_dir))
        except Exception as exc:
            raise EmbedderError(
                f"could not load the embedding model {self._model_name}: {exc}"
            ) from exc

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if self._model is None:
            raise EmbedderError("the embedding model is not loaded")
        if not texts:
            return []
        try:
            raw = list(self._model.embed(texts))
        except Exception as exc:
            raise EmbedderError(f"embedding failed: {exc}") from exc
        vectors: list[np.ndarray] = []
        for item in raw:
            vector = np.asarray(item, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            vectors.append(vector / norm if norm > 0 else vector)
        return vectors


def build_embedder(
    *,
    enabled: bool,
    model_name: str,
    cache_dir: Path,
    notify: Callable[[str], None] = print,
    factory: Callable[[str, Path], FastEmbedder] = FastEmbedder,
) -> Embedder | None:
    """Load the embedding model if enabled. Returns None (keyword recall only) on any failure."""
    if not enabled:
        return None
    embedder = factory(model_name, cache_dir)
    notify(f"Loading the memory embedding model {model_name} (first run downloads it)...")
    try:
        embedder.load()
    except EmbedderError as exc:
        notify(f"Memory recall will use keywords only: {exc}")
        return None
    return embedder


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine similarity for vectors that are already normalized."""
    return float(np.dot(first, second))
