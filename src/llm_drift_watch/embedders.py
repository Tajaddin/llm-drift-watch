"""Embedder protocol + two implementations.

* :class:`FastEmbedEmbedder` — ONNX-based, ~50MB model, no PyTorch required.
* :class:`HashEmbedder` — zero-dependency deterministic embedder for tests and
  for environments where a model download isn't possible. Uses character-trigram
  hashing into a fixed-size vector. **Not** a real semantic embedding; use only
  for tests and synthetic benchmarks where the texts already differ structurally.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder interface.

    Implementations must return an ``(n, d)`` float32 numpy array of L2-
    normalized row vectors when given an iterable of ``n`` texts.
    """

    dim: int

    def embed(self, texts: Iterable[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# HashEmbedder — zero-dep, deterministic
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic character-trigram hashing into a fixed-dim vector.

    This is *not* a real semantic embedding — two paraphrases of the same
    sentence will land in different regions of the vector space. It is the
    correct choice for:

    * unit tests (deterministic, no network, no GPU)
    * synthetic benchmarks where drift is structural (vocab, length, charset)
    * environments where you cannot install a real model

    For production LLM monitoring, use :class:`FastEmbedEmbedder` or another
    real embedder.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out = []
        for t in texts:
            out.append(self._one(t))
        arr = np.stack(out).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        # Sliding character trigrams capture both vocab and length differences.
        s = text.lower()
        if not s:
            return vec
        for i in range(max(1, len(s) - 2)):
            tri = s[i : i + 3]
            h = int(hashlib.blake2s(tri.encode("utf-8"), digest_size=8).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign
        # Length signal — small contribution so it doesn't dominate.
        length_idx = self.dim - 1
        vec[length_idx] += np.log1p(len(s)) * 0.5
        return vec


# ---------------------------------------------------------------------------
# FastEmbedEmbedder — production default
# ---------------------------------------------------------------------------


class FastEmbedEmbedder:
    """Wraps ``fastembed.TextEmbedding`` with a small, fast default model.

    Default model is ``BAAI/bge-small-en-v1.5`` (384-dim, ~50MB ONNX). Override
    via the ``model_name`` argument or ``FASTEMBED_MODEL`` env var.

    Tradeoffs vs other backends: no PyTorch, runs on CPU, model auto-downloads
    on first use. Embeddings are L2-normalized by fastembed already.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "fastembed is not installed. Install with `pip install llm-drift-watch[fastembed]` "
                "or switch to HashEmbedder."
            ) from exc
        self.model_name = model_name or os.environ.get("FASTEMBED_MODEL", self.DEFAULT_MODEL)
        self._model = TextEmbedding(model_name=self.model_name)
        # Probe dimensionality with a single embedding.
        probe = next(self._model.embed(["dim probe"]))
        self.dim = len(probe)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.stack(list(self._model.embed(texts_list))).astype(np.float32)
        # bge models are already normalized but be defensive.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def get_embedder(name: str | None = None, *, dim: int = 256) -> Embedder:
    """Resolve an embedder by name.

    * ``"hash"`` → :class:`HashEmbedder` (default if fastembed isn't installed)
    * ``"fastembed"`` → :class:`FastEmbedEmbedder`
    """
    name = (name or os.environ.get("DRIFT_WATCH_EMBEDDER") or "").lower().strip()
    if name == "hash":
        return HashEmbedder(dim=dim)
    if name == "fastembed":
        return FastEmbedEmbedder()
    if not name:
        try:
            return FastEmbedEmbedder()
        except RuntimeError:
            return HashEmbedder(dim=dim)
    raise ValueError(f"Unknown embedder: {name!r}")
