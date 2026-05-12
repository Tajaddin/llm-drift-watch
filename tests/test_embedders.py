"""Embedder tests."""

from __future__ import annotations

import numpy as np
import pytest

from llm_drift_watch import HashEmbedder
from llm_drift_watch.embedders import get_embedder


def test_hash_embedder_returns_correct_shape() -> None:
    e = HashEmbedder(dim=128)
    out = e.embed(["alpha", "beta", "gamma"])
    assert out.shape == (3, 128)
    assert out.dtype == np.float32


def test_hash_embedder_rows_are_normalized() -> None:
    e = HashEmbedder(dim=64)
    out = e.embed(["alpha", "beta", "gamma"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hash_embedder_deterministic() -> None:
    e = HashEmbedder(dim=64)
    a = e.embed(["hello world"])
    b = e.embed(["hello world"])
    assert np.allclose(a, b)


def test_hash_embedder_rejects_tiny_dim() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dim=4)


def test_hash_embedder_handles_empty() -> None:
    e = HashEmbedder(dim=32)
    out = e.embed([""])
    assert out.shape == (1, 32)


def test_get_embedder_hash() -> None:
    e = get_embedder("hash", dim=64)
    assert e.dim == 64


def test_get_embedder_unknown_rejected() -> None:
    with pytest.raises(ValueError):
        get_embedder("nonsense")
