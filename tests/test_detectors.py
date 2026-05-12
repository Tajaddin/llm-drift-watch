"""Detector tests using synthetic embeddings (no embedder call)."""

from __future__ import annotations

import numpy as np

from llm_drift_watch.detectors import CentroidZ, KSCosine, MMDDetector


def _rand_unit(n: int, d: int, *, seed: int, shift: float = 0.0) -> np.ndarray:
    """Sample n points near a cluster center. ``shift`` rotates the cluster
    center so two calls with different shift values land in different regions
    of the unit sphere."""
    rng = np.random.default_rng(seed)
    center = np.zeros(d, dtype=np.float32)
    center[0] = 1.0
    if shift:
        # Rotate the center in the (0, 1) plane by ``shift`` radians.
        c, s = np.cos(shift), np.sin(shift)
        rotated = np.zeros(d, dtype=np.float32)
        rotated[0] = c
        rotated[1] = s
        center = rotated
    noise = rng.normal(scale=0.15, size=(n, d)).astype(np.float32)
    arr = center[None, :] + noise
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


def test_centroid_z_no_drift_when_same_distribution() -> None:
    baseline = _rand_unit(200, 16, seed=0)
    recent = _rand_unit(50, 16, seed=1)  # same distribution
    res = CentroidZ(z_threshold=3.0).run(baseline, recent)
    assert res.drift_detected is False


def test_centroid_z_detects_shift() -> None:
    baseline = _rand_unit(200, 16, seed=0)
    recent = _rand_unit(50, 16, seed=2, shift=1.5)  # ~85° rotation
    res = CentroidZ(z_threshold=3.0).run(baseline, recent)
    assert res.drift_detected is True
    assert abs(res.score) >= 3.0


def test_ks_cosine_no_drift_when_same_distribution() -> None:
    baseline = _rand_unit(200, 16, seed=0)
    recent = _rand_unit(50, 16, seed=1)
    res = KSCosine(p_threshold=0.01).run(baseline, recent)
    assert res.drift_detected is False


def test_ks_cosine_detects_shift() -> None:
    baseline = _rand_unit(200, 16, seed=0)
    recent = _rand_unit(50, 16, seed=2, shift=1.5)
    res = KSCosine(p_threshold=0.01).run(baseline, recent)
    assert res.drift_detected is True
    assert res.p_value is not None and res.p_value < 0.01


def test_mmd_detects_shift() -> None:
    baseline = _rand_unit(80, 16, seed=0)
    recent = _rand_unit(40, 16, seed=2, shift=1.5)
    res = MMDDetector(p_threshold=0.05, n_permutations=100).run(baseline, recent)
    assert res.drift_detected is True


def test_mmd_no_drift_when_same() -> None:
    baseline = _rand_unit(80, 16, seed=0)
    recent = _rand_unit(40, 16, seed=1)
    res = MMDDetector(p_threshold=0.01, n_permutations=100).run(baseline, recent)
    assert res.drift_detected is False


def test_detectors_handle_empty_inputs() -> None:
    empty = np.zeros((0, 16), dtype=np.float32)
    some = _rand_unit(5, 16, seed=0)
    for det in (CentroidZ(), KSCosine(), MMDDetector(n_permutations=20)):
        res = det.run(empty, some)
        assert res.drift_detected is False
