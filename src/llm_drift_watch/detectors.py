"""Statistical drift detectors.

Each detector implements :class:`DetectionMethod` and returns a
:class:`DetectorResult`. The :class:`DriftWatch` runtime calls
``run(baseline_embeddings, recent_embeddings)`` whenever a drift assessment is
requested.

Three detectors are included:

* :class:`CentroidZ`  — z-score of cosine distance to baseline centroid
  (very fast, interpretable, default).
* :class:`KSCosine`   — Kolmogorov-Smirnov on cosine-distance distributions
  (parameter-free, returns a real p-value).
* :class:`MMDDetector` — Maximum Mean Discrepancy with Gaussian kernel and
  permutation test (robust to high-D embeddings, slowest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import stats


@dataclass
class DetectorResult:
    """One drift assessment."""

    drift_detected: bool
    score: float
    p_value: float | None
    method: str
    details: dict = field(default_factory=dict)


class DetectionMethod(Protocol):
    """All detectors expose ``run(baseline, recent) -> DetectorResult``."""

    name: str

    def run(self, baseline: np.ndarray, recent: np.ndarray) -> DetectorResult:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cosine_distance_to(point: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity) between a single point and a centroid.

    Both are assumed to be L2-normalized.
    """
    return float(1.0 - np.dot(point, centroid))


def _cosine_distance_batch(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return 1.0 - points @ centroid


def _l2_normalize(centroid: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(centroid)
    if n == 0:
        return centroid
    return centroid / n


# ---------------------------------------------------------------------------
# CentroidZ
# ---------------------------------------------------------------------------


@dataclass
class CentroidZ:
    """Z-score of recent cosine-distance distribution vs baseline.

    Fast path, default. Compute:

    1. Baseline centroid c = mean(baseline) / ||mean(baseline)||
    2. baseline_d = cosine_distance(b, c) for b in baseline
    3. recent_d   = cosine_distance(r, c) for r in recent
    4. z = (mean(recent_d) - mean(baseline_d)) / std(baseline_d)
    5. Alert when |z| >= z_threshold

    The p_value is the two-sided normal-CDF p for the same z statistic. It's
    reported but is **not** a calibrated p-value — for that, use ``KSCosine``
    or ``MMDDetector``.
    """

    z_threshold: float = 3.0
    name: str = "centroid_z"

    def run(self, baseline: np.ndarray, recent: np.ndarray) -> DetectorResult:
        if len(baseline) == 0 or len(recent) == 0:
            return DetectorResult(False, 0.0, None, self.name, {"reason": "empty input"})
        centroid = _l2_normalize(baseline.mean(axis=0))
        baseline_d = _cosine_distance_batch(baseline, centroid)
        recent_d = _cosine_distance_batch(recent, centroid)
        b_mean = float(baseline_d.mean())
        b_std = float(baseline_d.std(ddof=1)) if len(baseline_d) > 1 else 0.0
        if b_std == 0.0:
            b_std = 1e-9
        r_mean = float(recent_d.mean())
        z = (r_mean - b_mean) / b_std
        # Two-sided normal-CDF approximation for the p-value (not calibrated).
        p_approx = float(2 * (1.0 - stats.norm.cdf(abs(z))))
        return DetectorResult(
            drift_detected=bool(abs(z) >= self.z_threshold),
            score=float(z),
            p_value=p_approx,
            method=self.name,
            details={
                "baseline_mean_distance": b_mean,
                "baseline_std_distance": b_std,
                "recent_mean_distance": r_mean,
                "z_threshold": self.z_threshold,
            },
        )


# ---------------------------------------------------------------------------
# KSCosine
# ---------------------------------------------------------------------------


@dataclass
class KSCosine:
    """Kolmogorov-Smirnov test on cosine-distance distributions.

    Pros: parameter-free, returns a real p-value.
    Cons: less sensitive to small but systematic shifts than MMD; the cosine-
    distance reduction is 1D so subtle direction changes can be missed.
    """

    p_threshold: float = 0.01
    name: str = "ks_cosine"

    def run(self, baseline: np.ndarray, recent: np.ndarray) -> DetectorResult:
        if len(baseline) < 2 or len(recent) < 2:
            return DetectorResult(False, 0.0, None, self.name, {"reason": "n<2"})
        centroid = _l2_normalize(baseline.mean(axis=0))
        baseline_d = _cosine_distance_batch(baseline, centroid)
        recent_d = _cosine_distance_batch(recent, centroid)
        ks_stat, p_value = stats.ks_2samp(baseline_d, recent_d, alternative="two-sided")
        return DetectorResult(
            drift_detected=bool(p_value < self.p_threshold),
            score=float(ks_stat),
            p_value=float(p_value),
            method=self.name,
            details={"p_threshold": self.p_threshold},
        )


# ---------------------------------------------------------------------------
# MMD with permutation test
# ---------------------------------------------------------------------------


@dataclass
class MMDDetector:
    """Maximum Mean Discrepancy with Gaussian kernel and permutation test.

    More sensitive to multivariate drift than 1D reductions. Slower because of
    the permutation null distribution (default 200 permutations).
    """

    p_threshold: float = 0.01
    n_permutations: int = 200
    sigma: float | None = None  # None → median heuristic
    name: str = "mmd"

    def run(self, baseline: np.ndarray, recent: np.ndarray) -> DetectorResult:
        if len(baseline) < 5 or len(recent) < 5:
            return DetectorResult(False, 0.0, None, self.name, {"reason": "n<5"})

        # Optionally subsample to keep MMD fast.
        rng = np.random.default_rng(42)
        if len(baseline) > 200:
            idx = rng.choice(len(baseline), 200, replace=False)
            baseline = baseline[idx]
        if len(recent) > 200:
            idx = rng.choice(len(recent), 200, replace=False)
            recent = recent[idx]

        sigma = self.sigma if self.sigma is not None else _median_heuristic(np.vstack([baseline, recent]))
        observed = _mmd2(baseline, recent, sigma)
        # Permutation null.
        combined = np.vstack([baseline, recent])
        n_b = len(baseline)
        count = 0
        for _ in range(self.n_permutations):
            perm = rng.permutation(len(combined))
            b_perm = combined[perm[:n_b]]
            r_perm = combined[perm[n_b:]]
            if _mmd2(b_perm, r_perm, sigma) >= observed:
                count += 1
        p_value = (count + 1) / (self.n_permutations + 1)
        return DetectorResult(
            drift_detected=bool(p_value < self.p_threshold),
            score=float(observed),
            p_value=float(p_value),
            method=self.name,
            details={"sigma": sigma, "n_permutations": self.n_permutations},
        )


def _mmd2(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD² estimate with Gaussian RBF kernel."""
    xx = _gaussian_kernel(x, x, sigma)
    yy = _gaussian_kernel(y, y, sigma)
    xy = _gaussian_kernel(x, y, sigma)
    # Unbiased diagonal removal.
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return float(xx.mean() + yy.mean() - 2 * xy.mean())
    sum_xx = (xx.sum() - np.trace(xx)) / (m * (m - 1))
    sum_yy = (yy.sum() - np.trace(yy)) / (n * (n - 1))
    sum_xy = xy.mean()
    return float(sum_xx + sum_yy - 2 * sum_xy)


def _gaussian_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    a_sq = (a ** 2).sum(axis=1)[:, None]
    b_sq = (b ** 2).sum(axis=1)[None, :]
    d2 = np.maximum(a_sq + b_sq - 2 * a @ b.T, 0.0)
    return np.exp(-d2 / (2 * sigma ** 2))


def _median_heuristic(stacked: np.ndarray) -> float:
    """Median pairwise distance — the standard bandwidth heuristic for RBF MMD."""
    n = len(stacked)
    if n > 200:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 200, replace=False)
        stacked = stacked[idx]
    n = len(stacked)
    sq = (stacked ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * stacked @ stacked.T, 0.0)
    iu = np.triu_indices(n, k=1)
    distances = np.sqrt(d2[iu])
    med = float(np.median(distances))
    if med == 0:
        return 1.0
    return med
