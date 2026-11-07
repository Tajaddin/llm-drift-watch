"""DriftWatch runtime: collects embeddings, manages baseline + recent windows,
calls into a detector when ``observe`` is invoked.

Two states:

* **collecting baseline** — every ``observe`` appends to the baseline buffer
  until ``seal_baseline()`` is called.
* **monitoring** — every ``observe`` appends to a rolling recent window, and
  triggers a detector check. When the detector flags drift, the report has
  ``drift_detected=True`` and the optional ``on_drift`` callback fires once
  per drift episode.

To reduce flapping on individual outliers, ``min_consecutive`` requires N
consecutive flagged windows before the public report is marked drifted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from llm_drift_watch.detectors import (
    CentroidZ,
    DetectionMethod,
    DetectorResult,
    KSCosine,
    MMDDetector,
)
from llm_drift_watch.embedders import Embedder, get_embedder


@dataclass
class DriftReport:
    """Public report from an observation cycle."""

    drift_detected: bool
    score: float
    p_value: float | None
    method: str
    n_baseline: int
    n_recent: int
    consecutive_flags: int
    sample_text: str | None = None
    details: dict = field(default_factory=dict)


_METHOD_REGISTRY: dict[str, Callable[[], DetectionMethod]] = {
    "centroid_z": CentroidZ,
    "ks_cosine": KSCosine,
    "mmd": MMDDetector,
}


class DriftWatch:
    """Run a drift detector over a stream of texts.

    Parameters
    ----------
    embedder:
        An :class:`Embedder` or backend name (``"hash"``, ``"fastembed"``).
        ``None`` uses :func:`get_embedder` defaults.
    method:
        Detector name (``"centroid_z"``, ``"ks_cosine"``, ``"mmd"``) or an
        instantiated detector implementing :class:`DetectionMethod`.
    baseline_size:
        Soft cap on how many texts the baseline holds. Once exceeded, oldest
        observations roll off. Set to ``None`` for unbounded.
    window_size:
        Recent window length used in detector comparisons.
    min_consecutive:
        Require N consecutive flagged checks before reporting drift, to reduce
        flapping. Set to 1 to disable.
    check_every:
        Run the detector every Nth observation (default 1 = every call).
    on_drift:
        Optional callback ``(report) -> None`` fired when drift is confirmed.
        It fires once per drift episode; a check that comes back clean ends
        the episode and re-arms it.
    realert_every:
        Fire ``on_drift`` again every N checks while drift stays confirmed.
        ``None`` (default) never repeats within an episode. This is not a
        cooldown: a new episode alerts at once however recent the last alert.
    """

    def __init__(
        self,
        embedder: Embedder | str | None = None,
        *,
        method: DetectionMethod | str = "centroid_z",
        baseline_size: int | None = 500,
        window_size: int = 50,
        min_consecutive: int = 3,
        check_every: int = 1,
        on_drift: Callable[[DriftReport], None] | None = None,
        realert_every: int | None = None,
    ) -> None:
        if realert_every is not None and realert_every < 1:
            raise ValueError(f"realert_every must be >= 1 or None, got {realert_every}")
        self.embedder: Embedder = (
            embedder
            if isinstance(embedder, Embedder)
            else get_embedder(embedder if isinstance(embedder, str) else None)
        )
        if isinstance(method, str):
            if method not in _METHOD_REGISTRY:
                raise ValueError(f"Unknown method {method!r}")
            self.method: DetectionMethod = _METHOD_REGISTRY[method]()
        else:
            self.method = method
        self.baseline_size = baseline_size
        self.window_size = window_size
        self.min_consecutive = min_consecutive
        self.check_every = max(1, check_every)
        self.on_drift = on_drift
        self.realert_every = realert_every

        self._sealed = False
        self._baseline_emb: list[np.ndarray] = []
        self._baseline_text: list[str] = []
        self._recent_emb: deque[np.ndarray] = deque(maxlen=window_size)
        self._recent_text: deque[str] = deque(maxlen=window_size)
        self._consecutive_flags = 0
        self._n_since_check = 0
        self._checks_since_alert: int | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def n_baseline(self) -> int:
        return len(self._baseline_emb)

    @property
    def n_recent(self) -> int:
        return len(self._recent_emb)

    def seal_baseline(self) -> None:
        """Mark current baseline as final. Subsequent ``observe`` calls go to
        the rolling recent window and trigger detector checks."""
        if not self._baseline_emb:
            raise RuntimeError("Cannot seal an empty baseline. Call observe() first.")
        self._sealed = True
        self._consecutive_flags = 0
        self._n_since_check = 0
        self._checks_since_alert = None

    def reset(self, *, keep_baseline: bool = True) -> None:
        """Reset the recent window and consecutive-flag counter."""
        self._recent_emb.clear()
        self._recent_text.clear()
        self._consecutive_flags = 0
        self._n_since_check = 0
        self._checks_since_alert = None
        if not keep_baseline:
            self._baseline_emb = []
            self._baseline_text = []
            self._sealed = False

    def observe(self, text: str) -> DriftReport | None:
        """Add ``text`` to baseline (pre-seal) or recent (post-seal).

        Returns a :class:`DriftReport` post-seal. Pre-seal returns ``None``.
        """
        emb = self.embedder.embed([text])[0]
        if not self._sealed:
            self._add_baseline(text, emb)
            return None

        self._recent_emb.append(emb)
        self._recent_text.append(text)
        self._n_since_check += 1

        if self._n_since_check < self.check_every:
            return DriftReport(
                drift_detected=False,
                score=0.0,
                p_value=None,
                method=self.method.name,
                n_baseline=self.n_baseline,
                n_recent=self.n_recent,
                consecutive_flags=self._consecutive_flags,
                sample_text=text,
                details={"skipped_check": True},
            )

        self._n_since_check = 0
        det_result = self._run_detector()
        if det_result.drift_detected:
            self._consecutive_flags += 1
        else:
            self._consecutive_flags = 0
        confirmed = self._consecutive_flags >= self.min_consecutive
        report = DriftReport(
            drift_detected=confirmed,
            score=det_result.score,
            p_value=det_result.p_value,
            method=det_result.method,
            n_baseline=self.n_baseline,
            n_recent=self.n_recent,
            consecutive_flags=self._consecutive_flags,
            sample_text=text,
            details=det_result.details,
        )
        if not confirmed:
            self._checks_since_alert = None
        else:
            since = self._checks_since_alert
            if since is None or (self.realert_every is not None and since + 1 >= self.realert_every):
                self._checks_since_alert = 0
                if self.on_drift is not None:
                    self.on_drift(report)
            else:
                self._checks_since_alert = since + 1
        return report

    def observe_many(self, texts: Iterable[str]) -> list[DriftReport | None]:
        """Convenience: observe a batch one by one and return per-call reports."""
        return [self.observe(t) for t in texts]

    def run_check_now(self) -> DetectorResult | None:
        """Run the detector once on the current state (useful for debugging)."""
        if not self._sealed or not self._recent_emb:
            return None
        return self._run_detector()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _add_baseline(self, text: str, emb: np.ndarray) -> None:
        self._baseline_emb.append(emb)
        self._baseline_text.append(text)
        if self.baseline_size is not None and len(self._baseline_emb) > self.baseline_size:
            self._baseline_emb.pop(0)
            self._baseline_text.pop(0)

    def _run_detector(self) -> DetectorResult:
        baseline = np.stack(self._baseline_emb)
        recent = np.stack(list(self._recent_emb))
        return self.method.run(baseline, recent)
