"""DriftWatch core tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_drift_watch import DetectorResult, DriftWatch, HashEmbedder


@dataclass
class _ScriptedDetector:
    """Returns the next scripted verdict on each check."""

    flags: list[bool]
    name: str = "scripted"

    def run(self, baseline, recent) -> DetectorResult:
        return DetectorResult(
            drift_detected=self.flags.pop(0), score=0.0, p_value=None, method=self.name
        )


def _sealed_watch(flags: list[bool], **kwargs) -> tuple[DriftWatch, list]:
    events: list = []
    w = DriftWatch(
        embedder=HashEmbedder(dim=64),
        method=_ScriptedDetector(flags),
        window_size=5,
        min_consecutive=1,
        on_drift=events.append,
        **kwargs,
    )
    w.observe("baseline")
    w.seal_baseline()
    return w, events


def test_observe_pre_seal_returns_none() -> None:
    w = DriftWatch(embedder=HashEmbedder(dim=64))
    assert w.observe("baseline text 1") is None
    assert w.n_baseline == 1
    assert not w.is_sealed


def test_cannot_seal_empty_baseline() -> None:
    w = DriftWatch(embedder=HashEmbedder(dim=64))
    with pytest.raises(RuntimeError):
        w.seal_baseline()


def test_observe_post_seal_returns_report() -> None:
    w = DriftWatch(embedder=HashEmbedder(dim=64), window_size=5, min_consecutive=1)
    for i in range(20):
        w.observe(f"cooking question {i}")
    w.seal_baseline()
    report = w.observe("another cooking question")
    assert report is not None
    assert report.method == "centroid_z"


def test_seal_baseline_freezes_baseline() -> None:
    w = DriftWatch(embedder=HashEmbedder(dim=64), window_size=5)
    for i in range(10):
        w.observe(f"text {i}")
    w.seal_baseline()
    base_before = w.n_baseline
    # Post-seal observes should NOT grow the baseline.
    for _ in range(5):
        w.observe("more text")
    assert w.n_baseline == base_before


def test_drift_detected_on_distribution_shift() -> None:
    w = DriftWatch(
        embedder=HashEmbedder(dim=256),
        window_size=10,
        min_consecutive=2,
        baseline_size=None,
    )
    baseline_texts = [
        f"Recipe for {dish} with vegetables and spices."
        for dish in ["pasta", "rice", "chicken", "soup", "beans", "salad",
                     "steak", "fish", "curry", "pie", "cake", "bread", "eggs",
                     "pancakes", "tofu", "stew", "risotto", "tacos", "pizza",
                     "wings"]
    ] * 10
    for t in baseline_texts:
        w.observe(t)
    w.seal_baseline()

    drift_texts = [
        "How do I exploit SQL injection vulnerabilities in a web app?",
        "Detailed walk-through of OWASP top 10 with code samples.",
        "Reverse engineering ELF binaries with Ghidra and IDA.",
        "Buffer overflow exploitation on x86_64 systems.",
        "Privilege escalation techniques on modern Linux kernels.",
    ] * 4

    first_detection = None
    for i, t in enumerate(drift_texts):
        r = w.observe(t)
        if r and r.drift_detected and first_detection is None:
            first_detection = i + 1
            break
    assert first_detection is not None
    assert first_detection <= 20  # generous bound; smoke shows ~2


def test_on_drift_callback_fires() -> None:
    events: list = []
    w = DriftWatch(
        embedder=HashEmbedder(dim=256),
        window_size=5,
        min_consecutive=1,
        on_drift=lambda r: events.append(r),
    )
    for _ in range(20):
        w.observe("cooking text " + str(_))
    w.seal_baseline()
    for _ in range(5):
        w.observe("security exploit payload")
    assert len(events) >= 1
    assert events[0].drift_detected


def test_on_drift_fires_once_per_drift_episode() -> None:
    flags = [True, True, True, True, False, True, True]
    w, events = _sealed_watch(list(flags))
    reports = [w.observe("x") for _ in flags]
    assert [r.drift_detected for r in reports] == flags
    # One alert when drift is confirmed, none while it persists, one more
    # after a clean check ends the episode and drift comes back.
    assert len(events) == 2
    assert events[0] is reports[0]
    assert events[1] is reports[5]


def test_alert_cooldown_repeats_alert_while_drift_persists() -> None:
    w, events = _sealed_watch([True] * 6, alert_cooldown=2)
    reports = [w.observe("x") for _ in range(6)]
    assert events == [reports[0], reports[2], reports[4]]


def test_reset_rearms_on_drift() -> None:
    w, events = _sealed_watch([True, True])
    w.observe("x")
    w.reset()
    w.observe("x")
    assert len(events) == 2


def test_reset_clears_recent_but_keeps_baseline() -> None:
    w = DriftWatch(embedder=HashEmbedder(dim=64), window_size=5)
    for i in range(10):
        w.observe(f"t{i}")
    w.seal_baseline()
    for _ in range(3):
        w.observe("more")
    assert w.n_recent == 3
    w.reset(keep_baseline=True)
    assert w.n_recent == 0
    assert w.n_baseline == 10
    assert w.is_sealed


def test_reset_can_drop_baseline_too() -> None:
    w = DriftWatch(embedder=HashEmbedder(dim=64))
    w.observe("a")
    w.seal_baseline()
    w.reset(keep_baseline=False)
    assert w.n_baseline == 0
    assert not w.is_sealed


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        DriftWatch(method="ufo-test", embedder=HashEmbedder(dim=64))


def test_check_every_throttles_detector() -> None:
    w = DriftWatch(
        embedder=HashEmbedder(dim=64),
        window_size=5,
        min_consecutive=1,
        check_every=3,
    )
    for _ in range(10):
        w.observe("baseline")
    w.seal_baseline()
    reports = [w.observe("totally different text " + str(i)) for i in range(6)]
    # 6 observations, every 3rd runs the detector → 2 real checks
    skipped = sum(1 for r in reports if r and r.details.get("skipped_check"))
    assert skipped == 4
