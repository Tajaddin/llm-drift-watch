"""Synthetic drift benchmark.

Seeds a 200-text baseline of cooking questions. Then injects three drift
streams and measures, for each one:

* ``time_to_alert``  — index of the first observation that produced a
  ``drift_detected=True`` report after the baseline was sealed
* ``true_positive_rate`` over the first ``alert_horizon`` drifted texts
* ``false_positive_rate`` from a control stream of in-distribution cooking
  texts that follow the baseline

Three drift types are tested:

1. **vocabulary** — same intent (recipes) but in a different language
2. **intent** — switch domain entirely (cybersecurity questions)
3. **length** — same domain but bulky multi-paragraph prompts instead of one-liners

Each detector and drift type runs twice: once with the drift stream starting
right after ``seal_baseline()`` (empty recent window, so every check sees only
drifted texts), and once after ``WINDOW_SIZE`` control texts have filled the
window (``warmup``), which is how drift reaches a monitor in production.
``time_to_alert`` counts drifted observations in both cases.

Outputs ``bench/drift_results.json`` and a markdown table the README links to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_drift_watch import DriftWatch, HashEmbedder
from llm_drift_watch.detectors import CentroidZ, KSCosine, MMDDetector


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


BASELINE_DISHES = [
    "pasta", "rice", "chicken", "soup", "beans", "salad", "steak", "fish",
    "curry", "pie", "cake", "bread", "eggs", "pancakes", "tofu", "stew",
    "risotto", "tacos", "pizza", "wings", "burrito", "ramen", "sushi", "stir-fry",
    "lasagna", "omelette", "salmon", "shrimp", "lentils", "quinoa",
]


_BASELINE_TEMPLATES = [
    "How do I make {dish} for two people?",
    "What's the best technique for cooking {dish}?",
    "Recipe ideas for {dish} with seasonal vegetables.",
    "Simple weekday {dish} dinner under 30 minutes?",
    "How long should I cook {dish}?",
    "What's a healthy version of {dish}?",
    "Best spices to pair with {dish}.",
    "Quick prep tips for {dish} on a busy night.",
    "Side dishes that go well with {dish}.",
    "How to store leftover {dish} safely.",
]


def _full_cooking_corpus(seed: int = 7) -> list[str]:
    import random

    rng = random.Random(seed)
    out = [t.format(dish=d) for t in _BASELINE_TEMPLATES for d in BASELINE_DISHES]
    rng.shuffle(out)
    return out  # 10 templates × 30 dishes = 300 unique prompts, shuffled


def baseline_corpus(n: int = 200) -> list[str]:
    """First ``n`` prompts of the shuffled cooking corpus.

    Pairs with :func:`control_corpus`, which takes the *next* slice. Both come
    from the same shuffled list so they're i.i.d. samples from the same domain.
    """
    return _full_cooking_corpus()[:n]


# (1) Vocabulary drift — recipes but in Spanish
def vocab_drift_corpus(n: int = 50) -> list[str]:
    spanish = [
        "¿Cómo preparo paella valenciana auténtica para una cena familiar?",
        "Receta de tortilla española con cebolla caramelizada paso a paso.",
        "¿Cuál es la mejor manera de cocinar arroz con leche cremoso?",
        "Ingredientes principales para hacer un gazpacho andaluz tradicional.",
        "Técnicas para asar pollo al horno crujiente y jugoso.",
        "Receta sencilla de empanadas argentinas de carne picada.",
        "Cómo se prepara una salsa romesco para acompañar verduras.",
        "Tiempos de cocción para legumbres secas como garbanzos y lentejas.",
        "Recomendaciones para una paella vegetariana con champiñones.",
        "Trucos para que el arroz no se pegue en la cazuela.",
    ]
    return (spanish * ((n // len(spanish)) + 1))[:n]


# (2) Intent drift — cybersecurity
def intent_drift_corpus(n: int = 50) -> list[str]:
    sec = [
        "How do I detect SQL injection in a Django app?",
        "Explain CSRF protection in modern web frameworks.",
        "What is reverse engineering of an ELF binary?",
        "Privilege escalation techniques on Linux kernel 6.x.",
        "How does TLS certificate pinning work in mobile apps?",
        "Container escape vulnerabilities in Kubernetes.",
        "OWASP top 10 web vulnerabilities for 2026.",
        "Network reconnaissance methods using nmap.",
        "Buffer overflow exploitation on ARM64.",
        "Authentication bypass via JWT none algorithm.",
    ]
    return (sec * ((n // len(sec)) + 1))[:n]


# (3) Length drift — same domain, much longer prompts
def length_drift_corpus(n: int = 50) -> list[str]:
    long_prefix = (
        "I'm looking for a comprehensive, multi-paragraph answer because I'd like "
        "to understand the full context. Please include cultural background, "
        "ingredient sourcing tips, step-by-step preparation, common pitfalls, "
        "variations across regions, and storage recommendations. "
    )
    long_suffix = (
        " Also include the historical context of this dish, suggested wine "
        "pairings, dietary substitutions for vegetarians and people with "
        "common allergies, and a discussion of how the dish is served in "
        "different formal versus informal settings. Make sure the answer is "
        "thorough."
    )
    base = [
        f"How do I make {dish} for two people?" for dish in BASELINE_DISHES
    ]
    long_form = [long_prefix + b + long_suffix for b in base]
    return (long_form * ((n // len(long_form)) + 1))[:n]


# Control stream — more in-distribution cooking texts, used for FPR estimation.
def control_corpus(n: int = 50) -> list[str]:
    full = _full_cooking_corpus()
    return full[200 : 200 + n]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

WINDOW_SIZE = 20


def run_one(
    label: str,
    drift_texts: list[str],
    method_name: str,
    *,
    baseline_n: int = 200,
    window_size: int = WINDOW_SIZE,
    min_consecutive: int = 3,
    z_threshold: float = 3.0,
    p_threshold: float = 0.01,
    alert_horizon: int = 50,
    warmup: int = 0,
) -> dict:
    """``warmup`` control texts are observed after sealing and before either
    stream, so the recent window holds in-distribution traffic when drift
    starts. With ``warmup=0`` the window is all drift from the first check.
    """
    if method_name == "centroid_z":
        method = CentroidZ(z_threshold=z_threshold)
    elif method_name == "ks_cosine":
        method = KSCosine(p_threshold=p_threshold)
    elif method_name == "mmd":
        method = MMDDetector(p_threshold=p_threshold, n_permutations=100)
    else:
        raise ValueError(method_name)

    watcher = DriftWatch(
        embedder=HashEmbedder(dim=512),
        method=method,
        window_size=window_size,
        min_consecutive=min_consecutive,
        baseline_size=None,
    )
    for t in baseline_corpus(baseline_n):
        watcher.observe(t)
    watcher.seal_baseline()

    fill = control_corpus(warmup)
    alerts_during_warmup = sum(bool(r and r.drift_detected) for r in watcher.observe_many(fill))

    time_to_alert = None
    tp_in_horizon = 0
    for i, t in enumerate(drift_texts[:alert_horizon]):
        r = watcher.observe(t)
        if r and r.drift_detected:
            if time_to_alert is None:
                time_to_alert = i + 1
            tp_in_horizon += 1

    # FPR with control stream (no drift expected).
    watcher_ctrl = DriftWatch(
        embedder=HashEmbedder(dim=512),
        method=method,
        window_size=window_size,
        min_consecutive=min_consecutive,
        baseline_size=None,
    )
    for t in baseline_corpus(baseline_n):
        watcher_ctrl.observe(t)
    watcher_ctrl.seal_baseline()
    alerts_during_warmup += sum(
        bool(r and r.drift_detected) for r in watcher_ctrl.observe_many(fill)
    )
    fp_in_horizon = 0
    fp_first = None
    for i, t in enumerate(control_corpus(warmup + alert_horizon)[warmup:]):
        r = watcher_ctrl.observe(t)
        if r and r.drift_detected:
            if fp_first is None:
                fp_first = i + 1
            fp_in_horizon += 1

    return {
        "drift_type": label,
        "method": method_name,
        "warmup": warmup,
        "alerts_during_warmup": alerts_during_warmup,
        "time_to_alert": time_to_alert,
        "tp_in_horizon": tp_in_horizon,
        "horizon": alert_horizon,
        "true_positive_rate_in_horizon": round(tp_in_horizon / alert_horizon, 3),
        "false_positive_first": fp_first,
        "false_positives_in_horizon": fp_in_horizon,
        "false_positive_rate_in_horizon": round(fp_in_horizon / alert_horizon, 3),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--out", type=str, default="bench/drift_results.json")
    args = p.parse_args()

    drift_streams = {
        "vocabulary": vocab_drift_corpus(args.horizon),
        "intent": intent_drift_corpus(args.horizon),
        "length": length_drift_corpus(args.horizon),
    }
    methods = ["centroid_z", "ks_cosine", "mmd"]

    rows: list[dict] = []
    for warmup in (0, WINDOW_SIZE):
        for method in methods:
            for label, texts in drift_streams.items():
                row = run_one(label, texts, method, alert_horizon=args.horizon, warmup=warmup)
                rows.append(row)
                tta = row["time_to_alert"]
                tta_str = f"{tta}" if tta is not None else ">{}".format(args.horizon)
                print(
                    f"warmup={warmup:<3d} method={method:10s} drift={label:10s} "
                    f"time_to_alert={tta_str:6s} "
                    f"tpr={row['true_positive_rate_in_horizon']:.2f} "
                    f"fpr={row['false_positive_rate_in_horizon']:.2f} "
                    f"warmup_alerts={row['alerts_during_warmup']}"
                )

    out = {"rows": rows, "horizon": args.horizon}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out}")

    # Quick markdown table.
    print("\nMarkdown table:")
    print("| Warmup | Detector | Drift type | Time-to-alert | TPR @ horizon | FPR @ horizon |")
    print("|---:|---|---|---:|---:|---:|")
    for r in rows:
        tta = r["time_to_alert"]
        tta_cell = str(tta) if tta is not None else f">{args.horizon}"
        print(
            f"| {r['warmup']} | {r['method']} | {r['drift_type']} | {tta_cell} | "
            f"{r['true_positive_rate_in_horizon']:.2f} | "
            f"{r['false_positive_rate_in_horizon']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
