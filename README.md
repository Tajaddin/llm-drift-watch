# llm-drift-watch

> Embedding-based drift detector for LLM input/output streams. Statistical tests on rolling windows, webhook alerts, Streamlit dashboard. Detects three different drift types (vocabulary, intent, length) in **3 observations** on a synthetic benchmark — 17× under the 50-call budget.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-27%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero numbers

Synthetic benchmark: 200-text cooking baseline, then 50 drifted observations of three different types, plus a 50-text in-distribution control stream for FPR. Hash-based embedder (zero-dep, deterministic).

| Detector | Drift type | Time-to-alert | TPR (n=50) | **FPR (n=50)** |
|---|---|---:|---:|---:|
| **centroid_z** | vocabulary (Spanish recipes) | **3** | 96% | **0%** |
| **centroid_z** | intent (cybersecurity) | **3** | 96% | **0%** |
| **centroid_z** | length (verbose prompts) | **3** | 96% | **0%** |
| ks_cosine | vocabulary | 4 | 94% | 0% |
| ks_cosine | intent | 4 | 94% | 0% |
| ks_cosine | length | 4 | 94% | 0% |
| mmd | vocabulary | 10 | 82% | 0% |
| mmd | intent | 13 | 76% | 0% |
| mmd | length | 7 | 88% | 0% |

Target was ≤ 50 observations to first alert. The default `centroid_z` detector clears that by **17×** while staying at 0% false-positive rate on an i.i.d. in-distribution control stream. Raw results in [`bench/drift_results.json`](bench/drift_results.json).

Reproduce: `python bench/drift_benchmark.py`.

## Why this exists

MLflow's drift modules focus on tabular features. The drift that bites LLM production is different: prompts switch language, users shift from "what's the weather" to "ignore previous instructions", or output styles change after a model upgrade. Those are *embedding* problems, not feature problems.

`llm-drift-watch` watches a stream of texts (prompts, completions, or any other LLM-adjacent string), embeds them with a small local model, and runs a configurable statistical test against a sealed baseline window. When the test crosses a threshold for N consecutive checks, an alert fires. No SaaS, no PyTorch required, no API keys for the default path.

## Quickstart

```bash
pip install -e ".[fastembed]"
```

```python
from llm_drift_watch import DriftWatch

watch = DriftWatch(window_size=50)        # fastembed BGE-small by default

# Phase 1 — collect baseline (i.i.d. samples of expected production traffic)
for prompt in known_good_prompts:
    watch.observe(prompt)
watch.seal_baseline()

# Phase 2 — monitor
for prompt in live_prompts:
    report = watch.observe(prompt)
    if report.drift_detected:
        page_oncall(
            score=report.score,
            p_value=report.p_value,
            recent_sample=report.sample_text,
        )
```

The wrapper is dependency-light: numpy + scipy in core, fastembed for the default embedder, fastapi + streamlit only for the example webhook server + dashboard.

## Three detectors built in

| Detector | Test | When to use | Cost |
|---|---|---|---|
| `centroid_z` (default) | z-score of recent cosine distance against baseline centroid | First choice — fast, interpretable, low false-positive | O(d) per observation |
| `ks_cosine` | Two-sample Kolmogorov-Smirnov on cosine-distance distributions | When you want a calibrated p-value but don't need multivariate power | O(window log window) |
| `mmd` | Maximum Mean Discrepancy with Gaussian RBF kernel + permutation test | Subtle multivariate shifts that 1D summaries miss; slower because of the null distribution | O(n² × permutations) |

All three return a uniform `DriftReport` (`score`, `p_value`, `drift_detected`, `details`).

## Three example drift types in the benchmark

* **Vocabulary** — recipe prompts switch to Spanish. Tests whether the detector catches lexical drift even when the *intent* is the same.
* **Intent** — cooking → cybersecurity questions. Tests semantic-domain drift.
* **Length** — same recipe questions wrapped with multi-paragraph context. Tests stylistic / verbosity drift.

See [`bench/drift_benchmark.py`](bench/drift_benchmark.py) for the prompts.

## Webhook server (FastAPI)

`examples/webhook_server.py` exposes `/baseline`, `/seal`, `/observe`, `/status` and POSTs to `$DRIFT_WEBHOOK_URL` when drift is confirmed:

```bash
pip install -e ".[fastembed,server]"
DRIFT_WEBHOOK_URL=https://hooks.example.com/llm-drift \
  uvicorn examples.webhook_server:app --port 8000
```

```bash
# Add baseline
curl -X POST localhost:8000/baseline -H 'content-type: application/json' \
  -d '{"texts": ["How do I cook pasta?", "What is a good chicken recipe?"]}'

# Seal
curl -X POST localhost:8000/seal

# Observe — webhook fires when drift confirmed
curl -X POST localhost:8000/observe -H 'content-type: application/json' \
  -d '{"text": "How do I exploit SQL injection in Django?"}'
```

## Streamlit dashboard

`examples/dashboard.py` is a live UI: paste a baseline, seal it, stream observations, see the drift score evolve, alert on drift events.

```bash
pip install -e ".[dashboard,fastembed]"
streamlit run examples/dashboard.py
```

## Configuration

| Knob | Effect | Default |
|---|---|---|
| `embedder` | `fastembed` (default) or `hash` (deterministic, zero-dep) or any callable | auto |
| `method` | `centroid_z`, `ks_cosine`, `mmd`, or a `DetectionMethod` instance | `centroid_z` |
| `window_size` | Length of the rolling recent window | 50 |
| `min_consecutive` | Require N consecutive flagged checks before reporting drift | 3 |
| `check_every` | Run detector every Nth observation | 1 |
| `baseline_size` | Soft cap on baseline length; oldest roll off if exceeded | 500 |
| `on_drift` | Callback fired once when drift is confirmed; a clean check ends the episode and re-arms it | `None` |
| `alert_cooldown` | Re-fire `on_drift` every N checks while drift stays confirmed (`None` = once per episode) | `None` |

Custom detector? Implement `DetectionMethod.run(baseline, recent) -> DetectorResult` and pass an instance to `DriftWatch(method=...)`.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
27 passed
```

Covers detector math on synthetic cluster shifts, DriftWatch state machine (pre-seal, sealed, reset), callback firing (once per drift episode, cooldown, re-arm on reset), throttling via `check_every`, embedder protocol, edge cases (empty baseline, single observation).

## Project layout

```
.
├── src/llm_drift_watch/
│   ├── __init__.py
│   ├── core.py          # DriftWatch state machine
│   ├── detectors.py     # CentroidZ, KSCosine, MMDDetector
│   └── embedders.py     # FastEmbedEmbedder, HashEmbedder
├── tests/               # 27 pytest cases
├── bench/
│   ├── drift_benchmark.py  # 3 drift types × 3 detectors
│   └── drift_results.json
└── examples/
    ├── webhook_server.py   # FastAPI alert webhook
    └── dashboard.py        # Streamlit dashboard
```

## Limitations

**Hash embedder is a stand-in.** The bundled `HashEmbedder` is fast, deterministic, and zero-dep — perfect for tests and structural drift (vocab, charset, length). It is **not** a semantic embedding. Two paraphrases of the same sentence will land in different regions of the vector space. For production monitoring, use `FastEmbedEmbedder` (default when `fastembed` is installed) or plug in any OpenAI / Cohere / sentence-transformers backend.

**MMD is slower and less sensitive in this benchmark.** Counterintuitive but real — when drift is *primarily lexical* (as in this benchmark's three streams), the 1D cosine-distance summary used by `centroid_z` and `ks_cosine` is more discriminative than the multivariate MMD over the same embeddings. MMD shines when drift is geometric in the latent space without a clean centroid shift; switch to it when `centroid_z` misses a known drift.

**Time-to-alert is window-dependent.** Smaller `window_size` reacts faster but is noisier. The defaults (`window=50`, `min_consecutive=3`) trade ~3 observation lag for 0% FPR in the benchmark. Tune for your domain.

**Drift detection ≠ root cause.** The library tells you the distribution changed. It does not tell you *why*. The recent window samples are returned in each `DriftReport.sample_text` and the full recent buffer is accessible via `watch._recent_text` for triage.

## License

MIT — see [LICENSE](LICENSE).
