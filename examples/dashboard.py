"""Streamlit dashboard for llm-drift-watch.

Run with::

    pip install -e ".[dashboard,fastembed]"
    streamlit run examples/dashboard.py

The dashboard wires a DriftWatch into a session-scoped state object. You can
paste baseline texts into one field, seal, then stream new observations and
watch the drift score over time alongside a baseline-vs-recent distribution
histogram.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_drift_watch import DriftWatch, get_embedder


st.set_page_config(page_title="llm-drift-watch", layout="wide")
st.title("llm-drift-watch — live drift dashboard")


@st.cache_resource
def _build_watch(method: str, window: int, min_consec: int) -> DriftWatch:
    return DriftWatch(
        embedder=get_embedder(),
        method=method,
        window_size=window,
        min_consecutive=min_consec,
        baseline_size=None,
    )


with st.sidebar:
    st.header("Configuration")
    method = st.selectbox("Detector", ["centroid_z", "ks_cosine", "mmd"], index=0)
    window = st.slider("Window size", 5, 200, 50)
    min_consec = st.slider("Min consecutive flags", 1, 10, 3)
    if st.button("Rebuild watcher"):
        _build_watch.clear()
        st.session_state.history = []


watch = _build_watch(method, window, min_consec)
if "history" not in st.session_state:
    st.session_state.history = []  # type: list[dict]


col_baseline, col_observe = st.columns(2)

with col_baseline:
    st.subheader("Baseline")
    if not watch.is_sealed:
        baseline_text = st.text_area(
            "Paste baseline texts (one per line):", height=200,
            value="How do I make pasta?\nWhat's a good chicken recipe?\nHow long should I cook rice?",
        )
        if st.button("Add to baseline"):
            for line in baseline_text.splitlines():
                line = line.strip()
                if line:
                    watch.observe(line)
            st.success(f"Baseline now has {watch.n_baseline} texts.")
        if watch.n_baseline > 0 and st.button("Seal baseline"):
            watch.seal_baseline()
            st.session_state.history = []
            st.rerun()
    else:
        st.success(f"Baseline sealed with {watch.n_baseline} texts.")


with col_observe:
    st.subheader("Observe")
    if watch.is_sealed:
        observe_text = st.text_input("New observation:", "")
        if st.button("Observe") and observe_text.strip():
            report = watch.observe(observe_text.strip())
            if report is not None:
                st.session_state.history.append(
                    {
                        "step": len(st.session_state.history) + 1,
                        "score": report.score,
                        "p_value": report.p_value if report.p_value is not None else 1.0,
                        "drift_detected": report.drift_detected,
                        "text": observe_text.strip()[:80],
                        "consecutive_flags": report.consecutive_flags,
                        "ts": time.time(),
                    }
                )
                if report.drift_detected:
                    st.error(
                        f"DRIFT DETECTED — score={report.score:.2f} "
                        f"p={report.p_value} consecutive={report.consecutive_flags}"
                    )
                else:
                    st.info(f"OK — score={report.score:.2f} consecutive={report.consecutive_flags}")
    else:
        st.info("Add and seal a baseline first.")


# ---- Charts ----
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    st.subheader("Drift score over time")
    chart = alt.Chart(df).mark_line(point=True).encode(
        x="step:Q",
        y=alt.Y("score:Q", title="detector score (z or KS stat)"),
        color=alt.condition(
            "datum.drift_detected", alt.value("red"), alt.value("steelblue")
        ),
        tooltip=["step", "score", "p_value", "text"],
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Recent observations")
    st.dataframe(df[["step", "text", "score", "p_value", "drift_detected", "consecutive_flags"]])
