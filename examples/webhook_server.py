"""FastAPI server that runs a DriftWatch and forwards drift alerts to a webhook.

Run with::

    pip install -e ".[fastembed,server]"
    DRIFT_WEBHOOK_URL=https://hooks.example.com/drift uvicorn examples.webhook_server:app --reload

Endpoints:

* ``POST /baseline``  — body ``{"texts": ["..."]}`` — append to baseline
* ``POST /seal``      — freeze baseline, start monitoring
* ``POST /observe``   — body ``{"text": "..."}`` — observe and possibly fire webhook
* ``GET /status``     — current sizes + last detector result
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_drift_watch import DriftReport, DriftWatch, get_embedder


class BaselineRequest(BaseModel):
    texts: list[str]


class ObserveRequest(BaseModel):
    text: str


app = FastAPI(title="llm-drift-watch webhook server")

# Default to fastembed if installed, otherwise hash. Override via DRIFT_EMBEDDER.
_watch = DriftWatch(
    embedder=get_embedder(os.environ.get("DRIFT_EMBEDDER")),
    method=os.environ.get("DRIFT_METHOD", "centroid_z"),
    window_size=int(os.environ.get("DRIFT_WINDOW", "50")),
    min_consecutive=int(os.environ.get("DRIFT_MIN_CONSECUTIVE", "3")),
)
_webhook_url = os.environ.get("DRIFT_WEBHOOK_URL")
_last_report: DriftReport | None = None


def _fire_webhook(report: DriftReport) -> None:
    if not _webhook_url:
        return
    payload = {
        "drift_detected": report.drift_detected,
        "method": report.method,
        "score": report.score,
        "p_value": report.p_value,
        "n_baseline": report.n_baseline,
        "n_recent": report.n_recent,
        "sample_text": report.sample_text,
        "details": report.details,
    }
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(_webhook_url, json=payload)
    except httpx.HTTPError:
        # Webhook delivery is best-effort. Log to stdout so operators can see it.
        print(f"[drift] webhook delivery to {_webhook_url} failed")


_watch.on_drift = _fire_webhook


@app.post("/baseline")
def add_baseline(req: BaselineRequest) -> dict[str, Any]:
    if _watch.is_sealed:
        raise HTTPException(409, "Baseline is already sealed. POST /reset first.")
    for t in req.texts:
        _watch.observe(t)
    return {"n_baseline": _watch.n_baseline}


@app.post("/seal")
def seal() -> dict[str, Any]:
    if _watch.is_sealed:
        return {"already_sealed": True, "n_baseline": _watch.n_baseline}
    _watch.seal_baseline()
    return {"n_baseline": _watch.n_baseline, "sealed": True}


@app.post("/reset")
def reset() -> dict[str, Any]:
    _watch.reset(keep_baseline=False)
    return {"reset": True}


@app.post("/observe")
def observe(req: ObserveRequest) -> dict[str, Any]:
    global _last_report
    report = _watch.observe(req.text)
    if report is None:
        return {"phase": "baseline", "n_baseline": _watch.n_baseline}
    _last_report = report
    return {
        "phase": "monitoring",
        "drift_detected": report.drift_detected,
        "score": report.score,
        "p_value": report.p_value,
        "method": report.method,
        "consecutive_flags": report.consecutive_flags,
    }


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "is_sealed": _watch.is_sealed,
        "n_baseline": _watch.n_baseline,
        "n_recent": _watch.n_recent,
        "method": _watch.method.name,
        "last_report": (
            {
                "drift_detected": _last_report.drift_detected,
                "score": _last_report.score,
                "p_value": _last_report.p_value,
                "consecutive_flags": _last_report.consecutive_flags,
            }
            if _last_report is not None
            else None
        ),
    }
