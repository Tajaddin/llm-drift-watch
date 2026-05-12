"""Embedding-based drift detector for LLM input/output streams.

Track a baseline window of texts (prompts, completions, etc.), then watch a
rolling recent window for distributional drift. Three statistical detectors
are built in: ``centroid_z`` (default, fast), ``ks_cosine`` (KS test on cosine
distance distribution), and ``mmd`` (Maximum Mean Discrepancy with Gaussian
kernel and permutation test).
"""

from llm_drift_watch.core import DriftReport, DriftWatch
from llm_drift_watch.detectors import (
    DetectionMethod,
    CentroidZ,
    KSCosine,
    MMDDetector,
    DetectorResult,
)
from llm_drift_watch.embedders import Embedder, FastEmbedEmbedder, HashEmbedder, get_embedder

__version__ = "0.1.0"

__all__ = [
    # core
    "DriftWatch",
    "DriftReport",
    # detectors
    "DetectionMethod",
    "CentroidZ",
    "KSCosine",
    "MMDDetector",
    "DetectorResult",
    # embedders
    "Embedder",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "get_embedder",
]
