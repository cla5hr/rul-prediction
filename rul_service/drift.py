"""Population Stability Index (PSI) based data-drift detection.

Real predictive-maintenance models silently degrade when incoming sensor
distributions move away from the training distribution. We capture a per-feature
reference histogram at training time and score live data against it with PSI:

    PSI < 0.10  -> no significant drift
    0.10–0.25   -> moderate drift (investigate)
    > 0.25      -> significant drift (model may be unreliable)

PSI is implemented from scratch (no heavy dependency) and is fully testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

_EPS = 1e-6


def build_reference(scaled: np.ndarray, feature_columns: List[str], n_bins: int = 10) -> Dict:
    """Build per-feature reference histograms from scaled training data."""
    reference: Dict[str, Dict] = {}
    for j, name in enumerate(feature_columns):
        col = scaled[:, j]
        # Quantile-based edges give roughly balanced bins; dedupe for flat columns.
        edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([col.min() - 1e-6, col.max() + 1e-6])
        counts, _ = np.histogram(col, bins=edges)
        props = counts / max(counts.sum(), 1)
        reference[name] = {"edges": edges.tolist(), "props": props.tolist()}
    return reference


def _psi_for_column(values: np.ndarray, edges: List[float], ref_props: List[float]) -> float:
    edges_arr = np.asarray(edges)
    counts, _ = np.histogram(values, bins=edges_arr)
    actual = counts / max(counts.sum(), 1)
    expected = np.asarray(ref_props)
    actual = np.clip(actual, _EPS, None)
    expected = np.clip(expected, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


@dataclass
class DriftResult:
    per_feature_psi: Dict[str, float]
    max_psi: float
    drifted_features: List[str]
    status: str  # "ok" | "warning" | "alert"

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "max_psi": round(self.max_psi, 4),
            "drifted_features": self.drifted_features,
            "per_feature_psi": {k: round(v, 4) for k, v in self.per_feature_psi.items()},
        }


def compute_drift(
    scaled: np.ndarray,
    reference: Dict,
    feature_columns: List[str],
    warn: float = 0.1,
    alert: float = 0.25,
    min_samples: int = 100,
) -> DriftResult:
    """Score live (already-scaled) data against the training reference.

    PSI is only meaningful with enough samples to populate the histogram bins; for
    smaller inputs (e.g. a single 30-cycle engine window) we report
    ``insufficient_data`` rather than raising a false alarm.
    """
    if scaled.shape[0] < min_samples:
        return DriftResult({}, 0.0, [], "insufficient_data")

    per_feature: Dict[str, float] = {}
    for j, name in enumerate(feature_columns):
        if name not in reference:
            continue
        ref = reference[name]
        per_feature[name] = _psi_for_column(scaled[:, j], ref["edges"], ref["props"])

    max_psi = max(per_feature.values()) if per_feature else 0.0
    drifted = sorted(
        [k for k, v in per_feature.items() if v >= warn], key=lambda k: -per_feature[k]
    )
    status = "alert" if max_psi >= alert else "warning" if max_psi >= warn else "ok"
    return DriftResult(per_feature, max_psi, drifted, status)
