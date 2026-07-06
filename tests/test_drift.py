import numpy as np

from rul_service.drift import build_reference, compute_drift

FEATURES = ["a", "b", "c"]


def _ref(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((n, len(FEATURES)))
    return data, build_reference(data, FEATURES)


def test_no_drift_on_same_distribution():
    data, ref = _ref()
    fresh = np.random.default_rng(1).random((2000, len(FEATURES)))
    result = compute_drift(fresh, ref, FEATURES)
    assert result.status == "ok"
    assert result.max_psi < 0.1


def test_alert_on_shifted_distribution():
    data, ref = _ref()
    shifted = np.random.default_rng(2).random((2000, len(FEATURES))) + 0.6
    result = compute_drift(shifted, ref, FEATURES)
    assert result.status == "alert"
    assert result.max_psi >= 0.25
    assert result.drifted_features


def test_insufficient_data_guard():
    data, ref = _ref()
    tiny = np.random.default_rng(3).random((30, len(FEATURES)))
    result = compute_drift(tiny, ref, FEATURES, min_samples=100)
    assert result.status == "insufficient_data"
