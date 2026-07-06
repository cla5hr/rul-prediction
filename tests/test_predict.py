from rul_service.data import load_raw


def _engine_cycles(cfg, index=0):
    test = load_raw(cfg.test_file)
    eid = sorted(test["engine_id"].unique())[index]
    g = test[test["engine_id"] == eid].sort_values("cycle")
    return g.drop(columns=["engine_id", "cycle"]).to_dict(orient="records")


def test_predict_structure_and_bounds(predictor, trained):
    cycles = _engine_cycles(trained)
    res = predictor.predict(cycles, engine_id="1")
    assert 0.0 <= res["rul"] <= predictor.bundle.preprocessor.max_rul
    assert res["status"] in {"healthy", "warning", "critical"}
    assert res["model_type"] == "lstm"
    assert res["drift"]["status"] == "insufficient_data"  # single engine window


def test_alert_status_thresholds(predictor):
    assert predictor.alert_status(5) == "critical"
    assert predictor.alert_status(40) == "warning"
    assert predictor.alert_status(120) == "healthy"


def test_missing_columns_are_tolerated(predictor):
    # Only provide a couple of columns; the rest default to 0.
    cycles = [{"s2": 640.0, "s3": 1590.0} for _ in range(35)]
    res = predictor.predict(cycles, engine_id="x", with_drift=False)
    assert "rul" in res


def test_batch_fleet_drift(predictor, trained):
    engines = [
        {"engine_id": str(i), "cycles": _engine_cycles(trained, i)} for i in range(20)
    ]
    out = predictor.predict_batch(engines)
    assert len(out["results"]) == 20
    assert out["fleet_drift"]["status"] in {"ok", "warning", "alert"}
