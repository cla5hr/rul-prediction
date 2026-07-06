from fastapi.testclient import TestClient

import rul_service.api as api_module
from rul_service.data import load_raw


def _client(predictor):
    # Inject the tiny trained predictor so the API doesn't need the default bundle.
    api_module._predictor = predictor
    return TestClient(api_module.app)


def test_health(predictor):
    client = _client(predictor)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(predictor):
    client = _client(predictor)
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "rmse" in body["metrics"]
    assert body["feature_columns"]


def test_predict_endpoint(predictor, trained):
    client = _client(predictor)
    test = load_raw(trained.test_file)
    eid = sorted(test["engine_id"].unique())[0]
    g = test[test["engine_id"] == eid].sort_values("cycle")
    cycles = g.drop(columns=["engine_id", "cycle"]).to_dict(orient="records")

    r = client.post("/predict", json={"engine_id": "1", "cycles": cycles})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["rul"] <= body["max_rul"]
    assert body["status"] in {"healthy", "warning", "critical"}


def test_predict_validation_error(predictor):
    client = _client(predictor)
    r = client.post("/predict", json={"cycles": []})  # empty -> 422
    assert r.status_code == 422
