"""FastAPI service exposing RUL predictions, drift, and basic monitoring."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from rul_service.artifacts import bundle_exists
from rul_service.config import config
from rul_service.predict import Predictor
from rul_service.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

_predictor: Optional[Predictor] = None
_PRED_LOG = config.artifacts_dir / "predictions.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if bundle_exists(config.artifacts_dir):
        try:
            get_predictor()
        except Exception:
            pass
    yield


app = FastAPI(
    title="RUL Predictive Maintenance Service",
    description="Turbofan Remaining-Useful-Life predictions with drift monitoring "
                "(NASA CMAPSS FD001).",
    version="0.1.0",
    lifespan=lifespan,
)


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        if not bundle_exists(config.artifacts_dir):
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Run `rul-service train` to create artifacts.",
            )
        _predictor = Predictor.from_dir(config.artifacts_dir, config)
    return _predictor


def _log_prediction(result: dict) -> None:
    try:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "engine_id": result.get("engine_id"),
            "rul": result.get("rul"),
            "status": result.get("status"),
            "drift_status": (result.get("drift") or {}).get("status"),
        }
        with _PRED_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": bundle_exists(config.artifacts_dir)}


@app.get("/model/info")
def model_info() -> dict:
    p = get_predictor()
    return {
        "meta": p.bundle.meta,
        "metrics": p.bundle.metrics,
        "feature_columns": p.bundle.preprocessor.feature_columns,
        "sequence_length": p.bundle.preprocessor.sequence_length,
        "max_rul": p.bundle.preprocessor.max_rul,
        "thresholds": {"critical": config.critical_rul, "warning": config.warning_rul},
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    result = get_predictor().predict(req.cycles, engine_id=req.engine_id, with_drift=True)
    _log_prediction(result)
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> dict:
    p = get_predictor()
    out = p.predict_batch([e.model_dump() for e in req.engines])
    for r in out["results"]:
        _log_prediction(r)
    return out


@app.get("/stats")
def stats() -> dict:
    """Lightweight monitoring: summarise logged predictions."""
    if not _PRED_LOG.exists():
        return {"total": 0, "by_status": {}, "by_drift_status": {}}
    statuses, drifts = Counter(), Counter()
    total = 0
    for line in _PRED_LOG.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        statuses[e.get("status")] += 1
        drifts[e.get("drift_status")] += 1
    return {"total": total, "by_status": dict(statuses), "by_drift_status": dict(drifts)}
