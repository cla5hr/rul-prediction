"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    engine_id: Optional[str] = Field(None, description="Optional engine identifier")
    cycles: List[Dict[str, float]] = Field(
        ..., min_length=1,
        description="Chronological list of cycles; each is a {column: value} map "
                    "(e.g. {'op1':.., 's2':.., ...}). The last 30 are used.",
    )


class DriftInfo(BaseModel):
    status: str
    max_psi: float
    drifted_features: List[str]
    per_feature_psi: Dict[str, float]


class PredictResponse(BaseModel):
    engine_id: Optional[str]
    rul: float = Field(..., description="Predicted remaining useful life [cycles]")
    max_rul: int
    status: str = Field(..., description="healthy | warning | critical")
    n_cycles_used: int
    model_type: str
    drift: Optional[DriftInfo] = None


class BatchPredictRequest(BaseModel):
    engines: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    fleet_drift: Optional[DriftInfo] = None
