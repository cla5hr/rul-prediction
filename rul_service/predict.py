"""Inference: raw cycle history -> RUL prediction, maintenance alert, and drift."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from rul_service.artifacts import ArtifactBundle, load_bundle
from rul_service.config import Config, config
from rul_service.drift import DriftResult, compute_drift


class Predictor:
    """Loads an artifact bundle once and serves predictions."""

    def __init__(self, bundle: ArtifactBundle, cfg: Config = config, device: str = "cpu"):
        self.bundle = bundle
        self.cfg = cfg
        self.device = device

    @classmethod
    def from_dir(cls, artifacts_dir: Optional[Path] = None, cfg: Config = config) -> "Predictor":
        bundle = load_bundle(artifacts_dir or cfg.artifacts_dir)
        return cls(bundle, cfg=cfg)

    # ------------------------------------------------------------------ utils
    def _scaled_from_cycles(self, cycles: List[Dict[str, float]]) -> np.ndarray:
        """Select the model's feature columns from raw cycles and scale them."""
        pre = self.bundle.preprocessor
        df = pd.DataFrame(cycles)
        # Ensure every feature column exists; missing -> 0 (then scaled).
        for col in pre.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        arr = df[pre.feature_columns].to_numpy(dtype=float)
        return pre.scale_array(arr)

    def alert_status(self, rul: float) -> str:
        if rul <= self.cfg.critical_rul:
            return "critical"
        if rul <= self.cfg.warning_rul:
            return "warning"
        return "healthy"

    # --------------------------------------------------------------- predict
    def predict(
        self,
        cycles: List[Dict[str, float]],
        engine_id: Optional[str] = None,
        with_drift: bool = True,
    ) -> Dict:
        pre = self.bundle.preprocessor
        scaled = self._scaled_from_cycles(cycles)
        window = pre.window_from_scaled(scaled)  # (seq_len, n_features)
        x = torch.as_tensor(window[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            rul = float(self.bundle.model(x).item())
        rul = max(0.0, min(rul, float(pre.max_rul)))

        result: Dict = {
            "engine_id": engine_id,
            "rul": round(rul, 2),
            "max_rul": pre.max_rul,
            "status": self.alert_status(rul),
            "n_cycles_used": int(min(len(cycles), pre.sequence_length)),
            "model_type": self.bundle.model_type,
        }
        if with_drift:
            result["drift"] = self.drift(scaled).to_dict()
        return result

    def drift(self, scaled: np.ndarray) -> DriftResult:
        pre = self.bundle.preprocessor
        return compute_drift(
            scaled,
            self.bundle.reference,
            pre.feature_columns,
            warn=self.cfg.psi_warn,
            alert=self.cfg.psi_alert,
        )

    def predict_batch(self, engines: List[Dict]) -> Dict:
        results = [
            self.predict(e["cycles"], engine_id=e.get("engine_id"), with_drift=True)
            for e in engines
        ]
        # Fleet-level drift: stack every engine's scaled cycles.
        all_scaled = [self._scaled_from_cycles(e["cycles"]) for e in engines]
        fleet_drift = None
        if all_scaled:
            stacked = np.vstack(all_scaled)
            fleet_drift = self.drift(stacked).to_dict()
        return {"results": results, "fleet_drift": fleet_drift}
