"""Versioned artifact bundle: model weights + preprocessing + drift reference.

Everything needed to serve predictions lives in one directory, so deployment is
"copy the artifacts dir and run". No hidden state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch

from rul_service.data import Preprocessor
from rul_service.models import build_model

MODEL_FILE = "model.pt"
PREPROCESS_FILE = "preprocess.json"
REFERENCE_FILE = "drift_reference.json"
METRICS_FILE = "metrics.json"
META_FILE = "meta.json"


def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_bundle(
    out_dir: Path,
    model: torch.nn.Module,
    preprocessor: Preprocessor,
    reference: Dict,
    metrics: Dict,
    model_type: str,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / MODEL_FILE)
    _write_json(out_dir / PREPROCESS_FILE, preprocessor.to_dict())
    _write_json(out_dir / REFERENCE_FILE, reference)
    _write_json(out_dir / METRICS_FILE, metrics)
    _write_json(
        out_dir / META_FILE,
        {
            "model_type": model_type,
            "input_dim": preprocessor.n_features,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "framework": f"torch=={torch.__version__}",
        },
    )
    return out_dir


@dataclass
class ArtifactBundle:
    model: torch.nn.Module
    preprocessor: Preprocessor
    reference: Dict
    metrics: Dict
    meta: Dict

    @property
    def model_type(self) -> str:
        return self.meta.get("model_type", "lstm")


def load_bundle(in_dir: Path, device: str = "cpu") -> ArtifactBundle:
    in_dir = Path(in_dir)
    if not (in_dir / MODEL_FILE).exists():
        raise FileNotFoundError(
            f"No trained model found in {in_dir}. Run `rul-service train` first."
        )
    meta = _read_json(in_dir / META_FILE)
    preprocessor = Preprocessor.from_dict(_read_json(in_dir / PREPROCESS_FILE))
    model = build_model(meta["model_type"], meta["input_dim"])
    model.load_state_dict(torch.load(in_dir / MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()
    return ArtifactBundle(
        model=model,
        preprocessor=preprocessor,
        reference=_read_json(in_dir / REFERENCE_FILE),
        metrics=_read_json(in_dir / METRICS_FILE),
        meta=meta,
    )


def bundle_exists(in_dir: Path) -> bool:
    return (Path(in_dir) / MODEL_FILE).exists()
