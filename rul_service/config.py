"""Configuration and constants for the RUL service.

Values mirror the original notebook so results are reproducible, and are overridable
via environment variables for deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

# Raw CMAPSS column layout: engine id, cycle, 3 operational settings, 21 sensors.
OP_COLUMNS: List[str] = ["op1", "op2", "op3"]
SENSOR_COLUMNS: List[str] = [f"s{i}" for i in range(1, 22)]
ALL_COLUMNS: List[str] = ["engine_id", "cycle"] + OP_COLUMNS + SENSOR_COLUMNS


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


@dataclass(frozen=True)
class Config:
    # --- preprocessing (matches the notebook) ---
    sequence_length: int = int(os.environ.get("RUL_SEQ_LEN", "30"))
    max_rul: int = int(os.environ.get("RUL_MAX", "125"))
    constant_std_threshold: float = float(os.environ.get("RUL_STD_THRESH", "0.1"))

    # --- training ---
    model_type: str = os.environ.get("RUL_MODEL", "lstm")  # "lstm" | "transformer"
    epochs: int = int(os.environ.get("RUL_EPOCHS", "50"))
    batch_size: int = int(os.environ.get("RUL_BATCH", "64"))
    learning_rate: float = float(os.environ.get("RUL_LR", "1e-3"))
    seed: int = int(os.environ.get("RUL_SEED", "42"))

    # --- maintenance alert thresholds (in cycles) ---
    critical_rul: int = int(os.environ.get("RUL_CRITICAL", "20"))
    warning_rul: int = int(os.environ.get("RUL_WARNING", "50"))

    # --- drift ---
    psi_warn: float = float(os.environ.get("RUL_PSI_WARN", "0.1"))
    psi_alert: float = float(os.environ.get("RUL_PSI_ALERT", "0.25"))

    # --- paths ---
    data_dir: Path = field(default_factory=lambda: _env_path("RUL_DATA_DIR", PROJECT_DIR / "data"))
    artifacts_dir: Path = field(
        default_factory=lambda: _env_path("RUL_ARTIFACTS_DIR", PROJECT_DIR / "artifacts")
    )

    @property
    def train_file(self) -> Path:
        return self.data_dir / "train_FD001.txt"

    @property
    def test_file(self) -> Path:
        return self.data_dir / "test_FD001.txt"

    @property
    def rul_file(self) -> Path:
        return self.data_dir / "RUL_FD001.txt"


config = Config()
