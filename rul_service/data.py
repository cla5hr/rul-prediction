"""CMAPSS data loading, preprocessing, and windowing.

The :class:`Preprocessor` captures exactly what was learned from the training set
(which columns survive variance filtering and the MinMax scale parameters) so the
identical transform can be replayed at inference time. This is the bridge that lets
a notebook model become a reliable service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rul_service.config import ALL_COLUMNS, OP_COLUMNS, SENSOR_COLUMNS, Config, config


def load_raw(path: Path) -> pd.DataFrame:
    """Load a CMAPSS FD00x text file into a DataFrame with named columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Files may have trailing whitespace producing extra all-NaN columns.
    df = df.iloc[:, : len(ALL_COLUMNS)]
    df.columns = ALL_COLUMNS
    return df


def add_training_rul(train_df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    df = train_df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=max_rul)
    return df


@dataclass
class Preprocessor:
    """Fitted preprocessing state, fully serialisable."""

    feature_columns: List[str] = field(default_factory=list)
    data_min: List[float] = field(default_factory=list)
    data_max: List[float] = field(default_factory=list)
    sequence_length: int = 30
    max_rul: int = 125

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, train_df: pd.DataFrame, cfg: Config = config) -> "Preprocessor":
        # Drop sensors that are (near) constant on the training set + op3.
        constant = [
            s for s in SENSOR_COLUMNS if train_df[s].std() < cfg.constant_std_threshold
        ]
        feature_columns = [
            c for c in (OP_COLUMNS + SENSOR_COLUMNS)
            if c not in constant and c != "op3"
        ]
        sub = train_df[feature_columns]
        return cls(
            feature_columns=feature_columns,
            data_min=sub.min().tolist(),
            data_max=sub.max().tolist(),
            sequence_length=cfg.sequence_length,
            max_rul=cfg.max_rul,
        )

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    # ------------------------------------------------------------- transform
    def _range(self) -> np.ndarray:
        rng = np.asarray(self.data_max) - np.asarray(self.data_min)
        rng[rng == 0] = 1.0  # avoid division by zero for any flat column
        return rng

    def scale_array(self, arr: np.ndarray) -> np.ndarray:
        """Scale a (n, n_features) array of raw feature values to [0, 1]."""
        return (arr - np.asarray(self.data_min)) / self._range()

    def select_and_scale(self, df: pd.DataFrame) -> np.ndarray:
        return self.scale_array(df[self.feature_columns].to_numpy(dtype=float))

    # ----------------------------------------------------------- windowing
    def make_train_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding windows over each engine; label = RUL at the window's last step."""
        xs, ys = [], []
        for _, g in df.groupby("engine_id"):
            g = g.sort_values("cycle")
            feats = self.scale_array(g[self.feature_columns].to_numpy(dtype=float))
            rul = g["RUL"].to_numpy(dtype=float)
            n = len(g)
            for i in range(n - self.sequence_length + 1):
                xs.append(feats[i : i + self.sequence_length])
                ys.append(rul[i + self.sequence_length - 1])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def make_test_windows(self, df: pd.DataFrame) -> np.ndarray:
        """Last ``sequence_length`` cycles per engine (front-padded if shorter)."""
        windows = []
        for _, g in df.groupby("engine_id"):
            g = g.sort_values("cycle")
            windows.append(self._last_window(self.scale_array(
                g[self.feature_columns].to_numpy(dtype=float))))
        return np.asarray(windows, dtype=np.float32)

    def window_from_scaled(self, scaled: np.ndarray) -> np.ndarray:
        """Build a single model-ready window from already-scaled rows."""
        return self._last_window(scaled)

    def _last_window(self, feats: np.ndarray) -> np.ndarray:
        L = self.sequence_length
        if len(feats) >= L:
            return feats[-L:]
        pad = np.zeros((L - len(feats), self.n_features), dtype=feats.dtype)
        return np.vstack([pad, feats])

    # ----------------------------------------------------------- (de)serialise
    def to_dict(self) -> Dict:
        return {
            "feature_columns": self.feature_columns,
            "data_min": self.data_min,
            "data_max": self.data_max,
            "sequence_length": self.sequence_length,
            "max_rul": self.max_rul,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Preprocessor":
        return cls(**d)
