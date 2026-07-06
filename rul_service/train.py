"""Reproducible training pipeline that emits a deployable artifact bundle."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rul_service.artifacts import save_bundle
from rul_service.config import Config, config
from rul_service.data import Preprocessor, add_training_rul, load_raw
from rul_service.drift import build_reference
from rul_service.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: str) -> Dict:
    model.eval()
    with torch.no_grad():
        preds = model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    err = preds - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "n_test": int(len(y))}


def train(cfg: Config = config, verbose: bool = True) -> Dict:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data ---
    train_raw = load_raw(cfg.train_file)
    test_raw = load_raw(cfg.test_file)
    y_test = load_raw_rul(cfg.rul_file)

    train_df = add_training_rul(train_raw, cfg.max_rul)
    pre = Preprocessor.fit(train_df, cfg)
    X_train, y_train = pre.make_train_sequences(train_df)
    X_test = pre.make_test_windows(test_raw)

    if verbose:
        print(f"Features ({pre.n_features}): {pre.feature_columns}")
        print(f"X_train={X_train.shape}  X_test={X_test.shape}")

    # --- model ---
    model = build_model(cfg.model_type, pre.n_features).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model={cfg.model_type}  params={n_params:,}")

    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            avg = running / len(loader)
            print(f"epoch {epoch+1}/{cfg.epochs}  loss={avg:.4f}  rmse={avg**0.5:.4f}")

    # --- evaluate + drift reference ---
    metrics = evaluate(model, X_test, y_test, device)
    metrics["model_type"] = cfg.model_type
    metrics["n_params"] = n_params
    if verbose:
        print(f"TEST  RMSE={metrics['rmse']}  MAE={metrics['mae']}")

    scaled_train = pre.select_and_scale(train_df)
    reference = build_reference(scaled_train, pre.feature_columns)

    save_bundle(cfg.artifacts_dir, model, pre, reference, metrics, cfg.model_type)
    if verbose:
        print(f"Saved artifacts to {cfg.artifacts_dir}")
    return metrics


def load_raw_rul(path: Path) -> np.ndarray:
    import pandas as pd

    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)


if __name__ == "__main__":
    train()
