"""Command-line interface for the RUL service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rul_service.config import config

app = typer.Typer(add_completion=False, help="Train and serve the RUL predictive-maintenance model.")
console = Console()


@app.command()
def train(
    model: str = typer.Option(config.model_type, help="lstm | transformer"),
    epochs: int = typer.Option(config.epochs),
):
    """Train a model and write the deployable artifact bundle."""
    import os

    os.environ["RUL_MODEL"] = model
    os.environ["RUL_EPOCHS"] = str(epochs)
    # Re-read config from the freshly set environment.
    from importlib import reload

    from rul_service import config as cfgmod

    reload(cfgmod)
    from rul_service.train import train as run_train

    metrics = run_train(cfgmod.config)
    console.print(f"[green]Done.[/] Test RMSE={metrics['rmse']}  MAE={metrics['mae']}")


@app.command()
def evaluate():
    """Show the metrics stored in the current artifact bundle."""
    from rul_service.artifacts import load_bundle

    bundle = load_bundle(config.artifacts_dir)
    t = Table(title="RUL model")
    t.add_column("field")
    t.add_column("value")
    for k, v in {**bundle.meta, **bundle.metrics}.items():
        t.add_row(str(k), str(v))
    console.print(t)


@app.command()
def predict(
    file: Path = typer.Argument(..., help="JSON file: {engine_id?, cycles:[{col:val},...]}"),
):
    """Predict RUL for an engine described in a JSON file."""
    from rul_service.predict import Predictor

    payload = json.loads(Path(file).read_text())
    p = Predictor.from_dir(config.artifacts_dir, config)
    result = p.predict(payload["cycles"], engine_id=payload.get("engine_id"))
    console.print_json(data=result)


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)):
    """Launch the FastAPI server."""
    import uvicorn

    uvicorn.run("rul_service.api:app", host=host, port=port, reload=False)


@app.command("sample")
def sample(
    engine_index: int = typer.Option(0, help="Which test engine to dump (0-based)"),
    out: Optional[Path] = typer.Option(None, help="Write request JSON here"),
):
    """Dump a real test-engine cycle history as a ready-to-send request payload."""
    from rul_service.data import load_raw

    test_raw = load_raw(config.test_file)
    ids = sorted(test_raw["engine_id"].unique())
    eid = ids[engine_index]
    g = test_raw[test_raw["engine_id"] == eid].sort_values("cycle")
    cycles = g.drop(columns=["engine_id", "cycle"]).to_dict(orient="records")
    payload = {"engine_id": str(eid), "cycles": cycles}
    text = json.dumps(payload, indent=2)
    if out:
        Path(out).write_text(text)
        console.print(f"[green]Wrote[/] {out} ({len(cycles)} cycles)")
    else:
        console.print_json(data=payload)


if __name__ == "__main__":
    app()
