"""Production monitoring: compare live forecast error against the backtest
baseline and check for data drift (PSI). Emits a retrain decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Config, load_config
from .forecast import forecast_panel
from .generator import generate_demand
from .metrics import mae

_EPS = 1e-6


def psi(reference, current, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, bins=edges)[0] / max(len(reference), 1), _EPS, None)
    cur = np.clip(np.histogram(current, bins=edges)[0] / max(len(current), 1), _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def check_drift(cfg: Config, panel: pd.DataFrame | None = None) -> dict:
    metadata = json.loads(Path(cfg.paths.model_metadata).read_text())
    model = joblib.load(cfg.paths.model_path)
    series_ids = metadata["series_ids"]
    cols = metadata["feature_names"]
    backtest_mae = metadata["metrics"]["backtest_mae"]

    panel = generate_demand(cfg) if panel is None else panel
    panel = panel.assign(date=pd.to_datetime(panel["date"]))
    dates = np.sort(panel["date"].unique())
    cutoff = pd.Timestamp(dates[-cfg.horizon - 1])

    train_region = panel[panel["date"] <= cutoff]
    preds = forecast_panel(model, train_region, cfg, series_ids, cfg.horizon, cols)
    actuals = panel[panel["date"] > cutoff]
    merged = actuals.merge(preds, on=["date", "series_id"], how="inner")
    live_mae = mae(merged["demand"], merged["prediction"])

    drift_psi = psi(train_region["demand"], actuals["demand"])
    ratio = live_mae / backtest_mae if backtest_mae else float("nan")
    needs_retrain = bool(
        ratio > cfg.monitor.mae_degradation_ratio or drift_psi > cfg.monitor.psi_threshold
    )
    return {
        "live_mae": live_mae,
        "backtest_mae": backtest_mae,
        "mae_ratio": ratio,
        "demand_psi": drift_psi,
        "needs_retrain": needs_retrain,
    }


def main() -> None:
    argparse.ArgumentParser(description="Monitor the deployed forecaster.").parse_args()
    cfg = load_config()
    print(json.dumps(check_drift(cfg), indent=2))


if __name__ == "__main__":
    main()
