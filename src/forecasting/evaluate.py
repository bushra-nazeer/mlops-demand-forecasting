"""Final-holdout evaluation: LightGBM vs classical baselines (ETS, seasonal
naive), plus forecast-vs-actual and backtest-by-fold plots."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import ets_forecast, seasonal_naive_forecast
from .config import Config, load_config
from .forecast import forecast_panel
from .generator import generate_demand
from .metrics import all_metrics, smape

# Non-interactive backend so figures render in headless containers/CI.
matplotlib.use("Agg")


def evaluate(cfg: Config) -> dict:
    panel = generate_demand(cfg).assign(date=lambda d: pd.to_datetime(d["date"]))
    model = joblib.load(cfg.paths.model_path)
    metadata = json.loads(Path(cfg.paths.model_metadata).read_text())
    series_ids = metadata["series_ids"]
    cols = metadata["feature_names"]

    dates = np.sort(panel["date"].unique())
    cutoff = pd.Timestamp(dates[-cfg.horizon - 1])
    train_region = panel[panel["date"] <= cutoff]
    actuals = panel[panel["date"] > cutoff]

    preds = forecast_panel(model, train_region, cfg, series_ids, cfg.horizon, cols)
    merged = actuals.merge(preds, on=["date", "series_id"], how="inner")
    lgbm = all_metrics(merged["demand"], merged["prediction"])

    naive_smape, ets_smape = [], []
    for sid, group in train_region.groupby("series_id"):
        history = group.sort_values("date").set_index("date")["demand"]
        actual = actuals[actuals["series_id"] == sid].sort_values("date")["demand"].to_numpy()
        if len(actual) < cfg.horizon:
            continue
        naive_smape.append(smape(actual, seasonal_naive_forecast(history, cfg.horizon)))
        ets_smape.append(smape(actual, ets_forecast(history, cfg.horizon)))

    comparison = {
        "lightgbm_smape": lgbm["smape"],
        "ets_smape": float(np.mean(ets_smape)) if ets_smape else None,
        "seasonal_naive_smape": float(np.mean(naive_smape)) if naive_smape else None,
    }
    out = {"holdout": lgbm, "comparison": comparison, "backtest": metadata["metrics"]}

    reports = Path(cfg.paths.reports_dir)
    figures = Path(cfg.paths.figures_dir)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(out, indent=2))

    sid = series_ids[0]
    hist = train_region[train_region["series_id"] == sid].sort_values("date").tail(90)
    act = actuals[actuals["series_id"] == sid].sort_values("date")
    pr = preds[preds["series_id"] == sid].sort_values("date")
    plt.figure(figsize=(10, 4))
    plt.plot(hist["date"], hist["demand"], color="#46688E", label="history")
    plt.plot(act["date"], act["demand"], color="#15191C", label="actual")
    plt.plot(pr["date"], pr["prediction"], color="#0E7C66", linestyle="--", label="forecast")
    plt.title(f"Forecast vs actual — {sid} ({cfg.horizon}-day horizon)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "forecast_vs_actual.png", dpi=120)
    plt.close()

    folds = metadata.get("backtest_folds", [])
    if folds:
        plt.figure(figsize=(7, 4))
        plt.bar([f"fold {f['fold']}" for f in folds], [f["smape"] for f in folds], color="#0E7C66")
        plt.ylabel("sMAPE (%)")
        plt.title("Backtest sMAPE by fold (rolling origin)")
        plt.tight_layout()
        plt.savefig(figures / "backtest_by_fold.png", dpi=120)
        plt.close()

    return out


def main() -> None:
    cfg = load_config()
    print(json.dumps(evaluate(cfg), indent=2))


if __name__ == "__main__":
    main()
