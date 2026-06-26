"""Train the LightGBM demand forecaster: Optuna tuning on a holdout, an honest
rolling-origin backtest for reported metrics, MLflow registry, and persistence
(including recent history so the API can seed lag features)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from . import registry
from .backtest import rolling_origin_backtest
from .config import Config, load_config
from .features import build_training_frame, feature_names
from .forecast import forecast_panel
from .generator import generate_demand
from .metrics import smape
from .model import make_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _holdout_smape(panel: pd.DataFrame, cfg: Config, params: dict, series_ids: list[str]) -> float:
    panel = panel.assign(date=pd.to_datetime(panel["date"]))
    dates = np.sort(panel["date"].unique())
    cutoff = pd.Timestamp(dates[-cfg.horizon - 1])
    train_region = panel[panel["date"] <= cutoff]
    _, X, y, _ = build_training_frame(train_region, cfg, series_ids)
    model = make_model(params, cfg.random_state)
    model.fit(X, y)
    preds = forecast_panel(model, train_region, cfg, series_ids, cfg.horizon, feature_names(cfg))
    actuals = panel[panel["date"] > cutoff]
    merged = actuals.merge(preds, on=["date", "series_id"], how="inner")
    return smape(merged["demand"], merged["prediction"])


def train(cfg: Config, n_trials: int | None = None) -> dict:
    panel = generate_demand(cfg)
    series_ids = sorted(panel["series_id"].unique())
    n_trials = cfg.optuna.n_trials if n_trials is None else n_trials

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        return _holdout_smape(panel, cfg, params, series_ids)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=cfg.optuna.timeout_seconds)
    best_params = dict(study.best_params)

    _, X, y, _ = build_training_frame(panel, cfg, series_ids)
    model = make_model(best_params, cfg.random_state)
    model.fit(X, y)

    bt = rolling_origin_backtest(panel, cfg, best_params)
    metrics = {
        "backtest_mae": bt["mae"],
        "backtest_rmse": bt["rmse"],
        "backtest_mape": bt["mape"],
        "backtest_smape": bt["smape"],
    }

    Path(cfg.paths.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.paths.model_path)

    keep = max(max(cfg.lags), max(cfg.rolling_windows)) + 5
    history = panel.sort_values("date").groupby("series_id").tail(keep)[["date", "series_id", "demand"]].copy()
    history["date"] = history["date"].astype(str)
    metadata = {
        "best_params": best_params,
        "metrics": metrics,
        "backtest_folds": bt["folds"],
        "feature_names": feature_names(cfg),
        "series_ids": series_ids,
        "horizon": cfg.horizon,
        "recent_history": history.to_dict("records"),
        "dataset": "synthetic multi-series demand (forecasting.generator)",
    }
    Path(cfg.paths.model_metadata).write_text(json.dumps(metadata, indent=2))

    try:
        version = registry.register_champion(model, cfg, metrics)
        print(f"Registered '{cfg.registered_model_name}' version {version} (alias: champion)")
    except Exception as exc:
        print(f"MLflow registry step skipped: {exc}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the demand forecaster.")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    print(json.dumps(train(cfg, n_trials=args.trials), indent=2))


if __name__ == "__main__":
    main()
