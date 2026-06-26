"""Rolling-origin backtesting — the honest way to estimate forecast accuracy.

For each fold we train only on data up to a cutoff, recursively forecast the
horizon, and score against the held-out actuals. Folds step backwards from the
end of the series so each evaluates a different, unseen window.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import Config, load_config
from .features import build_training_frame, feature_names
from .forecast import forecast_panel
from .generator import generate_demand
from .metrics import all_metrics
from .model import make_model


def rolling_origin_backtest(panel: pd.DataFrame, cfg: Config, params: dict | None = None) -> dict:
    dates = np.sort(pd.to_datetime(panel["date"]).unique())
    series_ids = sorted(panel["series_id"].unique())
    cols = feature_names(cfg)
    horizon, step, n_folds = cfg.horizon, cfg.backtest.step_days, cfg.backtest.n_folds
    panel = panel.assign(date=pd.to_datetime(panel["date"]))

    fold_results = []
    for fold in range(n_folds):
        offset = (n_folds - fold) * step
        cutoff_pos = len(dates) - offset - 1
        if cutoff_pos < cfg.generator.n_days // 3:
            continue
        cutoff = pd.Timestamp(dates[cutoff_pos])

        train_panel = panel[panel["date"] <= cutoff]
        _, X, y, _ = build_training_frame(train_panel, cfg, series_ids)
        model = make_model(params, cfg.random_state)
        model.fit(X, y)

        preds = forecast_panel(model, train_panel, cfg, series_ids, horizon, cols)
        horizon_end = cutoff + pd.Timedelta(days=horizon)
        actuals = panel[(panel["date"] > cutoff) & (panel["date"] <= horizon_end)]
        merged = actuals.merge(preds, on=["date", "series_id"], how="inner")
        if merged.empty:
            continue
        m = all_metrics(merged["demand"], merged["prediction"])
        m["fold"] = fold
        m["cutoff"] = str(cutoff.date())
        fold_results.append(m)

    summary = {
        "folds": fold_results,
        "mae": float(np.mean([f["mae"] for f in fold_results])) if fold_results else float("nan"),
        "rmse": float(np.mean([f["rmse"] for f in fold_results])) if fold_results else float("nan"),
        "mape": float(np.mean([f["mape"] for f in fold_results])) if fold_results else float("nan"),
        "smape": float(np.mean([f["smape"] for f in fold_results])) if fold_results else float("nan"),
    }
    return summary


def main() -> None:
    argparse.ArgumentParser(description="Run rolling-origin backtest.").parse_args()
    cfg = load_config()
    panel = generate_demand(cfg)
    print(json.dumps(rolling_origin_backtest(panel, cfg), indent=2))


if __name__ == "__main__":
    main()
