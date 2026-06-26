"""Recursive multi-step forecasting.

Predicts one day at a time, feeding each prediction back in as the basis for the
next day's lag/rolling features — the standard way to roll a single-step model
out to a horizon. Feature construction mirrors `features.py` exactly.
"""

from __future__ import annotations

import holidays as holidays_pkg
import numpy as np
import pandas as pd

from .config import Config


def _calendar_row(d: pd.Timestamp, us_holidays) -> dict:
    return {
        "dayofweek": d.dayofweek,
        "is_weekend": int(d.dayofweek >= 5),
        "month": d.month,
        "dayofyear": d.dayofyear,
        "weekofyear": int(d.isocalendar().week),
        "is_holiday": int(d.date() in us_holidays),
    }


def forecast_series(
    model, history: pd.Series, cfg: Config, series_idx: int, horizon: int,
    feature_cols: list[str], us_holidays=None,
) -> pd.DataFrame:
    """Forecast ``horizon`` days ahead for one series.

    ``history`` is a demand Series indexed by ascending date.
    """
    if us_holidays is None:
        years = list(range(history.index.min().year, history.index.max().year + 3))
        us_holidays = holidays_pkg.US(years=years)

    demand = list(history.to_numpy(dtype=float))
    last_date = pd.Timestamp(history.index.max())
    predictions = []

    for step in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=step)
        row = {"series_idx": series_idx, "promo": 0, **_calendar_row(future_date, us_holidays)}
        values = np.asarray(demand, dtype=float)
        for lag in cfg.lags:
            row[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
        for window in cfg.rolling_windows:
            tail = values[-window:]
            row[f"rollmean_{window}"] = float(np.mean(tail)) if len(tail) else np.nan
            row[f"rollstd_{window}"] = float(np.std(tail, ddof=1)) if len(tail) >= 2 else 0.0

        X = pd.DataFrame([row])[feature_cols]
        yhat = max(float(model.predict(X)[0]), 0.0)
        predictions.append({"date": future_date, "prediction": yhat})
        demand.append(yhat)

    return pd.DataFrame(predictions)


def forecast_panel(
    model, panel: pd.DataFrame, cfg: Config, series_ids: list[str], horizon: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Forecast every series in ``panel`` and concatenate the results."""
    dates = pd.to_datetime(panel["date"])
    years = list(range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 3))
    us_holidays = holidays_pkg.US(years=years)
    index_map = {sid: i for i, sid in enumerate(series_ids)}

    out = []
    for sid, group in panel.groupby("series_id"):
        history = group.sort_values("date").set_index("date")["demand"]
        forecast = forecast_series(
            model, history, cfg, index_map[sid], horizon, feature_cols, us_holidays
        )
        forecast["series_id"] = sid
        out.append(forecast)
    return pd.concat(out, ignore_index=True)
