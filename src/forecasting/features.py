"""Feature engineering: calendar features, lags, and rolling statistics.

The training builder (vectorized) and the recursive forecaster (`forecast.py`)
produce the *same* feature columns in the same order, so there is no skew.
"""

from __future__ import annotations

import holidays as holidays_pkg
import pandas as pd

from .config import Config

CALENDAR_FEATURES = ["dayofweek", "is_weekend", "month", "dayofyear", "weekofyear", "is_holiday"]


def feature_names(cfg: Config) -> list[str]:
    names = ["series_idx", *CALENDAR_FEATURES, "promo"]
    names += [f"lag_{lag}" for lag in cfg.lags]
    for window in cfg.rolling_windows:
        names += [f"rollmean_{window}", f"rollstd_{window}"]
    return names


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    out["dayofweek"] = dates.dt.dayofweek
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    out["month"] = dates.dt.month
    out["dayofyear"] = dates.dt.dayofyear
    out["weekofyear"] = dates.dt.isocalendar().week.astype(int)
    years = list(range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1))
    us_holidays = holidays_pkg.US(years=years)
    out["is_holiday"] = dates.dt.date.map(lambda d: int(d in us_holidays)).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame, lags: list[int], rolling_windows: list[int], target: str = "demand"
) -> pd.DataFrame:
    out = df.sort_values(["series_id", "date"]).copy()
    grouped = out.groupby("series_id")[target]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    for window in rolling_windows:
        by_series = shifted.groupby(out["series_id"])
        out[f"rollmean_{window}"] = by_series.transform(lambda s, w=window: s.rolling(w).mean())
        out[f"rollstd_{window}"] = by_series.transform(
            lambda s, w=window: s.rolling(w, min_periods=2).std()
        )
    return out


def build_training_frame(panel: pd.DataFrame, cfg: Config, series_ids: list[str] | None = None):
    """Return (frame, X, y, series_ids) ready for model fitting."""
    if series_ids is None:
        series_ids = sorted(panel["series_id"].unique())
    index_map = {sid: i for i, sid in enumerate(series_ids)}

    df = add_calendar_features(panel)
    df = add_lag_features(df, cfg.lags, cfg.rolling_windows)
    df["series_idx"] = df["series_id"].map(index_map)
    df = df.dropna(subset=feature_names(cfg)).reset_index(drop=True)

    X = df[feature_names(cfg)]
    y = df["demand"]
    return df, X, y, series_ids
