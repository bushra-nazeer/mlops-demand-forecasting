"""LightGBM model factory (shared by training and backtesting)."""

from __future__ import annotations

from lightgbm import LGBMRegressor


def make_model(params: dict | None = None, random_state: int = 42) -> LGBMRegressor:
    params = dict(params or {})
    params.setdefault("n_estimators", 400)
    return LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1, **params)
