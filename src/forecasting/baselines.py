"""Classical forecasting baselines for comparison: seasonal-naive and
Holt-Winters exponential smoothing (statsmodels). Prophet is supported as an
optional extra and imported lazily if installed."""

from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    values = history.to_numpy(dtype=float)
    if len(values) < season:
        return np.repeat(values[-1] if len(values) else 0.0, horizon)
    return np.array([values[-season + ((h - 1) % season)] for h in range(1, horizon + 1)])


def ets_forecast(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    """Holt-Winters additive trend + weekly seasonality; falls back to
    seasonal-naive if the fit fails."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    values = history.to_numpy(dtype=float)
    try:
        fit = ExponentialSmoothing(
            values, trend="add", seasonal="add", seasonal_periods=season,
            initialization_method="estimated",
        ).fit()
        return np.clip(np.asarray(fit.forecast(horizon), dtype=float), 0.0, None)
    except Exception:
        return seasonal_naive_forecast(history, horizon, season)


def prophet_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Optional Prophet forecast (requires the `prophet` extra)."""
    from prophet import Prophet

    frame = pd.DataFrame({"ds": history.index, "y": history.to_numpy(dtype=float)})
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(frame)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return np.clip(forecast["yhat"].to_numpy()[-horizon:], 0.0, None)
