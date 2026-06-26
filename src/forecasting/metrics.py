"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred) -> float:
    diff = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(diff**2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def all_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
