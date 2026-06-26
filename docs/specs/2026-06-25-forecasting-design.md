# mlops-demand-forecasting, Design Spec

- **Date:** 2026-06-25
- **Status:** Approved
- **Portfolio role:** MLOps lifecycle flagship (repo 3 of 6), demonstrates the
  production discipline shared across all three resume personas.

## Overview

Demand forecasting on a synthetic multi-series panel, wrapped in a full MLOps
lifecycle: rolling-origin backtesting, MLflow tracking + model registry,
scheduled automated retraining gated by drift/error monitoring, and a FastAPI
forecasting service. Self-contained and reproducible, no external data.

## Resume claims this proves

| Claim | How |
|---|---|
| Time-series forecasting (Prophet/LSTM) | LightGBM recursive forecaster + ETS + seasonal-naive baselines; Prophet supported as an optional extra |
| MLOps, ML pipelines, CI/CD | Backtest → train → register → monitor → retrain, with a scheduled CI workflow |
| Model monitoring, drift detection, model versioning | `monitor.py` (live MAE + PSI) + MLflow registry champion alias |
| Experiment tracking (MLflow) | sqlite-backed MLflow tracking + registry |

## Key design decisions

- **Recursive single-step model** (LightGBM with lag/rolling/calendar features)
  rolled out to the horizon, one model, any horizon.
- **No train/serve skew**: training features (vectorized) and serving features
  (recursive) share definitions.
- **Rolling-origin backtest** for honest accuracy, not in-sample fit.
- **sqlite MLflow backend** so the model registry (versions + champion alias) works.
- **Closed retraining loop** gated by a monitor, runnable from a CI cron.
- **Prophet optional** (heavier install), lazily imported; LightGBM + ETS +
  seasonal-naive are the verified models.

## Components

`generator` → `features` → `train` (Optuna + backtest + register) →
`forecast` (recursive) ; `monitor` → `retrain` (closed loop) ; `evaluate`
(LightGBM vs ETS vs naive + plots) ; `api` (`/forecast`, `/series`, `/health`).

## Testing

Unit tests for the generator, feature builder (no NaN, correct columns), metrics,
recursive forecast (horizon length, non-negative), rolling-origin backtest (finite
metrics on a small config), PSI drift, and the API (health + forecast-or-503).

## Deliverable

Verified (tests + a real train→backtest→evaluate run + API forecast) →
`mlops-demand-forecasting.zip`, including the trained model + recent history so
`/forecast` works out of the box. Docker (`python:3.12-slim`) canonical runtime;
CI runs ruff + pytest; a second workflow performs scheduled retraining.
