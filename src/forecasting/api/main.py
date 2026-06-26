"""FastAPI forecasting service.

Loads the champion model plus the recent history saved at training time, and
serves recursive multi-step forecasts per series.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from ..config import load_config
from ..forecast import forecast_series
from .schemas import ForecastPoint, ForecastResponse

cfg = load_config()
_state: dict = {"model": None, "metadata": {}, "history": None}


def _load() -> None:
    model_path = Path(cfg.paths.model_path)
    if not model_path.exists():
        return
    _state["model"] = joblib.load(model_path)
    metadata = json.loads(Path(cfg.paths.model_metadata).read_text())
    _state["metadata"] = metadata
    history = pd.DataFrame(metadata["recent_history"])
    history["date"] = pd.to_datetime(history["date"])
    _state["history"] = history


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load()
    yield


app = FastAPI(title="Demand Forecasting API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/series")
def series() -> dict:
    return {"series_ids": _state["metadata"].get("series_ids", [])}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(series_id: str, horizon: int = 14) -> ForecastResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first (`make train`).")

    metadata = _state["metadata"]
    series_ids = metadata["series_ids"]
    if series_id not in series_ids:
        raise HTTPException(status_code=404, detail="Unknown series_id. See /series.")

    horizon = max(1, min(horizon, 90))
    history_df = _state["history"]
    history = (
        history_df[history_df["series_id"] == series_id]
        .sort_values("date")
        .set_index("date")["demand"]
    )
    forecast_df = forecast_series(
        model, history, cfg, series_ids.index(series_id), horizon, metadata["feature_names"]
    )
    points = [
        ForecastPoint(date=str(pd.Timestamp(r["date"]).date()), prediction=round(float(r["prediction"]), 2))
        for _, r in forecast_df.iterrows()
    ]
    return ForecastResponse(series_id=series_id, horizon=horizon, forecast=points)
