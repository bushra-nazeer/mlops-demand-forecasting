"""Pydantic response models for the forecast API."""

from __future__ import annotations

from pydantic import BaseModel


class ForecastPoint(BaseModel):
    date: str
    prediction: float


class ForecastResponse(BaseModel):
    series_id: str
    horizon: int
    forecast: list[ForecastPoint]
