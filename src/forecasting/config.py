"""Typed configuration loaded from ``config/config.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = "config/config.yaml"


class Paths(BaseModel):
    model_path: str
    model_metadata: str
    reports_dir: str
    figures_dir: str
    mlflow_tracking_uri: str


class GeneratorCfg(BaseModel):
    n_series: int = 12
    n_days: int = 1095
    start_date: str = "2021-01-01"


class BacktestCfg(BaseModel):
    n_folds: int = 4
    step_days: int = 28


class OptunaCfg(BaseModel):
    n_trials: int = 20
    timeout_seconds: int = 600


class MonitorCfg(BaseModel):
    mae_degradation_ratio: float = 1.25
    psi_threshold: float = 0.2
    window_days: int = 28


class Config(BaseModel):
    paths: Paths
    registered_model_name: str = "demand-forecaster"
    random_state: int = 42
    generator: GeneratorCfg = GeneratorCfg()
    horizon: int = 28
    lags: list[int] = Field(default_factory=lambda: [1, 7, 14, 28])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28])
    backtest: BacktestCfg = BacktestCfg()
    optuna: OptunaCfg = OptunaCfg()
    monitor: MonitorCfg = MonitorCfg()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Config(**data)
