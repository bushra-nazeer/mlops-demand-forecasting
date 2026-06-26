"""Synthetic multi-series daily demand.

Each series has its own base level, gentle trend, weekly + yearly seasonality,
occasional promotion spikes, and noise — a realistic forecasting panel with no
external data dependency.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import Config, load_config


def generate_demand(cfg: Config, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state if seed is None else seed)
    dates = pd.date_range(cfg.generator.start_date, periods=cfg.generator.n_days, freq="D")
    day_index = np.arange(len(dates))
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()

    frames = []
    for s in range(cfg.generator.n_series):
        base = rng.uniform(60, 220)
        trend_per_day = rng.uniform(-0.01, 0.04) * base / 100.0
        weekly_amp = rng.uniform(0.10, 0.35) * base
        yearly_amp = rng.uniform(0.10, 0.30) * base

        weekly = weekly_amp * np.sin(2 * np.pi * dow / 7.0) + 0.5 * weekly_amp * (dow >= 5)
        yearly = yearly_amp * np.sin(2 * np.pi * doy / 365.25)
        level = base + trend_per_day * day_index + weekly + yearly

        promo = rng.random(len(dates)) < 0.03
        promo_lift = promo * rng.uniform(0.3, 0.8, len(dates)) * base
        noise = rng.normal(0, 0.08 * base, len(dates))
        demand = np.clip(level + promo_lift + noise, 0.0, None)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "series_id": f"series_{s:02d}",
                    "demand": np.round(demand, 1),
                    "promo": promo.astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and summarize synthetic demand.")
    parser.parse_args()
    cfg = load_config()
    df = generate_demand(cfg)
    print(f"{df['series_id'].nunique()} series x {df['date'].nunique()} days = {len(df):,} rows")
    print(df.head())


if __name__ == "__main__":
    main()
