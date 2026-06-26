from forecasting.backtest import rolling_origin_backtest
from forecasting.config import load_config
from forecasting.generator import generate_demand


def test_backtest_returns_finite_metrics():
    cfg = load_config()
    cfg.generator.n_series = 2
    cfg.generator.n_days = 260
    cfg.horizon = 14
    cfg.backtest.n_folds = 2
    cfg.backtest.step_days = 14

    panel = generate_demand(cfg)
    summary = rolling_origin_backtest(panel, cfg)

    assert summary["mae"] == summary["mae"]  # not NaN
    assert summary["smape"] >= 0
    assert len(summary["folds"]) >= 1
