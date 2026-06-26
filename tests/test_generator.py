from forecasting.config import load_config
from forecasting.generator import generate_demand


def test_generate_demand_shape_and_validity():
    cfg = load_config()
    df = generate_demand(cfg)
    assert len(df) == cfg.generator.n_series * cfg.generator.n_days
    assert {"date", "series_id", "demand", "promo"} <= set(df.columns)
    assert (df["demand"] >= 0).all()
    assert df["series_id"].nunique() == cfg.generator.n_series
