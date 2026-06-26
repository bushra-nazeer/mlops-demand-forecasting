from forecasting.config import load_config
from forecasting.features import build_training_frame, feature_names
from forecasting.generator import generate_demand


def test_build_training_frame_columns_and_no_nan():
    cfg = load_config()
    panel = generate_demand(cfg)
    sids = sorted(panel["series_id"].unique())[:2]
    panel = panel[panel["series_id"].isin(sids)].groupby("series_id").head(200)

    df, X, y, series_ids = build_training_frame(panel, cfg, sids)
    assert list(X.columns) == feature_names(cfg)
    assert not X.isna().any().any()
    assert len(X) == len(y) > 0
    assert series_ids == sids
