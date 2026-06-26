from forecasting.config import load_config
from forecasting.features import build_training_frame, feature_names
from forecasting.forecast import forecast_series
from forecasting.generator import generate_demand
from forecasting.model import make_model


def test_forecast_series_returns_horizon_rows():
    cfg = load_config()
    panel = generate_demand(cfg)
    sids = sorted(panel["series_id"].unique())[:2]
    panel = panel[panel["series_id"].isin(sids)].groupby("series_id").head(300)

    _, X, y, series_ids = build_training_frame(panel, cfg, sids)
    model = make_model(None, cfg.random_state)
    model.fit(X, y)

    history = (
        panel[panel["series_id"] == sids[0]].sort_values("date").set_index("date")["demand"]
    )
    forecast = forecast_series(model, history, cfg, 0, 14, feature_names(cfg))
    assert len(forecast) == 14
    assert (forecast["prediction"] >= 0).all()
    assert list(forecast.columns) == ["date", "prediction"]
