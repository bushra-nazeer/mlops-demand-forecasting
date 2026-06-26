from forecasting.metrics import all_metrics, mae, rmse, smape


def test_known_values():
    y = [10, 20, 30]
    yhat = [12, 18, 33]
    assert abs(mae(y, yhat) - (2 + 2 + 3) / 3) < 1e-9
    assert rmse(y, yhat) > 0
    assert 0 <= smape(y, yhat) <= 200


def test_all_metrics_keys():
    m = all_metrics([1, 2, 3], [1, 2, 3])
    assert {"mae", "rmse", "mape", "smape"} <= set(m)
    assert m["mae"] == 0.0
