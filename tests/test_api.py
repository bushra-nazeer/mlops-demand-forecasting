from fastapi.testclient import TestClient

from forecasting.api.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_returns_series_or_503():
    response = client.get("/forecast", params={"series_id": "series_00", "horizon": 7})
    assert response.status_code in (200, 404, 503)
    if response.status_code == 200:
        body = response.json()
        assert body["horizon"] == 7
        assert len(body["forecast"]) == 7
        assert {"date", "prediction"} <= set(body["forecast"][0])
