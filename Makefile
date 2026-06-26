.PHONY: install train backtest evaluate monitor retrain serve test lint format clean mlflow-ui

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

train:
	uv run python -m forecasting.train

backtest:
	uv run python -m forecasting.backtest

evaluate:
	uv run python -m forecasting.evaluate

monitor:
	uv run python -m forecasting.monitor

retrain:
	uv run python -m forecasting.retrain

serve:
	uv run uvicorn forecasting.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf models/*.pkl models/metadata.json mlflow.db mlruns reports/figures .pytest_cache .ruff_cache
