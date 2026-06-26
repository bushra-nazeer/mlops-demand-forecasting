"""MLflow tracking + model registry helpers.

A sqlite tracking backend (configured in config.yaml) enables the model
registry, so each trained model is logged, versioned, and aliased "champion".
"""

from __future__ import annotations

import contextlib

import mlflow

from .config import Config


def setup_tracking(cfg: Config) -> None:
    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)


def register_champion(model, cfg: Config, metrics: dict) -> int | None:
    """Log + register the model and alias the new version 'champion'."""
    setup_tracking(cfg)
    mlflow.set_experiment("demand-forecasting")
    with mlflow.start_run(run_name="lightgbm-forecaster"):
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=cfg.registered_model_name,
            # cloudpickle avoids MLflow 3.x's skops "untrusted types" check for
            # the LightGBM Booster, which otherwise blocks registration.
            serialization_format="cloudpickle",
        )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{cfg.registered_model_name}'")
    latest = max(int(v.version) for v in versions)
    with contextlib.suppress(Exception):
        client.set_registered_model_alias(cfg.registered_model_name, "champion", str(latest))
    return latest
