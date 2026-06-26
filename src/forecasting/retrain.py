"""Automated retraining entrypoint (driven by the scheduled CI workflow).

Checks the monitor; retrains and re-registers only when drift or error
degradation is detected (or when forced). This is the closed MLOps loop.
"""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .monitor import check_drift


def retrain(cfg, force: bool = False) -> dict:
    status = check_drift(cfg)
    if not status["needs_retrain"] and not force:
        return {"action": "none", "reason": "model healthy", "status": status}

    from .train import train

    new_metrics = train(cfg)
    return {
        "action": "retrained",
        "reason": "forced" if force else "drift/degradation detected",
        "status": status,
        "new_metrics": new_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and retrain if needed.")
    parser.add_argument("--force", action="store_true", help="Retrain regardless of monitor.")
    args = parser.parse_args()
    cfg = load_config()
    print(json.dumps(retrain(cfg, force=args.force), indent=2))


if __name__ == "__main__":
    main()
