from __future__ import annotations

import pandas as pd


def build_executive_summary(champion_name: str, challenger_name: str | None, champion_metrics: dict, retraining_status: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "metric": ["champion_model", "challenger_model", "champion_mae", "champion_rmse", "champion_wmape", "retraining_recommended", "watch_status"],
        "value": [champion_name, challenger_name, champion_metrics.get("mae"), champion_metrics.get("rmse"), champion_metrics.get("wmape"), retraining_status.get("retraining_recommended"), retraining_status.get("status")],
    })
