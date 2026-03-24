from __future__ import annotations

import pandas as pd


def should_retrain(drift_df: pd.DataFrame, current_wmape: float, baseline_wmape: float, wmape_degradation_threshold: float = 0.05, min_drifted_features: int = 1) -> dict:
    drifted = int(drift_df["drift"].sum())
    degradation = current_wmape - baseline_wmape
    recommended = int((drifted >= min_drifted_features) and (degradation >= wmape_degradation_threshold))
    return {
        "drifted_features": drifted,
        "current_wmape": current_wmape,
        "baseline_wmape": baseline_wmape,
        "wmape_degradation": degradation,
        "retraining_recommended": recommended,
        "status": "Watch" if recommended == 0 else "Retrain Review",
    }


def build_retraining_audit(trigger_info: dict, champion_name: str, challenger_name: str | None) -> pd.DataFrame:
    return pd.DataFrame([{
        "champion_model": champion_name,
        "challenger_model": challenger_name,
        "drifted_features": trigger_info["drifted_features"],
        "current_wmape": trigger_info["current_wmape"],
        "baseline_wmape": trigger_info["baseline_wmape"],
        "wmape_degradation": trigger_info["wmape_degradation"],
        "retraining_recommended": trigger_info["retraining_recommended"],
        "status": trigger_info["status"],
    }])
