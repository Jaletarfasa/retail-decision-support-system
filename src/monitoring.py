from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp


def run_drift_monitor(train_df: pd.DataFrame, current_df: pd.DataFrame, monitored_features: list[str], p_threshold: float = 0.05) -> pd.DataFrame:
    rows = []
    for col in monitored_features:
        ref = train_df[col].dropna()
        cur = current_df[col].dropna()
        ks_stat, p_value = ks_2samp(ref, cur)
        rows.append({"feature": col, "ks_stat": float(ks_stat), "p_value": float(p_value), "drift": int(p_value < p_threshold)})
    return pd.DataFrame(rows)
