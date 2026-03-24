from __future__ import annotations

import math

import pandas as pd
from scipy.stats import ks_2samp


def run_drift_monitor(
    train_df: pd.DataFrame,
    current_df: pd.DataFrame,
    monitored_features: list[str],
    p_threshold: float = 0.05,
    min_sample_size: int = 2,
) -> pd.DataFrame:
    rows = []
    for col in monitored_features:
        ref = train_df[col].dropna()
        cur = current_df[col].dropna()
        sample_note = ""
        if len(ref) < min_sample_size or len(cur) < min_sample_size:
            ks_stat = math.nan
            p_value = math.nan
            drift = 0
            sample_note = "sample_too_small"
        else:
            ks_stat, p_value = ks_2samp(ref, cur)
            ks_stat = float(ks_stat)
            p_value = float(p_value)
            drift = int(p_value < p_threshold)
        rows.append(
            {
                "feature": col,
                "ks_stat": ks_stat,
                "p_value": p_value,
                "drift": drift,
                "sample_note": sample_note,
            }
        )
    return pd.DataFrame(rows)
