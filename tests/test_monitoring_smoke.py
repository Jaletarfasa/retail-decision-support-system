from __future__ import annotations

import math

import pandas as pd

from src.monitoring import run_drift_monitor


def test_run_drift_monitor_marks_small_samples_without_warning_path() -> None:
    train_df = pd.DataFrame({"lag_7": [1.0]})
    current_df = pd.DataFrame({"lag_7": [1.2]})

    drift = run_drift_monitor(
        train_df=train_df,
        current_df=current_df,
        monitored_features=["lag_7"],
        p_threshold=0.05,
        min_sample_size=5,
    )

    assert drift.loc[0, "sample_note"] == "sample_too_small"
    assert math.isnan(drift.loc[0, "ks_stat"])
    assert math.isnan(drift.loc[0, "p_value"])
    assert drift.loc[0, "drift"] == 0
