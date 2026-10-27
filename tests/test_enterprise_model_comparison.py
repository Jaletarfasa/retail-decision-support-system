from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_enterprise_upgrade_v4_schema_memory import load_pipeline_model_comparison


def test_pipeline_comparison_is_mapped_and_ranked(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "model_type": ["lstm", "lightgbm", "baseline_last28"],
            "mae": [3.08, 2.85, 3.04],
            "rmse": [4.20, 3.80, 4.13],
            "wmape": [0.230, 0.213, 0.227],
            "bias": [-0.14, -0.08, -0.14],
            "selected_flag": [0, 1, 0],
            "decision_note": ["Benchmarked only", "Selected for prototype deployment", "Benchmarked only"],
        }
    ).to_csv(tmp_path / "dashboard_model_comparison.csv", index=False)

    model = load_pipeline_model_comparison(tmp_path, "synthetic")

    assert model.iloc[0]["model_name"] == "lightgbm"
    assert model.iloc[0]["selected_flag"] == 1
    assert model.iloc[0]["test_mae"] == 2.85
    assert dict(zip(model["model_name"], model["model_type"])) == {
        "lightgbm": "classical_ml",
        "baseline_last28": "baseline",
        "lstm": "deep_sequence",
    }


def test_missing_pipeline_file_returns_empty(tmp_path: Path) -> None:
    assert load_pipeline_model_comparison(tmp_path, "synthetic").empty
