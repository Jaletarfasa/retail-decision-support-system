from __future__ import annotations

import pandas as pd

from src.agent_controller import AgentController


def test_agent_controller_runs_forecast_decision_chain() -> None:
    train_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "store_id": [1, 2] * 4,
            "sku_id": [101, 102] * 4,
            "price": [10.0, 12.0, 10.2, 12.1, 10.4, 12.2, 10.6, 12.3],
            "promo": [0, 1, 0, 1, 0, 0, 1, 0],
            "lag_1": [6, 7, 7, 8, 8, 9, 9, 10],
            "lag_7": [5, 6, 5, 6, 6, 7, 6, 7],
            "lag_28": [5] * 8,
            "rolling_mean_7": [6.0, 7.0, 6.3, 7.3, 6.6, 7.6, 6.9, 7.9],
            "rolling_mean_28": [6.0] * 8,
            "rolling_std_28": [1.0] * 8,
            "assortment_size": [12] * 8,
            "active_skus_28d_avg": [10.0] * 8,
            "units_28d_avg": [7.0] * 8,
            "assortment_health_ratio": [0.84] * 8,
            "traffic_index": [1.1, 0.9] * 4,
            "site_score": [1.0, 1.2] * 4,
            "units_sold": [7, 8, 8, 9, 9, 10, 10, 11],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-09", periods=4, freq="D"),
            "store_id": [1, 2, 1, 2],
            "sku_id": [101, 102, 101, 102],
            "price": [10.8, 12.4, 11.0, 12.5],
            "promo": [0, 1, 1, 0],
            "lag_1": [10, 11, 11, 12],
            "lag_7": [7, 8, 7, 8],
            "lag_28": [5] * 4,
            "rolling_mean_7": [7.2, 8.2, 7.5, 8.5],
            "rolling_mean_28": [6.0] * 4,
            "rolling_std_28": [1.0] * 4,
            "assortment_size": [12] * 4,
            "active_skus_28d_avg": [10.0] * 4,
            "units_28d_avg": [7.0] * 4,
            "assortment_health_ratio": [0.84] * 4,
            "traffic_index": [1.1, 0.9, 1.1, 0.9],
            "site_score": [1.0, 1.2, 1.0, 1.2],
            "units_sold": [11, 12, 12, 13],
        }
    )
    feature_cols = [
        "store_id",
        "sku_id",
        "price",
        "promo",
        "lag_1",
        "lag_7",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_28",
        "assortment_size",
        "active_skus_28d_avg",
        "units_28d_avg",
        "assortment_health_ratio",
        "traffic_index",
        "site_score",
    ]

    controller = AgentController()
    result = controller.run_forecast_decision_chain(
        forecast_payload={
            "train_df": train_df,
            "test_df": test_df,
            "feature_cols": feature_cols,
            "target_col": "units_sold",
            "candidate_model_names": ["mlp"],
            "model_config": {
                "candidate_models": ["mlp"],
                "deep_model_params": {
                    "mlp": {"hidden_dims": [8], "learning_rate": 0.01, "epochs": 6, "batch_size": 4, "seed": 42}
                },
            },
        }
    )

    assert result.status == "success"
    assert result.state.completed_steps == [
        "run_forecast_pipeline",
        "compare_candidate_models",
        "generate_decision_summary",
    ]
    assert result.final_output["comparison"][0]["model_name"] == "mlp"
    assert result.final_output["summary"][0]["metric"] == "champion_model"
