from __future__ import annotations

import pandas as pd

from src.forecasting import fit_and_score_models


def test_deep_models_execute_in_forecasting_pipeline() -> None:
    train_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "store_id": [1, 2] * 6,
            "sku_id": [101, 102] * 6,
            "price": [10.0, 12.0, 10.2, 12.1, 10.4, 12.2, 10.6, 12.3, 10.8, 12.4, 11.0, 12.5],
            "promo": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "lag_1": [6, 7, 6, 8, 7, 8, 8, 9, 9, 10, 10, 11],
            "lag_7": [5, 6, 5, 6, 5, 6, 6, 7, 6, 7, 7, 8],
            "lag_28": [5] * 12,
            "rolling_mean_7": [6.0, 7.0, 6.2, 7.2, 6.4, 7.4, 6.7, 7.7, 7.0, 8.0, 7.3, 8.3],
            "rolling_mean_28": [6.0] * 12,
            "rolling_std_28": [1.0] * 12,
            "assortment_size": [12] * 12,
            "active_skus_28d_avg": [10.0] * 12,
            "units_28d_avg": [7.0] * 12,
            "assortment_health_ratio": [0.84] * 12,
            "traffic_index": [1.1, 0.9] * 6,
            "site_score": [1.0, 1.2] * 6,
            "units_sold": [7, 8, 7, 9, 8, 9, 9, 10, 10, 11, 11, 12],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-13", periods=4, freq="D"),
            "store_id": [1, 2, 1, 2],
            "sku_id": [101, 102, 101, 102],
            "price": [11.1, 12.6, 11.2, 12.7],
            "promo": [0, 1, 1, 0],
            "lag_1": [11, 12, 12, 13],
            "lag_7": [8, 9, 8, 9],
            "lag_28": [5] * 4,
            "rolling_mean_7": [7.8, 8.8, 8.0, 9.0],
            "rolling_mean_28": [6.0] * 4,
            "rolling_std_28": [1.0] * 4,
            "assortment_size": [12] * 4,
            "active_skus_28d_avg": [10.0] * 4,
            "units_28d_avg": [7.0] * 4,
            "assortment_health_ratio": [0.84] * 4,
            "traffic_index": [1.1, 0.9, 1.1, 0.9],
            "site_score": [1.0, 1.2, 1.0, 1.2],
            "units_sold": [12, 13, 12, 14],
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
    model_config = {
        "candidate_models": ["mlp", "entity_embedding_nn"],
        "deep_model_params": {
            "mlp": {"hidden_dims": [16], "learning_rate": 0.01, "epochs": 8, "batch_size": 4, "seed": 42},
            "entity_embedding_nn": {
                "categorical_cols": ["store_id", "sku_id"],
                "hidden_dims": [16],
                "learning_rate": 0.01,
                "epochs": 8,
                "batch_size": 4,
                "seed": 42,
            },
        },
    }

    results = fit_and_score_models(
        train_df,
        test_df,
        feature_cols,
        "units_sold",
        model_config=model_config,
        candidate_model_names=["mlp", "entity_embedding_nn"],
    )

    assert [result.model_name for result in results] == ["mlp", "entity_embedding_nn"]
    assert all(len(result.preds) == len(test_df) for result in results)
    assert all({"mae", "rmse", "wmape", "bias"} <= set(result.metrics) for result in results)
