from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet

from src import forecasting


def test_fit_and_score_models_smoke(monkeypatch) -> None:
    train_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "store_id": [1] * 8,
            "sku_id": [101] * 8,
            "price": [10.0, 10.5, 10.1, 10.3, 10.4, 10.2, 10.6, 10.7],
            "promo": [0, 1, 0, 1, 0, 1, 0, 1],
            "lag_1": [5, 6, 5, 7, 6, 8, 7, 9],
            "lag_7": [5, 5, 5, 5, 5, 5, 5, 5],
            "lag_28": [5] * 8,
            "rolling_mean_7": [5.0, 5.5, 5.3, 5.8, 6.0, 6.4, 6.7, 7.1],
            "rolling_mean_28": [5.0] * 8,
            "rolling_std_28": [1.0] * 8,
            "assortment_size": [12] * 8,
            "active_skus_28d_avg": [10.0] * 8,
            "units_28d_avg": [6.0] * 8,
            "assortment_health_ratio": [0.83] * 8,
            "traffic_index": [1.1] * 8,
            "site_score": [1.0] * 8,
            "units_sold": [6, 7, 7, 8, 8, 9, 9, 10],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-09", periods=3, freq="D"),
            "store_id": [1] * 3,
            "sku_id": [101] * 3,
            "price": [10.8, 10.9, 11.0],
            "promo": [0, 1, 0],
            "lag_1": [10, 11, 12],
            "lag_7": [6, 7, 8],
            "lag_28": [5] * 3,
            "rolling_mean_7": [7.5, 8.0, 8.5],
            "rolling_mean_28": [5.0] * 3,
            "rolling_std_28": [1.0] * 3,
            "assortment_size": [12] * 3,
            "active_skus_28d_avg": [10.0] * 3,
            "units_28d_avg": [6.0] * 3,
            "assortment_health_ratio": [0.83] * 3,
            "traffic_index": [1.1] * 3,
            "site_score": [1.0] * 3,
            "units_sold": [10, 11, 12],
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

    monkeypatch.setattr(
        forecasting,
        "get_candidate_models",
        lambda model_config=None, candidate_model_names=None: {
            "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
            "decision_tree": DecisionTreeRegressor(max_depth=3, random_state=42),
        },
    )

    results = forecasting.fit_and_score_models(train_df, test_df, feature_cols, "units_sold")

    assert [result.model_name for result in results] == ["elasticnet", "decision_tree"]
    assert all(len(result.preds) == len(test_df) for result in results)
    assert all({"mae", "rmse", "wmape", "bias", "r2"} <= set(result.metrics) for result in results)
