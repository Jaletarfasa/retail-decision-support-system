from __future__ import annotations

import pandas as pd


def build_assortment_health_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["store_id", "date", "assortment_size", "active_skus_28d_avg", "units_28d_avg", "assortment_health_ratio"]
    return (
        feature_df[cols].drop_duplicates().groupby("store_id", as_index=False)
        .agg(avg_assortment_size=("assortment_size", "mean"), avg_active_skus_28d=("active_skus_28d_avg", "mean"), avg_units_28d=("units_28d_avg", "mean"), avg_assortment_health_ratio=("assortment_health_ratio", "mean"))
    )
