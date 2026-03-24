from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    df = sales_df.sort_values(["store_id", "sku_id", "date"]).copy()
    grp = df.groupby(["store_id", "sku_id"], sort=False)
    df["lag_1"] = grp["units_sold"].shift(1)
    df["lag_7"] = grp["units_sold"].shift(7)
    df["lag_28"] = grp["units_sold"].shift(28)
    df["rolling_mean_7"] = grp["units_sold"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["rolling_mean_28"] = grp["units_sold"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    df["rolling_std_28"] = grp["units_sold"].transform(lambda s: s.shift(1).rolling(28, min_periods=2).std())
    df["sales_dollars"] = df["units_sold"] * df["price"]
    df["gross_margin_dollars"] = df["sales_dollars"] * df["margin_pct"]
    daily = (
        df.groupby(["store_id", "date"], as_index=False)
        .agg(assortment_size=("sku_id", "nunique"), total_units=("units_sold", "sum"), total_sales=("sales_dollars", "sum"))
        .sort_values(["store_id", "date"])
    )
    dgrp = daily.groupby("store_id", sort=False)
    daily["active_skus_28d_avg"] = dgrp["assortment_size"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    daily["units_28d_avg"] = dgrp["total_units"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    daily["assortment_health_ratio"] = daily["active_skus_28d_avg"] / daily["assortment_size"].replace(0, np.nan)
    daily = daily.merge(store_df, on="store_id", how="left")
    out = df.merge(daily[["store_id", "date", "assortment_size", "active_skus_28d_avg", "units_28d_avg", "assortment_health_ratio", "traffic_index", "site_score"]], on=["store_id", "date"], how="left")
    required = ["lag_28", "rolling_mean_28", "rolling_std_28", "assortment_health_ratio"]
    return out.dropna(subset=required).reset_index(drop=True)
