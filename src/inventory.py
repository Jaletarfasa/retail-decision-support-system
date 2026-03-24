from __future__ import annotations

import numpy as np
import pandas as pd


def build_inventory_recommendations(forecast_df: pd.DataFrame, inventory_df: pd.DataFrame, safety_stock_ratio: float = 0.25, min_order_qty: int = 0, pack_size: int = 1) -> pd.DataFrame:
    latest_fc = forecast_df.groupby(["store_id", "sku_id"], as_index=False).agg(forecast_units=("forecast_units", "sum"))
    out = latest_fc.merge(inventory_df, on=["store_id", "sku_id"], how="left")
    out["on_hand_units"] = out["on_hand_units"].fillna(0)
    out["lead_time_days"] = out["lead_time_days"].fillna(7)
    out["safety_stock"] = np.ceil(safety_stock_ratio * out["forecast_units"])
    raw_qty = np.ceil(out["forecast_units"] + out["safety_stock"] - out["on_hand_units"])
    raw_qty = np.maximum(raw_qty, min_order_qty)
    if pack_size > 1:
        raw_qty = np.ceil(raw_qty / pack_size) * pack_size
    out["recommended_reorder_qty"] = np.maximum(raw_qty, 0)
    return out
