from __future__ import annotations

import pandas as pd


def build_promotion_summary(sales_df: pd.DataFrame) -> pd.DataFrame:
    promo_summary = sales_df.groupby("promo", as_index=False).agg(avg_units=("units_sold", "mean"), avg_sales=("sales_dollars", "mean"), avg_margin=("gross_margin_dollars", "mean"))
    if set(promo_summary["promo"]) == {0, 1}:
        base = promo_summary.loc[promo_summary["promo"] == 0].iloc[0]
        prom = promo_summary.loc[promo_summary["promo"] == 1].iloc[0]
        promo_lift = (prom["avg_units"] - base["avg_units"]) / max(base["avg_units"], 1e-9)
        promo_margin_lift = (prom["avg_margin"] - base["avg_margin"]) / max(base["avg_margin"], 1e-9)
    else:
        promo_lift, promo_margin_lift = None, None
    return pd.DataFrame({"metric": ["promo_lift", "promo_margin_lift"], "value": [promo_lift, promo_margin_lift]})
