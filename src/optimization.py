from __future__ import annotations

import pandas as pd
from scipy.optimize import linprog


def optimize_sites(site_scores: pd.DataFrame, budget_limit: float) -> pd.DataFrame:
    df = site_scores.copy()
    c = -df["projected_value_index"].values
    A_ub = [df["rent_cost"].values]
    b_ub = [budget_limit]
    bounds = [(0, 1) for _ in range(len(df))]
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    df["selected_fraction"] = result.x if result.success else 0.0
    df["selected_flag"] = (df["selected_fraction"] >= 0.5).astype(int)
    return df
