from __future__ import annotations

import pandas as pd


def score_candidate_sites(candidate_sites: pd.DataFrame) -> pd.DataFrame:
    df = candidate_sites.copy()
    df["site_score_model"] = 0.35 * (df["traffic_index"] / df["traffic_index"].mean()) + 0.30 * (df["household_income_index"] / df["household_income_index"].mean()) - 0.15 * (df["competition_index"] / df["competition_index"].mean()) - 0.20 * (df["rent_cost"] / df["rent_cost"].mean())
    df["projected_value_index"] = df["site_score_model"]
    return df.sort_values("projected_value_index", ascending=False).reset_index(drop=True)
