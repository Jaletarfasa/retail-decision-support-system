import itertools
import pandas as pd
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

BUDGET_LIMIT = 75000.0

store_path = OUT / "dashboard_store_forecast.csv"
watch_path = OUT / "store_watchlist.csv"

if not store_path.exists():
    raise FileNotFoundError("Missing outputs/dashboard_store_forecast.csv")

store = pd.read_csv(store_path)
watch = pd.read_csv(watch_path) if watch_path.exists() else pd.DataFrame()

df = store.copy()

if not watch.empty and "store_id" in df.columns and "store_id" in watch.columns:
    keep_cols = [c for c in ["store_id", "region", "watch_status", "watch_reason"] if c in watch.columns]
    df = df.merge(watch[keep_cols].drop_duplicates("store_id"), on="store_id", how="left", suffixes=("", "_watch"))

for col in ["actual_sales_28d", "forecast_sales_next_28d", "forecast_units_next_28d"]:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["candidate_site_id"] = "SITE_" + df["store_id"].astype(str)
df["site_name"] = "Candidate near " + df["store_id"].astype(str)
df["market_region"] = df["region"] if "region" in df.columns else ""

df["growth_gap"] = df["forecast_sales_next_28d"] - df["actual_sales_28d"]

# Synthetic demo cost model. This is NOT a real real-estate cost estimate.
# It gives the optimizer a budget constraint.
base_cost = 25000
scaled_cost = (df["forecast_sales_next_28d"].rank(pct=True) * 15000).fillna(0)
df["estimated_site_cost"] = (base_cost + scaled_cost).round(0)

# Synthetic benefit score from forecast scale and growth gap.
# This is NOT measured business impact.
df["benefit_score"] = (
    df["forecast_sales_next_28d"].rank(pct=True) * 60
    + df["growth_gap"].rank(pct=True) * 40
).round(4)

# Exact 0/1 budget-constrained optimization for small demo size.
# Maximize total benefit_score subject to sum(estimated_site_cost) <= BUDGET_LIMIT.
n = len(df)
best_subset = []
best_score = -1.0
best_cost = 0.0

indices = list(range(n))
for r in range(0, n + 1):
    for combo in itertools.combinations(indices, r):
        cost = float(df.loc[list(combo), "estimated_site_cost"].sum()) if combo else 0.0
        if cost <= BUDGET_LIMIT:
            score = float(df.loc[list(combo), "benefit_score"].sum()) if combo else 0.0
            if score > best_score:
                best_score = score
                best_cost = cost
                best_subset = list(combo)

df["selected_flag"] = 0
df.loc[best_subset, "selected_flag"] = 1
df["site_budget_limit"] = BUDGET_LIMIT
df["selected_portfolio_cost"] = best_cost
df["selected_portfolio_score"] = best_score
df["optimization_method"] = "exact_0_1_budget_constrained_enumeration"
df["optimization_objective"] = "maximize synthetic benefit_score subject to estimated_site_cost <= site_budget_limit"
df["claim_boundary"] = (
    "Synthetic budget-constrained site-selection demo. "
    "This is not real estate optimization, not an investment recommendation, "
    "and not validated production location analytics."
)

df["recommendation"] = df["selected_flag"].map({1: "Selected under demo budget", 0: "Not selected under demo budget"})

cols = [
    "candidate_site_id",
    "site_name",
    "store_id",
    "market_region",
    "actual_sales_28d",
    "forecast_sales_next_28d",
    "growth_gap",
    "benefit_score",
    "estimated_site_cost",
    "site_budget_limit",
    "selected_flag",
    "selected_portfolio_cost",
    "selected_portfolio_score",
    "recommendation",
    "optimization_method",
    "optimization_objective",
    "claim_boundary",
]

out = df[cols].sort_values(["selected_flag", "benefit_score"], ascending=[False, False]).reset_index(drop=True)

out.to_csv(OUT / "optimized_site_selection.csv", index=False)
out.to_csv(OUT / "site_selection.csv", index=False)

print("Wrote outputs/optimized_site_selection.csv and outputs/site_selection.csv")
print("Rows:", len(out))
print("Selected sites:", int(out["selected_flag"].sum()))
print("Budget limit:", BUDGET_LIMIT)
print("Selected portfolio cost:", best_cost)
print("Selected portfolio score:", best_score)
