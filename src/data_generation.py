from __future__ import annotations

import numpy as np
import pandas as pd


def generate_store_metadata(n_stores: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "store_id": np.arange(1, n_stores + 1),
        "region": rng.choice(["Atlantic", "Quebec", "Ontario", "Prairies", "BC"], size=n_stores),
        "store_format": rng.choice(["Urban", "Suburban", "Rural"], size=n_stores),
        "traffic_index": rng.uniform(0.7, 1.4, size=n_stores),
        "site_score": rng.uniform(0.65, 1.35, size=n_stores),
    })


def generate_sku_metadata(n_skus: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    return pd.DataFrame({
        "sku_id": np.arange(1, n_skus + 1),
        "department": rng.choice(["Automotive", "Home", "Tools", "Outdoor Living"], size=n_skus),
        "base_price": rng.uniform(5, 200, size=n_skus),
        "margin_pct": rng.uniform(0.15, 0.45, size=n_skus),
    })


def generate_inventory_snapshot(store_ids: pd.Series, sku_ids: pd.Series, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    rows = []
    for store_id in store_ids:
        for sku_id in rng.choice(sku_ids, size=min(25, len(sku_ids)), replace=False):
            rows.append({
                "store_id": int(store_id),
                "sku_id": int(sku_id),
                "on_hand_units": int(rng.integers(0, 120)),
                "lead_time_days": int(rng.integers(2, 15)),
            })
    return pd.DataFrame(rows)


def generate_candidate_sites(n_sites: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    return pd.DataFrame({
        "candidate_site_id": np.arange(1, n_sites + 1),
        "region": rng.choice(["Atlantic", "Quebec", "Ontario", "Prairies", "BC"], size=n_sites),
        "traffic_index": rng.uniform(0.7, 1.5, size=n_sites),
        "household_income_index": rng.uniform(0.8, 1.4, size=n_sites),
        "competition_index": rng.uniform(0.5, 1.5, size=n_sites),
        "rent_cost": rng.uniform(25000, 120000, size=n_sites),
    })


def generate_daily_sales(stores: pd.DataFrame, skus: pd.DataFrame, n_days: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 4)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for _, store in stores.iterrows():
        for _, sku in skus.sample(min(40, len(skus)), random_state=seed).iterrows():
            level = store["traffic_index"] * store["site_score"] * rng.uniform(0.7, 1.3)
            for date in dates:
                promo = int(rng.binomial(1, 0.12))
                price = max(1.0, sku["base_price"] * (1 - 0.08 * promo + rng.normal(0, 0.03)))
                demand = max(0.1, level * rng.uniform(0.7, 1.4) * (1.20 if promo else 1.0))
                units = rng.poisson(demand)
                rows.append({
                    "date": date,
                    "store_id": int(store["store_id"]),
                    "sku_id": int(sku["sku_id"]),
                    "department": sku["department"],
                    "price": float(price),
                    "promo": promo,
                    "units_sold": int(units),
                    "margin_pct": float(sku["margin_pct"]),
                })
    return pd.DataFrame(rows)
