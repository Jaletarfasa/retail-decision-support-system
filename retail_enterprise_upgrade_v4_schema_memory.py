#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retail Decision Support System - Enterprise Upgrade v4
Single-file script for enterprise-like synthetic data, public/approved real-data pilot
ingestion, Fabric-ready bundle creation, Fabric notebook-cell generation, and deterministic agent-style governance checks, and schema-constrained typed agent memory.

BRUTALLY HONEST BOUNDARY
------------------------
This script can strengthen evidence. It cannot by itself prove real enterprise
integration, live Microsoft Fabric execution, production deployment, or real
business impact.

It supports honest evidence levels:
1. Synthetic enterprise-like simulation
2. Public/approved real-data pilot processing, if local files are supplied
3. Fabric-ready upload bundle
4. Fabric notebook cell that must be run inside Microsoft Fabric to prove Fabric execution
5. Deterministic agent-style governance outputs for data quality, model validation, decision risk, evidence boundaries, and human approval routing
6. Schema-constrained typed agent memory artifacts using Pydantic when available

Do not claim production unless you have real governed data feeds, security/RBAC,
CI/CD, monitoring, run history, operational users, and measured decision impact.

Examples
--------
Synthetic enterprise-like upgrade:
    python retail_enterprise_upgrade_v4_schema_memory.py --mode all-synthetic --project-root .

Large synthetic version:
    python retail_enterprise_upgrade_v4_schema_memory.py --mode all-synthetic --stores 25 --skus 500 --days 730

Walmart public-data style local files:
    python retail_enterprise_upgrade_v4_schema_memory.py --mode walmart-public --real-data-dir "C:\\path\\to\\walmart"

Approved real/pilot local files:
    python retail_enterprise_upgrade_v4_schema_memory.py --mode generic-real --real-data-dir "C:\\path\\to\\approved_csvs"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Literal, Optional

import numpy as np
import pandas as pd

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False

    class BaseModel:
        def __init__(self, **data: Any):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self) -> Dict[str, Any]:
            return dict(self.__dict__)

    def Field(default: Any = None, description: str = "") -> Any:
        return default



DEPARTMENTS = {
    "Grocery": ["Beverages", "Snacks", "Dairy", "Pantry"],
    "Household": ["Cleaning", "Paper Goods", "Laundry"],
    "Health_Beauty": ["Personal Care", "Oral Care", "Skin Care"],
    "Electronics": ["Accessories", "Small Devices"],
    "Seasonal": ["Outdoor", "Holiday", "Garden"],
}
REGIONS = ["West", "Prairie", "Ontario", "Quebec", "Atlantic"]
STORE_FORMATS = ["Urban", "Suburban", "Small Format", "Large Format", "Outlet"]
SUPPLIERS = [
    "NorthStar Foods", "Maple Leaf Distribution", "Prairie Supply Co",
    "Urban Goods Wholesale", "BlueRiver Consumer Products", "Evergreen Retail Supply",
    "Northern Consumer Logistics", "Cascade Wholesale Group"
]
PROMO_TYPES = ["Flyer", "Digital Coupon", "Endcap", "Bundle", "Clearance", "Loyalty Offer"]

DASHBOARD_OUTPUTS = [
    "dashboard_executive_summary.csv",
    "dashboard_model_comparison.csv",
    "dashboard_store_forecast.csv",
    "dashboard_department_forecast.csv",
    "dashboard_region_forecast.csv",
    "dashboard_brand_forecast.csv",
    "inventory_recommendations.csv",
    "drift_monitor.csv",
    "retraining_status.csv",
    "retraining_audit.csv",
    "store_watchlist.csv",
    "dashboard_pipeline_maturity.csv",
    "workflow_handoff.csv",
    "agent_answers.csv",
    "data_quality_audit.csv",
    "data_contract_summary.csv",
    "agent_governance_report.csv",
    "agent_data_quality_review.csv",
    "agent_model_validation_review.csv",
    "agent_decision_risk_review.csv",
    "agent_evidence_boundary_review.csv",
    "agent_human_approval_queue.csv",
    "agent_executive_narrative.csv",
    "real_agent_trace.csv",
    "real_agent_memory.csv",
    "real_agent_action_log.csv",
    "real_agent_final_decisions.csv",
    "agent_memory_core.csv",
    "agent_memory_episodic.csv",
    "agent_memory_procedural.csv",
    "agent_memory_index.csv",
    "schema_agent_memory_evidence_artifacts.csv",
    "schema_agent_memory_model_runs.csv",
    "schema_agent_memory_drift_signals.csv",
    "schema_agent_memory_inventory_actions.csv",
    "schema_agent_memory_business_claims.csv",
    "schema_agent_memory_temporal_facts.csv",
    "schema_agent_memory_edges.csv",
    "schema_agent_memory_validation_summary.csv",
    "schema_agent_memory_validation_errors.csv",
    "schema_agent_memory_jsonl_records.csv",
    "schema_agent_memory_answer_examples.csv",
    "api_connector_registry.csv",
    "api_ingestion_plan.csv",
    "api_call_audit.csv",
    "api_security_checklist.csv",
    "api_integration_maturity.csv",
]


@dataclass
class Config:
    mode: str = "all-synthetic"
    project_root: str = "."
    real_data_dir: str = ""
    stores: int = 12
    skus: int = 150
    days: int = 365
    seed: int = 20260518
    start_date: str = ""
    write_app_outputs: bool = True
    inject_quality_issues: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "base": root / "data" / "enterprise_upgrade",
        "raw": root / "data" / "enterprise_upgrade" / "raw",
        "bronze": root / "data" / "enterprise_upgrade" / "bronze",
        "silver": root / "data" / "enterprise_upgrade" / "silver",
        "gold": root / "data" / "enterprise_upgrade" / "gold",
        "contracts": root / "data" / "enterprise_upgrade" / "contracts",
        "audit": root / "data" / "enterprise_upgrade" / "audit",
        "fabric_upload": root / "data" / "enterprise_upgrade" / "fabric_bundle" / "retail_decision_support_upload",
        "notebooks": root / "data" / "enterprise_upgrade" / "fabric_notebook",
        "outputs": root / "outputs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    return out


def make_date_range(days: int, start_date: str = "") -> pd.DatetimeIndex:
    if start_date:
        start = pd.Timestamp(start_date).normalize()
    else:
        end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days - 1)
    return pd.date_range(start=start, periods=days, freq="D")


def weighted_choice(rng: np.random.Generator, values: List[str], probs: List[float]) -> str:
    return str(rng.choice(values, p=probs))


# ---------------------------------------------------------------------
# Synthetic enterprise-like source generation
# ---------------------------------------------------------------------

def generate_store_master(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        region = REGIONS[(i - 1) % len(REGIONS)]
        sqft = int(rng.integers(8000, 95000))
        rows.append({
            "store_id": f"ST{i:03d}",
            "store_name": f"Store {i:03d}",
            "region": region,
            "city": f"City_{region}_{i:02d}",
            "province": {"West": "BC", "Prairie": "AB", "Ontario": "ON", "Quebec": "QC", "Atlantic": "NS"}[region],
            "store_format": str(rng.choice(STORE_FORMATS)),
            "sales_area_sqft": sqft,
            "open_date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=int(rng.integers(0, 5000)))).date().isoformat(),
            "active_flag": 1,
            "local_market_index": round(float(rng.normal(1.0, 0.15)), 4),
        })
    return pd.DataFrame(rows)


def generate_supplier_master(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame([{
        "supplier_id": f"SUP{i:03d}",
        "supplier_name": name,
        "lead_time_days_mean": int(rng.integers(3, 18)),
        "lead_time_days_sd": round(float(rng.uniform(0.5, 4.5)), 2),
        "fill_rate_target": round(float(rng.uniform(0.90, 0.99)), 3),
        "supplier_risk_score": round(float(rng.uniform(0.05, 0.40)), 3),
        "active_flag": 1,
    } for i, name in enumerate(SUPPLIERS, start=1)])


def generate_product_master(n: int, suppliers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    dept_items = [(dept, cat) for dept, cats in DEPARTMENTS.items() for cat in cats]
    supplier_ids = suppliers["supplier_id"].tolist()
    for i in range(1, n + 1):
        dept, cat = dept_items[(i - 1) % len(dept_items)]
        base_cost = round(float(rng.uniform(0.75, 120.0)), 2)
        margin = float(rng.uniform(0.16, 0.58))
        regular_price = round(base_cost / (1 - margin), 2)
        rows.append({
            "sku_id": f"SKU{i:05d}",
            "product_name": f"{cat} Product {i:05d}",
            "department": dept,
            "category": cat,
            "brand": f"Brand_{int(rng.integers(1, 28)):02d}",
            "pack_size": str(rng.choice(["Single", "Small", "Medium", "Large", "Family", "Bulk"])),
            "supplier_id": str(rng.choice(supplier_ids)),
            "unit_cost": base_cost,
            "regular_price": regular_price,
            "active_flag": 1,
            "perishable_flag": int(cat in ["Dairy", "Beverages", "Garden"]),
            "private_label_flag": int(rng.random() < 0.24),
            "discontinued_flag": int(rng.random() < 0.03),
        })
    return pd.DataFrame(rows)


def generate_customer_master(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        rows.append({
            "customer_hash": f"CUST_{uuid.uuid4().hex[:16]}",
            "loyalty_tier": weighted_choice(rng, ["None", "Bronze", "Silver", "Gold"], [0.42, 0.26, 0.21, 0.11]),
            "customer_segment": str(rng.choice(["Value", "Convenience", "Family", "Premium", "Occasional"])),
            "marketing_opt_in": int(rng.random() < 0.36),
            "synthetic_only_flag": 1,
        })
    return pd.DataFrame(rows)


def generate_promotions(products: pd.DataFrame, stores: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    promo_id = 1
    sample_n = min(max(15, len(products) // 3), len(products))
    for _, product in products.sample(sample_n, random_state=99).iterrows():
        for _ in range(int(rng.integers(1, 5))):
            start_idx = int(rng.integers(0, max(1, len(dates) - 28)))
            start = dates[start_idx]
            end = min(dates[-1], start + pd.Timedelta(days=int(rng.integers(7, 29))))
            scope = weighted_choice(rng, ["National", "Region", "Store"], [0.42, 0.38, 0.20])
            rows.append({
                "promotion_id": f"PROMO{promo_id:05d}",
                "sku_id": product["sku_id"],
                "promo_type": str(rng.choice(PROMO_TYPES)),
                "scope": scope,
                "region": str(rng.choice(REGIONS)) if scope == "Region" else "",
                "store_id": str(stores.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]["store_id"]) if scope == "Store" else "",
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "discount_depth": round(float(rng.uniform(0.05, 0.40)), 3),
                "campaign_name": f"{product['category']} campaign {promo_id:05d}",
            })
            promo_id += 1
    return pd.DataFrame(rows)


def active_promo(promos: pd.DataFrame, sku_id: str, store_id: str, region: str, date: pd.Timestamp) -> Tuple[float, str]:
    if promos.empty:
        return 0.0, ""
    p = promos[
        (promos["sku_id"].astype(str) == str(sku_id)) &
        (pd.to_datetime(promos["start_date"]) <= date) &
        (pd.to_datetime(promos["end_date"]) >= date)
    ]
    for _, row in p.iterrows():
        applies = row["scope"] == "National" or (row["scope"] == "Region" and row["region"] == region) or (row["scope"] == "Store" and row["store_id"] == store_id)
        if applies:
            return float(row["discount_depth"]), str(row["promotion_id"])
    return 0.0, ""


def generate_pos(products: pd.DataFrame, stores: pd.DataFrame, customers: pd.DataFrame, promos: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    tx = 1
    sku_pop = {sku: float(rng.lognormal(mean=0.0, sigma=0.75)) for sku in products["sku_id"]}
    store_factor = {
        r["store_id"]: float(rng.uniform(0.70, 1.45)) * (float(r["sales_area_sqft"]) / 40000) ** 0.20 * float(r["local_market_index"])
        for _, r in stores.iterrows()
    }
    for d in dates:
        weekend = d.dayofweek >= 5
        seasonal = 1.0 + 0.10 * math.sin(2 * math.pi * (d.dayofyear / 365.25)) + (0.15 if d.month in [11, 12] else 0.0)
        for _, store in stores.iterrows():
            for _, product in products.sample(frac=float(rng.uniform(0.35, 0.68)), random_state=int(rng.integers(0, 1_000_000))).iterrows():
                sku = str(product["sku_id"])
                discount, promo_id = active_promo(promos, sku, str(store["store_id"]), str(store["region"]), d)
                category_factor = 1.15 if product["category"] in ["Beverages", "Snacks"] and weekend else 1.0
                lam = max(0.05, sku_pop[sku] * store_factor[store["store_id"]] * seasonal * category_factor * (1 + 1.7 * discount))
                units = int(max(0, rng.poisson(lam=lam)))
                if units == 0:
                    continue
                regular = float(product["regular_price"])
                price = round(regular * (1 - discount), 2)
                cost = float(product["unit_cost"])
                customer_hash = ""
                if rng.random() < 0.35:
                    customer_hash = str(customers.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]["customer_hash"])
                rows.append({
                    "transaction_id": f"TX{tx:010d}",
                    "transaction_date": d.date().isoformat(),
                    "store_id": store["store_id"],
                    "sku_id": sku,
                    "customer_hash": customer_hash,
                    "units_sold": units,
                    "gross_sales": round(units * regular, 2),
                    "net_sales": round(units * price, 2),
                    "unit_price": price,
                    "discount_amount": round(units * regular * discount, 2),
                    "promotion_id": promo_id,
                    "gross_margin": round((price - cost) * units, 2),
                })
                tx += 1
    return pd.DataFrame(rows)


def generate_pricing(products: pd.DataFrame, stores: pd.DataFrame, dates: pd.DatetimeIndex, promos: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _, product in products.iterrows():
        for _, store in stores.iterrows():
            for d in dates[::7]:
                discount, _ = active_promo(promos, str(product["sku_id"]), str(store["store_id"]), str(store["region"]), d)
                regular = float(product["regular_price"])
                rows.append({
                    "price_date": d.date().isoformat(),
                    "store_id": store["store_id"],
                    "sku_id": product["sku_id"],
                    "regular_price": round(regular, 2),
                    "selling_price": round(regular * (1 - discount) * float(rng.normal(1.0, 0.012)), 2),
                    "discount_depth": round(discount, 3),
                })
    return pd.DataFrame(rows)


def generate_inventory(products: pd.DataFrame, stores: pd.DataFrame, pos: pd.DataFrame, suppliers: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    daily = pos.groupby(["transaction_date", "store_id", "sku_id"], as_index=False)["units_sold"].sum().rename(columns={"transaction_date": "date"})
    lead_lookup = products[["sku_id", "supplier_id"]].merge(suppliers[["supplier_id", "lead_time_days_mean"]], on="supplier_id", how="left").set_index("sku_id")["lead_time_days_mean"].to_dict()
    rows = []
    sku_ids = products["sku_id"].tolist()
    for d in dates[::7]:
        for _, store in stores.iterrows():
            for sku in rng.choice(sku_ids, size=min(len(sku_ids), max(20, len(sku_ids) // 2)), replace=False):
                recent = daily[(daily["store_id"] == store["store_id"]) & (daily["sku_id"] == sku) & (pd.to_datetime(daily["date"]) <= d) & (pd.to_datetime(daily["date"]) > d - pd.Timedelta(days=28))]["units_sold"].sum()
                avg_daily = recent / 28 if recent > 0 else float(rng.uniform(0.1, 2.0))
                lead = float(lead_lookup.get(sku, 7))
                safety = avg_daily * float(rng.uniform(3, 10))
                reorder = int(max(3, round(avg_daily * lead + safety)))
                on_hand = int(max(0, round(reorder * float(rng.uniform(0.1, 3.5)))))
                rows.append({
                    "date": d.date().isoformat(),
                    "store_id": store["store_id"],
                    "sku_id": sku,
                    "stock_on_hand": on_hand,
                    "stock_on_order": int(max(0, round(reorder * float(rng.uniform(0, 1.2))))),
                    "reorder_point": reorder,
                    "stockout_flag": int(on_hand == 0),
                    "days_of_supply_est": round(on_hand / max(avg_daily, 0.1), 1),
                    "safety_stock_est": round(safety, 1),
                })
    return pd.DataFrame(rows)


def generate_finance(pos: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    out = pos.groupby(["transaction_date", "store_id"], as_index=False).agg(
        net_sales=("net_sales", "sum"),
        gross_margin=("gross_margin", "sum"),
        units_sold=("units_sold", "sum"),
        transactions=("transaction_id", "nunique"),
    ).rename(columns={"transaction_date": "date"})
    out = out.merge(stores[["store_id", "region", "sales_area_sqft"]], on="store_id", how="left")
    out["sales_per_sqft"] = (out["net_sales"] / out["sales_area_sqft"].replace(0, np.nan)).fillna(0).round(4)
    out["gross_margin_rate"] = (out["gross_margin"] / out["net_sales"].replace(0, np.nan)).fillna(0).round(4)
    return out


def generate_synthetic_raw(config: Config) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)
    dates = make_date_range(config.days, config.start_date)
    stores = generate_store_master(config.stores, rng)
    suppliers = generate_supplier_master(rng)
    products = generate_product_master(config.skus, suppliers, rng)
    customers = generate_customer_master(max(500, config.stores * config.skus * 5), rng)
    promos = generate_promotions(products, stores, dates, rng)
    pricing = generate_pricing(products, stores, dates, promos, rng)
    pos = generate_pos(products, stores, customers, promos, dates, rng)
    inventory = generate_inventory(products, stores, pos, suppliers, dates, rng)
    finance = generate_finance(pos, stores)
    raw = {
        "store_master": stores,
        "supplier_master": suppliers,
        "product_master": products,
        "customer_master_synthetic": customers,
        "promotions_calendar": promos,
        "pricing_history": pricing,
        "pos_transaction_lines": pos,
        "inventory_snapshots": inventory,
        "finance_daily_summary": finance,
    }
    if config.inject_quality_issues and len(raw["pos_transaction_lines"]) > 100:
        idx = raw["pos_transaction_lines"].sample(5, random_state=123).index
        raw["pos_transaction_lines"].loc[idx, "customer_hash"] = ""
    return raw


# ---------------------------------------------------------------------
# Public or approved real/pilot data ingestion
# ---------------------------------------------------------------------

def load_walmart_public(real_dir: Path) -> Dict[str, pd.DataFrame]:
    train_path, stores_path, features_path = real_dir / "train.csv", real_dir / "stores.csv", real_dir / "features.csv"
    missing = [str(p) for p in [train_path, stores_path, features_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Walmart-style files: " + "; ".join(missing))
    train = normalize_columns(read_csv_safe(train_path))
    stores_src = normalize_columns(read_csv_safe(stores_path))
    _features = normalize_columns(read_csv_safe(features_path))
    req = {"store", "dept", "date", "weekly_sales"}
    if not req.issubset(set(train.columns)):
        raise ValueError("train.csv must contain Store, Dept, Date, Weekly_Sales.")
    store_master = stores_src.copy()
    store_master["store_id"] = "WM_ST" + store_master["store"].astype(str).str.zfill(3)
    store_master["store_name"] = "Walmart Public Store " + store_master["store"].astype(str)
    store_master["region"] = "Public_Walmart"
    store_master["city"] = "Not provided"
    store_master["province"] = "Not provided"
    store_master["store_format"] = store_master.get("type", "Not provided")
    store_master["sales_area_sqft"] = store_master.get("size", 0)
    store_master["open_date"] = ""
    store_master["active_flag"] = 1
    store_master = store_master[["store_id", "store_name", "region", "city", "province", "store_format", "sales_area_sqft", "open_date", "active_flag"]]
    depts = sorted(train["dept"].dropna().unique().tolist())
    product_master = pd.DataFrame({
        "sku_id": [f"WM_DEPT_{int(d):03d}" for d in depts],
        "product_name": [f"Department {int(d):03d}" for d in depts],
        "department": [f"Dept_{int(d):03d}" for d in depts],
        "category": ["Public Walmart Department"] * len(depts),
        "brand": ["Not provided"] * len(depts),
        "pack_size": ["Not provided"] * len(depts),
        "supplier_id": ["SUP_PUBLIC"] * len(depts),
        "unit_cost": [np.nan] * len(depts),
        "regular_price": [np.nan] * len(depts),
        "active_flag": [1] * len(depts),
    })
    supplier_master = pd.DataFrame([{"supplier_id": "SUP_PUBLIC", "supplier_name": "Not provided in public Walmart dataset"}])
    pos = train.copy()
    pos["transaction_id"] = ["WM_TX_" + str(i).zfill(10) for i in range(1, len(pos) + 1)]
    pos["transaction_date"] = pd.to_datetime(pos["date"]).dt.date.astype(str)
    pos["store_id"] = "WM_ST" + pos["store"].astype(str).str.zfill(3)
    pos["sku_id"] = "WM_DEPT_" + pos["dept"].astype(int).astype(str).str.zfill(3)
    pos["customer_hash"] = ""
    pos["units_sold"] = np.nan
    pos["gross_sales"] = pos["weekly_sales"]
    pos["net_sales"] = pos["weekly_sales"]
    pos["unit_price"] = np.nan
    pos["discount_amount"] = 0
    pos["promotion_id"] = ""
    pos["gross_margin"] = np.nan
    pos = pos[["transaction_id", "transaction_date", "store_id", "sku_id", "customer_hash", "units_sold", "gross_sales", "net_sales", "unit_price", "discount_amount", "promotion_id", "gross_margin"]]
    finance = pos.groupby(["transaction_date", "store_id"], as_index=False).agg(net_sales=("net_sales", "sum")).rename(columns={"transaction_date": "date"})
    finance["gross_margin"] = np.nan
    finance["units_sold"] = np.nan
    finance["transactions"] = np.nan
    finance = finance.merge(store_master[["store_id", "region", "sales_area_sqft"]], on="store_id", how="left")
    finance["sales_per_sqft"] = (finance["net_sales"] / finance["sales_area_sqft"].replace(0, np.nan)).fillna(0).round(4)
    finance["gross_margin_rate"] = np.nan
    return {
        "store_master": store_master,
        "supplier_master": supplier_master,
        "product_master": product_master,
        "customer_master_synthetic": pd.DataFrame(columns=["customer_hash", "loyalty_tier", "customer_segment", "marketing_opt_in", "synthetic_only_flag"]),
        "promotions_calendar": pd.DataFrame(columns=["promotion_id", "sku_id", "promo_type", "scope", "region", "store_id", "start_date", "end_date", "discount_depth", "campaign_name"]),
        "pricing_history": pd.DataFrame(columns=["price_date", "store_id", "sku_id", "regular_price", "selling_price", "discount_depth"]),
        "pos_transaction_lines": pos,
        "inventory_snapshots": pd.DataFrame(columns=["date", "store_id", "sku_id", "stock_on_hand", "stock_on_order", "reorder_point", "stockout_flag", "days_of_supply_est", "safety_stock_est"]),
        "finance_daily_summary": finance,
    }


def load_generic_real(real_dir: Path) -> Dict[str, pd.DataFrame]:
    required = ["pos_transaction_lines.csv", "product_master.csv", "store_master.csv"]
    missing = [f for f in required if not (real_dir / f).exists()]
    if missing:
        raise FileNotFoundError("Missing required approved real/pilot files: " + "; ".join(missing))
    raw = {}
    for fname in ["store_master.csv", "supplier_master.csv", "product_master.csv", "customer_master_synthetic.csv", "promotions_calendar.csv", "pricing_history.csv", "pos_transaction_lines.csv", "inventory_snapshots.csv", "finance_daily_summary.csv"]:
        p = real_dir / fname
        if p.exists():
            raw[Path(fname).stem] = normalize_columns(read_csv_safe(p))
    raw.setdefault("supplier_master", pd.DataFrame(columns=["supplier_id", "supplier_name"]))
    raw.setdefault("customer_master_synthetic", pd.DataFrame(columns=["customer_hash", "loyalty_tier", "customer_segment", "marketing_opt_in", "synthetic_only_flag"]))
    raw.setdefault("promotions_calendar", pd.DataFrame(columns=["promotion_id", "sku_id", "promo_type", "scope", "region", "store_id", "start_date", "end_date", "discount_depth", "campaign_name"]))
    raw.setdefault("pricing_history", pd.DataFrame(columns=["price_date", "store_id", "sku_id", "regular_price", "selling_price", "discount_depth"]))
    raw.setdefault("inventory_snapshots", pd.DataFrame(columns=["date", "store_id", "sku_id", "stock_on_hand", "stock_on_order", "reorder_point", "stockout_flag", "days_of_supply_est", "safety_stock_est"]))
    if "finance_daily_summary" not in raw:
        pos = raw["pos_transaction_lines"]
        if {"transaction_date", "store_id", "net_sales"}.issubset(pos.columns):
            raw["finance_daily_summary"] = pos.groupby(["transaction_date", "store_id"], as_index=False).agg(net_sales=("net_sales", "sum")).rename(columns={"transaction_date": "date"})
        else:
            raw["finance_daily_summary"] = pd.DataFrame(columns=["date", "store_id", "net_sales"])
    return raw


# ---------------------------------------------------------------------
# Validation, contracts, silver and gold outputs
# ---------------------------------------------------------------------

def validate_table(name: str, df: pd.DataFrame, layer: str, source_type: str) -> Dict[str, Any]:
    required = {
        "store_master": ["store_id"],
        "product_master": ["sku_id"],
        "pos_transaction_lines": ["transaction_date", "store_id", "sku_id", "net_sales"],
        "finance_daily_summary": ["date", "store_id", "net_sales"],
    }.get(name, [])
    missing_req = [c for c in required if c not in df.columns]
    status = "PASS"
    issues = []
    if len(df) == 0 and required:
        status = "FAIL"
        issues.append("zero_rows")
    elif len(df) == 0:
        status = "WARNING"
        issues.append("zero_rows_optional")
    if missing_req:
        status = "FAIL"
        issues.append("missing_required:" + ",".join(missing_req))
    missing_cells = int(df.isna().sum().sum()) if len(df.columns) else 0
    duplicates = int(df.duplicated().sum()) if len(df) else 0
    if duplicates and status != "FAIL":
        status = "WARNING"
        issues.append("duplicate_rows")
    if missing_cells and status != "FAIL":
        status = "WARNING"
        issues.append("missing_cells")
    return {
        "table_name": name,
        "layer": layer,
        "source_type": source_type,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicates,
        "missing_required_columns": ",".join(missing_req),
        "status": status,
        "issues": ";".join(issues),
        "decision_implication": "Synthetic only" if source_type == "synthetic" else "Candidate pilot/public evidence, not production by itself",
    }


def build_silver(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    pos = raw.get("pos_transaction_lines", pd.DataFrame()).copy()
    products = raw.get("product_master", pd.DataFrame()).copy()
    stores = raw.get("store_master", pd.DataFrame()).copy()
    inv = raw.get("inventory_snapshots", pd.DataFrame()).copy()
    if "transaction_date" in pos.columns:
        pos["transaction_date"] = pd.to_datetime(pos["transaction_date"], errors="coerce")
    pos["units_sold"] = pd.to_numeric(pos["units_sold"], errors="coerce") if "units_sold" in pos.columns else np.nan
    pos["net_sales"] = pd.to_numeric(pos["net_sales"], errors="coerce").fillna(0) if "net_sales" in pos.columns else 0
    pos["gross_margin"] = pd.to_numeric(pos["gross_margin"], errors="coerce") if "gross_margin" in pos.columns else np.nan
    pos["unit_price"] = pd.to_numeric(pos["unit_price"], errors="coerce") if "unit_price" in pos.columns else np.nan
    if "promotion_id" not in pos.columns:
        pos["promotion_id"] = ""
    sales = pos.copy()
    if not products.empty and "sku_id" in products.columns and "sku_id" in sales.columns:
        sales = sales.merge(products, on="sku_id", how="left")
    if not stores.empty and "store_id" in stores.columns and "store_id" in sales.columns:
        sales = sales.merge(stores, on="store_id", how="left", suffixes=("", "_store"))
    group_cols = [c for c in ["transaction_date", "store_id", "region", "department", "category", "brand", "sku_id"] if c in sales.columns]
    if group_cols:
        daily = sales.groupby(group_cols, as_index=False).agg(
            units_sold=("units_sold", "sum"),
            net_sales=("net_sales", "sum"),
            gross_margin=("gross_margin", "sum"),
            avg_unit_price=("unit_price", "mean"),
            promo_flag=("promotion_id", lambda s: int((s.astype(str) != "").any())),
        )
    else:
        daily = pd.DataFrame()
    inv_enriched = inv.copy()
    if not inv_enriched.empty:
        if "date" in inv_enriched.columns:
            inv_enriched["date"] = pd.to_datetime(inv_enriched["date"], errors="coerce")
        if not products.empty and "sku_id" in inv_enriched.columns and "sku_id" in products.columns:
            cols = [c for c in ["sku_id", "product_name", "department", "category", "brand", "supplier_id"] if c in products.columns]
            inv_enriched = inv_enriched.merge(products[cols], on="sku_id", how="left")
        if not stores.empty and "store_id" in inv_enriched.columns and "store_id" in stores.columns:
            cols = [c for c in ["store_id", "region", "store_format"] if c in stores.columns]
            inv_enriched = inv_enriched.merge(stores[cols], on="store_id", how="left")
        if {"reorder_point", "stock_on_hand"}.issubset(inv_enriched.columns):
            inv_enriched["inventory_gap"] = pd.to_numeric(inv_enriched["reorder_point"], errors="coerce") - pd.to_numeric(inv_enriched["stock_on_hand"], errors="coerce")
    return {
        "silver_pos_transaction_lines": pos,
        "silver_sales_enriched": sales,
        "silver_daily_sku_store_demand": daily,
        "silver_inventory_enriched": inv_enriched,
    }


def build_contracts(raw: Dict[str, pd.DataFrame], source_type: str) -> Dict[str, Any]:
    keys = {
        "store_master": ["store_id"],
        "supplier_master": ["supplier_id"],
        "product_master": ["sku_id"],
        "customer_master_synthetic": ["customer_hash"],
        "promotions_calendar": ["promotion_id"],
        "pricing_history": ["price_date", "store_id", "sku_id"],
        "pos_transaction_lines": ["transaction_id"],
        "inventory_snapshots": ["date", "store_id", "sku_id"],
        "finance_daily_summary": ["date", "store_id"],
    }
    contracts = {}
    for name, df in raw.items():
        fields = [{"name": c, "observed_dtype": str(df[c].dtype), "nullable_observed": bool(df[c].isna().any()) if len(df) else True, "example_value": None if df.empty else str(df[c].iloc[0])} for c in df.columns]
        contracts[name] = {
            "table_name": name,
            "source_type": source_type,
            "fields": fields,
            "business_keys": keys.get(name, []),
            "claim_boundary": "Synthetic contract" if source_type == "synthetic" else "Draft contract requiring source-owner confirmation",
        }
    return contracts


def build_quality(raw: Dict[str, pd.DataFrame], silver: Dict[str, pd.DataFrame], source_type: str) -> pd.DataFrame:
    rows = [validate_table(k, v, "raw", source_type) for k, v in raw.items()]
    rows += [validate_table(k, v, "silver", source_type) for k, v in silver.items()]
    return pd.DataFrame(rows)


def build_model_comparison(source_type: str, rng: np.random.Generator) -> pd.DataFrame:
    models = [
        ("seasonal_naive", "baseline"),
        ("moving_average_28d", "baseline"),
        ("elasticnet", "classical_ml"),
        ("random_forest", "classical_ml"),
        ("extra_trees", "classical_ml"),
        ("hist_gradient_boosting", "classical_ml"),
        ("xgboost_like", "classical_ml"),
        ("lightgbm_like", "classical_ml"),
        ("mlp_deep_tabular", "deep_tabular"),
        ("lstm_gru_sequence", "deep_sequence"),
    ]
    rows = []
    for i, (m, t) in enumerate(models):
        mae = max(2.85, 3.95 - i * 0.12 + float(rng.normal(0, 0.07)))
        if i == len(models) - 1:
            mae = 2.85
        rows.append({
            "model_name": m,
            "model_type": t,
            "test_mae": round(mae, 4),
            "test_rmse": round(mae * float(rng.uniform(1.20, 1.45)), 4),
            "test_wmape": round(float(rng.uniform(0.17, 0.29)), 4),
            "bias": round(float(rng.normal(0, 0.08)), 4),
            "selected_flag": 1 if i == len(models) - 1 else 0,
            "decision_note": "Demo metric; real temporal validation required.",
            "source_type": source_type,
            "claim_boundary": "Not proof of real enterprise model performance unless trained/validated on approved real data.",
        })
    return pd.DataFrame(rows).sort_values("test_mae").reset_index(drop=True)


def forecast_group(daily: pd.DataFrame, group_col: str, rng: np.random.Generator, source_type: str) -> pd.DataFrame:
    if daily.empty or group_col not in daily.columns or "net_sales" not in daily.columns:
        return pd.DataFrame(columns=[group_col, "actual_units_28d", "actual_sales_28d", "forecast_units_next_28d", "forecast_sales_next_28d", "forecast_status", "claim_boundary"])
    if "transaction_date" in daily.columns:
        latest = pd.to_datetime(daily["transaction_date"], errors="coerce").max()
        last_28 = daily[pd.to_datetime(daily["transaction_date"], errors="coerce") > latest - pd.Timedelta(days=28)].copy()
    else:
        last_28 = daily.copy()
    agg = {"net_sales": "sum"}
    if "units_sold" in last_28.columns:
        agg["units_sold"] = "sum"
    grp = last_28.groupby(group_col, as_index=False).agg(agg).rename(columns={"net_sales": "actual_sales_28d", "units_sold": "actual_units_28d"})
    if "actual_units_28d" not in grp.columns:
        grp["actual_units_28d"] = np.nan
    grp["forecast_units_next_28d"] = np.where(grp["actual_units_28d"].notna(), (grp["actual_units_28d"].fillna(0) * rng.uniform(0.92, 1.12, size=len(grp))).round(0), np.nan)
    grp["forecast_sales_next_28d"] = (grp["actual_sales_28d"].fillna(0) * rng.uniform(0.92, 1.12, size=len(grp))).round(2)
    grp["forecast_status"] = np.where(grp["forecast_sales_next_28d"] > grp["actual_sales_28d"], "Growth", "Stable/Decline")
    grp["source_type"] = source_type
    grp["claim_boundary"] = "Forecast-style output; production use requires validated forecasting model and thresholds."
    return grp


def inventory_recs(inv: pd.DataFrame, source_type: str) -> pd.DataFrame:
    if inv.empty:
        return pd.DataFrame(columns=["date", "store_id", "sku_id", "recommended_action", "priority", "human_review_required", "claim_boundary"])
    out = inv.copy()
    if "date" in out.columns:
        latest = pd.to_datetime(out["date"], errors="coerce").max()
        if pd.notna(latest):
            out = out[pd.to_datetime(out["date"], errors="coerce") == latest].copy()
    for c in ["stockout_flag", "inventory_gap", "days_of_supply_est"]:
        if c not in out.columns:
            out[c] = np.nan
    out["recommended_action"] = np.select(
        [out["stockout_flag"].fillna(0).astype(float) == 1, out["inventory_gap"].fillna(0).astype(float) > 5, out["days_of_supply_est"].fillna(999).astype(float) > 45],
        ["Urgent Reorder", "Reorder", "Reduce / Review"],
        default="Maintain"
    )
    out["priority"] = np.select(
        [out["recommended_action"] == "Urgent Reorder", out["recommended_action"] == "Reorder", out["recommended_action"] == "Reduce / Review"],
        ["High", "Medium", "Low"],
        default="Normal"
    )
    out["human_review_required"] = np.where(out["priority"].isin(["High", "Medium"]), 1, 0)
    out["source_type"] = source_type
    out["claim_boundary"] = "Recommendation requires real constraints and human approval before operational use."
    keep = [c for c in ["date", "store_id", "region", "sku_id", "product_name", "department", "category", "brand", "stock_on_hand", "stock_on_order", "reorder_point", "days_of_supply_est", "recommended_action", "priority", "human_review_required", "source_type", "claim_boundary"] if c in out.columns]
    return out[keep].head(500)


def monitoring_outputs(source_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drift = pd.DataFrame([
        {"feature": "price_log", "ks_stat": 0.0026141971403264, "p_value": 0.9998384213866576, "status": "No material drift", "recommended_response": "Continue monitoring"},
        {"feature": "lag_7", "ks_stat": 0.0125150786621114, "p_value": 0.0106158802835363, "status": "Watch", "recommended_response": "Review feature drift context"},
        {"feature": "rolling_mean_28", "ks_stat": 0.0197182504534143, "p_value": 0.000004533244956014451, "status": "Watch", "recommended_response": "Review demand trend changes"},
        {"feature": "assortment_health_ratio", "ks_stat": 0.0605955544308068, "p_value": 9.818204481279467e-54, "status": "Watch", "recommended_response": "Check assortment changes before acting"},
        {"feature": "sales_per_sqft_28d", "ks_stat": 0.0855683495718925, "p_value": 8.34916387145772e-107, "status": "Watch", "recommended_response": "Review store productivity shifts"},
    ])
    drift["source_type"] = source_type
    drift["claim_boundary"] = "Demo monitoring output; thresholds require production calibration."
    status = pd.DataFrame([
        {"metric": "drift_detected", "value": 1},
        {"metric": "num_drifted_features", "value": 4},
        {"metric": "max_ks_stat", "value": 0.08556834957189252},
        {"metric": "current_wmape", "value": 0.2119774701690027},
        {"metric": "reference_wmape", "value": 0.21302451943126682},
        {"metric": "wmape_degradation", "value": -0.004915158428989936},
        {"metric": "retraining_recommended", "value": 0},
        {"metric": "watch_status", "value": "Watch"},
        {"metric": "source_type", "value": source_type},
    ])
    audit = pd.DataFrame([
        {"metric": "triggered", "value": "False"},
        {"metric": "trigger_reason", "value": ""},
        {"metric": "retrained_model_name", "value": ""},
        {"metric": "candidate_mae", "value": ""},
        {"metric": "current_champion_mae", "value": 2.857676956866885},
        {"metric": "relative_improvement", "value": ""},
        {"metric": "promoted", "value": "False"},
        {"metric": "deployment_action", "value": "No retraining triggered"},
        {"metric": "claim_boundary", "value": "Demo audit output, not production retraining governance."},
    ])
    return drift, status, audit


def store_watchlist(finance: pd.DataFrame, source_type: str) -> pd.DataFrame:
    if finance.empty or "store_id" not in finance.columns or "net_sales" not in finance.columns:
        return pd.DataFrame(columns=["store_id", "watch_status", "watch_reason", "claim_boundary"])
    cols = ["store_id"] + (["region"] if "region" in finance.columns else [])
    agg = {"net_sales": "sum"}
    if "gross_margin_rate" in finance.columns:
        agg["gross_margin_rate"] = "mean"
    if "sales_per_sqft" in finance.columns:
        agg["sales_per_sqft"] = "mean"
    if "transactions" in finance.columns:
        agg["transactions"] = "sum"
    out = finance.groupby(cols, as_index=False).agg(agg)
    if "gross_margin_rate" in out.columns and out["gross_margin_rate"].notna().any():
        cut = out["gross_margin_rate"].quantile(0.25)
        out["watch_reason"] = np.where(out["gross_margin_rate"] < cut, "Low margin rate", "Normal")
    elif "sales_per_sqft" in out.columns and out["sales_per_sqft"].notna().any():
        cut = out["sales_per_sqft"].quantile(0.25)
        out["watch_reason"] = np.where(out["sales_per_sqft"] < cut, "Low sales productivity", "Normal")
    else:
        out["watch_reason"] = "Insufficient productivity/margin fields"
    out["watch_status"] = np.where(out["watch_reason"] == "Normal", "OK", "Watch")
    out["source_type"] = source_type
    out["claim_boundary"] = "Watchlist thresholds require business-owner calibration."
    return out


def pipeline_maturity(source_type: str) -> pd.DataFrame:
    rows = [
        ("Source data loaded", "Implemented", "Demo / Portfolio" if source_type == "synthetic" else "Pilot Evidence Candidate", "Confirm source authorization and quality."),
        ("Data contracts", "Generated as JSON documentation", "Demo / Portfolio", "Validate with real source owners."),
        ("Raw/Bronze layer", "Implemented locally as CSV", "Demo / Portfolio", "Move to governed Lakehouse/Warehouse for Fabric evidence."),
        ("Silver cleaned/joined layer", "Implemented locally as CSV", "Demo / Portfolio", "Schedule and monitor transformations."),
        ("Gold/dashboard outputs", "Implemented locally as CSV", "Demo / Portfolio", "Connect to governed BI/dashboard layer."),
        ("Model comparison", "Demo table generated", "Demo / Portfolio", "Train and validate with temporal backtesting."),
        ("Deep learning", "Represented as candidate model family", "Demo / Portfolio", "Prove lift over baselines on approved real data."),
        ("Monitoring and retraining status", "Demo outputs generated", "Demo / Portfolio", "Calibrate thresholds and alerts."),
        ("Human-in-the-loop workflow", "Workflow handoff table generated", "Demo / Portfolio", "Integrate approvals with operational tools."),
        ("Real enterprise source integration", "Not implemented unless generic-real uses approved live source exports", "Not Confirmed", "Connect governed source feeds."),
        ("Live Microsoft Fabric execution", "Not implemented by local script", "Not Confirmed", "Run generated Fabric notebook/pipeline and capture history."),
        ("Enterprise security/RBAC", "Not implemented", "Not Confirmed", "Add authentication, roles, secrets, and data access policy."),
        ("Production SLA/observability", "Not implemented", "Not Confirmed", "Add alerting, incident response, rollback, uptime tracking."),
    ]
    return pd.DataFrame(rows, columns=["capability", "status", "maturity", "upgrade_needed"])


def workflow_handoff(source_type: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"workflow_step": "Recommendations Generated", "status": "Completed", "owner": "System", "action_required": "Review high-priority inventory recommendations", "auto_execute": 0},
        {"workflow_step": "Analyst Review", "status": "Pending", "owner": "Analyst", "action_required": "Validate drift/watch items and recommendation logic", "auto_execute": 0},
        {"workflow_step": "Manager Approval", "status": "Pending", "owner": "Business Manager", "action_required": "Approve operational action before execution", "auto_execute": 0},
        {"workflow_step": "Actions Published", "status": "Blocked until approval", "owner": "Operations", "action_required": "Do not auto-execute in demo/pilot mode", "auto_execute": 0},
        {"workflow_step": "Source Type", "status": source_type, "owner": "Documentation", "action_required": "Do not overclaim source maturity", "auto_execute": 0},
    ])


def agent_answers(source_type: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"question": "Which model is best?", "answer": "The best-ranked model is currently best under active filters, but production performance requires real temporal validation.", "source_table": "dashboard_model_comparison.csv", "claim_boundary": "Demo or pilot evidence only."},
        {"question": "Is drift elevated?", "answer": "Drift monitoring has rows in Watch status. Production thresholds must be calibrated before blocking outputs or retraining.", "source_table": "drift_monitor.csv", "claim_boundary": "Demo monitoring evidence."},
        {"question": "Can the system auto-execute inventory decisions?", "answer": "No. The workflow requires human review and approval before operational action.", "source_table": "workflow_handoff.csv", "claim_boundary": "Not enterprise workflow integration by itself."},
        {"question": "Does this prove real enterprise integration?", "answer": "Only if approved real enterprise data and source documentation are provided. Synthetic/public data alone does not prove enterprise integration.", "source_table": "run_manifest.json", "claim_boundary": f"Current source type: {source_type}"},
    ])


def build_gold(raw: Dict[str, pd.DataFrame], silver: Dict[str, pd.DataFrame], quality: pd.DataFrame, contracts: Dict[str, Any], source_type: str, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    finance = raw.get("finance_daily_summary", pd.DataFrame())
    daily = silver.get("silver_daily_sku_store_demand", pd.DataFrame())
    stores = raw.get("store_master", pd.DataFrame())
    products = raw.get("product_master", pd.DataFrame())
    model = build_model_comparison(source_type, rng)
    drift, status, audit = monitoring_outputs(source_type)
    total_sales = float(pd.to_numeric(finance.get("net_sales", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not finance.empty else 0.0
    total_units = float(pd.to_numeric(finance.get("units_sold", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not finance.empty else np.nan
    gross_margin = float(pd.to_numeric(finance.get("gross_margin", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not finance.empty else np.nan
    executive = pd.DataFrame([
        {"metric": "total_sales", "value": round(total_sales, 2), "decision_note": "Sales from generated or provided source data."},
        {"metric": "total_units", "value": total_units, "decision_note": "Units where available."},
        {"metric": "gross_margin", "value": gross_margin, "decision_note": "Margin where available."},
        {"metric": "num_stores", "value": stores["store_id"].nunique() if "store_id" in stores.columns else np.nan, "decision_note": "Stores in source."},
        {"metric": "num_skus", "value": products["sku_id"].nunique() if "sku_id" in products.columns else np.nan, "decision_note": "SKUs/departments in source."},
        {"metric": "model_rows", "value": len(model), "decision_note": "Model comparison rows."},
        {"metric": "best_mae", "value": float(model["test_mae"].min()), "decision_note": "Demo model metric; not production proof."},
        {"metric": "drift_rows", "value": len(drift), "decision_note": "Monitoring rows."},
        {"metric": "retraining_rows", "value": len(status), "decision_note": "Retraining status rows."},
        {"metric": "audit_rows", "value": len(audit), "decision_note": "Retraining audit rows."},
        {"metric": "source_type", "value": source_type, "decision_note": "Claim boundary depends on source type."},
    ])
    contract_summary = pd.DataFrame([{
        "table_name": n,
        "field_count": len(c["fields"]),
        "business_keys": ",".join(c["business_keys"]),
        "claim_boundary": c["claim_boundary"]
    } for n, c in contracts.items()])
    return {
        "dashboard_executive_summary": executive,
        "dashboard_model_comparison": model,
        "dashboard_store_forecast": forecast_group(daily, "store_id", rng, source_type),
        "dashboard_department_forecast": forecast_group(daily, "department", rng, source_type),
        "dashboard_region_forecast": forecast_group(daily, "region", rng, source_type),
        "dashboard_brand_forecast": forecast_group(daily, "brand", rng, source_type),
        "inventory_recommendations": inventory_recs(silver.get("silver_inventory_enriched", pd.DataFrame()), source_type),
        "drift_monitor": drift,
        "retraining_status": status,
        "retraining_audit": audit,
        "store_watchlist": store_watchlist(finance, source_type),
        "dashboard_pipeline_maturity": pipeline_maturity(source_type),
        "workflow_handoff": workflow_handoff(source_type),
        "agent_answers": agent_answers(source_type),
        "data_quality_audit": quality,
        "data_contract_summary": contract_summary,
    }




# ---------------------------------------------------------------------
# Deterministic agent-style governance layer
# ---------------------------------------------------------------------

def agent_data_quality_review(quality: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """
    Deterministic Data Quality Agent.
    This is not an autonomous AI agent. It is an auditable rule-based reviewer.
    """
    rows = []
    fail_count = int((quality["status"] == "FAIL").sum()) if "status" in quality.columns else 0
    warning_count = int((quality["status"] == "WARNING").sum()) if "status" in quality.columns else 0

    if fail_count > 0:
        decision = "BLOCK"
        reason = "At least one required data table failed validation."
    elif warning_count > 0:
        decision = "REVIEW"
        reason = "One or more data-quality warnings exist."
    else:
        decision = "PASS"
        reason = "No blocking data-quality issues detected."

    rows.append({
        "agent_name": "Data Quality Agent",
        "review_area": "schema, rows, missingness, duplicates",
        "input_table": "data_quality_audit.csv",
        "decision": decision,
        "risk_level": "High" if decision == "BLOCK" else ("Medium" if decision == "REVIEW" else "Low"),
        "reason": reason,
        "required_human_action": "Fix failed tables before using model outputs." if decision == "BLOCK" else ("Review warnings before operational use." if decision == "REVIEW" else "No immediate action."),
        "claim_boundary": "Rule-based governance check; not autonomous AI.",
        "source_type": source_type,
    })
    return pd.DataFrame(rows)


def agent_model_validation_review(model_comparison: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """
    Deterministic Model Validation Agent.
    Blocks overclaiming when model evidence is synthetic/demo or not temporally validated.
    """
    selected = model_comparison[model_comparison.get("selected_flag", 0) == 1] if not model_comparison.empty else pd.DataFrame()
    best_model = selected.iloc[0]["model_name"] if not selected.empty and "model_name" in selected.columns else "Not confirmed"
    best_mae = selected.iloc[0]["test_mae"] if not selected.empty and "test_mae" in selected.columns else None

    if source_type == "synthetic":
        decision = "REVIEW"
        risk = "Medium"
        reason = "Model metrics are generated from synthetic/demo evidence and do not prove real enterprise performance."
        action = "Use for portfolio demonstration only; validate on approved real temporal data before operational use."
    elif source_type == "walmart-public":
        decision = "REVIEW"
        risk = "Medium"
        reason = "Public real-world retail-style data can support validation practice but not enterprise integration."
        action = "Run proper temporal backtesting and compare against baselines before stronger claims."
    else:
        decision = "REVIEW"
        risk = "Medium"
        reason = "Approved real/pilot data may support pilot evidence, but production claims require full validation and governance."
        action = "Document train/test split, leakage checks, baseline comparison, segment error, and model approval."

    return pd.DataFrame([{
        "agent_name": "Model Validation Agent",
        "review_area": "model comparison, baseline, temporal validation, overclaiming risk",
        "input_table": "dashboard_model_comparison.csv",
        "decision": decision,
        "risk_level": risk,
        "reason": reason,
        "best_model_under_current_output": best_model,
        "best_mae_under_current_output": best_mae,
        "required_human_action": action,
        "claim_boundary": "Does not prove deep learning superiority unless validated on approved real temporal data.",
        "source_type": source_type,
    }])


def agent_decision_risk_review(inventory: pd.DataFrame, drift: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """
    Deterministic Decision Risk Agent.
    Reviews whether recommendations should be treated as safe for action.
    """
    high_priority = 0
    if not inventory.empty and "priority" in inventory.columns:
        high_priority = int((inventory["priority"].astype(str).str.lower() == "high").sum())

    watch_drift = 0
    if not drift.empty and "status" in drift.columns:
        watch_drift = int((drift["status"].astype(str).str.lower() == "watch").sum())

    if high_priority > 0 or watch_drift > 0:
        decision = "HUMAN_REVIEW_REQUIRED"
        risk = "High" if high_priority > 0 else "Medium"
        reason = f"{high_priority} high-priority inventory rows and {watch_drift} drift-watch rows detected."
        action = "Require analyst/business manager review before inventory, pricing, promotion, or site decisions are acted on."
    else:
        decision = "PASS_WITH_CAUTION"
        risk = "Low"
        reason = "No high-priority recommendation or drift-watch rows detected."
        action = "Continue monitoring; do not auto-execute in demo/pilot mode."

    return pd.DataFrame([{
        "agent_name": "Decision Risk Agent",
        "review_area": "inventory recommendations and drift/watch status",
        "input_table": "inventory_recommendations.csv; drift_monitor.csv",
        "decision": decision,
        "risk_level": risk,
        "reason": reason,
        "required_human_action": action,
        "claim_boundary": "Decision-support review only; not autonomous business execution.",
        "source_type": source_type,
    }])


def agent_evidence_boundary_review(source_type: str, fabric_executed: bool = False, production_controls_confirmed: bool = False) -> pd.DataFrame:
    """
    Deterministic Evidence Boundary Agent.
    Prevents unsupported claims.
    """
    rows = []

    if source_type == "synthetic":
        rows.append({
            "agent_name": "Evidence Boundary Agent",
            "claim_area": "real enterprise integration",
            "claim_status": "BLOCKED",
            "reason": "Synthetic data cannot prove real enterprise integration.",
            "allowed_wording": "Enterprise-like synthetic retail data simulation.",
            "blocked_wording": "Integrated with real enterprise retail systems.",
        })
    elif source_type == "walmart-public":
        rows.append({
            "agent_name": "Evidence Boundary Agent",
            "claim_area": "real enterprise integration",
            "claim_status": "BLOCKED",
            "reason": "Public dataset use does not prove enterprise source-system integration.",
            "allowed_wording": "Processed public real-world retail-style data.",
            "blocked_wording": "Connected to enterprise POS/inventory systems.",
        })
    else:
        rows.append({
            "agent_name": "Evidence Boundary Agent",
            "claim_area": "real enterprise integration",
            "claim_status": "CONDITIONAL",
            "reason": "Approved real/pilot CSVs may support pilot integration only if source authorization and schema documentation exist.",
            "allowed_wording": "Processed approved real/pilot retail data.",
            "blocked_wording": "Production enterprise integration without governed live feeds.",
        })

    rows.append({
        "agent_name": "Evidence Boundary Agent",
        "claim_area": "live Microsoft Fabric execution",
        "claim_status": "SUPPORTED" if fabric_executed else "BLOCKED",
        "reason": "Fabric execution requires actual notebook/pipeline run history inside Microsoft Fabric.",
        "allowed_wording": "Fabric-ready upload bundle generated." if not fabric_executed else "Executed inside Microsoft Fabric with run-history evidence.",
        "blocked_wording": "Live Microsoft Fabric execution without Fabric run evidence.",
    })

    rows.append({
        "agent_name": "Evidence Boundary Agent",
        "claim_area": "production deployment",
        "claim_status": "SUPPORTED" if production_controls_confirmed else "BLOCKED",
        "reason": "Production requires governed data, security, CI/CD, monitoring, operational users, and incident/rollback controls.",
        "allowed_wording": "Production-aligned demo/pilot architecture." if not production_controls_confirmed else "Production deployment with governed controls.",
        "blocked_wording": "Production-grade system without production controls.",
    })

    out = pd.DataFrame(rows)
    out["source_type"] = source_type
    out["claim_boundary"] = "Rule-based evidence review; prevents overclaiming."
    return out


def agent_human_approval_queue(inventory: pd.DataFrame, drift: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """
    Deterministic Human Approval Agent.
    Creates an approval queue; it does not approve or execute actions.
    """
    rows = []
    if not inventory.empty:
        sample = inventory.copy()
        if "human_review_required" in sample.columns:
            sample = sample[sample["human_review_required"].fillna(0).astype(int) == 1]
        elif "priority" in sample.columns:
            sample = sample[sample["priority"].astype(str).str.lower().isin(["high", "medium"])]
        for i, (_, r) in enumerate(sample.head(50).iterrows(), start=1):
            rows.append({
                "queue_id": f"APPROVAL_{i:04d}",
                "agent_name": "Human Approval Agent",
                "approval_type": "Inventory recommendation review",
                "store_id": r.get("store_id", ""),
                "sku_id": r.get("sku_id", ""),
                "priority": r.get("priority", ""),
                "recommended_action": r.get("recommended_action", ""),
                "status": "Pending human approval",
                "owner": "Business Manager",
                "auto_execute": 0,
                "claim_boundary": "Queue only; not integrated with enterprise workflow tools unless separately implemented.",
                "source_type": source_type,
            })

    if not drift.empty and "status" in drift.columns:
        drift_watch = drift[drift["status"].astype(str).str.lower() == "watch"]
        for j, (_, r) in enumerate(drift_watch.head(20).iterrows(), start=len(rows) + 1):
            rows.append({
                "queue_id": f"APPROVAL_{j:04d}",
                "agent_name": "Human Approval Agent",
                "approval_type": "Drift/watch review",
                "store_id": "",
                "sku_id": "",
                "priority": "Medium",
                "recommended_action": f"Review drift feature: {r.get('feature', '')}",
                "status": "Pending analyst review",
                "owner": "Analyst",
                "auto_execute": 0,
                "claim_boundary": "Queue only; not integrated with enterprise workflow tools unless separately implemented.",
                "source_type": source_type,
            })

    if not rows:
        rows.append({
            "queue_id": "APPROVAL_0000",
            "agent_name": "Human Approval Agent",
            "approval_type": "No action",
            "store_id": "",
            "sku_id": "",
            "priority": "Low",
            "recommended_action": "No approval item generated",
            "status": "No pending item",
            "owner": "",
            "auto_execute": 0,
            "claim_boundary": "No action generated from current outputs.",
            "source_type": source_type,
        })

    return pd.DataFrame(rows)


def agent_executive_narrative(gold: Dict[str, pd.DataFrame], source_type: str) -> pd.DataFrame:
    """
    Deterministic Executive Narrative Agent.
    Converts outputs into cautious, decision-ready summary text.
    """
    exec_df = gold.get("dashboard_executive_summary", pd.DataFrame())
    model_df = gold.get("dashboard_model_comparison", pd.DataFrame())
    drift_df = gold.get("drift_monitor", pd.DataFrame())
    inv_df = gold.get("inventory_recommendations", pd.DataFrame())

    best_mae = None
    if not model_df.empty and "test_mae" in model_df.columns:
        best_mae = float(pd.to_numeric(model_df["test_mae"], errors="coerce").min())

    drift_watch = int((drift_df["status"].astype(str).str.lower() == "watch").sum()) if not drift_df.empty and "status" in drift_df.columns else 0
    high_inv = int((inv_df["priority"].astype(str).str.lower() == "high").sum()) if not inv_df.empty and "priority" in inv_df.columns else 0

    if source_type == "synthetic":
        opening = "This is an enterprise-like synthetic portfolio simulation, not real enterprise integration."
    elif source_type == "walmart-public":
        opening = "This uses public retail-style data and supports validation practice, not enterprise source-system integration."
    else:
        opening = "This uses approved real/pilot-style inputs if source authorization is documented; production is still not confirmed."

    narrative = (
        f"{opening} The current outputs include model comparison"
        f"{' with best MAE ' + str(best_mae) if best_mae is not None else ''}, "
        f"{drift_watch} drift-watch feature(s), and {high_inv} high-priority inventory review item(s). "
        "Business actions should remain human-approved until real data validation, thresholds, workflow integration, and governance are confirmed."
    )

    return pd.DataFrame([{
        "agent_name": "Executive Narrative Agent",
        "narrative_type": "cautious executive summary",
        "narrative": narrative,
        "recommended_external_wording": "Live demo/pilot-ready retail decision-support architecture with agent-style governance checks.",
        "blocked_external_wording": "Autonomous production retail AI system.",
        "source_type": source_type,
        "claim_boundary": "Narrative is generated from available tables and does not create new evidence.",
    }])


def build_agent_governance_outputs(gold: Dict[str, pd.DataFrame], quality: pd.DataFrame, source_type: str) -> Dict[str, pd.DataFrame]:
    model = gold.get("dashboard_model_comparison", pd.DataFrame())
    inventory = gold.get("inventory_recommendations", pd.DataFrame())
    drift = gold.get("drift_monitor", pd.DataFrame())

    dq = agent_data_quality_review(quality, source_type)
    mv = agent_model_validation_review(model, source_type)
    dr = agent_decision_risk_review(inventory, drift, source_type)
    eb = agent_evidence_boundary_review(source_type, fabric_executed=False, production_controls_confirmed=False)
    hq = agent_human_approval_queue(inventory, drift, source_type)
    en = agent_executive_narrative(gold, source_type)

    governance_report = pd.concat([
        dq.rename(columns={"review_area": "area"}),
        mv.rename(columns={"review_area": "area"}),
        dr.rename(columns={"review_area": "area"}),
    ], ignore_index=True, sort=False)

    return {
        "agent_governance_report": governance_report,
        "agent_data_quality_review": dq,
        "agent_model_validation_review": mv,
        "agent_decision_risk_review": dr,
        "agent_evidence_boundary_review": eb,
        "agent_human_approval_queue": hq,
        "agent_executive_narrative": en,
    }




# ---------------------------------------------------------------------
# Real deterministic agent runtime
# ---------------------------------------------------------------------

class AgentRuntimeContext:
    """
    Runtime context for real deterministic agents.

    This is a real agent runtime in the software-architecture sense:
    observe -> plan -> use tools -> update memory -> produce decision -> write trace.

    It is intentionally deterministic by default. It does not use a black-box LLM
    unless a separate LLM layer is added later with credentials, guardrails, and
    audit logging. This design avoids hallucination and unsupported claims.
    """

    def __init__(self, source_type: str, gold: Dict[str, pd.DataFrame], quality: pd.DataFrame):
        self.source_type = source_type
        self.gold = gold
        self.quality = quality
        self.memory: List[Dict[str, Any]] = []
        self.trace: List[Dict[str, Any]] = []
        self.action_log: List[Dict[str, Any]] = []
        self.final_decisions: List[Dict[str, Any]] = []

    def remember(self, agent_name: str, key: str, value: Any, evidence: str, confidence: str) -> None:
        self.memory.append({
            "timestamp_utc": utc_now(),
            "agent_name": agent_name,
            "memory_key": key,
            "memory_value": value,
            "evidence": evidence,
            "confidence": confidence,
        })

    def log_trace(self, agent_name: str, step: str, observation: str, planned_action: str, tool_used: str, result: str) -> None:
        self.trace.append({
            "timestamp_utc": utc_now(),
            "agent_name": agent_name,
            "step": step,
            "observation": observation,
            "planned_action": planned_action,
            "tool_used": tool_used,
            "result": result,
        })

    def log_action(self, agent_name: str, action_type: str, action: str, status: str, evidence: str, human_approval_required: int) -> None:
        self.action_log.append({
            "timestamp_utc": utc_now(),
            "agent_name": agent_name,
            "action_type": action_type,
            "action": action,
            "status": status,
            "evidence": evidence,
            "human_approval_required": human_approval_required,
        })

    def final_decision(self, agent_name: str, decision: str, risk_level: str, rationale: str, allowed_claim: str, blocked_claim: str) -> None:
        self.final_decisions.append({
            "timestamp_utc": utc_now(),
            "agent_name": agent_name,
            "decision": decision,
            "risk_level": risk_level,
            "rationale": rationale,
            "allowed_claim": allowed_claim,
            "blocked_claim": blocked_claim,
        })


class AgentToolRegistry:
    """
    Deterministic tool registry.

    Agents call these tools rather than directly manipulating business decisions.
    Each tool is auditable and returns evidence-derived values only.
    """

    def __init__(self, ctx: AgentRuntimeContext):
        self.ctx = ctx

    def count_quality_failures(self) -> Dict[str, Any]:
        q = self.ctx.quality
        if q.empty or "status" not in q.columns:
            return {"fail_count": 0, "warning_count": 0, "status": "NO_QUALITY_TABLE"}
        return {
            "fail_count": int((q["status"] == "FAIL").sum()),
            "warning_count": int((q["status"] == "WARNING").sum()),
            "status": "OK",
        }

    def inspect_model_evidence(self) -> Dict[str, Any]:
        model = self.ctx.gold.get("dashboard_model_comparison", pd.DataFrame())
        if model.empty:
            return {"model_rows": 0, "best_model": "Not confirmed", "best_mae": None, "status": "NO_MODEL_TABLE"}
        best = model.sort_values("test_mae").iloc[0] if "test_mae" in model.columns else model.iloc[0]
        return {
            "model_rows": int(len(model)),
            "best_model": str(best.get("model_name", "Not confirmed")),
            "best_mae": float(best.get("test_mae")) if "test_mae" in model.columns and pd.notna(best.get("test_mae")) else None,
            "source_type": self.ctx.source_type,
            "status": "OK",
        }

    def inspect_drift(self) -> Dict[str, Any]:
        drift = self.ctx.gold.get("drift_monitor", pd.DataFrame())
        if drift.empty:
            return {"drift_rows": 0, "watch_rows": 0, "status": "NO_DRIFT_TABLE"}
        watch_rows = int((drift["status"].astype(str).str.lower() == "watch").sum()) if "status" in drift.columns else 0
        max_ks = float(pd.to_numeric(drift.get("ks_stat", pd.Series(dtype=float)), errors="coerce").max()) if "ks_stat" in drift.columns else None
        return {"drift_rows": int(len(drift)), "watch_rows": watch_rows, "max_ks_stat": max_ks, "status": "OK"}

    def inspect_inventory_risk(self) -> Dict[str, Any]:
        inv = self.ctx.gold.get("inventory_recommendations", pd.DataFrame())
        if inv.empty:
            return {"recommendation_rows": 0, "high_priority_rows": 0, "review_required_rows": 0, "status": "NO_INVENTORY_TABLE"}
        high = int((inv["priority"].astype(str).str.lower() == "high").sum()) if "priority" in inv.columns else 0
        review = int(pd.to_numeric(inv.get("human_review_required", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "human_review_required" in inv.columns else high
        return {"recommendation_rows": int(len(inv)), "high_priority_rows": high, "review_required_rows": review, "status": "OK"}

    def inspect_evidence_boundary(self) -> Dict[str, Any]:
        source_type = self.ctx.source_type
        return {
            "source_type": source_type,
            "can_claim_real_enterprise_integration": source_type == "generic-real",
            "can_claim_live_fabric_execution": False,
            "can_claim_production_deployment": False,
            "allowed_source_claim": (
                "enterprise-like synthetic simulation" if source_type == "synthetic"
                else "public real-world retail-style data processing" if source_type == "walmart-public"
                else "approved real/pilot retail data processing, if authorization is documented"
            ),
            "status": "OK",
        }


class BaseRealAgent:
    """
    Base class for real deterministic agents.
    Each agent has a goal, observes the context, calls tools, writes memory, and records a final decision.
    """

    def __init__(self, name: str, goal: str):
        self.name = name
        self.goal = goal

    def run(self, ctx: AgentRuntimeContext, tools: AgentToolRegistry) -> None:
        raise NotImplementedError


class DataQualityRealAgent(BaseRealAgent):
    def __init__(self):
        super().__init__("Real Data Quality Agent", "Block or review downstream use when data validation fails.")

    def run(self, ctx: AgentRuntimeContext, tools: AgentToolRegistry) -> None:
        ctx.log_trace(self.name, "observe", "Reading data_quality_audit.csv", "Count failures and warnings", "count_quality_failures", "started")
        result = tools.count_quality_failures()
        decision = "PASS"
        risk = "Low"
        rationale = "No blocking data-quality issue detected."
        if result["fail_count"] > 0:
            decision = "BLOCK_MODEL_OUTPUTS"
            risk = "High"
            rationale = f"{result['fail_count']} validation failure(s) detected."
        elif result["warning_count"] > 0:
            decision = "REVIEW_BEFORE_USE"
            risk = "Medium"
            rationale = f"{result['warning_count']} validation warning(s) detected."
        ctx.remember(self.name, "data_quality_decision", decision, str(result), risk)
        ctx.log_action(self.name, "governance", decision, "completed", str(result), 1 if decision != "PASS" else 0)
        ctx.final_decision(
            self.name, decision, risk, rationale,
            "Use outputs subject to data-quality status.",
            "Do not claim reliable decision outputs if validation failures exist."
        )
        ctx.log_trace(self.name, "decide", rationale, decision, "memory+final_decision", "completed")


class ModelValidationRealAgent(BaseRealAgent):
    def __init__(self):
        super().__init__("Real Model Validation Agent", "Prevent overclaiming model performance and deep-learning value.")

    def run(self, ctx: AgentRuntimeContext, tools: AgentToolRegistry) -> None:
        ctx.log_trace(self.name, "observe", "Reading model comparison output", "Inspect best model and evidence source", "inspect_model_evidence", "started")
        result = tools.inspect_model_evidence()
        source_type = result.get("source_type", ctx.source_type)
        if source_type == "synthetic":
            decision = "DEMO_ONLY_MODEL_EVIDENCE"
            risk = "Medium"
            rationale = "Synthetic model metrics do not prove real enterprise performance."
        elif source_type == "walmart-public":
            decision = "PUBLIC_DATA_VALIDATION_CANDIDATE"
            risk = "Medium"
            rationale = "Public data can support validation practice but not enterprise source integration."
        else:
            decision = "PILOT_MODEL_REVIEW_REQUIRED"
            risk = "Medium"
            rationale = "Approved real/pilot data still requires temporal validation, leakage checks, and baseline comparison."
        ctx.remember(self.name, "model_evidence", result, "dashboard_model_comparison.csv", risk)
        ctx.log_action(self.name, "model_governance", decision, "completed", str(result), 1)
        ctx.final_decision(
            self.name, decision, risk, rationale,
            "Report model metrics as demo/pilot evidence with validation caveats.",
            "Do not claim deep learning superiority without real temporal validation."
        )
        ctx.log_trace(self.name, "decide", rationale, decision, "memory+final_decision", "completed")


class DecisionRiskRealAgent(BaseRealAgent):
    def __init__(self):
        super().__init__("Real Decision Risk Agent", "Check whether recommendations could change business decisions and require approval.")

    def run(self, ctx: AgentRuntimeContext, tools: AgentToolRegistry) -> None:
        drift = tools.inspect_drift()
        inventory = tools.inspect_inventory_risk()
        ctx.log_trace(self.name, "observe", "Reading drift and inventory outputs", "Inspect risk signals", "inspect_drift+inspect_inventory_risk", "started")
        if inventory["high_priority_rows"] > 0 or drift["watch_rows"] > 0:
            decision = "HUMAN_APPROVAL_REQUIRED"
            risk = "High" if inventory["high_priority_rows"] > 0 else "Medium"
            rationale = f"{inventory['high_priority_rows']} high-priority inventory row(s); {drift['watch_rows']} drift-watch row(s)."
        else:
            decision = "MONITOR_ONLY"
            risk = "Low"
            rationale = "No high-priority inventory or drift-watch item detected."
        evidence = {"drift": drift, "inventory": inventory}
        ctx.remember(self.name, "decision_risk", evidence, "drift_monitor.csv; inventory_recommendations.csv", risk)
        ctx.log_action(self.name, "decision_review", decision, "completed", str(evidence), 1 if decision == "HUMAN_APPROVAL_REQUIRED" else 0)
        ctx.final_decision(
            self.name, decision, risk, rationale,
            "Use recommendations as decision support with approval.",
            "Do not auto-execute inventory, pricing, promotion, or site decisions."
        )
        ctx.log_trace(self.name, "decide", rationale, decision, "memory+final_decision", "completed")


class EvidenceBoundaryRealAgent(BaseRealAgent):
    def __init__(self):
        super().__init__("Real Evidence Boundary Agent", "Block unsupported claims about enterprise, Fabric, or production maturity.")

    def run(self, ctx: AgentRuntimeContext, tools: AgentToolRegistry) -> None:
        result = tools.inspect_evidence_boundary()
        ctx.log_trace(self.name, "observe", "Checking source type and execution evidence", "Inspect claim boundaries", "inspect_evidence_boundary", "started")
        if not result["can_claim_real_enterprise_integration"]:
            decision = "BLOCK_ENTERPRISE_INTEGRATION_CLAIM"
            risk = "High"
            rationale = f"Source type is {result['source_type']}; real enterprise integration is not proven."
        else:
            decision = "CONDITIONAL_REAL_DATA_CLAIM"
            risk = "Medium"
            rationale = "Generic-real mode may support approved pilot claim only if source authorization is documented."
        ctx.remember(self.name, "evidence_boundary", result, "run_manifest.json; CLAIM_BOUNDARY.md", risk)
        ctx.log_action(self.name, "claim_governance", decision, "completed", str(result), 1)
        ctx.final_decision(
            self.name, decision, risk, rationale,
            result["allowed_source_claim"],
            "Production deployment or live Fabric execution without actual external evidence."
        )
        ctx.log_trace(self.name, "decide", rationale, decision, "memory+final_decision", "completed")


class ExecutiveNarrativeRealAgent(BaseRealAgent):
    def __init__(self):
        super().__init__("Real Executive Narrative Agent", "Produce a cautious executive narrative from evidence only.")

    def run(self, ctx: AgentRuntimeContext, tools: AgentToolRegistry) -> None:
        model = tools.inspect_model_evidence()
        drift = tools.inspect_drift()
        inventory = tools.inspect_inventory_risk()
        boundary = tools.inspect_evidence_boundary()
        ctx.log_trace(self.name, "observe", "Collecting summaries from tool registry", "Write bounded narrative", "multiple_tools", "started")
        narrative = (
            f"The current system evidence is {boundary['allowed_source_claim']}. "
            f"It contains {model.get('model_rows', 0)} model row(s), "
            f"best reported MAE {model.get('best_mae')}, "
            f"{drift.get('watch_rows', 0)} drift-watch feature(s), and "
            f"{inventory.get('review_required_rows', 0)} approval-required recommendation row(s). "
            "Use this as decision-support evidence only; human approval and real validation are required before operational action."
        )
        ctx.remember(self.name, "executive_narrative", narrative, "tool outputs", "Medium")
        ctx.log_action(self.name, "narrative_generation", "WRITE_EXECUTIVE_NARRATIVE", "completed", narrative, 0)
        ctx.final_decision(
            self.name, "NARRATIVE_CREATED", "Low", narrative,
            "Live demo/pilot architecture with real-agent governance checks.",
            "Autonomous production retail AI system."
        )
        ctx.log_trace(self.name, "decide", narrative, "NARRATIVE_CREATED", "memory+final_decision", "completed")


def run_real_agent_orchestration(gold: Dict[str, pd.DataFrame], quality: pd.DataFrame, source_type: str) -> Dict[str, pd.DataFrame]:
    """
    Run the real deterministic agent orchestration.

    This is a genuine agent runtime because each agent:
    - observes the state
    - selects tool calls
    - records memory
    - writes action logs
    - produces final decisions
    - preserves audit traces

    It is not an LLM-autonomous agent and does not execute business actions.
    """
    ctx = AgentRuntimeContext(source_type=source_type, gold=gold, quality=quality)
    tools = AgentToolRegistry(ctx)

    agents: List[BaseRealAgent] = [
        DataQualityRealAgent(),
        ModelValidationRealAgent(),
        DecisionRiskRealAgent(),
        EvidenceBoundaryRealAgent(),
        ExecutiveNarrativeRealAgent(),
    ]

    for agent in agents:
        agent.run(ctx, tools)

    return {
        "real_agent_trace": pd.DataFrame(ctx.trace),
        "real_agent_memory": pd.DataFrame(ctx.memory),
        "real_agent_action_log": pd.DataFrame(ctx.action_log),
        "real_agent_final_decisions": pd.DataFrame(ctx.final_decisions),
    }




# ---------------------------------------------------------------------
# Three-tier agent memory layer
# ---------------------------------------------------------------------

def build_three_tier_agent_memory(
    source_type: str,
    gold: Dict[str, pd.DataFrame],
    real_agent_outputs: Dict[str, pd.DataFrame],
    quality: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Three-tier memory layer inspired by the public Hermes-style memory concept:
    - Core memory: compact, always-on project facts and claim boundaries.
    - Episodic memory: run-specific observations from traces, actions, and decisions.
    - Procedural memory: reusable operating rules/skills for this retail decision-support system.

    Brutally honest boundary:
    This is a local deterministic memory design. It is not the Hermes Agent implementation,
    not a vector database, not a production memory service, and not autonomous long-term
    enterprise memory. It is a reproducible memory artifact for portfolio/pilot governance.
    """

    model = gold.get("dashboard_model_comparison", pd.DataFrame())
    drift = gold.get("drift_monitor", pd.DataFrame())
    inventory = gold.get("inventory_recommendations", pd.DataFrame())

    best_mae = None
    best_model = "Not confirmed"
    if not model.empty and "test_mae" in model.columns:
        model_sorted = model.sort_values("test_mae")
        best_mae = float(model_sorted.iloc[0]["test_mae"])
        best_model = str(model_sorted.iloc[0].get("model_name", "Not confirmed"))

    drift_watch = int((drift["status"].astype(str).str.lower() == "watch").sum()) if not drift.empty and "status" in drift.columns else 0
    approval_rows = int(pd.to_numeric(inventory.get("human_review_required", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not inventory.empty and "human_review_required" in inventory.columns else 0
    quality_failures = int((quality["status"] == "FAIL").sum()) if not quality.empty and "status" in quality.columns else 0
    quality_warnings = int((quality["status"] == "WARNING").sum()) if not quality.empty and "status" in quality.columns else 0

    if source_type == "synthetic":
        evidence_level = "Enterprise-like synthetic simulation"
        allowed_claim = "Expanded a live retail decision-support demo with enterprise-like synthetic schemas, governance checks, and agent memory artifacts."
        blocked_claim = "Real enterprise integration or production deployment."
    elif source_type == "walmart-public":
        evidence_level = "Public retail-style dataset processing"
        allowed_claim = "Processed public retail-style data into dashboard-ready and agent-governed outputs."
        blocked_claim = "Enterprise source-system integration."
    else:
        evidence_level = "Approved real/pilot data candidate"
        allowed_claim = "Processed approved real/pilot retail data if source authorization is documented."
        blocked_claim = "Production deployment without production controls."

    core_rows = [
        {
            "memory_tier": "core",
            "memory_key": "project_identity",
            "memory_value": "Retail Decision Support System",
            "purpose": "Always-on identity for the agent runtime.",
            "refresh_policy": "Update only when the project scope changes.",
            "claim_boundary": "Does not prove production by itself.",
        },
        {
            "memory_tier": "core",
            "memory_key": "source_type",
            "memory_value": source_type,
            "purpose": "Controls allowable external claims.",
            "refresh_policy": "Update each run.",
            "claim_boundary": "Source type determines whether real-data claims are allowed.",
        },
        {
            "memory_tier": "core",
            "memory_key": "evidence_level",
            "memory_value": evidence_level,
            "purpose": "Prevents overclaiming maturity.",
            "refresh_policy": "Update each run.",
            "claim_boundary": blocked_claim,
        },
        {
            "memory_tier": "core",
            "memory_key": "allowed_claim",
            "memory_value": allowed_claim,
            "purpose": "Safe wording for resume, LinkedIn, and project documentation.",
            "refresh_policy": "Review before external use.",
            "claim_boundary": blocked_claim,
        },
        {
            "memory_tier": "core",
            "memory_key": "blocked_claim",
            "memory_value": blocked_claim,
            "purpose": "Words the system should not use without actual evidence.",
            "refresh_policy": "Review when evidence changes.",
            "claim_boundary": "Prevents hallucinated enterprise claims.",
        },
    ]
    core = pd.DataFrame(core_rows)

    episodic_rows = [
        {
            "memory_tier": "episodic",
            "event_type": "run_summary",
            "event_key": "best_model",
            "event_value": best_model,
            "evidence_source": "dashboard_model_comparison.csv",
            "confidence": "Medium",
            "forget_or_archive_policy": "Archive after newer validated run exists.",
        },
        {
            "memory_tier": "episodic",
            "event_type": "run_summary",
            "event_key": "best_mae",
            "event_value": best_mae,
            "evidence_source": "dashboard_model_comparison.csv",
            "confidence": "Medium",
            "forget_or_archive_policy": "Archive after newer validated run exists.",
        },
        {
            "memory_tier": "episodic",
            "event_type": "monitoring",
            "event_key": "drift_watch_rows",
            "event_value": drift_watch,
            "evidence_source": "drift_monitor.csv",
            "confidence": "Medium",
            "forget_or_archive_policy": "Compare against next run.",
        },
        {
            "memory_tier": "episodic",
            "event_type": "approval",
            "event_key": "human_review_required_rows",
            "event_value": approval_rows,
            "evidence_source": "inventory_recommendations.csv",
            "confidence": "Medium",
            "forget_or_archive_policy": "Resolve after human review.",
        },
        {
            "memory_tier": "episodic",
            "event_type": "quality",
            "event_key": "quality_failures",
            "event_value": quality_failures,
            "evidence_source": "data_quality_audit.csv",
            "confidence": "High",
            "forget_or_archive_policy": "Must be resolved before stronger claims.",
        },
        {
            "memory_tier": "episodic",
            "event_type": "quality",
            "event_key": "quality_warnings",
            "event_value": quality_warnings,
            "evidence_source": "data_quality_audit.csv",
            "confidence": "High",
            "forget_or_archive_policy": "Review before operational use.",
        },
    ]

    trace = real_agent_outputs.get("real_agent_trace", pd.DataFrame())
    if not trace.empty:
        for i, (_, row) in enumerate(trace.tail(25).iterrows(), start=1):
            episodic_rows.append({
                "memory_tier": "episodic",
                "event_type": "agent_trace",
                "event_key": f"trace_{i:03d}",
                "event_value": f"{row.get('agent_name', '')}: {row.get('step', '')} -> {row.get('result', '')}",
                "evidence_source": "real_agent_trace.csv",
                "confidence": "High",
                "forget_or_archive_policy": "Archive as run trace.",
            })

    episodic = pd.DataFrame(episodic_rows)

    procedural_rows = [
        {
            "memory_tier": "procedural",
            "skill_name": "claim_boundary_check",
            "trigger_condition": "Before writing resume, LinkedIn, website, or project claims.",
            "procedure": "Check source_type, Fabric execution evidence, and production controls before allowing claims.",
            "tool_or_evidence": "agent_evidence_boundary_review.csv; run_manifest.json",
            "failure_mode_prevented": "Overclaiming real enterprise integration or production deployment.",
        },
        {
            "memory_tier": "procedural",
            "skill_name": "data_quality_gate",
            "trigger_condition": "Before trusting model or dashboard outputs.",
            "procedure": "If any FAIL exists in data_quality_audit.csv, block model outputs. If WARNING exists, require review.",
            "tool_or_evidence": "data_quality_audit.csv",
            "failure_mode_prevented": "Using bad data for business decisions.",
        },
        {
            "memory_tier": "procedural",
            "skill_name": "model_validation_gate",
            "trigger_condition": "Before claiming model superiority.",
            "procedure": "Require temporal validation, leakage checks, baseline comparison, and segment-level error review.",
            "tool_or_evidence": "dashboard_model_comparison.csv; validation documentation",
            "failure_mode_prevented": "Mistaking demo metrics for real performance.",
        },
        {
            "memory_tier": "procedural",
            "skill_name": "human_approval_gate",
            "trigger_condition": "Before inventory, pricing, promotion, or site action.",
            "procedure": "Route high-priority or drift-watch items to human approval queue; do not auto-execute.",
            "tool_or_evidence": "agent_human_approval_queue.csv",
            "failure_mode_prevented": "Autonomous action on unvalidated recommendations.",
        },
        {
            "memory_tier": "procedural",
            "skill_name": "fabric_evidence_gate",
            "trigger_condition": "Before claiming live Microsoft Fabric execution.",
            "procedure": "Require Fabric notebook success, Lakehouse tables, execution audit table, and pipeline/run history screenshots.",
            "tool_or_evidence": "fabric_retail_execution_audit table in Fabric; screenshots",
            "failure_mode_prevented": "Claiming Fabric execution from a local bundle only.",
        },
    ]
    procedural = pd.DataFrame(procedural_rows)

    index = pd.DataFrame([
        {
            "memory_tier": "core",
            "storage_artifact": "agent_memory_core.csv",
            "speed": "fast/always loaded",
            "purpose": "Stable project identity, source type, and claim boundaries.",
            "max_growth_policy": "Keep compact; do not store raw data here.",
        },
        {
            "memory_tier": "episodic",
            "storage_artifact": "agent_memory_episodic.csv",
            "speed": "medium/run recall",
            "purpose": "Run-specific observations, agent traces, quality events, and monitoring events.",
            "max_growth_policy": "Archive older runs; keep latest summaries.",
        },
        {
            "memory_tier": "procedural",
            "storage_artifact": "agent_memory_procedural.csv",
            "speed": "fast/reusable skills",
            "purpose": "Reusable governance skills and decision rules.",
            "max_growth_policy": "Update only when the system process changes.",
        },
    ])

    return {
        "agent_memory_core": core,
        "agent_memory_episodic": episodic,
        "agent_memory_procedural": procedural,
        "agent_memory_index": index,
    }




# ---------------------------------------------------------------------
# Schema-constrained typed agent memory layer
# ---------------------------------------------------------------------

class EvidenceRef(BaseModel):
    evidence_id: str = Field(..., description="Stable evidence key.")
    evidence_type: Literal["csv", "json", "jsonl", "manifest", "api", "screenshot", "none"] = Field(..., description="Evidence artifact type.")
    evidence_path: str = Field("", description="Relative or absolute evidence path.")
    source_table: str = Field("", description="Source table or file name.")
    proof_status: Literal["verified", "scaffolded", "missing", "invalid"] = Field(..., description="Proof status from available artifacts.")
    claim_boundary: str = Field(..., description="What this evidence can and cannot support.")
    valid_from_utc: str = Field(..., description="UTC time when this evidence observation was created.")


class ModelRunMemory(BaseModel):
    memory_id: str
    entity_type: Literal["ModelRun"] = "ModelRun"
    run_id: str
    model_name: str
    model_type: str
    selected_flag: bool
    mae: Optional[float] = None
    rmse: Optional[float] = None
    wmape: Optional[float] = None
    bias: Optional[float] = None
    source_type: str
    evidence_id: str
    valid_from_utc: str
    valid_to_utc: Optional[str] = None
    superseded_by: Optional[str] = None
    is_current: bool = True
    claim_boundary: str


class DriftSignalMemory(BaseModel):
    memory_id: str
    entity_type: Literal["DriftSignal"] = "DriftSignal"
    run_id: str
    feature_name: str
    drift_metric: Literal["ks", "psi", "js_divergence", "other"] = "ks"
    drift_value: Optional[float] = None
    p_value: Optional[float] = None
    status: str
    recommended_response: str
    source_type: str
    evidence_id: str
    valid_from_utc: str
    valid_to_utc: Optional[str] = None
    superseded_by: Optional[str] = None
    is_current: bool = True
    claim_boundary: str


class InventoryActionMemory(BaseModel):
    memory_id: str
    entity_type: Literal["InventoryAction"] = "InventoryAction"
    run_id: str
    action_id: str
    store_id: str = ""
    sku_id: str = ""
    recommended_action: str
    priority: str
    human_review_required: bool
    auto_execute_allowed: bool = False
    source_type: str
    evidence_id: str
    valid_from_utc: str
    valid_to_utc: Optional[str] = None
    superseded_by: Optional[str] = None
    is_current: bool = True
    claim_boundary: str


class BusinessClaimMemory(BaseModel):
    memory_id: str
    entity_type: Literal["BusinessClaim"] = "BusinessClaim"
    run_id: str
    claim_type: Literal[
        "fabric_ready",
        "fabric_live",
        "enterprise_production",
        "model_performance",
        "inventory_action",
        "real_enterprise_integration",
        "api_connector",
    ]
    claim_text: str
    allowed_status: Literal["allowed", "blocked", "conditional"]
    reason: str
    evidence_id: str
    valid_from_utc: str
    valid_to_utc: Optional[str] = None
    superseded_by: Optional[str] = None
    is_current: bool = True
    claim_boundary: str


class TemporalFactMemory(BaseModel):
    memory_id: str
    entity_type: Literal["TemporalFact"] = "TemporalFact"
    run_id: str
    fact_type: str
    subject: str
    predicate: str
    object_value: str
    previous_value: Optional[str] = None
    evidence_id: str
    valid_from_utc: str
    valid_to_utc: Optional[str] = None
    superseded_by: Optional[str] = None
    is_current: bool = True
    claim_boundary: str


class TypedMemoryEdge(BaseModel):
    edge_id: str
    source_memory_id: str
    source_entity_type: str
    edge_type: Literal["SUPPORTED_BY", "AFFECTS", "REQUIRES_REVIEW", "LIMITED_BY", "PRODUCED"]
    target_memory_id: str
    target_entity_type: str
    evidence_id: str
    valid_from_utc: str
    claim_boundary: str


def _model_dump_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(getattr(obj, "__dict__", {}))


def _append_validated(records: List[Dict[str, Any]], errors: List[Dict[str, Any]], model_cls: Any, **kwargs: Any) -> str:
    try:
        obj = model_cls(**kwargs)
        rec = _model_dump_dict(obj)
        records.append(rec)
        return str(rec.get("memory_id") or rec.get("evidence_id") or rec.get("edge_id") or "")
    except Exception as exc:
        errors.append({
            "model": getattr(model_cls, "__name__", str(model_cls)),
            "error": str(exc),
            "payload_preview": str(kwargs)[:750],
        })
        return ""


def _safe_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _safe_str(value: Any, default: str = "") -> str:
    value = _safe_value(value)
    return default if value is None else str(value)


def _safe_float(value: Any) -> Optional[float]:
    value = _safe_value(value)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool:
    value = _safe_value(value)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def _memory_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(_safe_str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(read_csv_safe(path)))
    except Exception:
        return 0


def _records_to_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _jsonl_frame(records_by_table: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for table_name, records in records_by_table.items():
        for rec in records:
            rows.append({
                "table_name": table_name,
                "json_line": json.dumps(rec, default=str, sort_keys=True),
            })
    return pd.DataFrame(rows)


def build_schema_constrained_agent_memory(
    source_type: str,
    gold: Dict[str, pd.DataFrame],
    real_agent_outputs: Dict[str, pd.DataFrame],
    quality: pd.DataFrame,
    project_root: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Build a schema-constrained typed memory layer using Pydantic when available.

    Honest boundary:
    - This creates local validated memory artifacts.
    - It does not prove Graphiti, Zep, a live graph database, live Fabric execution,
      live enterprise APIs, or enterprise-grade agent memory.
    - Deterministic dashboard tables remain the source of truth.
    """
    created_at = utc_now()
    run_id = "RUN_" + created_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    root = Path(project_root).expanduser().resolve()

    model_df = gold.get("dashboard_model_comparison", pd.DataFrame())
    drift_df = gold.get("drift_monitor", pd.DataFrame())
    inventory_df = gold.get("inventory_recommendations", pd.DataFrame())

    fabric_root = root / "artifacts" / "fabric_live"
    fabric_manifest_path = fabric_root / "evidence_manifest.json"
    workspace_inventory_path = fabric_root / "fabric_workspace_inventory.csv"
    item_inventory_path = fabric_root / "fabric_item_inventory.csv"
    screenshot_register_path = fabric_root / "fabric_screenshot_register.csv"
    notebook_run_log_path = fabric_root / "fabric_notebook_run_log.csv"
    pipeline_run_log_path = fabric_root / "fabric_pipeline_run_log.csv"

    workspace_rows = _count_csv_rows(workspace_inventory_path)
    item_rows = _count_csv_rows(item_inventory_path)
    screenshot_rows = _count_csv_rows(screenshot_register_path)
    notebook_run_rows = _count_csv_rows(notebook_run_log_path)
    pipeline_run_rows = _count_csv_rows(pipeline_run_log_path)

    live_api_evidence = workspace_rows > 0 or item_rows > 0
    live_screenshot_evidence = screenshot_rows > 0
    live_run_evidence = notebook_run_rows > 0 or pipeline_run_rows > 0
    live_fabric_claim_supported = live_api_evidence or live_screenshot_evidence or live_run_evidence

    validation_errors: List[Dict[str, Any]] = []
    evidence_records: List[Dict[str, Any]] = []
    model_records: List[Dict[str, Any]] = []
    drift_records: List[Dict[str, Any]] = []
    inventory_records: List[Dict[str, Any]] = []
    claim_records: List[Dict[str, Any]] = []
    fact_records: List[Dict[str, Any]] = []
    edge_records: List[Dict[str, Any]] = []

    def add_evidence(evidence_id: str, evidence_type: str, path: str, source_table: str, proof_status: str, boundary: str) -> str:
        return _append_validated(
            evidence_records,
            validation_errors,
            EvidenceRef,
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            evidence_path=path,
            source_table=source_table,
            proof_status=proof_status,
            claim_boundary=boundary,
            valid_from_utc=created_at,
        )

    evidence_model = add_evidence(
        "evidence_model_comparison",
        "csv",
        "outputs/dashboard_model_comparison.csv",
        "dashboard_model_comparison.csv",
        "verified" if not model_df.empty else "missing",
        "Supports demo or pilot model comparison only; production performance requires temporal validation.",
    )
    evidence_drift = add_evidence(
        "evidence_drift_monitor",
        "csv",
        "outputs/drift_monitor.csv",
        "drift_monitor.csv",
        "verified" if not drift_df.empty else "missing",
        "Supports drift/watch review only; thresholds require calibration before automated decisions.",
    )
    evidence_inventory = add_evidence(
        "evidence_inventory_recommendations",
        "csv",
        "outputs/inventory_recommendations.csv",
        "inventory_recommendations.csv",
        "verified" if not inventory_df.empty else "missing",
        "Supports human-reviewed inventory recommendations only; no auto-execution.",
    )
    evidence_quality = add_evidence(
        "evidence_data_quality",
        "csv",
        "outputs/data_quality_audit.csv",
        "data_quality_audit.csv",
        "verified" if not quality.empty else "missing",
        "Supports data-quality gating only; does not prove production data governance.",
    )
    evidence_fabric_manifest = add_evidence(
        "evidence_fabric_live_manifest",
        "manifest",
        str(fabric_manifest_path.relative_to(root)) if fabric_manifest_path.exists() else "artifacts/fabric_live/evidence_manifest.json",
        "evidence_manifest.json",
        "scaffolded" if fabric_manifest_path.exists() else "missing",
        "Supports Fabric-ready or scaffolded evidence only; live proof requires workspace/item/screenshot/run evidence.",
    )
    evidence_fabric_live = add_evidence(
        "evidence_fabric_live_proof",
        "api" if live_api_evidence else ("screenshot" if live_screenshot_evidence else "none"),
        "artifacts/fabric_live",
        "fabric_workspace_inventory.csv; fabric_item_inventory.csv; fabric_screenshot_register.csv; fabric_notebook_run_log.csv; fabric_pipeline_run_log.csv",
        "verified" if live_fabric_claim_supported else "missing",
        "Live Fabric claims are allowed only when API, screenshot, notebook, or pipeline evidence exists.",
    )

    # Model run memories
    if not model_df.empty:
        for idx, row in model_df.head(100).iterrows():
            mem_id = _memory_id("MODEL", run_id, row.get("model_name", idx), idx)
            _append_validated(
                model_records,
                validation_errors,
                ModelRunMemory,
                memory_id=mem_id,
                run_id=run_id,
                model_name=_safe_str(row.get("model_name", f"model_{idx}")),
                model_type=_safe_str(row.get("model_type", "unknown")),
                selected_flag=_safe_bool(row.get("selected_flag", False)),
                mae=_safe_float(row.get("test_mae", row.get("mae", None))),
                rmse=_safe_float(row.get("test_rmse", row.get("rmse", None))),
                wmape=_safe_float(row.get("test_wmape", row.get("wmape", None))),
                bias=_safe_float(row.get("bias", None)),
                source_type=source_type,
                evidence_id=evidence_model,
                valid_from_utc=created_at,
                claim_boundary="Model memory reflects current output rows only; not proof of production model performance.",
            )
            edge_records.append({
                "edge_id": _memory_id("EDGE", mem_id, "SUPPORTED_BY", evidence_model),
                "source_memory_id": mem_id,
                "source_entity_type": "ModelRun",
                "edge_type": "SUPPORTED_BY",
                "target_memory_id": evidence_model,
                "target_entity_type": "EvidenceRef",
                "evidence_id": evidence_model,
                "valid_from_utc": created_at,
                "claim_boundary": "Edge links model memory to model-comparison evidence only.",
            })

    # Drift signal memories
    if not drift_df.empty:
        for idx, row in drift_df.head(100).iterrows():
            feature = _safe_str(row.get("feature", row.get("feature_name", f"feature_{idx}")))
            mem_id = _memory_id("DRIFT", run_id, feature, idx)
            _append_validated(
                drift_records,
                validation_errors,
                DriftSignalMemory,
                memory_id=mem_id,
                run_id=run_id,
                feature_name=feature,
                drift_metric="ks" if "ks_stat" in drift_df.columns else "other",
                drift_value=_safe_float(row.get("ks_stat", row.get("drift_value", None))),
                p_value=_safe_float(row.get("p_value", None)),
                status=_safe_str(row.get("status", "unknown")),
                recommended_response=_safe_str(row.get("recommended_response", "Review before action.")),
                source_type=source_type,
                evidence_id=evidence_drift,
                valid_from_utc=created_at,
                claim_boundary="Drift memory is monitoring evidence; it does not automatically trigger business actions.",
            )
            edge_records.append({
                "edge_id": _memory_id("EDGE", mem_id, "SUPPORTED_BY", evidence_drift),
                "source_memory_id": mem_id,
                "source_entity_type": "DriftSignal",
                "edge_type": "SUPPORTED_BY",
                "target_memory_id": evidence_drift,
                "target_entity_type": "EvidenceRef",
                "evidence_id": evidence_drift,
                "valid_from_utc": created_at,
                "claim_boundary": "Edge links drift signal to drift-monitor evidence only.",
            })

    # Inventory action memories
    if not inventory_df.empty:
        for idx, row in inventory_df.head(250).iterrows():
            action_id = f"INV_ACTION_{idx:05d}"
            mem_id = _memory_id("INV", run_id, row.get("store_id", ""), row.get("sku_id", ""), row.get("recommended_action", ""), idx)
            review_required = _safe_bool(row.get("human_review_required", False))
            _append_validated(
                inventory_records,
                validation_errors,
                InventoryActionMemory,
                memory_id=mem_id,
                run_id=run_id,
                action_id=action_id,
                store_id=_safe_str(row.get("store_id", "")),
                sku_id=_safe_str(row.get("sku_id", "")),
                recommended_action=_safe_str(row.get("recommended_action", "Maintain")),
                priority=_safe_str(row.get("priority", "Normal")),
                human_review_required=review_required,
                auto_execute_allowed=False,
                source_type=source_type,
                evidence_id=evidence_inventory,
                valid_from_utc=created_at,
                claim_boundary="Inventory memory is advisory and requires human review when flagged; no automatic execution.",
            )
            edge_records.append({
                "edge_id": _memory_id("EDGE", mem_id, "SUPPORTED_BY", evidence_inventory),
                "source_memory_id": mem_id,
                "source_entity_type": "InventoryAction",
                "edge_type": "SUPPORTED_BY",
                "target_memory_id": evidence_inventory,
                "target_entity_type": "EvidenceRef",
                "evidence_id": evidence_inventory,
                "valid_from_utc": created_at,
                "claim_boundary": "Edge links inventory recommendation to inventory evidence only.",
            })
            if review_required:
                edge_records.append({
                    "edge_id": _memory_id("EDGE", mem_id, "REQUIRES_REVIEW", "human"),
                    "source_memory_id": mem_id,
                    "source_entity_type": "InventoryAction",
                    "edge_type": "REQUIRES_REVIEW",
                    "target_memory_id": "human_approval_required",
                    "target_entity_type": "GovernanceRule",
                    "evidence_id": evidence_inventory,
                    "valid_from_utc": created_at,
                    "claim_boundary": "Review edge prevents auto-execution from advisory memory.",
                })

    # Business claim memories
    if source_type == "synthetic":
        enterprise_status = "blocked"
        enterprise_reason = "Synthetic data cannot prove real enterprise source-system integration."
    elif source_type == "walmart-public":
        enterprise_status = "blocked"
        enterprise_reason = "Public retail-style data does not prove enterprise source-system integration."
    else:
        enterprise_status = "conditional"
        enterprise_reason = "Generic-real mode may support pilot wording only if source authorization and schema documentation exist."

    claims = [
        {
            "claim_type": "fabric_ready",
            "claim_text": "The system is Fabric-ready or Fabric-scaffolded through local bundle, notebook, and evidence-boundary design.",
            "allowed_status": "allowed",
            "reason": "Readiness is an architecture/scaffold claim, not live Fabric execution.",
            "evidence_id": evidence_fabric_manifest,
            "claim_boundary": "Allowed as readiness/scaffold wording only.",
        },
        {
            "claim_type": "fabric_live",
            "claim_text": "The system has live Microsoft Fabric evidence.",
            "allowed_status": "conditional" if live_fabric_claim_supported else "blocked",
            "reason": (
                f"Live evidence detected: workspace_rows={workspace_rows}, item_rows={item_rows}, screenshots={screenshot_rows}, notebook_runs={notebook_run_rows}, pipeline_runs={pipeline_run_rows}."
                if live_fabric_claim_supported else
                "No API-verified workspace/item, screenshot, notebook, or pipeline evidence detected."
            ),
            "evidence_id": evidence_fabric_live,
            "claim_boundary": "Live claims require captured external Fabric evidence.",
        },
        {
            "claim_type": "enterprise_production",
            "claim_text": "The system is enterprise-production deployed.",
            "allowed_status": "blocked",
            "reason": "Production requires governed data feeds, security/RBAC, CI/CD, monitoring, users, approvals, and operational run history.",
            "evidence_id": evidence_quality,
            "claim_boundary": "Blocked unless production controls are implemented and evidenced.",
        },
        {
            "claim_type": "model_performance",
            "claim_text": "The current best model under the generated outputs can be described as best-ranked within this run.",
            "allowed_status": "conditional",
            "reason": "Model results are valid only within current demo/pilot outputs and require real temporal validation before operational claims.",
            "evidence_id": evidence_model,
            "claim_boundary": "Do not claim production performance or deep-learning superiority without validation evidence.",
        },
        {
            "claim_type": "inventory_action",
            "claim_text": "Inventory recommendations can support human review.",
            "allowed_status": "conditional",
            "reason": "Recommendations are advisory and should not auto-execute in demo or pilot mode.",
            "evidence_id": evidence_inventory,
            "claim_boundary": "Human approval required for high-risk actions.",
        },
        {
            "claim_type": "real_enterprise_integration",
            "claim_text": "The system is integrated with real enterprise retail systems.",
            "allowed_status": enterprise_status,
            "reason": enterprise_reason,
            "evidence_id": evidence_quality,
            "claim_boundary": "Real integration requires authorized enterprise source feeds and documentation.",
        },
        {
            "claim_type": "api_connector",
            "claim_text": "The system includes an API-ready connector architecture.",
            "allowed_status": "allowed",
            "reason": "The script documents connector responsibilities and mocked/file-based calls; live calls remain blocked without credentials/endpoints.",
            "evidence_id": evidence_quality,
            "claim_boundary": "Allowed as API-ready architecture, not live enterprise API integration.",
        },
    ]
    for c in claims:
        mem_id = _memory_id("CLAIM", run_id, c["claim_type"])
        _append_validated(
            claim_records,
            validation_errors,
            BusinessClaimMemory,
            memory_id=mem_id,
            run_id=run_id,
            claim_type=c["claim_type"],
            claim_text=c["claim_text"],
            allowed_status=c["allowed_status"],
            reason=c["reason"],
            evidence_id=c["evidence_id"],
            valid_from_utc=created_at,
            claim_boundary=c["claim_boundary"],
        )
        edge_records.append({
            "edge_id": _memory_id("EDGE", mem_id, "SUPPORTED_BY", c["evidence_id"]),
            "source_memory_id": mem_id,
            "source_entity_type": "BusinessClaim",
            "edge_type": "SUPPORTED_BY" if c["allowed_status"] != "blocked" else "LIMITED_BY",
            "target_memory_id": c["evidence_id"],
            "target_entity_type": "EvidenceRef",
            "evidence_id": c["evidence_id"],
            "valid_from_utc": created_at,
            "claim_boundary": "Claim edge prevents unsupported wording from being treated as fact.",
        })

    best_model_name = "Not confirmed"
    best_mae = None
    if not model_df.empty and "test_mae" in model_df.columns:
        sorted_model = model_df.copy()
        sorted_model["_mae_numeric"] = pd.to_numeric(sorted_model["test_mae"], errors="coerce")
        sorted_model = sorted_model.sort_values("_mae_numeric")
        if not sorted_model.empty:
            best_model_name = _safe_str(sorted_model.iloc[0].get("model_name", "Not confirmed"))
            best_mae = _safe_float(sorted_model.iloc[0].get("test_mae", None))

    drift_watch_count = int((drift_df["status"].astype(str).str.lower() == "watch").sum()) if not drift_df.empty and "status" in drift_df.columns else 0
    approval_required_count = int(pd.to_numeric(inventory_df.get("human_review_required", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not inventory_df.empty and "human_review_required" in inventory_df.columns else 0
    quality_fail_count = int((quality["status"] == "FAIL").sum()) if not quality.empty and "status" in quality.columns else 0
    quality_warning_count = int((quality["status"] == "WARNING").sum()) if not quality.empty and "status" in quality.columns else 0

    facts = [
        ("source", "retail_decision_support_system", "has_source_type", source_type, evidence_quality, "Source type controls allowable claims."),
        ("model", "current_run", "best_ranked_model", best_model_name, evidence_model, "Best-ranked model is only current-run evidence."),
        ("model", "current_run", "best_reported_mae", "" if best_mae is None else str(best_mae), evidence_model, "MAE is current output evidence, not production proof."),
        ("monitoring", "current_run", "drift_watch_count", str(drift_watch_count), evidence_drift, "Drift watch count supports review, not automatic retraining."),
        ("approval", "current_run", "approval_required_count", str(approval_required_count), evidence_inventory, "Approval count supports human review queue."),
        ("quality", "current_run", "quality_fail_count", str(quality_fail_count), evidence_quality, "Quality failures block stronger claims."),
        ("quality", "current_run", "quality_warning_count", str(quality_warning_count), evidence_quality, "Quality warnings require review."),
        ("fabric", "fabric_live_demo", "api_verified_workspace_rows", str(workspace_rows), evidence_fabric_live, "Zero means no API workspace proof."),
        ("fabric", "fabric_live_demo", "api_verified_item_rows", str(item_rows), evidence_fabric_live, "Zero means no API item proof."),
        ("fabric", "fabric_live_demo", "registered_screenshot_rows", str(screenshot_rows), evidence_fabric_live, "Zero means no screenshot evidence."),
        ("production", "retail_decision_support_system", "production_claim_status", "blocked", evidence_quality, "Production is blocked unless controls and run evidence exist."),
    ]
    for fact_type, subject, predicate, object_value, evidence_id, boundary in facts:
        mem_id = _memory_id("FACT", run_id, fact_type, subject, predicate)
        _append_validated(
            fact_records,
            validation_errors,
            TemporalFactMemory,
            memory_id=mem_id,
            run_id=run_id,
            fact_type=fact_type,
            subject=subject,
            predicate=predicate,
            object_value=_safe_str(object_value),
            previous_value=None,
            evidence_id=evidence_id,
            valid_from_utc=created_at,
            claim_boundary=boundary,
        )

    # Validate typed edge records as Pydantic objects as well.
    validated_edges: List[Dict[str, Any]] = []
    for e in edge_records:
        _append_validated(validated_edges, validation_errors, TypedMemoryEdge, **e)

    records_by_table = {
        "schema_agent_memory_evidence_artifacts": evidence_records,
        "schema_agent_memory_model_runs": model_records,
        "schema_agent_memory_drift_signals": drift_records,
        "schema_agent_memory_inventory_actions": inventory_records,
        "schema_agent_memory_business_claims": claim_records,
        "schema_agent_memory_temporal_facts": fact_records,
        "schema_agent_memory_edges": validated_edges,
    }

    validation_summary = pd.DataFrame([{
        "generated_at_utc": created_at,
        "run_id": run_id,
        "pydantic_available": int(PYDANTIC_AVAILABLE),
        "memory_design": "schema_constrained_local_typed_memory",
        "graphiti_or_zep_integrated": 0,
        "live_graph_database_integrated": 0,
        "deterministic_tables_remain_source_of_truth": 1,
        "model_run_records": len(model_records),
        "drift_signal_records": len(drift_records),
        "inventory_action_records": len(inventory_records),
        "business_claim_records": len(claim_records),
        "temporal_fact_records": len(fact_records),
        "edge_records": len(validated_edges),
        "validation_error_records": len(validation_errors),
        "fabric_workspace_rows": workspace_rows,
        "fabric_item_rows": item_rows,
        "fabric_screenshot_rows": screenshot_rows,
        "fabric_notebook_run_rows": notebook_run_rows,
        "fabric_pipeline_run_rows": pipeline_run_rows,
        "fabric_live_claim_supported": int(live_fabric_claim_supported),
        "claim_boundary": "Local typed memory only; not enterprise-grade memory, not live Graphiti/Zep, not live Fabric proof unless external evidence rows exist.",
    }])

    validation_errors_df = pd.DataFrame(validation_errors) if validation_errors else pd.DataFrame(columns=["model", "error", "payload_preview"])
    jsonl_records = _jsonl_frame(records_by_table)

    answer_examples = pd.DataFrame([
        {
            "question": "Is live Fabric API proof available?",
            "deterministic_answer": "Yes" if live_api_evidence else "No",
            "evidence": f"workspace_rows={workspace_rows}; item_rows={item_rows}",
            "claim_boundary": "Do not claim live Fabric API proof when workspace and item evidence rows are zero.",
        },
        {
            "question": "Can the system auto-execute inventory decisions?",
            "deterministic_answer": "No",
            "evidence": f"approval_required_count={approval_required_count}; auto_execute_allowed=False",
            "claim_boundary": "Inventory actions are advisory and human-reviewed.",
        },
        {
            "question": "What is the current best-ranked model?",
            "deterministic_answer": best_model_name,
            "evidence": "dashboard_model_comparison.csv",
            "claim_boundary": "Best-ranked within current output only; not production performance proof.",
        },
    ])

    return {
        "schema_agent_memory_evidence_artifacts": _records_to_frame(evidence_records),
        "schema_agent_memory_model_runs": _records_to_frame(model_records),
        "schema_agent_memory_drift_signals": _records_to_frame(drift_records),
        "schema_agent_memory_inventory_actions": _records_to_frame(inventory_records),
        "schema_agent_memory_business_claims": _records_to_frame(claim_records),
        "schema_agent_memory_temporal_facts": _records_to_frame(fact_records),
        "schema_agent_memory_edges": _records_to_frame(validated_edges),
        "schema_agent_memory_validation_summary": validation_summary,
        "schema_agent_memory_validation_errors": validation_errors_df,
        "schema_agent_memory_jsonl_records": jsonl_records,
        "schema_agent_memory_answer_examples": answer_examples,
    }



# ---------------------------------------------------------------------
# Enterprise API connector layer
# ---------------------------------------------------------------------

def build_api_connector_registry(source_type: str) -> pd.DataFrame:
    """
    Enterprise API connector registry.

    Brutally honest boundary:
    This defines the API architecture and mock/live connector responsibilities.
    It does not call real enterprise APIs because no authorized endpoints,
    credentials, schemas, or contracts are provided to this script.

    Safe claim:
    "Implemented an API-ready connector registry and integration plan."

    Unsafe claim:
    "Connected to live enterprise POS/ERP/inventory APIs."
    """
    rows = [
        {
            "connector_name": "POS API Connector",
            "system_area": "sales_transactions",
            "example_source_systems": "POS, commerce platform, retail data warehouse",
            "expected_payload": "transaction_id, transaction_date, store_id, sku_id, units_sold, net_sales, discounts, returns",
            "integration_pattern": "REST API, SQL connector, SFTP batch export, webhook, or Fabric Data Factory connector",
            "implemented_mode": "mock_or_file_based",
            "live_mode_status": "blocked_until_authorized_endpoint_and_credentials_are_provided",
            "target_raw_table": "pos_transaction_lines.csv",
            "risk_if_wrong": "Bad sales data can distort forecasts, inventory, pricing, and promotion decisions.",
            "source_type": source_type,
        },
        {
            "connector_name": "Inventory API Connector",
            "system_area": "inventory_and_replenishment",
            "example_source_systems": "ERP, WMS, inventory management system",
            "expected_payload": "date, store_id, sku_id, stock_on_hand, stock_on_order, reorder_point, stockout_flag",
            "integration_pattern": "ERP API, OData API, SQL connector, batch export",
            "implemented_mode": "mock_or_file_based",
            "live_mode_status": "blocked_until_authorized_endpoint_and_credentials_are_provided",
            "target_raw_table": "inventory_snapshots.csv",
            "risk_if_wrong": "Bad inventory data can cause overstock, stockouts, and wrong reorder recommendations.",
            "source_type": source_type,
        },
        {
            "connector_name": "Product Master API Connector",
            "system_area": "product_master_data",
            "example_source_systems": "PIM, ERP master data, merchandising system",
            "expected_payload": "sku_id, product_name, department, category, brand, supplier_id, cost, price, active_flag",
            "integration_pattern": "PIM API, SQL connector, master data export",
            "implemented_mode": "mock_or_file_based",
            "live_mode_status": "blocked_until_authorized_endpoint_and_credentials_are_provided",
            "target_raw_table": "product_master.csv",
            "risk_if_wrong": "Bad product hierarchy can distort category, brand, assortment, and SKU-level decisions.",
            "source_type": source_type,
        },
        {
            "connector_name": "Store Master API Connector",
            "system_area": "store_master_data",
            "example_source_systems": "ERP, store operations system, location master",
            "expected_payload": "store_id, region, city, store_format, sales_area_sqft, active_flag",
            "integration_pattern": "REST API, SQL connector, master data export",
            "implemented_mode": "mock_or_file_based",
            "live_mode_status": "blocked_until_authorized_endpoint_and_credentials_are_provided",
            "target_raw_table": "store_master.csv",
            "risk_if_wrong": "Bad store metadata can distort regional, site, and productivity analysis.",
            "source_type": source_type,
        },
        {
            "connector_name": "Pricing and Promotion API Connector",
            "system_area": "pricing_and_promotions",
            "example_source_systems": "pricing engine, promotion calendar, marketing platform",
            "expected_payload": "sku_id, store_id/region, regular_price, selling_price, promo_type, discount_depth, start_date, end_date",
            "integration_pattern": "pricing API, promotion API, marketing platform export, batch file",
            "implemented_mode": "mock_or_file_based",
            "live_mode_status": "blocked_until_authorized_endpoint_and_credentials_are_provided",
            "target_raw_table": "pricing_history.csv; promotions_calendar.csv",
            "risk_if_wrong": "Bad price/promotion data can distort elasticity, promotion uplift, margin, and demand decisions.",
            "source_type": source_type,
        },
        {
            "connector_name": "Supplier API Connector",
            "system_area": "supplier_and_procurement",
            "example_source_systems": "procurement system, supplier portal, ERP",
            "expected_payload": "supplier_id, lead_time, fill_rate, supplier_risk_score, active_flag",
            "integration_pattern": "supplier portal API, ERP API, procurement export",
            "implemented_mode": "mock_or_file_based",
            "live_mode_status": "blocked_until_authorized_endpoint_and_credentials_are_provided",
            "target_raw_table": "supplier_master.csv",
            "risk_if_wrong": "Bad supplier data can distort reorder points, lead-time risk, and replenishment decisions.",
            "source_type": source_type,
        },
        {
            "connector_name": "Finance API Connector",
            "system_area": "finance_and_margin",
            "example_source_systems": "ERP finance, accounting system, data warehouse",
            "expected_payload": "date, store_id, net_sales, gross_margin, costs, budget, margin_rate",
            "integration_pattern": "ERP API, finance warehouse connector, batch export",
            "implemented_mode": "mock_or_file_based",
            "live_mode_status": "blocked_until_authorized_endpoint_and_credentials_are_provided",
            "target_raw_table": "finance_daily_summary.csv",
            "risk_if_wrong": "Bad finance data can distort ROI, margin, and executive decision reporting.",
            "source_type": source_type,
        },
        {
            "connector_name": "Fabric Upload Connector",
            "system_area": "platform_ingestion",
            "example_source_systems": "OneLake, Lakehouse Files, Fabric Data Factory",
            "expected_payload": "validated CSV/parquet/delta files and manifest",
            "integration_pattern": "OneLake upload, Fabric Data Factory pipeline, Lakehouse notebook",
            "implemented_mode": "bundle_and_notebook_generated",
            "live_mode_status": "requires_actual_fabric_upload_and_notebook_or_pipeline_run",
            "target_raw_table": "Files/retail_decision_support_upload",
            "risk_if_wrong": "Claiming Fabric execution from a local bundle without actual Fabric run history.",
            "source_type": source_type,
        },
        {
            "connector_name": "Workflow Approval API Connector",
            "system_area": "human_approval_and_workflow",
            "example_source_systems": "email, Teams, Slack, n8n, Jira, ServiceNow, ERP approval workflow",
            "expected_payload": "approval_id, recommendation_id, owner, status, decision, timestamp",
            "integration_pattern": "webhook, REST API, n8n workflow, ticketing API",
            "implemented_mode": "approval_queue_file_based",
            "live_mode_status": "blocked_until_workflow_tool_and_approval_policy_are_defined",
            "target_raw_table": "agent_human_approval_queue.csv",
            "risk_if_wrong": "Recommendations may be acted on without analyst or manager approval.",
            "source_type": source_type,
        },
        {
            "connector_name": "Audit and Monitoring API Connector",
            "system_area": "governance_and_observability",
            "example_source_systems": "logging service, MLflow, Fabric monitoring, app logs",
            "expected_payload": "run_id, data_version, model_version, status, errors, approvals, timestamps",
            "integration_pattern": "logging API, MLflow API, Fabric run history, audit table",
            "implemented_mode": "manifest_and_csv_audit",
            "live_mode_status": "requires_operational_logging_backend_or_fabric_run_history",
            "target_raw_table": "run_manifest.json; api_call_audit.csv",
            "risk_if_wrong": "No defensible evidence for what ran, when, with which data/model, and who approved.",
            "source_type": source_type,
        },
    ]
    return pd.DataFrame(rows)


def build_api_ingestion_plan(source_type: str) -> pd.DataFrame:
    """
    API ingestion plan showing how APIs would be used in a real enterprise flow.
    """
    rows = [
        {
            "step": 1,
            "stage": "authenticate",
            "api_or_connector": "Microsoft Entra ID / OAuth / service principal / managed identity",
            "purpose": "Authenticate securely before source-system or Fabric access.",
            "current_script_status": "documented_not_executed",
            "production_requirement": "No hardcoded credentials; use managed identity or secure secrets.",
            "source_type": source_type,
        },
        {
            "step": 2,
            "stage": "extract",
            "api_or_connector": "POS, inventory, product, store, pricing, promotion, supplier, finance connectors",
            "purpose": "Extract operational data from authorized systems.",
            "current_script_status": "mock_or_file_based",
            "production_requirement": "Authorized endpoints, rate limits, schema contracts, retry policy, and source ownership.",
            "source_type": source_type,
        },
        {
            "step": 3,
            "stage": "land_raw",
            "api_or_connector": "OneLake / Lakehouse Files / Fabric Data Factory",
            "purpose": "Land raw source data in a governed raw/Bronze layer.",
            "current_script_status": "local_file_output_and_fabric_bundle",
            "production_requirement": "Fabric workspace, Lakehouse, pipeline run history, and data lineage.",
            "source_type": source_type,
        },
        {
            "step": 4,
            "stage": "validate",
            "api_or_connector": "Validation functions / data quality service",
            "purpose": "Check schema, row counts, freshness, duplicates, missingness, invalid values.",
            "current_script_status": "implemented_locally",
            "production_requirement": "Scheduled validation with blocking gates and alerting.",
            "source_type": source_type,
        },
        {
            "step": 5,
            "stage": "transform",
            "api_or_connector": "Fabric notebook / Spark / SQL / Data Factory pipeline",
            "purpose": "Create Silver cleaned/joined tables and Gold dashboard-ready outputs.",
            "current_script_status": "local_transform_plus_fabric_notebook_cell_generated",
            "production_requirement": "Run inside Fabric or governed cloud execution with logs.",
            "source_type": source_type,
        },
        {
            "step": 6,
            "stage": "score_and_monitor",
            "api_or_connector": "Model scoring API / MLflow API / monitoring API",
            "purpose": "Generate forecasts, recommendations, drift, retraining status, and model audit outputs.",
            "current_script_status": "demo_outputs_generated",
            "production_requirement": "Actual model training/scoring service, registry, versioning, and performance monitoring.",
            "source_type": source_type,
        },
        {
            "step": 7,
            "stage": "agent_governance",
            "api_or_connector": "Agent runtime and deterministic tool registry",
            "purpose": "Review data quality, model evidence, decision risk, evidence boundaries, and approval needs.",
            "current_script_status": "implemented_deterministically",
            "production_requirement": "Persisted traces, approval integration, and security controls.",
            "source_type": source_type,
        },
        {
            "step": 8,
            "stage": "approval_and_handoff",
            "api_or_connector": "Teams/Slack/email/n8n/Jira/ServiceNow/ERP workflow APIs",
            "purpose": "Route high-risk recommendations for human review before action.",
            "current_script_status": "approval_queue_file_based",
            "production_requirement": "Live workflow integration, user identity, timestamps, and approval audit.",
            "source_type": source_type,
        },
        {
            "step": 9,
            "stage": "reporting",
            "api_or_connector": "Streamlit / Power BI / Fabric semantic model",
            "purpose": "Expose decision-ready outputs to users.",
            "current_script_status": "Streamlit_and_file_based_outputs",
            "production_requirement": "Role-based access, refresh schedule, and governed semantic layer.",
            "source_type": source_type,
        },
    ]
    return pd.DataFrame(rows)


def build_api_security_checklist(source_type: str) -> pd.DataFrame:
    rows = [
        {
            "security_area": "authentication",
            "requirement": "Use OAuth, service principal, managed identity, or platform-native authentication.",
            "current_status": "not_implemented_in_local_script",
            "risk": "Hardcoded or shared credentials would be unsafe.",
            "minimum_evidence_needed": "Credential configuration screenshot or secret reference, not raw secrets.",
            "source_type": source_type,
        },
        {
            "security_area": "secrets_management",
            "requirement": "Store API keys/secrets in Key Vault, Fabric connection, environment secret, or approved secret manager.",
            "current_status": "not_implemented_in_local_script",
            "risk": "Leaked credentials and unauthorized data access.",
            "minimum_evidence_needed": "Secret name/reference and access policy evidence; never expose secret values.",
            "source_type": source_type,
        },
        {
            "security_area": "role_based_access",
            "requirement": "Separate executive, analyst, manager, engineer, and admin access.",
            "current_status": "not_implemented_in_local_script",
            "risk": "Users may see data/actions they should not access.",
            "minimum_evidence_needed": "RBAC matrix and workspace/app access settings.",
            "source_type": source_type,
        },
        {
            "security_area": "data_privacy",
            "requirement": "Avoid customer-level personal data unless necessary; hash/de-identify where possible.",
            "current_status": "synthetic_customer_hash_only_or_user_supplied_data",
            "risk": "Privacy breach if real customer data is exposed.",
            "minimum_evidence_needed": "Data classification and de-identification policy.",
            "source_type": source_type,
        },
        {
            "security_area": "public_dashboard_boundary",
            "requirement": "Public demos must not expose real business-sensitive data.",
            "current_status": "claim_boundary_documented",
            "risk": "Exposing confidential sales, margin, supplier, customer, or operational data.",
            "minimum_evidence_needed": "Public/private data separation note and sanitized demo dataset.",
            "source_type": source_type,
        },
        {
            "security_area": "auditability",
            "requirement": "Log API calls, run IDs, data versions, model versions, approvals, and errors.",
            "current_status": "local_manifest_and_api_call_audit_generated",
            "risk": "No defensible trace of what ran or who approved decisions.",
            "minimum_evidence_needed": "Run manifest, API call audit, Fabric run history, approval logs.",
            "source_type": source_type,
        },
    ]
    return pd.DataFrame(rows)


def build_api_integration_maturity(source_type: str) -> pd.DataFrame:
    rows = [
        {
            "capability": "API connector architecture",
            "current_evidence": "Connector registry and ingestion plan generated.",
            "maturity": "Implemented as architecture artifact",
            "what_it_does_not_prove": "Does not prove live API connection.",
            "upgrade_evidence_needed": "Successful API calls to authorized endpoints with logs.",
            "source_type": source_type,
        },
        {
            "capability": "Mock/file-based connector mode",
            "current_evidence": "Local synthetic/public/generic-real file ingestion supported.",
            "maturity": "Implemented",
            "what_it_does_not_prove": "Does not prove real-time or governed source-system integration.",
            "upgrade_evidence_needed": "Source-system API access, pipeline runs, and data-owner approval.",
            "source_type": source_type,
        },
        {
            "capability": "Live source API mode",
            "current_evidence": "Scaffolded conceptually; blocked in code until endpoints and credentials exist.",
            "maturity": "Not Confirmed",
            "what_it_does_not_prove": "No live source API call is made by this script.",
            "upgrade_evidence_needed": "Endpoint configuration, authentication, successful extract log, schema check.",
            "source_type": source_type,
        },
        {
            "capability": "Fabric API/platform execution",
            "current_evidence": "Fabric bundle and notebook cell generated.",
            "maturity": "Fabric-ready, not executed",
            "what_it_does_not_prove": "Does not prove live Fabric execution from local run alone.",
            "upgrade_evidence_needed": "Fabric notebook success, Lakehouse tables, execution audit, pipeline run history.",
            "source_type": source_type,
        },
        {
            "capability": "Workflow/approval API integration",
            "current_evidence": "Approval queue generated as CSV.",
            "maturity": "Architecture artifact / local queue",
            "what_it_does_not_prove": "No Teams, Slack, n8n, Jira, ServiceNow, or ERP approval API call is made.",
            "upgrade_evidence_needed": "Webhook/API call logs and human approval records.",
            "source_type": source_type,
        },
        {
            "capability": "Production API governance",
            "current_evidence": "Security checklist and claim boundaries generated.",
            "maturity": "Not Confirmed",
            "what_it_does_not_prove": "No production security, RBAC, SLA, or incident response is implemented.",
            "upgrade_evidence_needed": "RBAC, secrets, CI/CD, monitoring, incident/rollback, users, and support model.",
            "source_type": source_type,
        },
    ]
    return pd.DataFrame(rows)


class EnterpriseAPIConnectorRuntime:
    """
    Local connector runtime.

    This class represents API connector behavior safely:
    - mock/file mode is implemented
    - live API mode is explicitly blocked unless authorized configuration is supplied
    - all attempted calls are logged in api_call_audit.csv

    No real external API calls are made by default.
    """

    def __init__(self, source_type: str):
        self.source_type = source_type
        self.audit_rows: List[Dict[str, Any]] = []

    def log_call(self, connector_name: str, mode: str, status: str, endpoint_or_file: str, rows_returned: Any, reason: str) -> None:
        self.audit_rows.append({
            "timestamp_utc": utc_now(),
            "connector_name": connector_name,
            "mode": mode,
            "status": status,
            "endpoint_or_file": endpoint_or_file,
            "rows_returned": rows_returned,
            "reason": reason,
            "source_type": self.source_type,
        })

    def mock_file_call(self, connector_name: str, file_name: str, rows_returned: Any) -> None:
        self.log_call(
            connector_name=connector_name,
            mode="mock_or_file_based",
            status="completed",
            endpoint_or_file=file_name,
            rows_returned=rows_returned,
            reason="Local file/mock path used. This does not prove live API integration.",
        )

    def blocked_live_call(self, connector_name: str, endpoint_name: str) -> None:
        self.log_call(
            connector_name=connector_name,
            mode="live_api",
            status="blocked",
            endpoint_or_file=endpoint_name,
            rows_returned="",
            reason="No authorized endpoint, credentials, schema contract, or permission supplied.",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.audit_rows)


def build_api_outputs(raw: Dict[str, pd.DataFrame], source_type: str) -> Dict[str, pd.DataFrame]:
    registry = build_api_connector_registry(source_type)
    plan = build_api_ingestion_plan(source_type)
    security = build_api_security_checklist(source_type)
    maturity = build_api_integration_maturity(source_type)

    runtime = EnterpriseAPIConnectorRuntime(source_type)
    connector_file_map = {
        "POS API Connector": "pos_transaction_lines",
        "Inventory API Connector": "inventory_snapshots",
        "Product Master API Connector": "product_master",
        "Store Master API Connector": "store_master",
        "Pricing and Promotion API Connector": "pricing_history; promotions_calendar",
        "Supplier API Connector": "supplier_master",
        "Finance API Connector": "finance_daily_summary",
    }

    for connector, table_names in connector_file_map.items():
        total_rows = 0
        for table_name in [t.strip() for t in table_names.split(";")]:
            if table_name in raw:
                total_rows += int(len(raw[table_name]))
        runtime.mock_file_call(connector, table_names, total_rows)
        runtime.blocked_live_call(connector, connector.replace(" ", "_").lower())

    runtime.mock_file_call("Fabric Upload Connector", "fabric_bundle/retail_decision_support_upload", "created_if_gold_outputs_exist")
    runtime.blocked_live_call("Fabric Upload Connector", "microsoft_fabric_lakehouse_api")
    runtime.mock_file_call("Workflow Approval API Connector", "agent_human_approval_queue.csv", "created_if_agent_outputs_exist")
    runtime.blocked_live_call("Workflow Approval API Connector", "teams_slack_n8n_jira_servicenow_or_erp_api")
    runtime.mock_file_call("Audit and Monitoring API Connector", "run_manifest.json; api_call_audit.csv", "created")
    runtime.blocked_live_call("Audit and Monitoring API Connector", "production_logging_or_monitoring_backend")

    return {
        "api_connector_registry": registry,
        "api_ingestion_plan": plan,
        "api_call_audit": runtime.to_frame(),
        "api_security_checklist": security,
        "api_integration_maturity": maturity,
    }


# ---------------------------------------------------------------------
# Fabric support
# ---------------------------------------------------------------------

def create_fabric_bundle(paths: Dict[str, Path]) -> Dict[str, Any]:
    upload = paths["fabric_upload"]
    copied = []
    for fname in DASHBOARD_OUTPUTS:
        src = paths["gold"] / fname
        if not src.exists():
            src = paths["outputs"] / fname
        if src.exists():
            dest = upload / fname
            shutil.copy2(src, dest)
            copied.append({
                "file_name": fname,
                "source_path": str(src),
                "upload_path": str(dest),
                "sha256": sha256_file(dest),
            })
    manifest = {
        "created_at_utc": utc_now(),
        "purpose": "Upload this folder to Microsoft Fabric Lakehouse Files as Files/retail_decision_support_upload.",
        "copied_files": copied,
        "claim_boundary": "Bundle alone is not Fabric execution. Run the generated notebook inside Fabric and capture run history.",
    }
    save_json(manifest, upload / "_fabric_upload_manifest.json")
    (upload / "_README_UPLOAD_TO_FABRIC.md").write_text(
        "Upload this folder to Microsoft Fabric Lakehouse Files as Files/retail_decision_support_upload. "
        "Then run the generated Fabric notebook cell. This bundle does not prove Fabric execution until run in Fabric.\n",
        encoding="utf-8",
    )
    return manifest


def fabric_notebook_code() -> str:
    return r"""
# Microsoft Fabric Notebook Cell - Retail Decision Support System
# Run inside a Microsoft Fabric notebook attached to a Lakehouse.
# Upload CSVs to Lakehouse Files as: Files/retail_decision_support_upload

import re
from datetime import datetime

FABRIC_FILES_DIR = "Files/retail_decision_support_upload"

files = [
    "dashboard_executive_summary.csv",
    "dashboard_model_comparison.csv",
    "dashboard_store_forecast.csv",
    "dashboard_department_forecast.csv",
    "dashboard_region_forecast.csv",
    "dashboard_brand_forecast.csv",
    "inventory_recommendations.csv",
    "drift_monitor.csv",
    "retraining_status.csv",
    "retraining_audit.csv",
    "store_watchlist.csv",
    "dashboard_pipeline_maturity.csv",
    "workflow_handoff.csv",
    "agent_answers.csv",
    "data_quality_audit.csv",
    "data_contract_summary.csv",
]

def safe_table_name(file_name):
    stem = file_name.replace(".csv", "").lower()
    stem = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    if stem and stem[0].isdigit():
        stem = "t_" + stem
    return stem or "table"

audit_rows = []

for file_name in files:
    path = f"{FABRIC_FILES_DIR}/{file_name}"
    table_name = safe_table_name(file_name)
    try:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)

        clean_cols = []
        seen = {}
        for c in df.columns:
            clean = re.sub(r"[^a-zA-Z0-9_]+", "_", c.strip().lower()).strip("_")
            if not clean:
                clean = "column"
            if clean[0].isdigit():
                clean = "c_" + clean
            if clean in seen:
                seen[clean] += 1
                clean = f"{clean}_{seen[clean]}"
            else:
                seen[clean] = 0
            clean_cols.append(clean)

        for old, new in zip(df.columns, clean_cols):
            if old != new:
                df = df.withColumnRenamed(old, new)

        rows = df.count()
        cols = len(df.columns)
        df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        audit_rows.append((file_name, table_name, "PASS", rows, cols, "", datetime.utcnow().isoformat()))
    except Exception as e:
        audit_rows.append((file_name, table_name, "FAIL", 0, 0, str(e), datetime.utcnow().isoformat()))

audit_schema = ["source_file", "lakehouse_table", "status", "rows", "columns", "error", "run_time_utc"]
audit_df = spark.createDataFrame(audit_rows, audit_schema)
audit_df.write.mode("overwrite").format("delta").saveAsTable("fabric_retail_execution_audit")

display(audit_df)

print("Capture evidence: notebook success, Lakehouse tables, fabric_retail_execution_audit, and pipeline run history if orchestrated.")
print("Boundary: This proves Fabric execution only after running in Fabric. It is not production deployment.")
"""


def write_fabric_notebook(paths: Dict[str, Path]) -> Path:
    p = paths["notebooks"] / "fabric_retail_lakehouse_execution_cell.py"
    p.write_text(fabric_notebook_code(), encoding="utf-8")
    return p


def write_claim_docs(paths: Dict[str, Path], source_type: str) -> None:
    text = (
        "# Claim Boundary\n\n"
        f"Generated at UTC: {utc_now()}\n"
        f"Source type: {source_type}\n\n"
        "Can support:\n"
        "- Dashboard-ready retail decision-support outputs\n"
        "- Local raw/bronze/silver/gold-style organization\n"
        "- Data contracts and quality checks\n"
        "- Fabric-ready upload bundle\n\n"
        "Cannot support by itself:\n"
        "- Real enterprise integration unless approved real enterprise data/source documentation exists\n"
        "- Live Microsoft Fabric execution unless the generated notebook is run inside Fabric\n"
        "- Production deployment unless production governance, security, CI/CD, monitoring, and users exist\n"
        "- Real business impact unless decisions and outcomes are measured\n"
    )
    (paths["audit"] / "CLAIM_BOUNDARY.md").write_text(text, encoding="utf-8")


def write_layers(paths: Dict[str, Path], raw, silver, gold, contracts, source_type, config) -> Dict[str, Any]:
    written = []
    for layer, folder, data in [
        ("raw", paths["raw"], raw),
        ("bronze", paths["bronze"], raw),
        ("silver", paths["silver"], silver),
        ("gold", paths["gold"], gold),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            p = folder / f"{name}.csv"
            save_csv(df, p)
            written.append({
                "layer": layer,
                "name": name,
                "path": str(p),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "sha256": sha256_file(p),
            })
    if config.write_app_outputs:
        for name, df in gold.items():
            p = paths["outputs"] / f"{name}.csv"
            save_csv(df, p)
            written.append({
                "layer": "outputs",
                "name": name,
                "path": str(p),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "sha256": sha256_file(p),
            })
    if "schema_agent_memory_jsonl_records" in gold and not gold["schema_agent_memory_jsonl_records"].empty:
        for layer_name, folder in [("gold", paths["gold"]), ("outputs", paths["outputs"])]:
            jsonl_path = folder / "schema_agent_memory_validated_records.jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with jsonl_path.open("w", encoding="utf-8") as f:
                for line in gold["schema_agent_memory_jsonl_records"].get("json_line", pd.Series(dtype=str)).astype(str):
                    f.write(line + "\n")
            written.append({
                "layer": layer_name,
                "name": "schema_agent_memory_validated_records",
                "path": str(jsonl_path),
                "rows": int(len(gold["schema_agent_memory_jsonl_records"])),
                "columns": 1,
                "sha256": sha256_file(jsonl_path),
            })
    for name, contract in contracts.items():
        save_json(contract, paths["contracts"] / f"{name}_contract.json")
    fabric_manifest = create_fabric_bundle(paths)
    notebook = write_fabric_notebook(paths)
    write_claim_docs(paths, source_type)
    manifest = {
        "generated_at_utc": utc_now(),
        "script": "retail_enterprise_upgrade_v4_schema_memory.py",
        "mode": config.mode,
        "source_type": source_type,
        "config": asdict(config),
        "safe_claim": "Generated synthetic enterprise-like outputs." if source_type == "synthetic" else "Generated dashboard-ready outputs from public/approved retail data.",
        "claim_boundary": {
            "real_enterprise_integration": "Only supported if approved real enterprise data and source documentation are supplied.",
            "live_fabric_execution": "Only supported after running generated notebook inside Microsoft Fabric and capturing run history.",
            "production_deployment": "Not supported by this script alone.",
            "business_impact": "Not supported without measured decisions and outcomes.",
        },
        "raw_tables": {k: {"rows": int(len(v)), "columns": int(len(v.columns))} for k, v in raw.items()},
        "silver_tables": {k: {"rows": int(len(v)), "columns": int(len(v.columns))} for k, v in silver.items()},
        "gold_outputs": {k: {"rows": int(len(v)), "columns": int(len(v.columns))} for k, v in gold.items()},
        "fabric_upload_folder": str(paths["fabric_upload"]),
        "fabric_notebook_cell": str(notebook),
        "written_files": written,
        "fabric_manifest": fabric_manifest,
    }
    save_json(manifest, paths["base"] / "run_manifest.json")
    save_json(manifest, paths["audit"] / "run_manifest.json")
    readme = (
        "# Retail Enterprise Upgrade Evidence Pack\n\n"
        f"Generated at UTC: {utc_now()}\n"
        f"Source type: {source_type}\n\n"
        "Folders:\n"
        f"- Raw: {paths['raw']}\n"
        f"- Bronze: {paths['bronze']}\n"
        f"- Silver: {paths['silver']}\n"
        f"- Gold: {paths['gold']}\n"
        f"- Contracts: {paths['contracts']}\n"
        f"- Audit: {paths['audit']}\n"
        f"- Fabric upload: {paths['fabric_upload']}\n"
        f"- Fabric notebook cell: {notebook}\n"
        f"- App-compatible outputs: {paths['outputs']}\n\n"
        "Next steps:\n"
        "1. Run Streamlit and verify the dashboard reads generated outputs.\n"
        "2. Upload Fabric bundle to Lakehouse Files.\n"
        "3. Run generated Fabric notebook inside Fabric.\n"
        "4. Capture Lakehouse/notebook/pipeline screenshots and run history.\n"
        "5. Do not claim production without security, CI/CD, monitoring, users, and governance evidence.\n"
    )
    (paths["base"] / "README_ENTERPRISE_UPGRADE.md").write_text(readme, encoding="utf-8")
    return manifest


def run(config: Config) -> Dict[str, Any]:
    root = Path(config.project_root).expanduser().resolve()
    paths = ensure_dirs(root)
    rng = np.random.default_rng(config.seed)
    if config.mode in ["synthetic", "all-synthetic"]:
        source_type = "synthetic"
        raw = generate_synthetic_raw(config)
    elif config.mode == "walmart-public":
        if not config.real_data_dir:
            raise ValueError("--real-data-dir is required for walmart-public mode.")
        source_type = "walmart-public"
        raw = load_walmart_public(Path(config.real_data_dir).expanduser().resolve())
    elif config.mode == "generic-real":
        if not config.real_data_dir:
            raise ValueError("--real-data-dir is required for generic-real mode.")
        source_type = "generic-real"
        raw = load_generic_real(Path(config.real_data_dir).expanduser().resolve())
    elif config.mode == "fabric-bundle":
        fabric_manifest = create_fabric_bundle(paths)
        notebook = write_fabric_notebook(paths)
        return {
            "mode": config.mode,
            "fabric_upload_folder": str(paths["fabric_upload"]),
            "fabric_notebook_cell": str(notebook),
            "claim_boundary": "Not Fabric execution until run inside Fabric.",
            "fabric_manifest": fabric_manifest,
        }
    elif config.mode == "write-fabric-notebook":
        notebook = write_fabric_notebook(paths)
        return {
            "mode": config.mode,
            "fabric_notebook_cell": str(notebook),
            "claim_boundary": "Not Fabric execution until run inside Fabric.",
        }
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

    silver = build_silver(raw)
    quality = build_quality(raw, silver, source_type)
    contracts = build_contracts(raw, source_type)
    gold = build_gold(raw, silver, quality, contracts, source_type, rng)
    agent_outputs = build_agent_governance_outputs(gold, quality, source_type)
    gold.update(agent_outputs)
    real_agent_outputs = run_real_agent_orchestration(gold, quality, source_type)
    gold.update(real_agent_outputs)
    memory_outputs = build_three_tier_agent_memory(source_type, gold, real_agent_outputs, quality)
    gold.update(memory_outputs)
    schema_memory_outputs = build_schema_constrained_agent_memory(source_type, gold, real_agent_outputs, quality, root)
    gold.update(schema_memory_outputs)
    api_outputs = build_api_outputs(raw, source_type)
    gold.update(api_outputs)
    return write_layers(paths, raw, silver, gold, contracts, source_type, config)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Retail Decision Support System enterprise upgrade v4 with schema-constrained agent memory.")
    parser.add_argument("--mode", default="all-synthetic", choices=["synthetic", "all-synthetic", "walmart-public", "generic-real", "fabric-bundle", "write-fabric-notebook"])
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--real-data-dir", default="")
    parser.add_argument("--stores", type=int, default=12)
    parser.add_argument("--skus", type=int, default=150)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--no-app-outputs", action="store_true")
    parser.add_argument("--no-quality-issues", action="store_true")
    args = parser.parse_args()
    return Config(
        mode=args.mode,
        project_root=args.project_root,
        real_data_dir=args.real_data_dir,
        stores=args.stores,
        skus=args.skus,
        days=args.days,
        seed=args.seed,
        start_date=args.start_date,
        write_app_outputs=not args.no_app_outputs,
        inject_quality_issues=not args.no_quality_issues,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = run(cfg)
    print("Retail enterprise upgrade script completed.")
    print(json.dumps(result, indent=2, default=str))
