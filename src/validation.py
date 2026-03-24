from __future__ import annotations

import pandas as pd


def validate_non_empty(df: pd.DataFrame, name: str) -> list[str]:
    return [f"{name} is empty."] if df is None or df.empty else []


def validate_required_columns(df: pd.DataFrame, name: str, required: list[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return [f"{name} is missing required columns: {missing}"] if missing else []


def validate_positive_numeric(df: pd.DataFrame, col: str, name: str) -> list[str]:
    if col in df.columns and (df[col] <= 0).any():
        return [f"{name}.{col} contains non-positive values."]
    return []


def run_source_validations(source_tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    if "sales_transactions" in source_tables:
        df = source_tables["sales_transactions"]
        errs = []
        errs += validate_non_empty(df, "sales_transactions")
        errs += validate_required_columns(df, "sales_transactions", ["date", "store_id", "sku_id", "units_sold", "price"])
        errs += validate_positive_numeric(df, "price", "sales_transactions")
        results["sales_transactions"] = errs
    return results
