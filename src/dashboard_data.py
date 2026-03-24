from __future__ import annotations

import pandas as pd


def build_store_dashboard(forecast_output: pd.DataFrame) -> pd.DataFrame:
    out = forecast_output.groupby("store_id", as_index=False).agg(forecast_units=("forecast_units", "sum"), actual_units=("units_sold", "sum"), total_margin=("gross_margin_dollars", "sum"))
    out["forecast_error"] = out["actual_units"] - out["forecast_units"]
    out["abs_error"] = out["forecast_error"].abs()
    return out


def build_department_dashboard(forecast_output: pd.DataFrame) -> pd.DataFrame:
    out = forecast_output.groupby("department", as_index=False).agg(forecast_units=("forecast_units", "sum"), actual_units=("units_sold", "sum"))
    out["forecast_error"] = out["actual_units"] - out["forecast_units"]
    return out
