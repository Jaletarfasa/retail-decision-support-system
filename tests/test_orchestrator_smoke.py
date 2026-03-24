from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from src import orchestrator
from src.forecasting import ModelResult
from src.storage import RunStorage


def test_orchestrator_main_smoke(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    temp_root = repo_root / ".pytest_tmp" / "orchestrator_smoke"
    if temp_root.exists():
        shutil.rmtree(temp_root)

    def small_store_metadata(_: int, seed: int = 42) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "store_id": [1, 2],
                "region": ["Ontario", "BC"],
                "store_format": ["Urban", "Suburban"],
                "traffic_index": [1.1, 0.9],
                "site_score": [1.0, 1.2],
            }
        )

    def small_sku_metadata(_: int, seed: int = 42) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sku_id": [101, 102],
                "department": ["Home", "Tools"],
                "base_price": [10.0, 15.0],
                "margin_pct": [0.25, 0.30],
            }
        )

    def small_inventory_snapshot(store_ids: pd.Series, sku_ids: pd.Series, seed: int = 42) -> pd.DataFrame:
        rows = []
        for store_id in store_ids:
            for sku_id in sku_ids:
                rows.append(
                    {
                        "store_id": int(store_id),
                        "sku_id": int(sku_id),
                        "on_hand_units": 20,
                        "lead_time_days": 5,
                    }
                )
        return pd.DataFrame(rows)

    def small_candidate_sites(_: int, seed: int = 42) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "candidate_site_id": [1, 2, 3],
                "region": ["Ontario", "BC", "Prairies"],
                "traffic_index": [1.2, 0.95, 1.05],
                "household_income_index": [1.1, 0.9, 1.0],
                "competition_index": [0.8, 1.1, 0.95],
                "rent_cost": [40000.0, 35000.0, 45000.0],
            }
        )

    def small_daily_sales(stores: pd.DataFrame, skus: pd.DataFrame, n_days: int = 120, seed: int = 42) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        rows = []
        for _, store in stores.iterrows():
            for _, sku in skus.iterrows():
                for day_idx, date in enumerate(dates, start=1):
                    promo = int(day_idx % 7 == 0)
                    units = 5 + int(store["store_id"]) + int(sku["sku_id"] == 102) + (1 if promo else 0)
                    rows.append(
                        {
                            "date": date,
                            "store_id": int(store["store_id"]),
                            "sku_id": int(sku["sku_id"]),
                            "department": sku["department"],
                            "price": float(sku["base_price"]) * (0.95 if promo else 1.0),
                            "promo": promo,
                            "units_sold": units,
                            "margin_pct": float(sku["margin_pct"]),
                        }
                    )
        return pd.DataFrame(rows)

    def fake_fit_and_score_models(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> list[ModelResult]:
        preds = test_df[target_col].astype(float).rename("forecast_units")
        return [
            ModelResult(
                model_name="smoke_baseline",
                model_obj=None,
                metrics={"mae": 0.0, "rmse": 0.0, "wmape": 0.0, "bias": 0.0, "r2": 1.0},
                preds=preds,
            )
        ]

    class TempRunStorage(RunStorage):
        def __init__(self, artifact_root: str | Path, run_id: str) -> None:
            super().__init__(temp_root, run_id)

    monkeypatch.setattr(orchestrator, "generate_store_metadata", small_store_metadata)
    monkeypatch.setattr(orchestrator, "generate_sku_metadata", small_sku_metadata)
    monkeypatch.setattr(orchestrator, "generate_inventory_snapshot", small_inventory_snapshot)
    monkeypatch.setattr(orchestrator, "generate_candidate_sites", small_candidate_sites)
    monkeypatch.setattr(orchestrator, "generate_daily_sales", small_daily_sales)
    monkeypatch.setattr(orchestrator, "fit_and_score_models", fake_fit_and_score_models)
    monkeypatch.setattr(orchestrator, "RunStorage", TempRunStorage)
    monkeypatch.chdir(repo_root)

    orchestrator.main()

    run_dirs = sorted((temp_root / "runs").glob("run_*"))
    assert run_dirs, "orchestrator did not create a run directory"

    run_dir = run_dirs[-1]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "selection" / "model_comparison.csv").exists()
    assert (run_dir / "dashboard" / "dashboard_executive_summary.csv").exists()
