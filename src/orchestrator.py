from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

from src.utils import make_run_id
from src.storage import RunStorage
from src.data_generation import (
    generate_store_metadata,
    generate_sku_metadata,
    generate_inventory_snapshot,
    generate_candidate_sites,
    generate_daily_sales,
)
from src.validation import run_source_validations
from src.feature_engineering import build_features
from src.forecasting import time_split, fit_and_score_models
from src.model_selection import build_model_comparison, select_champion_and_challenger
from src.promotion_analytics import build_promotion_summary
from src.assortment_health import build_assortment_health_summary
from src.inventory import build_inventory_recommendations
from src.site_scoring import score_candidate_sites
from src.optimization import optimize_sites
from src.monitoring import run_drift_monitor
from src.retraining import should_retrain, build_retraining_audit
from src.reporting import build_executive_summary
from src.dashboard_data import build_store_dashboard, build_department_dashboard


def main() -> None:
    app_cfg = yaml.safe_load(Path("config/app_config.yaml").read_text())
    model_cfg = yaml.safe_load(Path("config/model_config.yaml").read_text())
    monitoring_cfg = yaml.safe_load(Path("config/monitoring_config.yaml").read_text())
    storage_cfg = yaml.safe_load(Path("config/storage_config.yaml").read_text())

    run_id = make_run_id()
    storage = RunStorage(storage_cfg["artifact_root"], run_id)
    storage.write_manifest({"run_id": run_id, "status": "started", "app_version": app_cfg["app_version"]})

    stores = generate_store_metadata(30)
    skus = generate_sku_metadata(60)
    inventory = generate_inventory_snapshot(stores["store_id"], skus["sku_id"])
    sites = generate_candidate_sites(20)
    sales = generate_daily_sales(stores, skus, n_days=120)
    sales["sales_dollars"] = sales["units_sold"] * sales["price"]
    sales["gross_margin_dollars"] = sales["sales_dollars"] * sales["margin_pct"]

    for name, df in {
        "sales_transactions": sales,
        "store_metadata": stores,
        "sku_metadata": skus,
        "inventory_snapshot": inventory,
        "candidate_sites": sites,
    }.items():
        storage.save_table_sqlite(df, name)
        storage.save_csv(df, f"{name}.csv", "source")

    validation_results = run_source_validations({"sales_transactions": sales})
    pd.DataFrame([
        {"table": k, "errors": "; ".join(v)} for k, v in validation_results.items()
    ]).to_csv(storage.paths["logs"] / "validation_results.csv", index=False)

    feat = build_features(sales, stores)
    storage.save_csv(feat.head(5000), "feature_sample.csv", "features")

    train_df, test_df = time_split(feat, test_days=model_cfg["test_days"])
    feature_cols = [
        "store_id", "sku_id", "price", "promo", "lag_1", "lag_7", "lag_28",
        "rolling_mean_7", "rolling_mean_28", "rolling_std_28", "assortment_size",
        "active_skus_28d_avg", "units_28d_avg", "assortment_health_ratio", "traffic_index", "site_score"
    ]
    target_col = "units_sold"

    results = fit_and_score_models(train_df, test_df, feature_cols, target_col)
    comp = build_model_comparison(results)
    champion, challenger = select_champion_and_challenger(results)

    storage.save_csv(comp, "model_comparison.csv", "selection")
    storage.save_table_sqlite(comp, "dashboard_model_comparison")

    scored = test_df.copy()
    scored["forecast_units"] = champion.preds

    promotion = build_promotion_summary(scored)
    assortment = build_assortment_health_summary(feat)
    inv = build_inventory_recommendations(scored[["store_id", "sku_id", "forecast_units"]], inventory)
    site_scores = score_candidate_sites(sites)
    site_opt = optimize_sites(site_scores, budget_limit=75000.0)

    drift = run_drift_monitor(train_df, test_df, monitoring_cfg["monitored_features"], monitoring_cfg["p_threshold"])
    retrain = should_retrain(drift, champion.metrics["wmape"], baseline_wmape=champion.metrics["wmape"] - 0.01,
                             wmape_degradation_threshold=monitoring_cfg["wmape_degradation_threshold"],
                             min_drifted_features=monitoring_cfg["min_drifted_features"])
    retrain_audit = build_retraining_audit(retrain, champion.model_name, challenger.model_name if challenger else None)

    executive = build_executive_summary(champion.model_name, challenger.model_name if challenger else None, champion.metrics, retrain)
    store_dash = build_store_dashboard(scored)
    dept_dash = build_department_dashboard(scored)

    outputs = [
        (promotion, "promotion_analytics.csv", "promotion", "promotion_analytics"),
        (assortment, "assortment_health.csv", "assortment", "assortment_health"),
        (inv, "inventory_recommendations.csv", "inventory", "inventory_recommendations"),
        (site_scores, "site_selection_rankings.csv", "site", "site_selection_rankings"),
        (site_opt, "optimized_site_selection.csv", "optimization", "optimized_site_selection"),
        (drift, "drift_monitor.csv", "monitoring", "drift_monitor"),
        (retrain_audit, "retraining_audit.csv", "retraining", "retraining_audit"),
        (executive, "dashboard_executive_summary.csv", "dashboard", "dashboard_executive_summary"),
        (store_dash, "dashboard_store_forecast.csv", "dashboard", "dashboard_store_forecast"),
        (dept_dash, "dashboard_department_forecast.csv", "dashboard", "dashboard_department_forecast"),
    ]

    for df, fn, section, table in outputs:
        storage.save_csv(df, fn, section)
        storage.save_table_sqlite(df, table)

    storage.write_manifest({"run_id": run_id, "status": "completed", "app_version": app_cfg["app_version"], "champion_model": champion.model_name})
    print(f"Run complete: {run_id}")


if __name__ == "__main__":
    main()
