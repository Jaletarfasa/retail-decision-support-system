import sqlite3
import pandas as pd
import pathlib

run_dirs = sorted(pathlib.Path("artifacts/runs").glob("run_*"))
if not run_dirs:
    raise FileNotFoundError("No run folders found under artifacts/runs")

latest_run = run_dirs[-1]
print(f"Latest run: {latest_run}")

db_path = latest_run / "retail_system.db"
if not db_path.exists():
    raise FileNotFoundError(f"No database found at {db_path}")

print(f"Using database: {db_path}")

conn = sqlite3.connect(db_path)

print("\n=== TABLES ===")
tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
    conn
)
print(tables)

print("\n=== EXECUTIVE SUMMARY ===")
exec_df = pd.read_sql("SELECT * FROM dashboard_executive_summary", conn)
print(exec_df)

print("\n=== MODEL COMPARISON ===")
model_df = pd.read_sql("SELECT * FROM dashboard_model_comparison", conn)
print(model_df)

print("\n=== STORE FORECAST (TOP 10) ===")
store_df = pd.read_sql("SELECT * FROM dashboard_store_forecast LIMIT 10", conn)
print(store_df)

print("\n=== DEPARTMENT FORECAST ===")
dept_df = pd.read_sql("SELECT * FROM dashboard_department_forecast", conn)
print(dept_df)

print("\n=== PROMOTION ANALYTICS ===")
promo_df = pd.read_sql("SELECT * FROM promotion_analytics", conn)
print(promo_df)

print("\n=== INVENTORY RECOMMENDATIONS (TOP 10) ===")
inv_df = pd.read_sql(
    """
    SELECT *
    FROM inventory_recommendations
    ORDER BY recommended_reorder_qty DESC
    LIMIT 10
    """,
    conn,
)
print(inv_df)

print("\n=== OPTIMIZED SITE SELECTION ===")
site_df = pd.read_sql(
    """
    SELECT *
    FROM optimized_site_selection
    ORDER BY projected_value_index DESC
    LIMIT 10
    """,
    conn,
)
print(site_df)

print("\n=== DRIFT MONITOR ===")
drift_df = pd.read_sql("SELECT * FROM drift_monitor", conn)
print(drift_df)

print("\n=== RETRAINING AUDIT ===")
audit_df = pd.read_sql("SELECT * FROM retraining_audit", conn)
print(audit_df)

conn.close()
print("\nInspection complete.")
