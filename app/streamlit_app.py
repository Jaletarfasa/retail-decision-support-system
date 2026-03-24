from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Retail Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom styling
# -------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #eef3f9 100%);
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #dbe7ff;
        margin-top: 0;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 1.1rem;
        margin-bottom: 0.6rem;
    }

    .kpi-card {
        border-radius: 18px;
        padding: 1rem 1rem 0.8rem 1rem;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.10);
        margin-bottom: 0.6rem;
    }

    .kpi-label {
        font-size: 0.85rem;
        opacity: 0.92;
        margin-bottom: 0.2rem;
    }

    .kpi-value {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.1;
        word-break: break-word;
    }

    .blue-card { background: linear-gradient(135deg, #2563eb, #1d4ed8); }
    .green-card { background: linear-gradient(135deg, #059669, #047857); }
    .amber-card { background: linear-gradient(135deg, #d97706, #b45309); }
    .purple-card { background: linear-gradient(135deg, #7c3aed, #6d28d9); }
    .slate-card { background: linear-gradient(135deg, #334155, #1e293b); }
    .rose-card { background: linear-gradient(135deg, #e11d48, #be123c); }

    .panel-box {
        background: white;
        border-radius: 18px;
        padding: 1rem 1rem 1rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }

    .status-good {
        background: #dcfce7;
        color: #166534;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        font-weight: 700;
        border-left: 6px solid #16a34a;
        margin-bottom: 1rem;
    }

    .status-watch {
        background: #fef3c7;
        color: #92400e;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        font-weight: 700;
        border-left: 6px solid #f59e0b;
        margin-bottom: 1rem;
    }

    .status-alert {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        font-weight: 700;
        border-left: 6px solid #ef4444;
        margin-bottom: 1rem;
    }

    .callout-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dbeafe;
        border-left: 6px solid #2563eb;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.06);
    }

    .callout-title {
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.25rem;
    }

    .callout-text {
        color: #334155;
        font-size: 0.95rem;
    }

    .mini-badge {
        display: inline-block;
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .badge-blue { background: #dbeafe; color: #1d4ed8; }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-amber { background: #fef3c7; color: #92400e; }
    .badge-rose { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None

def get_metric(df: pd.DataFrame | None, metric_name: str, default="N/A"):
    if df is None or df.empty:
        return default
    row = df.loc[df["metric"] == metric_name, "value"]
    if row.empty:
        return default
    return row.iloc[0]

def metric_card(label: str, value: str, color_class: str):
    st.markdown(
        f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def callout(title: str, text: str):
    st.markdown(
        f"""
        <div class="callout-box">
            <div class="callout-title">{title}</div>
            <div class="callout-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def short_model_name(name: str) -> str:
    mapping = {
        "hist_gradient_boosting": "HGB",
        "xgboost": "XGBoost",
        "elasticnet": "ElasticNet",
        "random_forest": "RF",
        "extra_trees": "ExtraTrees",
    }
    return mapping.get(name, name)

# -------------------------------------------------
# Run selection
# -------------------------------------------------
run_dir = Path("artifacts/runs")
available_runs = sorted([p.name for p in run_dir.glob("run_*")], reverse=True)

if not available_runs:
    st.warning("No runs found.")
    st.stop()

st.markdown("""
<div class="hero-box">
    <div class="main-title">Retail Decision Support System</div>
    <div class="sub-title">Forecasting, business actions, monitoring, and retraining review in one interface</div>
</div>
""", unsafe_allow_html=True)

selected_run = st.selectbox("Select run", available_runs)
base = run_dir / selected_run

# -------------------------------------------------
# Load outputs
# -------------------------------------------------
exec_df = safe_read_csv(base / "dashboard" / "dashboard_executive_summary.csv")
model_df = safe_read_csv(base / "selection" / "model_comparison.csv")
store_df = safe_read_csv(base / "dashboard" / "dashboard_store_forecast.csv")
dept_df = safe_read_csv(base / "dashboard" / "dashboard_department_forecast.csv")
promo_df = safe_read_csv(base / "promotion" / "promotion_analytics.csv")
inv_df = safe_read_csv(base / "inventory" / "inventory_recommendations.csv")
site_df = safe_read_csv(base / "optimization" / "optimized_site_selection.csv")
drift_df = safe_read_csv(base / "monitoring" / "drift_monitor.csv")
audit_df = safe_read_csv(base / "retraining" / "retraining_audit.csv")

# -------------------------------------------------
# Sidebar slicers
# -------------------------------------------------
st.sidebar.header("Filters")

top_n = st.sidebar.slider("Top N rows", min_value=5, max_value=50, value=10, step=5)

selected_sites_only = st.sidebar.checkbox("Selected sites only", value=False)

region_options = []
if site_df is not None and "region" in site_df.columns:
    region_options = sorted(site_df["region"].dropna().astype(str).unique().tolist())

region_filter = st.sidebar.multiselect("Region", region_options, default=region_options)

store_options = []
if store_df is not None and "store_id" in store_df.columns:
    store_options = sorted(store_df["store_id"].dropna().tolist())

store_filter = st.sidebar.multiselect("Store ID", store_options, default=[])

dept_options = []
if dept_df is not None and "department" in dept_df.columns:
    dept_options = sorted(dept_df["department"].dropna().astype(str).unique().tolist())

dept_filter = st.sidebar.multiselect("Department", dept_options, default=dept_options)

# -------------------------------------------------
# Apply filters
# -------------------------------------------------
if site_df is not None and not site_df.empty:
    if region_filter:
        site_df = site_df[site_df["region"].astype(str).isin(region_filter)]
    if selected_sites_only and "selected_flag" in site_df.columns:
        site_df = site_df[site_df["selected_flag"] == 1]

if store_df is not None and not store_df.empty and store_filter:
    store_df = store_df[store_df["store_id"].isin(store_filter)]

if dept_df is not None and not dept_df.empty and dept_filter:
    dept_df = dept_df[dept_df["department"].astype(str).isin(dept_filter)]

if inv_df is not None and not inv_df.empty and store_filter and "store_id" in inv_df.columns:
    inv_df = inv_df[inv_df["store_id"].isin(store_filter)]

# -------------------------------------------------
# Extract summary values
# -------------------------------------------------
champion_model = str(get_metric(exec_df, "champion_model", "N/A"))
challenger_model = str(get_metric(exec_df, "challenger_model", "N/A"))
champion_mae = get_metric(exec_df, "champion_mae")
champion_wmape = get_metric(exec_df, "champion_wmape")
watch_status = str(get_metric(exec_df, "watch_status", "N/A"))
retrain_flag = str(get_metric(exec_df, "retraining_recommended", "N/A"))

champion_display = short_model_name(champion_model)
challenger_display = short_model_name(challenger_model)

try:
    champion_mae_display = f"{float(champion_mae):.3f}"
except Exception:
    champion_mae_display = str(champion_mae)

try:
    champion_wmape_display = f"{float(champion_wmape):.3f}"
except Exception:
    champion_wmape_display = str(champion_wmape)

promo_lift = None
promo_margin_lift = None
if promo_df is not None and not promo_df.empty:
    try:
        promo_lift = float(promo_df.loc[promo_df["metric"] == "promo_lift", "value"].iloc[0])
    except Exception:
        pass
    try:
        promo_margin_lift = float(promo_df.loc[promo_df["metric"] == "promo_margin_lift", "value"].iloc[0])
    except Exception:
        pass

selected_sites = None
if site_df is not None and not site_df.empty and "selected_flag" in site_df.columns:
    selected_sites = int(site_df["selected_flag"].sum())

top_reorders = None
if inv_df is not None and not inv_df.empty and "recommended_reorder_qty" in inv_df.columns:
    top_reorders = int((inv_df["recommended_reorder_qty"] > 0).sum())

drifted_features = None
if drift_df is not None and not drift_df.empty and "drift" in drift_df.columns:
    drifted_features = int(drift_df["drift"].sum())

top_site_region = "N/A"
top_site_id = "N/A"
if site_df is not None and not site_df.empty:
    ranked_sites = site_df.sort_values("projected_value_index", ascending=False)
    if "region" in ranked_sites.columns:
        top_site_region = str(ranked_sites.iloc[0]["region"])
    if "candidate_site_id" in ranked_sites.columns:
        top_site_id = str(ranked_sites.iloc[0]["candidate_site_id"])

# -------------------------------------------------
# Executive snapshot
# -------------------------------------------------
st.markdown('<div class="section-title">Executive Snapshot</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Champion", champion_display, "blue-card")
with c2:
    metric_card("Challenger", challenger_display, "purple-card")
with c3:
    metric_card("MAE", champion_mae_display, "green-card")
with c4:
    metric_card("WMAPE", champion_wmape_display, "amber-card")
with c5:
    metric_card("Watch Status", watch_status, "slate-card")
with c6:
    metric_card("Retrain?", retrain_flag, "rose-card")

if str(retrain_flag) == "1":
    st.markdown('<div class="status-alert">Retraining review recommended.</div>', unsafe_allow_html=True)
elif str(watch_status).lower() == "watch":
    st.markdown('<div class="status-watch">System is in WATCH mode. Monitor before retraining.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-good">System stable. No immediate retraining action flagged.</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Decision callouts
# -------------------------------------------------
st.markdown('<div class="section-title">Decision Callouts</div>', unsafe_allow_html=True)
dc1, dc2 = st.columns(2)

with dc1:
    callout(
        "Forecast Decision",
        f"{champion_display} remains the current champion, with {challenger_display} retained as challenger for future review."
    )
    callout(
        "Inventory Decision",
        f"{top_reorders if top_reorders is not None else 'N/A'} reorder lines are currently flagged from forecast-to-stock translation."
    )

with dc2:
    callout(
        "Site Decision",
        f"{selected_sites if selected_sites is not None else 'N/A'} sites are selected under the current optimization constraint. Highest projected value site: {top_site_id} in {top_site_region}."
    )
    callout(
        "Monitoring Decision",
        f"{drifted_features if drifted_features is not None else 'N/A'} monitored features drifted; current system status remains {watch_status}."
    )

st.markdown('<div class="section-title">Monitoring Snapshot</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <span class="mini-badge badge-blue">Drifted Features: {drifted_features if drifted_features is not None else 'N/A'}</span>
    <span class="mini-badge badge-amber">Status: {watch_status}</span>
    <span class="mini-badge {'badge-rose' if str(retrain_flag) == '1' else 'badge-green'}">Retraining: {'Yes' if str(retrain_flag) == '1' else 'No'}</span>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    with st.expander("Show raw executive summary table"):
        if exec_df is not None:
            st.dataframe(exec_df, use_container_width=True, height=240)
        else:
            st.info("Executive summary not found.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    if model_df is not None and not model_df.empty:
        model_chart = model_df.sort_values("mae").copy()
        colors = []
        for m in model_chart["model_name"]:
            if m == champion_model:
                colors.append("#2563eb")
            elif m == challenger_model:
                colors.append("#7c3aed")
            else:
                colors.append("#94a3b8")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar([short_model_name(x) for x in model_chart["model_name"]], model_chart["mae"], color=colors)
        ax.set_title("Model MAE Comparison")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Model comparison not found.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Business Decision Snapshot</div>', unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)

with b1:
    metric_card("Promo Lift", f"{promo_lift:.2%}" if promo_lift is not None else "N/A", "green-card")
with b2:
    metric_card("Promo Margin Lift", f"{promo_margin_lift:.2%}" if promo_margin_lift is not None else "N/A", "blue-card")
with b3:
    metric_card("Selected Sites", str(selected_sites) if selected_sites is not None else "N/A", "purple-card")
with b4:
    metric_card("Reorder Lines", str(top_reorders) if top_reorders is not None else "N/A", "amber-card")

st.markdown('<div class="section-title">Forecast Views</div>', unsafe_allow_html=True)
f1, f2 = st.columns(2)

with f1:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("**Department Forecast Chart**")
    if dept_df is not None and not dept_df.empty:
        dept_chart = dept_df.sort_values("forecast_units", ascending=False).head(top_n).copy()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(dept_chart["department"], dept_chart["forecast_units"])
        ax.set_title("Forecast Units by Department")
        ax.set_ylabel("Forecast Units")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Department forecast not found.")
    st.markdown('</div>', unsafe_allow_html=True)

with f2:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("**Top Store Errors**")
    if store_df is not None and not store_df.empty:
        show_cols = [c for c in ["store_id", "forecast_units", "actual_units", "forecast_error", "abs_error"] if c in store_df.columns]
        top_err = store_df[show_cols].sort_values("abs_error", ascending=False).head(top_n)
        st.dataframe(top_err, use_container_width=True, height=320)
    else:
        st.info("Store forecast not found.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Inventory Recommendations</div>', unsafe_allow_html=True)
i1, i2 = st.columns(2)

with i1:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("**Top Reorder Quantities**")
    if inv_df is not None and not inv_df.empty:
        chart_df = inv_df.sort_values("recommended_reorder_qty", ascending=False).head(top_n).copy()
        chart_df["label"] = chart_df["store_id"].astype(str) + "-" + chart_df["sku_id"].astype(str)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(chart_df["label"], chart_df["recommended_reorder_qty"])
        ax.set_title("Top Reorder Lines")
        ax.set_ylabel("Recommended Reorder Qty")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Inventory recommendations not found.")
    st.markdown('</div>', unsafe_allow_html=True)

with i2:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("**Top Inventory Table**")
    if inv_df is not None and not inv_df.empty:
        show_cols = [c for c in ["store_id", "sku_id", "forecast_units", "on_hand_units", "safety_stock", "recommended_reorder_qty"] if c in inv_df.columns]
        st.dataframe(
            inv_df[show_cols].sort_values("recommended_reorder_qty", ascending=False).head(top_n),
            use_container_width=True,
            height=320
        )
    else:
        st.info("Inventory recommendations not found.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Optimized Site Selection</div>', unsafe_allow_html=True)
s1, s2 = st.columns(2)

with s1:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("**Top Site Scores**")
    if site_df is not None and not site_df.empty:
        chart_df = site_df.sort_values("projected_value_index", ascending=False).head(top_n).copy()
        chart_df["label"] = chart_df["candidate_site_id"].astype(str)
        colors = ["#7c3aed" if int(flag) == 1 else "#cbd5e1" for flag in chart_df["selected_flag"]]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(chart_df["label"], chart_df["projected_value_index"], color=colors)
        ax.set_title("Projected Value Index by Site")
        ax.set_ylabel("Projected Value Index")
        ax.set_xlabel("Candidate Site ID")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Optimized site selection not found.")
    st.markdown('</div>', unsafe_allow_html=True)

with s2:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("**Selected Sites Table**")
    if site_df is not None and not site_df.empty:
        show_cols = [c for c in ["candidate_site_id", "region", "projected_value_index", "rent_cost", "selected_flag"] if c in site_df.columns]
        st.dataframe(
            site_df[show_cols].sort_values("projected_value_index", ascending=False).head(top_n),
            use_container_width=True,
            height=320
        )
    else:
        st.info("Optimized site selection not found.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Monitoring and Retraining</div>', unsafe_allow_html=True)
m1, m2 = st.columns(2)

with m1:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("**Drift Monitor**")
    if drift_df is not None:
        st.dataframe(drift_df.sort_values("ks_stat", ascending=False), use_container_width=True, height=260)
    else:
        st.info("Drift monitor not found.")
    st.markdown('</div>', unsafe_allow_html=True)

with m2:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("**Retraining Audit**")
    if audit_df is not None:
        st.dataframe(audit_df, use_container_width=True, height=260)
    else:
        st.info("Retraining audit not found.")
    st.markdown('</div>', unsafe_allow_html=True)