# ================================================================
# GUARDRAIL BLOCK — DO NOT REMOVE
# ================================================================
# PURPOSE:
# Expand this app incrementally while preserving the full original
# retail decision support system.
#
# NON-NEGOTIABLE RULES:
# 1. Do NOT remove or simplify existing functionality.
# 2. Do NOT convert this into a toy demo.
# 3. Preserve all original datasets and business-facing sections.
# 4. Only make incremental additions or modifications.
# 5. Prefer add/extend over rewrite.
# 6. Do not delete sections unless explicitly instructed.
#
# ORIGINAL DATASETS THAT MUST REMAIN AVAILABLE:
# - dashboard_executive_summary.csv
# - dashboard_model_comparison.csv
# - dashboard_store_forecast.csv
# - dashboard_department_forecast.csv
# - dashboard_region_forecast.csv
# - dashboard_brand_forecast.csv
# - drift_monitor.csv
# - retraining_status.csv
# - retraining_audit.csv
# - inventory_recommendations.csv
# - optimized_site_selection.csv OR site_selection_rankings.csv
# - agent_answers.csv
# - store_watchlist.csv
# - dashboard_pipeline_maturity.csv
#
# STEP POLICY:
# Current phase: STEP 4A = page-aware slicers only.
# Refine filters only. Do NOT remove or rename any existing page,
# dataset, or section. Preserve original CSV source mode.
# ================================================================

from __future__ import annotations

# -------------------------------------------------
# ENFORCED GUARDRAILS — DO NOT REMOVE
# -------------------------------------------------
REQUIRED_DATASETS = [
    "dashboard_executive_summary.csv",
    "dashboard_model_comparison.csv",
    "dashboard_store_forecast.csv",
    "dashboard_department_forecast.csv",
    "dashboard_region_forecast.csv",
    "dashboard_brand_forecast.csv",
    "drift_monitor.csv",
    "retraining_status.csv",
    "retraining_audit.csv",
    "inventory_recommendations.csv",
    "agent_answers.csv",
    "store_watchlist.csv",
    "dashboard_pipeline_maturity.csv",
]

REQUIRED_SECTIONS = [
    "Overview",
    "Executive Summary",
    "Model Comparison",
    "Forecasts",
    "Inventory & Actions",
    "Monitoring",
    "Agent & Watchlist",
    "Pipeline Maturity",
    "Data Browser",
    "Explainers",
]

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from app.explainers import list_explainers, load_explainer_markup
except ModuleNotFoundError:
    from explainers import list_explainers, load_explainer_markup


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Retail Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets" / "animations"


# -------------------------------------------------
# Styling
# -------------------------------------------------
st.markdown(
    """
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

    .panel-box {
        background: white;
        border-radius: 18px;
        padding: 1rem;
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
    .explainer-frame {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 0.8rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.8rem;
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
    .small-note {
        color: #475569;
        font-size: 0.9rem;
    }
    .filter-chip-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }
    .filter-chip-title {
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.45rem;
    }
    .filter-chip {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 0.35rem 0.6rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-bottom: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def locate_named_csv(filename: str) -> Optional[Path]:
    direct_candidates = [
        PROJECT_ROOT / filename,
        APP_DIR / filename,
        OUTPUTS_DIR / filename,
        DATA_DIR / filename,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    return None


def locate_any_csv(candidates: List[str]) -> Optional[Path]:
    for filename in candidates:
        path = locate_named_csv(filename)
        if path is not None:
            return path
    return None


@st.cache_data(show_spinner=False)
def load_named_csv(filename: str) -> pd.DataFrame:
    path = locate_named_csv(filename)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_any_csv(candidates: List[str]) -> pd.DataFrame:
    path = locate_any_csv(candidates)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def check_required_datasets() -> pd.DataFrame:
    records = []
    for filename in REQUIRED_DATASETS:
        path = locate_named_csv(filename)
        records.append(
            {
                "dataset": filename,
                "found": path is not None,
                "path": str(path) if path is not None else "",
            }
        )
    return pd.DataFrame(records)


def enforce_required_sections(nav_options: List[str]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in nav_options]
    if missing:
        st.error(
            "Guardrail violation: required sections missing from navigation: "
            + ", ".join(missing)
        )
        st.stop()


def enforce_required_datasets(strict: bool = False) -> pd.DataFrame:
    status_df = check_required_datasets()
    missing = status_df.loc[~status_df["found"], "dataset"].tolist()
    if missing and strict:
        st.error(
            "Guardrail violation: required datasets missing: " + ", ".join(missing)
        )
        st.stop()
    return status_df


def numeric_summary_card(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df.empty:
        return out
    out["rows"] = float(len(df))
    out["cols"] = float(df.shape[1])

    if "forecast_units" in df.columns:
        out["forecast_units_sum"] = float(
            pd.to_numeric(df["forecast_units"], errors="coerce").fillna(0).sum()
        )
    if "recommended_reorder_qty" in df.columns:
        out["reorder_sum"] = float(
            pd.to_numeric(df["recommended_reorder_qty"], errors="coerce").fillna(0).sum()
        )
    if "store_id" in df.columns:
        out["stores"] = float(df["store_id"].nunique())
    if "sku_id" in df.columns:
        out["skus"] = float(df["sku_id"].nunique())
    return out


def build_filter_options(frames: List[pd.DataFrame]) -> Dict[str, List[str]]:
    options = {
        "region": [],
        "store_id": [],
        "department": [],
        "category": [],
        "sku_id": [],
        "brand": [],
    }

    for df in frames:
        if df is None or df.empty:
            continue
        for col in options.keys():
            if col in df.columns:
                vals = df[col].dropna().astype(str).unique().tolist()
                options[col].extend(vals)

    for col in options:
        options[col] = ["All"] + sorted(set(options[col]))
    return options


def apply_filters(
    df: pd.DataFrame,
    region_filter: str,
    store_filter: str,
    department_filter: str,
    category_filter: str,
    sku_filter: str,
    brand_filter: str,
) -> pd.DataFrame:
    out = df.copy()

    if region_filter != "All" and "region" in out.columns:
        out = out[out["region"].astype(str) == region_filter]
    if store_filter != "All" and "store_id" in out.columns:
        out = out[out["store_id"].astype(str) == store_filter]
    if department_filter != "All" and "department" in out.columns:
        out = out[out["department"].astype(str) == department_filter]
    if category_filter != "All" and "category" in out.columns:
        out = out[out["category"].astype(str) == category_filter]
    if sku_filter != "All" and "sku_id" in out.columns:
        out = out[out["sku_id"].astype(str) == sku_filter]
    if brand_filter != "All" and "brand" in out.columns:
        out = out[out["brand"].astype(str) == brand_filter]

    return out


def format_kpi(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value:,.1f}M".replace(".0M", "M")
    if abs(value) >= 1_000:
        return f"{value:,.1f}K".replace(".0K", "K")
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def status_box(text: str, status: str = "good") -> None:
    css_class = "status-good" if status == "good" else "status-watch"
    st.markdown(f"<div class='{css_class}'>{text}</div>", unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, css_class: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card {css_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_bar_chart(
    df: pd.DataFrame, category_col: str, value_col: str, title: str
) -> None:
    if category_col not in df.columns or value_col not in df.columns:
        st.info("Required chart columns are not available.")
        return

    plot_df = df[[category_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(value_col, ascending=False).head(10)

    if plot_df.empty:
        st.info("No plottable data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plot_df[category_col].astype(str), plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)


def render_dataframe_panel(
    title: str,
    df: pd.DataFrame,
    sort_col: Optional[str] = None,
    ascending: bool = False,
) -> None:
    st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
    st.markdown(f"**{title}**")
    if df.empty:
        st.info(f"No data available for {title}.")
    else:
        if sort_col and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=ascending)
        st.dataframe(df, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_empty_state(title: str, filters: Dict[str, str]) -> None:
    if filters:
        applied = ", ".join([f"{k}={v}" for k, v in filters.items()])
        st.info(f"No {title.lower()} rows match the current filters: {applied}.")
    else:
        st.info(f"No data available for {title}.")


def render_top_chart(
    df: pd.DataFrame,
    category_candidates: List[str],
    value_col: str,
    title: str,
    top_n: int = 10,
) -> None:
    if df.empty or value_col not in df.columns:
        st.info(f"No chart data available for {title}.")
        return

    category_col = None
    for col in category_candidates:
        if col in df.columns:
            category_col = col
            break

    if category_col is None:
        st.info(f"No valid category column available for {title}.")
        return

    plot_df = df[[category_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(value_col, ascending=False).head(top_n)

    if plot_df.empty:
        st.info(f"No plottable rows available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plot_df[category_col].astype(str), plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

def render_kpi_row(items: List[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, css_class) in zip(cols, items):
        with col:
            render_kpi_card(label, value, css_class)

def render_decision_summary(title: str, bullets: List[str]) -> None:
    if not bullets:
        return
    items = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(
        f"""
        <div class="panel-box">
            <div class="filter-chip-title">{title}</div>
            <ul style="margin-top: 0.4rem; padding-left: 1.2rem;">
                {items}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def render_download_button(label: str, df: pd.DataFrame, filename: str) -> None:
    if df is None or df.empty:
        return
    st.download_button(
        label=label,
        data=to_csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )
   
def render_decision_narrative(title: str, paragraphs: List[str]) -> None:
    if not paragraphs:
        return
    body = "".join([f"<p style='margin-bottom: 0.55rem;'>{p}</p>" for p in paragraphs])
    st.markdown(
        f"""
        <div class="panel-box">
            <div class="filter-chip-title">{title}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def narrative_model_page(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["No model evidence is available under the current filters."]

    lines = [f"The current filtered view contains {len(df)} model rows."]
    if "mae" in df.columns:
        tmp = df.copy()
        tmp["mae_num"] = pd.to_numeric(tmp["mae"], errors="coerce")
        tmp = tmp.dropna(subset=["mae_num"])
        if not tmp.empty:
            best = tmp.sort_values("mae_num").iloc[0]
            best_name = str(best["model_name"]) if "model_name" in tmp.columns else "the best-ranked model"
            lines.append(
                f"{best_name} currently provides the lowest MAE at {format_kpi(float(best['mae_num']))}."
            )
    return lines


def narrative_forecast_page(
    store_df: pd.DataFrame,
    dept_df: pd.DataFrame,
    region_df: pd.DataFrame,
    brand_df: pd.DataFrame,
) -> List[str]:
    lines: List[str] = []

    def top_line(df: pd.DataFrame, label_col: str, value_col: str, label_name: str) -> Optional[str]:
        if df.empty or label_col not in df.columns or value_col not in df.columns:
            return None
        tmp = df[[label_col, value_col]].copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna().sort_values(value_col, ascending=False)
        if tmp.empty:
            return None
        return f"The highest {label_name} forecast is {tmp.iloc[0][label_col]} with {format_kpi(float(tmp.iloc[0][value_col]))} units."

    for line in [
        top_line(store_df, "store_id", "forecast_units", "store"),
        top_line(region_df, "region", "forecast_units", "region"),
        top_line(
            dept_df,
            "department" if "department" in dept_df.columns else "category",
            "forecast_units",
            "department",
        ),
        top_line(brand_df, "brand", "forecast_units", "brand"),
    ]:
        if line:
            lines.append(line)

    if not lines:
        lines.append("No forecast narrative can be generated under the current filters.")
    return lines


def narrative_inventory_page(reorder_df: pd.DataFrame, site_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []

    if not reorder_df.empty and "recommended_reorder_qty" in reorder_df.columns:
        tmp = reorder_df.copy()
        tmp["recommended_reorder_qty"] = pd.to_numeric(tmp["recommended_reorder_qty"], errors="coerce")
        tmp = tmp.dropna(subset=["recommended_reorder_qty"]).sort_values("recommended_reorder_qty", ascending=False)
        if not tmp.empty:
            label_col = "sku_id" if "sku_id" in tmp.columns else ("store_id" if "store_id" in tmp.columns else None)
            if label_col:
                lines.append(
                    f"The top reorder priority is {tmp.iloc[0][label_col]} with {format_kpi(float(tmp.iloc[0]['recommended_reorder_qty']))} recommended units."
                )

    if not site_df.empty and "projected_value_index" in site_df.columns:
        tmp = site_df.copy()
        tmp["projected_value_index"] = pd.to_numeric(tmp["projected_value_index"], errors="coerce")
        tmp = tmp.dropna(subset=["projected_value_index"]).sort_values("projected_value_index", ascending=False)
        if not tmp.empty:
            label_col = "site_id" if "site_id" in tmp.columns else ("store_id" if "store_id" in tmp.columns else ("region" if "region" in tmp.columns else None))
            if label_col:
                lines.append(
                    f"The strongest site opportunity is {tmp.iloc[0][label_col]} with projected value {format_kpi(float(tmp.iloc[0]['projected_value_index']))}."
                )

    if not lines:
        lines.append("No inventory or site-selection narrative can be generated under the current filters.")
    return lines


def narrative_monitoring_page(
    drift_df: pd.DataFrame,
    retrain_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> List[str]:
    lines: List[str] = []

    if not drift_df.empty:
        lines.append(f"Drift monitoring currently contains {len(drift_df)} rows in scope.")
        if "psi" in drift_df.columns:
            psi_series = pd.to_numeric(drift_df["psi"], errors="coerce").dropna()
            if not psi_series.empty:
                lines.append(f"The highest PSI visible in the filtered view is {format_kpi(float(psi_series.max()))}.")

    if not retrain_df.empty:
        lines.append(f"Retraining status currently contains {len(retrain_df)} rows.")

    if not audit_df.empty:
        lines.append(f"Retraining audit currently contains {len(audit_df)} rows.")

    if not lines:
        lines.append("No monitoring narrative can be generated under the current filters.")
    return lines
 
def summarize_model_page(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["No model rows available under the current filters."]

    bullets = [f"{len(df)} model rows are currently in scope."]
    if "mae" in df.columns:
        mae_series = pd.to_numeric(df["mae"], errors="coerce").dropna()
        if not mae_series.empty:
            bullets.append(f"Best observed MAE is {format_kpi(float(mae_series.min()))}.")
    if "model_name" in df.columns and "mae" in df.columns:
        tmp = df.copy()
        tmp["mae_num"] = pd.to_numeric(tmp["mae"], errors="coerce")
        tmp = tmp.dropna(subset=["mae_num"])
        if not tmp.empty:
            best_model = str(tmp.sort_values("mae_num").iloc[0]["model_name"])
            bullets.append(f"Top-ranked model under the current filters is {best_model}.")
    return bullets


def summarize_forecast_page(
    store_df: pd.DataFrame,
    dept_df: pd.DataFrame,
    region_df: pd.DataFrame,
    brand_df: pd.DataFrame,
) -> List[str]:
    bullets: List[str] = []

    def add_top(df: pd.DataFrame, label_col: str, value_col: str, label_name: str) -> None:
        if df.empty or label_col not in df.columns or value_col not in df.columns:
            return
        tmp = df[[label_col, value_col]].copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna().sort_values(value_col, ascending=False)
        if not tmp.empty:
            top_label = str(tmp.iloc[0][label_col])
            top_value = format_kpi(float(tmp.iloc[0][value_col]))
            bullets.append(f"Highest {label_name} forecast is {top_label} with {top_value} units.")

    add_top(store_df, "store_id", "forecast_units", "store")
    add_top(region_df, "region", "forecast_units", "region")
    add_top(dept_df, "department" if "department" in dept_df.columns else "category", "forecast_units", "department")
    add_top(brand_df, "brand", "forecast_units", "brand")

    if not bullets:
        bullets.append("No forecast rows are available under the current filters.")
    return bullets


def summarize_inventory_page(reorder_df: pd.DataFrame, site_df: pd.DataFrame) -> List[str]:
    bullets: List[str] = []

    if not reorder_df.empty and "recommended_reorder_qty" in reorder_df.columns:
        tmp = reorder_df.copy()
        tmp["recommended_reorder_qty"] = pd.to_numeric(tmp["recommended_reorder_qty"], errors="coerce")
        tmp = tmp.dropna(subset=["recommended_reorder_qty"]).sort_values("recommended_reorder_qty", ascending=False)
        if not tmp.empty:
            label_col = "sku_id" if "sku_id" in tmp.columns else ("store_id" if "store_id" in tmp.columns else None)
            if label_col is not None:
                bullets.append(
                    f"Top reorder priority is {tmp.iloc[0][label_col]} with {format_kpi(float(tmp.iloc[0]['recommended_reorder_qty']))} recommended units."
                )

    if not site_df.empty:
        sort_col = "projected_value_index" if "projected_value_index" in site_df.columns else None
        if sort_col is not None:
            tmp = site_df.copy()
            tmp[sort_col] = pd.to_numeric(tmp[sort_col], errors="coerce")
            tmp = tmp.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False)
            if not tmp.empty:
                label_col = "site_id" if "site_id" in tmp.columns else ("store_id" if "store_id" in tmp.columns else ("region" if "region" in tmp.columns else None))
                if label_col is not None:
                    bullets.append(
                        f"Top site opportunity is {tmp.iloc[0][label_col]} with projected value {format_kpi(float(tmp.iloc[0][sort_col]))}."
                    )

    if not bullets:
        bullets.append("No inventory or site-selection rows are available under the current filters.")
    return bullets


def summarize_monitoring_page(
    drift_df: pd.DataFrame,
    retrain_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> List[str]:
    bullets: List[str] = []

    if not drift_df.empty:
        bullets.append(f"Drift monitor currently shows {len(drift_df)} rows in scope.")
        if "psi" in drift_df.columns:
            psi_series = pd.to_numeric(drift_df["psi"], errors="coerce").dropna()
            if not psi_series.empty:
                bullets.append(f"Highest PSI currently visible is {format_kpi(float(psi_series.max()))}.")

    if not retrain_df.empty:
        bullets.append(f"Retraining status currently has {len(retrain_df)} rows in scope.")

    if not audit_df.empty:
        bullets.append(f"Retraining audit currently has {len(audit_df)} rows in scope.")

    if not bullets:
        bullets.append("No monitoring rows are available under the current filters.")
    return bullets

def build_forecast_kpis(df: pd.DataFrame, label_prefix: str) -> List[tuple[str, str, str]]:
    if df.empty:
        return [
            (f"{label_prefix} Rows", "0", "slate-card"),
            (f"{label_prefix} Units", "0", "slate-card"),
        ]

    rows = format_kpi(float(len(df)))
    units = "0"
    if "forecast_units" in df.columns:
        units = format_kpi(float(pd.to_numeric(df["forecast_units"], errors="coerce").fillna(0).sum()))

    return [
        (f"{label_prefix} Rows", rows, "blue-card"),
        (f"{label_prefix} Units", units, "green-card"),
    ]


def build_monitoring_kpis(
    drift_df: pd.DataFrame,
    retrain_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> List[tuple[str, str, str]]:
    drift_rows = format_kpi(float(len(drift_df))) if not drift_df.empty else "0"
    retrain_rows = format_kpi(float(len(retrain_df))) if not retrain_df.empty else "0"
    audit_rows = format_kpi(float(len(audit_df))) if not audit_df.empty else "0"

    return [
        ("Drift Rows", drift_rows, "amber-card"),
        ("Retraining Rows", retrain_rows, "purple-card"),
        ("Audit Rows", audit_rows, "blue-card"),
    ]


def build_inventory_kpis(reorder_df: pd.DataFrame, site_df: pd.DataFrame) -> List[tuple[str, str, str]]:
    reorder_rows = format_kpi(float(len(reorder_df))) if not reorder_df.empty else "0"
    reorder_qty = "0"
    if not reorder_df.empty and "recommended_reorder_qty" in reorder_df.columns:
        reorder_qty = format_kpi(
            float(pd.to_numeric(reorder_df["recommended_reorder_qty"], errors="coerce").fillna(0).sum())
        )

    site_rows = format_kpi(float(len(site_df))) if not site_df.empty else "0"

    return [
        ("Reorder Rows", reorder_rows, "amber-card"),
        ("Reorder Qty", reorder_qty, "green-card"),
        ("Site Rows", site_rows, "blue-card"),
    ]

def render_empty_state(title: str, filters: Dict[str, str]) -> None:
    if filters:
        applied = ", ".join([f"{k}={v}" for k, v in filters.items()])
        st.info(f"No {title.lower()} rows match the current filters: {applied}.")
    else:
        st.info(f"No data available for {title}.")


def render_top_chart(
    df: pd.DataFrame,
    category_candidates: List[str],
    value_col: str,
    title: str,
    top_n: int = 10,
) -> None:
    if df.empty or value_col not in df.columns:
        st.info(f"No chart data available for {title}.")
        return

    category_col = None
    for col in category_candidates:
        if col in df.columns:
            category_col = col
            break

    if category_col is None:
        st.info(f"No valid category column available for {title}.")
        return

    plot_df = df[[category_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(value_col, ascending=False).head(top_n)

    if plot_df.empty:
        st.info(f"No plottable rows available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plot_df[category_col].astype(str), plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

def render_explainers() -> None:
    st.markdown(
        "<div class='section-title'>Model and System Explainers</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small-note'>Lightweight SVG explainers that describe the end-to-end system, model logic, and MCP-style orchestration.</div>",
        unsafe_allow_html=True,
    )

    explainers = list_explainers()
    if not explainers:
        st.warning("No explainer assets found.")
        return

    normalized = []
    if isinstance(explainers, dict):
        for key, value in explainers.items():
            normalized.append(
                {
                    "key": str(getattr(value, "key", key)),
                    "title": str(getattr(value, "title", key)),
                    "obj": value,
                }
            )
    else:
        for item in explainers:
            key = str(getattr(item, "key", str(item)))
            title = str(getattr(item, "title", key))
            normalized.append({"key": key, "title": title, "obj": item})

    keys = [x["key"] for x in normalized]
    titles = {x["key"]: x["title"] for x in normalized}
    objects = {x["key"]: x["obj"] for x in normalized}

    selected = st.selectbox(
        "Choose explainer",
        keys,
        index=0,
        format_func=lambda x: titles.get(x, x),
    )

    left, right = st.columns([1.3, 1.7])

    with left:
        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.markdown("**Available explainers**")
        for item in normalized:
            badge_class = "badge-blue" if item["key"] == selected else "badge-green"
            st.markdown(
                f"<span class='mini-badge {badge_class}'>{item['title']}</span>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='explainer-frame'>", unsafe_allow_html=True)
        try:
            markup = load_explainer_markup(objects[selected])
        except Exception as e:
            st.error(f"Failed to load explainer: {e}")
            markup = None

        if markup:
            st.markdown(markup, unsafe_allow_html=True)
        else:
            st.warning("Unable to load the selected explainer asset.")
        st.markdown("</div>", unsafe_allow_html=True)


def active_filter_dict(
    region_filter: str,
    store_filter: str,
    department_filter: str,
    category_filter: str,
    sku_filter: str,
    brand_filter: str,
) -> Dict[str, str]:
    filters = {
        "Region": region_filter,
        "Store": store_filter,
        "Department": department_filter,
        "Category": category_filter,
        "SKU": sku_filter,
        "Brand": brand_filter,
    }
    return {k: v for k, v in filters.items() if v != "All"}


def render_active_filters(filters: Dict[str, str]) -> None:
    if not filters:
        return
    chips = "".join(
        [f"<span class='filter-chip'>{k}: {v}</span>" for k, v in filters.items()]
    )
    st.markdown(
        f"""
        <div class="filter-chip-box">
            <div class="filter-chip-title">Active Filters</div>
            {chips}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_filter_frames(page_name: str) -> List[pd.DataFrame]:
    mapping = {
        "Overview": [
            exec_df,
            model_df,
            store_df,
            dept_df,
            region_df,
            brand_df,
            reorder_df,
            drift_df,
            retrain_df,
            retrain_audit_df,
            agent_df,
            watch_df,
            maturity_df,
            site_df,
        ],
        "Executive Summary": [exec_df],
        "Model Comparison": [model_df],
        "Forecasts": [store_df, dept_df, region_df, brand_df],
        "Inventory & Actions": [reorder_df, site_df],
        "Monitoring": [drift_df, retrain_df, retrain_audit_df],
        "Agent & Watchlist": [agent_df, watch_df],
        "Pipeline Maturity": [maturity_df],
        "Data Browser": [
            exec_df,
            model_df,
            store_df,
            dept_df,
            region_df,
            brand_df,
            reorder_df,
            drift_df,
            retrain_df,
            retrain_audit_df,
            agent_df,
            watch_df,
            maturity_df,
            site_df,
        ],
        "Explainers": [],
    }
    return mapping.get(page_name, [])


# -------------------------------------------------
# Single-source original CSV loading
# -------------------------------------------------
exec_df = load_named_csv("dashboard_executive_summary.csv")
model_df = load_named_csv("dashboard_model_comparison.csv")
store_df = load_named_csv("dashboard_store_forecast.csv")
dept_df = load_named_csv("dashboard_department_forecast.csv")
region_df = load_named_csv("dashboard_region_forecast.csv")
brand_df = load_named_csv("dashboard_brand_forecast.csv")
drift_df = load_named_csv("drift_monitor.csv")
retrain_df = load_named_csv("retraining_status.csv")
retrain_audit_df = load_named_csv("retraining_audit.csv")
reorder_df = load_named_csv("inventory_recommendations.csv")
site_df = load_any_csv(["optimized_site_selection.csv", "site_selection_rankings.csv"])
agent_df = load_named_csv("agent_answers.csv")
watch_df = load_named_csv("store_watchlist.csv")
maturity_df = load_named_csv("dashboard_pipeline_maturity.csv")

dataset_guardrail_status = enforce_required_datasets(strict=False)

browser_tables: Dict[str, pd.DataFrame] = {
    "dashboard_executive_summary.csv": exec_df,
    "dashboard_model_comparison.csv": model_df,
    "dashboard_store_forecast.csv": store_df,
    "dashboard_department_forecast.csv": dept_df,
    "dashboard_region_forecast.csv": region_df,
    "dashboard_brand_forecast.csv": brand_df,
    "drift_monitor.csv": drift_df,
    "retraining_status.csv": retrain_df,
    "retraining_audit.csv": retrain_audit_df,
    "inventory_recommendations.csv": reorder_df,
    "site_selection.csv": site_df,
    "agent_answers.csv": agent_df,
    "store_watchlist.csv": watch_df,
    "dashboard_pipeline_maturity.csv": maturity_df,
}


# -------------------------------------------------
# Hero
# -------------------------------------------------
st.markdown(
    """
<div class="hero-box">
    <div class="main-title">Retail Decision Support System</div>
    <div class="sub-title">
        End-to-end retail analytics with classical ML, deep tabular models, lightweight MCP-style orchestration,
        explainable visuals, and demo-safe execution.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
NAV_OPTIONS = [
    "Overview",
    "Executive Summary",
    "Model Comparison",
    "Forecasts",
    "Inventory & Actions",
    "Monitoring",
    "Agent & Watchlist",
    "Pipeline Maturity",
    "Data Browser",
    "Explainers",
]

enforce_required_sections(NAV_OPTIONS)

st.sidebar.title("Decision Modules")
page = st.sidebar.radio("Choose a view", NAV_OPTIONS)

page_frames = page_filter_frames(page)
page_filter_options = build_filter_options(page_frames)

st.sidebar.markdown("---")
st.sidebar.markdown("**Retail Filters**")
region_filter = st.sidebar.selectbox("Region", page_filter_options["region"], key=f"{page}_region")
store_filter = st.sidebar.selectbox("Store", page_filter_options["store_id"], key=f"{page}_store")
department_filter = st.sidebar.selectbox("Department", page_filter_options["department"], key=f"{page}_department")
category_filter = st.sidebar.selectbox("Category", page_filter_options["category"], key=f"{page}_category")
sku_filter = st.sidebar.selectbox("SKU", page_filter_options["sku_id"], key=f"{page}_sku")
brand_filter = st.sidebar.selectbox("Brand", page_filter_options["brand"], key=f"{page}_brand")

st.sidebar.markdown("---")
st.sidebar.markdown("**Source mode**")
st.sidebar.caption("Primary source: original flat CSV files")
st.sidebar.caption(f"Assets: {ASSETS_DIR}")

current_filters = active_filter_dict(
    region_filter,
    store_filter,
    department_filter,
    category_filter,
    sku_filter,
    brand_filter,
)

# Filtered frames
exec_df_f = apply_filters(exec_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
model_df_f = apply_filters(model_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
store_df_f = apply_filters(store_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
dept_df_f = apply_filters(dept_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
region_df_f = apply_filters(region_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
brand_df_f = apply_filters(brand_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
drift_df_f = apply_filters(drift_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
retrain_df_f = apply_filters(retrain_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
retrain_audit_df_f = apply_filters(retrain_audit_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
reorder_df_f = apply_filters(reorder_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
site_df_f = apply_filters(site_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
agent_df_f = apply_filters(agent_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
watch_df_f = apply_filters(watch_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)
maturity_df_f = apply_filters(maturity_df, region_filter, store_filter, department_filter, category_filter, sku_filter, brand_filter)


# -------------------------------------------------
# Pages
# -------------------------------------------------
if page == "Overview":
    st.markdown("<div class='section-title'>Executive Overview</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)

    missing_required = dataset_guardrail_status.loc[
        ~dataset_guardrail_status["found"], "dataset"
    ].tolist()

    if missing_required:
        status_box(
            "Required dataset guardrail warning: " + ", ".join(missing_required),
            "watch",
        )
    else:
        status_box(
            "Dashboard loaded successfully from the original CSV source.",
            "good",
        )

    kpi_cols = st.columns(4)
    kpi_payloads: List[tuple[str, str, str]] = []

    if not exec_df_f.empty:
        summary_stats = numeric_summary_card(exec_df_f)
        for idx, (k, v) in enumerate(summary_stats.items()):
            if idx >= 4:
                break
            kpi_payloads.append(
                (
                    k.replace("_", " ").title(),
                    format_kpi(v),
                    ["blue-card", "green-card", "amber-card", "purple-card"][idx],
                )
            )

    if not kpi_payloads and not reorder_df_f.empty:
        inv_stats = numeric_summary_card(reorder_df_f)
        for idx, (k, v) in enumerate(inv_stats.items()):
            if idx >= 4:
                break
            kpi_payloads.append(
                (
                    k.replace("_", " ").title(),
                    format_kpi(v),
                    ["blue-card", "green-card", "amber-card", "purple-card"][idx],
                )
            )

    if not kpi_payloads and not model_df_f.empty:
        kpi_payloads.append(("Model Rows", format_kpi(float(len(model_df_f))), "blue-card"))

    while len(kpi_payloads) < 4:
        kpi_payloads.append((f"Metric {len(kpi_payloads)+1}", "N/A", "slate-card"))

    for col, (label, value, css) in zip(kpi_cols, kpi_payloads):
        with col:
            render_kpi_card(label, value, css)

    left, right = st.columns([1.2, 1.8])

    with left:
        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.markdown("**System scope**")
        st.markdown(
            """
- Executive Summary
- Model Comparison
- Store / Department / Region / Brand Forecasts
- Inventory Recommendations
- Site Selection
- Drift Monitor
- Retraining Status + Audit
- Agent Answers
- Store Watchlist
- Pipeline Maturity
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        restoration_status = pd.DataFrame(
            {
                "dataset": list(browser_tables.keys()),
                "rows_loaded": [len(df) for df in browser_tables.values()],
            }
        )
        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.markdown("**Loaded source tables**")
        st.dataframe(restoration_status, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Enforced Guardrail Status"):
        st.dataframe(dataset_guardrail_status, width="stretch", hide_index=True)

elif page == "Executive Summary":
    st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)
    render_dataframe_panel("Executive Summary", exec_df_f)

elif page == "Model Comparison":
    st.markdown("<div class='section-title'>Model Comparison</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)

    model_rows = format_kpi(float(len(model_df_f))) if not model_df_f.empty else "0"
    best_mae = "N/A"
    if not model_df_f.empty and "mae" in model_df_f.columns:
        best_mae = format_kpi(float(pd.to_numeric(model_df_f["mae"], errors="coerce").min()))
        render_kpi_row([
        ("Model Rows", model_rows, "blue-card"),
        ("Best MAE", best_mae, "green-card"),
    ])

    render_decision_summary("Decision Summary", summarize_model_page(model_df_f))
    render_decision_narrative("Decision Narrative", narrative_model_page(model_df_f))
    render_download_button(
        "Download Model Comparison CSV",
        model_df_f,
        "model_comparison_filtered.csv",
    )
    render_dataframe_panel("Model Comparison", model_df_f, sort_col="mae", ascending=True)

elif page == "Forecasts":
    st.markdown("<div class='section-title'>Forecasts</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)

    render_kpi_row(build_forecast_kpis(store_df_f, "Store"))
    render_kpi_row(build_forecast_kpis(region_df_f, "Region"))
    render_kpi_row(build_forecast_kpis(dept_df_f, "Department"))
    render_kpi_row(build_forecast_kpis(brand_df_f, "Brand"))
    render_decision_summary(
        "Decision Summary",
        summarize_forecast_page(store_df_f, dept_df_f, region_df_f, brand_df_f),
    )
    render_decision_narrative(
        "Decision Narrative",
        narrative_forecast_page(store_df_f, dept_df_f, region_df_f, brand_df_f),
    )

    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        render_download_button("Download Store Forecast", store_df_f, "store_forecast_filtered.csv")
    with dl2:
        render_download_button("Download Department Forecast", dept_df_f, "department_forecast_filtered.csv")
    with dl3:
        render_download_button("Download Region Forecast", region_df_f, "region_forecast_filtered.csv")
    with dl4:
        render_download_button("Download Brand Forecast", brand_df_f, "brand_forecast_filtered.csv")

    col1, col2 = st.columns(2)

    with col1:
        render_dataframe_panel("Store Forecast", store_df_f, sort_col="forecast_units", ascending=False)
        render_top_chart(
            store_df_f,
            ["store_id", "store", "site_id"],
            "forecast_units",
            "Top Store Forecasts",
        )

        render_dataframe_panel("Region Forecast", region_df_f, sort_col="forecast_units", ascending=False)
        render_top_chart(
            region_df_f,
            ["region"],
            "forecast_units",
            "Top Region Forecasts",
        )

    with col2:
        render_dataframe_panel("Department Forecast", dept_df_f, sort_col="forecast_units", ascending=False)
        render_top_chart(
            dept_df_f,
            ["department", "category"],
            "forecast_units",
            "Top Department Forecasts",
        )

        render_dataframe_panel("Brand Forecast", brand_df_f, sort_col="forecast_units", ascending=False)
        render_top_chart(
            brand_df_f,
            ["brand"],
            "forecast_units",
            "Top Brand Forecasts",
        )
        
elif page == "Inventory & Actions":
    st.markdown("<div class='section-title'>Inventory & Actions</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)
    render_kpi_row(build_inventory_kpis(reorder_df_f, site_df_f))
    render_decision_summary(
        "Decision Summary",
        summarize_inventory_page(reorder_df_f, site_df_f),
    )
    render_decision_narrative(
        "Decision Narrative",
        narrative_inventory_page(reorder_df_f, site_df_f),
    )

    dl1, dl2 = st.columns(2)
    with dl1:
        render_download_button(
            "Download Inventory Recommendations",
            reorder_df_f,
            "inventory_recommendations_filtered.csv",
        )
    with dl2:
        render_download_button(
            "Download Site Selection",
            site_df_f,
            "site_selection_filtered.csv",
        )

    if reorder_df_f.empty:
        render_empty_state("Inventory Recommendations", current_filters)
    else:
        render_dataframe_panel("Inventory Recommendations", reorder_df_f)
        render_top_chart(
            reorder_df_f,
            ["sku_id", "store_id", "category"],
            "recommended_reorder_qty",
            "Top Reorder Recommendations",
        )

    sort_col = "projected_value_index" if "projected_value_index" in site_df_f.columns else None

    if site_df_f.empty:
        render_empty_state("Optimized Site Selection", current_filters)
    else:
        render_dataframe_panel("Optimized Site Selection", site_df_f, sort_col=sort_col, ascending=False)
        if sort_col is not None:
            render_top_chart(
                site_df_f,
                ["site_id", "store_id", "region", "location"],
                "projected_value_index",
                "Top Site Selection Rankings",
            )

elif page == "Monitoring":
    st.markdown("<div class='section-title'>Monitoring</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)
    render_kpi_row(build_monitoring_kpis(drift_df_f, retrain_df_f, retrain_audit_df_f))
    render_decision_summary(
        "Decision Summary",
        summarize_monitoring_page(drift_df_f, retrain_df_f, retrain_audit_df_f),
    )
    render_decision_narrative(
        "Decision Narrative",
        narrative_monitoring_page(drift_df_f, retrain_df_f, retrain_audit_df_f),
    )

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        render_download_button("Download Drift Monitor", drift_df_f, "drift_monitor_filtered.csv")
    with dl2:
        render_download_button("Download Retraining Status", retrain_df_f, "retraining_status_filtered.csv")
    with dl3:
        render_download_button("Download Retraining Audit", retrain_audit_df_f, "retraining_audit_filtered.csv")

    render_dataframe_panel("Drift Monitor", drift_df_f)
    if not drift_df_f.empty:
        if "psi" in drift_df_f.columns:
            render_top_chart(
                drift_df_f,
                ["feature", "metric", "column"],
                "psi",
                "Top Drift Signals by PSI",
            )
        elif "drift_score" in drift_df_f.columns:
            render_top_chart(
                drift_df_f,
                ["feature", "metric", "column"],
                "drift_score",
                "Top Drift Signals",
            )

    render_dataframe_panel("Retraining Status", retrain_df_f)
    if not retrain_df_f.empty:
        status_cols = [c for c in ["status", "retrain_flag", "decision"] if c in retrain_df_f.columns]
        if status_cols:
            status_col = status_cols[0]
            counts = retrain_df_f[status_col].astype(str).value_counts().reset_index()
            counts.columns = [status_col, "count"]
            render_top_chart(
                counts,
                [status_col],
                "count",
                "Retraining Status Counts",
            )

    render_dataframe_panel("Retraining Audit", retrain_audit_df_f)

elif page == "Agent & Watchlist":
    st.markdown("<div class='section-title'>Agent & Watchlist</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)

    dl1, dl2 = st.columns(2)
    with dl1:
        render_download_button("Download Agent Answers", agent_df_f, "agent_answers_filtered.csv")
    with dl2:
        render_download_button("Download Store Watchlist", watch_df_f, "store_watchlist_filtered.csv")

    render_dataframe_panel("Agent Answers", agent_df_f)
    render_dataframe_panel("Store Watchlist", watch_df_f)

elif page == "Pipeline Maturity":
    st.markdown("<div class='section-title'>Pipeline Maturity</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)
    render_download_button(
        "Download Pipeline Maturity",
        maturity_df_f,
        "pipeline_maturity_filtered.csv",
    )
    render_dataframe_panel("Implementation Maturity", maturity_df_f)

elif page == "Data Browser":
    st.markdown("<div class='section-title'>Data Browser</div>", unsafe_allow_html=True)
    render_active_filters(current_filters)

    filtered_browser_tables: Dict[str, pd.DataFrame] = {
        "dashboard_executive_summary.csv": exec_df_f,
        "dashboard_model_comparison.csv": model_df_f,
        "dashboard_store_forecast.csv": store_df_f,
        "dashboard_department_forecast.csv": dept_df_f,
        "dashboard_region_forecast.csv": region_df_f,
        "dashboard_brand_forecast.csv": brand_df_f,
        "drift_monitor.csv": drift_df_f,
        "retraining_status.csv": retrain_df_f,
        "retraining_audit.csv": retrain_audit_df_f,
        "inventory_recommendations.csv": reorder_df_f,
        "site_selection.csv": site_df_f,
        "agent_answers.csv": agent_df_f,
        "store_watchlist.csv": watch_df_f,
        "dashboard_pipeline_maturity.csv": maturity_df_f,
    }

    selected_table = st.selectbox("Select a restored table", list(filtered_browser_tables.keys()))
    df = filtered_browser_tables[selected_table]

    render_dataframe_panel(f"Preview: {selected_table}", df)

    if not df.empty:
        dtype_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(t) for t in df.dtypes],
                "missing": [int(df[c].isna().sum()) for c in df.columns],
            }
        )
        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.markdown("**Column types**")
        st.dataframe(dtype_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Explainers":
    render_explainers()

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "Retail Decision Support System • Step 4A Page-Aware Filters • Original CSV Source Mode • Explainable Dashboard"
)