from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEMO_DIR = ARTIFACTS_DIR / "demo"
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
    .rose-card { background: linear-gradient(135deg, #e11d48, #be123c); }

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

    .explainer-frame {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 0.8rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.8rem;
    }

    .small-note {
        color: #475569;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def find_csv_files() -> List[Path]:
    if not DEMO_DIR.exists():
        return []

    csvs: List[Path] = []
    for path in DEMO_DIR.rglob("*.csv"):
        # skip raw source files for the public demo
        if "source" in path.parts:
            continue
        csvs.append(path)

    return sorted(csvs, key=lambda p: (p.parent.as_posix(), p.name.lower()))


@st.cache_data(show_spinner=False)
def load_all_csvs() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for path in find_csv_files():
        try:
            out[str(path.relative_to(PROJECT_ROOT))] = pd.read_csv(path)
        except Exception:
            continue
    return out


def safe_pick_table(
    tables: Dict[str, pd.DataFrame], keywords: List[str]
) -> Optional[Tuple[str, pd.DataFrame]]:
    for name, df in tables.items():
        lower_name = name.lower()
        if all(k.lower() in lower_name for k in keywords):
            return name, df
    for name, df in tables.items():
        lower_name = name.lower()
        if any(k.lower() in lower_name for k in keywords):
            return name, df
    return None


def numeric_summary_card(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df.empty:
        return out

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
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

    if not out and numeric_cols:
        first_col = numeric_cols[0]
        out[f"{first_col}_sum"] = float(
            pd.to_numeric(df[first_col], errors="coerce").fillna(0).sum()
        )
    return out


def detect_model_table(
    tables: Dict[str, pd.DataFrame],
) -> Optional[Tuple[str, pd.DataFrame]]:
    candidates = [
        ["model"],
        ["comparison"],
        ["metrics"],
        ["forecast", "model"],
    ]
    for keys in candidates:
        hit = safe_pick_table(tables, keys)
        if hit is not None:
            return hit

    for name, df in tables.items():
        cols = {c.lower() for c in df.columns}
        if {"model_name", "mae", "rmse"}.issubset(cols) or {
            "model",
            "mae",
            "rmse",
        }.issubset(cols):
            return name, df
    return None


def detect_inventory_table(
    tables: Dict[str, pd.DataFrame],
) -> Optional[Tuple[str, pd.DataFrame]]:
    candidates = [
        ["inventory"],
        ["reorder"],
        ["stock"],
    ]
    for keys in candidates:
        hit = safe_pick_table(tables, keys)
        if hit is not None:
            return hit

    for name, df in tables.items():
        cols = {c.lower() for c in df.columns}
        if "recommended_reorder_qty" in cols:
            return name, df
    return None


def detect_exec_summary_table(
    tables: Dict[str, pd.DataFrame],
) -> Optional[Tuple[str, pd.DataFrame]]:
    candidates = [
        ["executive", "summary"],
        ["summary"],
        ["dashboard"],
    ]
    for keys in candidates:
        hit = safe_pick_table(tables, keys)
        if hit is not None:
            return hit
    return None


def format_kpi(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value:,.1f}M".replace(".0M", "M")
    if abs(value) >= 1_000:
        return f"{value:,.1f}K".replace(".0K", "K")
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def status_box(text: str, status: str = "good") -> None:
    css_class = {
        "good": "status-good",
        "watch": "status-watch",
        "alert": "status-alert",
    }.get(status, "status-good")
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

    # Normalize while preserving the original object for loading
    if isinstance(explainers, dict):
        for key, value in explainers.items():
            if isinstance(value, dict):
                normalized.append(
                    {
                        "key": str(value.get("key", key)),
                        "title": str(value.get("title", key)),
                        "obj": value,
                    }
                )
            else:
                normalized.append(
                    {
                        "key": str(key),
                        "title": str(getattr(value, "title", key)),
                        "obj": value,
                    }
                )
    else:
        for item in explainers:
            if isinstance(item, str):
                normalized.append(
                    {
                        "key": item,
                        "title": item,
                        "obj": item,
                    }
                )
            elif isinstance(item, dict):
                normalized.append(
                    {
                        "key": str(item.get("key", "unknown")),
                        "title": str(item.get("title", item.get("key", "unknown"))),
                        "obj": item,
                    }
                )
            else:
                key = str(getattr(item, "key", str(item)))
                title = str(getattr(item, "title", key))
                normalized.append(
                    {
                        "key": key,
                        "title": title,
                        "obj": item,
                    }
                )

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
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a view",
    [
        "Overview",
        "Model Comparison",
        "Inventory & Actions",
        "Data Browser",
        "Explainers",
    ],
)

tables = load_all_csvs()
model_hit = detect_model_table(tables)
inventory_hit = detect_inventory_table(tables)
summary_hit = detect_exec_summary_table(tables)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project folders**")
st.sidebar.caption(f"Outputs: {OUTPUTS_DIR}")
st.sidebar.caption(f"Artifacts: {ARTIFACTS_DIR}")
st.sidebar.caption(f"Assets: {ASSETS_DIR}")

# -------------------------------------------------
# Overview
# -------------------------------------------------
if page == "Overview":
    st.markdown(
        "<div class='section-title'>Executive Overview</div>",
        unsafe_allow_html=True,
    )

    if not tables:
        status_box(
            "No CSV outputs were found yet. Run the demo pipeline first, then refresh the app.",
            "watch",
        )
        st.info("Try running the demo pipeline before launching the dashboard.")
    else:
        status_box(
            "Dashboard loaded successfully. Data assets were discovered and summarized.",
            "good",
        )

        kpi_cols = st.columns(4)
        kpi_payloads: List[Tuple[str, str, str]] = []

        if summary_hit is not None:
            _, summary_df = summary_hit
            summary_stats = numeric_summary_card(summary_df)
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

        if not kpi_payloads and inventory_hit is not None:
            _, inv_df = inventory_hit
            inv_stats = numeric_summary_card(inv_df)
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

        if not kpi_payloads and model_hit is not None:
            _, model_df = model_hit
            kpi_payloads.append(
                ("Model Rows", format_kpi(float(len(model_df))), "blue-card")
            )

        while len(kpi_payloads) < 4:
            kpi_payloads.append((f"Metric {len(kpi_payloads)+1}", "N/A", "slate-card"))

        for col, (label, value, css) in zip(kpi_cols, kpi_payloads):
            with col:
                render_kpi_card(label, value, css)

        left, right = st.columns([1.2, 1.8])

        with left:
            st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
            st.markdown("**Architecture highlights**")
            st.markdown(
                """
- Classical ML candidate models
- Deep tabular models: MLP and entity embeddings
- MCP-style schemas, tool registry, and controller
- Streamlit dashboard with explainers
- Demo-safe execution mode and smoke tests
"""
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
            st.markdown("**Discovered tables**")
            discovered = pd.DataFrame(
                {
                    "table": list(tables.keys()),
                    "rows": [len(df) for df in tables.values()],
                    "cols": [df.shape[1] for df in tables.values()],
                }
            )
            st.dataframe(discovered, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Model Comparison
# -------------------------------------------------
elif page == "Model Comparison":
    st.markdown(
        "<div class='section-title'>Model Comparison</div>",
        unsafe_allow_html=True,
    )

    if model_hit is None:
        status_box(
            "No model comparison table was detected in outputs/artifacts/data.",
            "watch",
        )
        st.info("Run the pipeline first or verify that model metrics CSV outputs exist.")
    else:
        model_name, model_df = model_hit
        st.caption(f"Source: {model_name}")

        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.dataframe(model_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        candidate_col = None
        for col in ["model_name", "model", "candidate"]:
            if col in model_df.columns:
                candidate_col = col
                break

        metric_col = None
        for col in ["rmse", "mae", "wmape", "bias", "r2"]:
            if col in model_df.columns:
                metric_col = col
                break

        if candidate_col and metric_col:
            make_bar_chart(
                model_df,
                candidate_col,
                metric_col,
                f"Top models by {metric_col.upper()}",
            )
        else:
            st.info("Model table found, but no standard plotting columns were detected.")

        st.markdown("<div class='callout-box'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='callout-title'>Interpretation</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='callout-text'>Use this section to compare classical ML and deep tabular candidates under the same metrics framework.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Inventory & Actions
# -------------------------------------------------
elif page == "Inventory & Actions":
    st.markdown(
        "<div class='section-title'>Inventory & Actions</div>",
        unsafe_allow_html=True,
    )

    if inventory_hit is None:
        status_box("No inventory or reorder table was detected.", "watch")
        st.info("Run the pipeline or verify that inventory outputs are available.")
    else:
        inventory_name, inventory_df = inventory_hit
        st.caption(f"Source: {inventory_name}")

        left, right = st.columns([1.2, 1.8])

        with left:
            stats = numeric_summary_card(inventory_df)
            st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
            st.markdown("**Inventory KPIs**")
            for key, val in stats.items():
                st.markdown(
                    f"- **{key.replace('_', ' ').title()}**: {format_kpi(val)}"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
            st.dataframe(
                inventory_df.head(50), use_container_width=True, hide_index=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if "recommended_reorder_qty" in inventory_df.columns:
            category_col = None
            for c in ["sku_id", "store_id", "product", "category"]:
                if c in inventory_df.columns:
                    category_col = c
                    break
            if category_col is not None:
                make_bar_chart(
                    inventory_df,
                    category_col,
                    "recommended_reorder_qty",
                    "Top reorder recommendations",
                )

# -------------------------------------------------
# Data Browser
# -------------------------------------------------
elif page == "Data Browser":
    st.markdown(
        "<div class='section-title'>Data Browser</div>",
        unsafe_allow_html=True,
    )

    if not tables:
        st.info("No CSV files found in outputs/, artifacts/, or data/.")
    else:
        selected_table = st.selectbox("Select a discovered table", list(tables.keys()))
        df = tables[selected_table]

        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.markdown(f"**Preview: {selected_table}**")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.markdown("**Column types**")
        dtype_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(t) for t in df.dtypes],
                "missing": [int(df[c].isna().sum()) for c in df.columns],
            }
        )
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Explainers
# -------------------------------------------------
elif page == "Explainers":
    render_explainers()

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "Retail Decision Support System • Classical ML + Deep Tabular Models • Lightweight MCP-style Orchestration • Explainable Dashboard"
)