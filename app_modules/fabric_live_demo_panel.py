"""
Streamlit helper module for Fabric live demo evidence.

Add to your Streamlit app:

from app_modules.fabric_live_demo_panel import render_fabric_live_demo_panel

render_fabric_live_demo_panel(project_root=Path("."))

This module reads local evidence artifacts only.
It does not call Fabric directly from Streamlit.
"""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


FABRIC_ROOT = Path("artifacts") / "fabric_live"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read {path}: {exc}")
        return {}


def _latest_timestamp(df: pd.DataFrame, column: str = "captured_at_utc") -> str:
    if df.empty or column not in df.columns:
        return "not available"
    values = df[column].dropna().astype(str)
    if values.empty:
        return "not available"
    return sorted(values)[-1]


def _api_verified_count(df: pd.DataFrame) -> int:
    if df.empty or "evidence_status" not in df.columns:
        return 0
    return int((df["evidence_status"].astype(str) == "api_verified").sum())


def render_fabric_live_demo_panel(project_root: Path | str = Path(".")) -> None:
    project_root = Path(project_root)
    evidence_root = project_root / FABRIC_ROOT

    st.subheader("Fabric Live Demo Evidence")
    st.caption(
        "This page separates Fabric-ready architecture from live Fabric evidence. "
        "Live claims require API or screenshot evidence."
    )

    manifest = _read_json(evidence_root / "fabric_demo_manifest.json")
    workspaces = _read_csv(evidence_root / "fabric_workspace_inventory.csv")
    items = _read_csv(evidence_root / "fabric_item_inventory.csv")
    claims = _read_csv(evidence_root / "fabric_claim_boundary.csv")
    screenshots = _read_csv(evidence_root / "fabric_screenshot_register.csv")

    workspace_api_count = _api_verified_count(workspaces)
    item_api_count = _api_verified_count(items)
    screenshot_count = 0 if screenshots.empty else len(screenshots)

    c1, c2, c3 = st.columns(3)
    c1.metric("API-verified workspaces", workspace_api_count)
    c2.metric("API-verified items", item_api_count)
    c3.metric("Registered screenshots", screenshot_count)

    if workspace_api_count > 0 and item_api_count > 0:
        st.success("Fabric live-demo evidence is present.")
    elif workspace_api_count > 0:
        st.warning("Workspace evidence is present, but item inventory is missing or empty.")
    else:
        st.info("No live Fabric API evidence detected yet. Current status remains Fabric-ready or scaffolded.")

    st.markdown("#### Evidence manifest")
    if manifest:
        st.json(manifest)
    else:
        st.write("No Fabric demo manifest found.")

    st.markdown("#### Workspace inventory")
    if workspaces.empty:
        st.write("No workspace inventory found.")
    else:
        st.dataframe(workspaces, use_container_width=True)

    st.markdown("#### Item inventory")
    if items.empty:
        st.write("No item inventory found.")
    else:
        st.dataframe(items, use_container_width=True)

    st.markdown("#### Claim boundary")
    if claims.empty:
        st.write("No claim boundary file found.")
    else:
        st.dataframe(claims, use_container_width=True)

    st.markdown("#### Screenshot register")
    if screenshots.empty:
        st.write("No screenshot register entries found.")
    else:
        st.dataframe(screenshots, use_container_width=True)

    st.caption(f"Latest workspace evidence timestamp: {_latest_timestamp(workspaces)}")
