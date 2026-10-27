from __future__ import annotations

import ast
from pathlib import Path

APP_FILE = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _load_format_kpi():
    # Importing the app would run the Streamlit page, so compile only this helper.
    tree = ast.parse(APP_FILE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "format_kpi")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP_FILE), "exec"), namespace)
    return namespace["format_kpi"]


def test_format_kpi_scales_thousands_and_millions() -> None:
    format_kpi = _load_format_kpi()

    assert format_kpi(1129.0) == "1.1K"
    assert format_kpi(374607.0) == "374.6K"
    assert format_kpi(320010.8) == "320K"
    assert format_kpi(2_500_000.0) == "2.5M"


def test_format_kpi_leaves_small_values_unscaled() -> None:
    format_kpi = _load_format_kpi()

    assert format_kpi(12.0) == "12"
    assert format_kpi(2.85) == "2.85"
