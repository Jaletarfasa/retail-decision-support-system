from __future__ import annotations

import pandas as pd

from src.forecasting import ModelResult
from src.schemas import ToolRequest
from src.tool_registry import get_tool_registry, invoke_tool, list_tool_specs


def test_tool_registry_exposes_requested_tools() -> None:
    registry = get_tool_registry()

    assert set(registry) >= {
        "run_forecast_pipeline",
        "compare_candidate_models",
        "get_inventory_actions",
        "get_drift_status",
        "generate_decision_summary",
    }
    assert {spec.name for spec in list_tool_specs()} >= set(registry)


def test_compare_candidate_models_tool_smoke() -> None:
    results = [
        ModelResult(
            model_name="mlp",
            model_obj=None,
            metrics={"mae": 1.0, "rmse": 1.2, "wmape": 0.1, "bias": 0.0, "r2": 0.8},
            preds=pd.Series([10.0, 11.0], name="forecast_units"),
        ),
        ModelResult(
            model_name="elasticnet",
            model_obj=None,
            metrics={"mae": 1.5, "rmse": 1.7, "wmape": 0.12, "bias": 0.1, "r2": 0.7},
            preds=pd.Series([9.5, 10.5], name="forecast_units"),
        ),
    ]

    response = invoke_tool(
        ToolRequest(
            tool_name="compare_candidate_models",
            payload={"model_results": results},
        )
    )

    assert response.status == "success"
    comparison = response.payload["comparison"]
    assert list(comparison["model_name"]) == ["mlp", "elasticnet"]
    assert response.payload["champion_model"] == "mlp"
    assert response.payload["challenger_model"] == "elasticnet"
