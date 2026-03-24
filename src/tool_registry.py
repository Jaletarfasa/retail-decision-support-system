from __future__ import annotations

from typing import Callable

import pandas as pd

from src.forecasting import ModelResult, fit_and_score_models
from src.inventory import build_inventory_recommendations
from src.model_selection import build_model_comparison, select_champion_and_challenger
from src.monitoring import run_drift_monitor
from src.reporting import build_executive_summary
from src.retraining import should_retrain
from src.schemas import (
    CompareCandidateModelsRequest,
    DecisionSummaryRequest,
    DriftStatusRequest,
    ForecastPipelineRequest,
    InventoryActionsRequest,
    ToolRequest,
    ToolResponse,
    ToolSpec,
    dataframe_to_records,
)


ToolHandler = Callable[[dict], dict]


def _serialize_model_result(result: ModelResult) -> dict:
    return {
        "model_name": result.model_name,
        "metrics": result.metrics,
        "preds": result.preds.tolist(),
    }


def _deserialize_model_result(data: ModelResult | dict) -> ModelResult:
    if isinstance(data, ModelResult):
        return data
    preds = data.get("preds", [])
    if not isinstance(preds, pd.Series):
        preds = pd.Series(preds, name="forecast_units")
    return ModelResult(
        model_name=data["model_name"],
        model_obj=None,
        metrics=data["metrics"],
        preds=preds,
    )


def run_forecast_pipeline_tool(payload: dict) -> dict:
    request = ForecastPipelineRequest(**payload)
    results = fit_and_score_models(
        train_df=request.train_df,
        test_df=request.test_df,
        feature_cols=request.feature_cols,
        target_col=request.target_col,
        model_config=request.model_config,
        candidate_model_names=request.candidate_model_names,
    )
    return {"model_results": [_serialize_model_result(result) for result in results]}


def compare_candidate_models_tool(payload: dict) -> dict:
    request = CompareCandidateModelsRequest(**payload)
    model_results = [_deserialize_model_result(result) for result in request.model_results]
    comparison = build_model_comparison(model_results)
    champion, challenger = select_champion_and_challenger(model_results)
    return {
        "comparison": dataframe_to_records(comparison),
        "champion_model": champion.model_name,
        "challenger_model": challenger.model_name if challenger else None,
        "champion_metrics": champion.metrics,
    }


def get_inventory_actions_tool(payload: dict) -> dict:
    request = InventoryActionsRequest(**payload)
    inventory_actions = build_inventory_recommendations(
        forecast_df=request.forecast_df,
        inventory_df=request.inventory_df,
        safety_stock_ratio=request.safety_stock_ratio,
        min_order_qty=request.min_order_qty,
        pack_size=request.pack_size,
    )
    return {"inventory_actions": dataframe_to_records(inventory_actions)}


def get_drift_status_tool(payload: dict) -> dict:
    request = DriftStatusRequest(**payload)
    drift_table = run_drift_monitor(
        train_df=request.train_df,
        current_df=request.current_df,
        monitored_features=request.monitored_features,
        p_threshold=request.p_threshold,
    )
    retraining_status = None
    if request.current_wmape is not None and request.baseline_wmape is not None:
        retraining_status = should_retrain(
            drift_df=drift_table,
            current_wmape=request.current_wmape,
            baseline_wmape=request.baseline_wmape,
            wmape_degradation_threshold=request.wmape_degradation_threshold,
            min_drifted_features=request.min_drifted_features,
        )
    return {"drift_table": dataframe_to_records(drift_table), "retraining_status": retraining_status}


def generate_decision_summary_tool(payload: dict) -> dict:
    request = DecisionSummaryRequest(**payload)
    summary = build_executive_summary(
        champion_name=request.champion_name,
        challenger_name=request.challenger_name,
        champion_metrics=request.champion_metrics,
        retraining_status=request.retraining_status,
    )
    return {"summary": dataframe_to_records(summary)}


def get_tool_registry() -> dict[str, dict[str, ToolSpec | ToolHandler]]:
    return {
        "run_forecast_pipeline": {
            "spec": ToolSpec(
                name="run_forecast_pipeline",
                description="Run candidate model fitting and evaluation on prepared train/test data.",
                request_type="ForecastPipelineRequest",
                response_type="ForecastPipelineResponse",
            ),
            "handler": run_forecast_pipeline_tool,
        },
        "compare_candidate_models": {
            "spec": ToolSpec(
                name="compare_candidate_models",
                description="Build a comparison table and select champion and challenger models.",
                request_type="CompareCandidateModelsRequest",
                response_type="CompareCandidateModelsResponse",
            ),
            "handler": compare_candidate_models_tool,
        },
        "get_inventory_actions": {
            "spec": ToolSpec(
                name="get_inventory_actions",
                description="Translate forecast output and inventory state into reorder recommendations.",
                request_type="InventoryActionsRequest",
                response_type="InventoryActionsResponse",
            ),
            "handler": get_inventory_actions_tool,
        },
        "get_drift_status": {
            "spec": ToolSpec(
                name="get_drift_status",
                description="Compute drift signals and optionally derive retraining status.",
                request_type="DriftStatusRequest",
                response_type="DriftStatusResponse",
            ),
            "handler": get_drift_status_tool,
        },
        "generate_decision_summary": {
            "spec": ToolSpec(
                name="generate_decision_summary",
                description="Generate an executive summary table from model and retraining status.",
                request_type="DecisionSummaryRequest",
                response_type="DecisionSummaryResponse",
            ),
            "handler": generate_decision_summary_tool,
        },
    }


def list_tool_specs() -> list[ToolSpec]:
    return [entry["spec"] for entry in get_tool_registry().values()]


def invoke_tool(request: ToolRequest) -> ToolResponse:
    registry = get_tool_registry()
    entry = registry.get(request.tool_name)
    if entry is None:
        return ToolResponse(tool_name=request.tool_name, status="error", error_message="Unknown tool.")

    try:
        payload = entry["handler"](request.payload)
        return ToolResponse(tool_name=request.tool_name, status="success", payload=payload)
    except Exception as exc:
        return ToolResponse(tool_name=request.tool_name, status="error", error_message=str(exc))
