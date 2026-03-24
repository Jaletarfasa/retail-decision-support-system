from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd


ToolName = Literal[
    "run_forecast_pipeline",
    "compare_candidate_models",
    "get_inventory_actions",
    "get_drift_status",
    "generate_decision_summary",
]


@dataclass
class ToolSpec:
    name: ToolName
    description: str
    request_type: str
    response_type: str


@dataclass
class ToolRequest:
    tool_name: ToolName
    payload: dict[str, Any]


@dataclass
class ToolResponse:
    tool_name: ToolName
    status: Literal["success", "error"]
    payload: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class ControllerState:
    completed_steps: list[str] = field(default_factory=list)
    tool_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ControllerResult:
    status: Literal["success", "error"]
    state: ControllerState
    final_output: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class ForecastPipelineRequest:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    target_col: str
    model_config: dict[str, Any] | None = None
    candidate_model_names: list[str] | None = None


@dataclass
class ForecastPipelineResponse:
    model_results: list[dict[str, Any]]


@dataclass
class CompareCandidateModelsRequest:
    model_results: list[Any]


@dataclass
class CompareCandidateModelsResponse:
    comparison: pd.DataFrame
    champion_model: str
    challenger_model: str | None


@dataclass
class InventoryActionsRequest:
    forecast_df: pd.DataFrame
    inventory_df: pd.DataFrame
    safety_stock_ratio: float = 0.25
    min_order_qty: int = 0
    pack_size: int = 1


@dataclass
class InventoryActionsResponse:
    inventory_actions: pd.DataFrame


@dataclass
class DriftStatusRequest:
    train_df: pd.DataFrame
    current_df: pd.DataFrame
    monitored_features: list[str]
    p_threshold: float = 0.05
    current_wmape: float | None = None
    baseline_wmape: float | None = None
    wmape_degradation_threshold: float = 0.05
    min_drifted_features: int = 1


@dataclass
class DriftStatusResponse:
    drift_table: pd.DataFrame
    retraining_status: dict[str, Any] | None = None


@dataclass
class DecisionSummaryRequest:
    champion_name: str
    champion_metrics: dict[str, Any]
    retraining_status: dict[str, Any]
    challenger_name: str | None = None


@dataclass
class DecisionSummaryResponse:
    summary: pd.DataFrame


def dataclass_to_payload(data: Any) -> dict[str, Any]:
    return asdict(data)


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.to_dict(orient="records")
