from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.schemas import ControllerResult, ControllerState, ToolRequest
from src.tool_registry import invoke_tool


class AgentController:
    def __init__(self) -> None:
        self.state = ControllerState()

    def invoke(self, tool_name: str, payload: dict[str, Any]) -> ControllerResult:
        response = invoke_tool(ToolRequest(tool_name=tool_name, payload=payload))
        if response.status == "error":
            return ControllerResult(
                status="error",
                state=replace(self.state),
                error_message=response.error_message,
            )

        self.state.completed_steps.append(tool_name)
        self.state.tool_outputs[tool_name] = response.payload
        return ControllerResult(
            status="success",
            state=replace(self.state),
            final_output=response.payload,
        )

    def run_forecast_decision_chain(
        self,
        forecast_payload: dict[str, Any],
        drift_payload: dict[str, Any] | None = None,
    ) -> ControllerResult:
        forecast_result = self.invoke("run_forecast_pipeline", forecast_payload)
        if forecast_result.status == "error":
            return forecast_result

        compare_result = self.invoke(
            "compare_candidate_models",
            {"model_results": forecast_result.final_output["model_results"]},
        )
        if compare_result.status == "error":
            return compare_result

        retraining_status = {"retraining_recommended": 0, "status": "Watch"}
        if drift_payload is not None:
            drift_result = self.invoke("get_drift_status", drift_payload)
            if drift_result.status == "error":
                return drift_result
            retraining_status = drift_result.final_output.get("retraining_status") or retraining_status

        summary_result = self.invoke(
            "generate_decision_summary",
            {
                "champion_name": compare_result.final_output["champion_model"],
                "challenger_name": compare_result.final_output["challenger_model"],
                "champion_metrics": compare_result.final_output["champion_metrics"],
                "retraining_status": retraining_status,
            },
        )
        if summary_result.status == "error":
            return summary_result

        return ControllerResult(
            status="success",
            state=replace(self.state),
            final_output={
                "model_results": forecast_result.final_output["model_results"],
                "comparison": compare_result.final_output["comparison"],
                "retraining_status": retraining_status,
                "summary": summary_result.final_output["summary"],
            },
        )
