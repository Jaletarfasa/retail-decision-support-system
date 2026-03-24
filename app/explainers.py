from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExplainerAsset:
    key: str
    title: str
    caption: str
    filename: str


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "animations"

EXPLAINERS = [
    ExplainerAsset(
        key="system_flow",
        title="End-to-End System Flow",
        caption="Source data becomes features, model evaluations, decision outputs, monitoring, and reporting artifacts.",
        filename="system_flow.svg",
    ),
    ExplainerAsset(
        key="tree_interactions",
        title="Tree Model Interaction Learning",
        caption="Tree ensembles split across price, promo, and context signals to capture nonlinear retail interactions.",
        filename="tree_interactions.svg",
    ),
    ExplainerAsset(
        key="mlp_flow",
        title="MLP Model Flow",
        caption="Dense layers transform standardized tabular signals into a demand forecast through stacked nonlinear representations.",
        filename="mlp_flow.svg",
    ),
    ExplainerAsset(
        key="embedding_flow",
        title="Entity Embedding Model Flow",
        caption="Categorical IDs become learned embeddings that combine with continuous signals before prediction.",
        filename="embedding_flow.svg",
    ),
    ExplainerAsset(
        key="mcp_orchestration",
        title="MCP-Style Orchestration Flow",
        caption="Structured tool requests flow through a registry and controller to produce deterministic outputs for downstream use.",
        filename="mcp_orchestration.svg",
    ),
]


def list_explainers() -> list[ExplainerAsset]:
    return EXPLAINERS


def get_explainer_path(explainer: ExplainerAsset) -> Path:
    return ASSET_DIR / explainer.filename


def load_explainer_markup(explainer: ExplainerAsset) -> str:
    return get_explainer_path(explainer).read_text(encoding="utf-8")
