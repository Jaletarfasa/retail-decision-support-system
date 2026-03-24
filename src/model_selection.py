from __future__ import annotations

import pandas as pd
from src.forecasting import ModelResult


def build_model_comparison(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        row = {"model_name": res.model_name}
        row.update(res.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def select_champion_and_challenger(results: list[ModelResult]) -> tuple[ModelResult, ModelResult | None]:
    ranked = sorted(results, key=lambda r: r.metrics["mae"])
    return ranked[0], ranked[1] if len(ranked) > 1 else None
