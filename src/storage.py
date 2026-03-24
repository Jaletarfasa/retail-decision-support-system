from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import sqlite3

from src.utils import ensure_dir, write_json


class RunStorage:
    def __init__(self, artifact_root: str | Path, run_id: str) -> None:
        self.artifact_root = Path(artifact_root)
        self.run_id = run_id
        self.run_dir = ensure_dir(self.artifact_root / "runs" / run_id)
        self.paths = {name: ensure_dir(self.run_dir / name) for name in [
            "source","features","models","tuning","selection","promotion","assortment",
            "inventory","site","optimization","monitoring","retraining","retrieval",
            "routing","dashboard","logs","exports"
        ]}
        self.sqlite_path = self.run_dir / "retail_system.db"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(manifest, self.run_dir / "manifest.json")

    def save_csv(self, df: pd.DataFrame, filename: str, section: str) -> Path:
        out = self.paths[section] / filename
        df.to_csv(out, index=False)
        return out

    def save_table_sqlite(self, df: pd.DataFrame, table_name: str) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

    def read_sql(self, query: str) -> pd.DataFrame:
        with sqlite3.connect(self.sqlite_path) as conn:
            return pd.read_sql(query, conn)
