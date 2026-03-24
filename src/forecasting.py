from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import yaml

from src.deep_models import EntityEmbeddingRegressor, TabularMLPRegressor


@dataclass
class ModelResult:
    model_name: str
    model_obj: object
    metrics: dict
    preds: pd.Series


DEFAULT_MODEL_CONFIG = {
    "candidate_models": [
        "elasticnet",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
        "mlp",
        "entity_embedding_nn",
    ],
    "deep_model_params": {
        "mlp": {
            "hidden_dims": [64, 32],
            "learning_rate": 0.001,
            "epochs": 25,
            "batch_size": 128,
            "seed": 42,
        },
        "entity_embedding_nn": {
            "categorical_cols": ["store_id", "sku_id"],
            "hidden_dims": [64, 32],
            "learning_rate": 0.001,
            "epochs": 25,
            "batch_size": 128,
            "seed": 42,
        },
    },
    "serial_safe": False,
}


def time_split(df: pd.DataFrame, test_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    wmape = abs(y_true - y_pred).sum() / max(abs(y_true).sum(), 1e-9)
    bias = (y_pred - y_true).mean()
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "wmape": float(wmape), "bias": float(bias), "r2": float(r2)}


def _load_model_config() -> dict:
    config_path = Path("config/model_config.yaml")
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        merged = dict(DEFAULT_MODEL_CONFIG)
        merged.update(loaded)
        merged["deep_model_params"] = {
            **DEFAULT_MODEL_CONFIG["deep_model_params"],
            **loaded.get("deep_model_params", {}),
        }
        return merged
    return DEFAULT_MODEL_CONFIG


def _build_all_models(model_config: dict | None = None) -> dict:
    config = model_config or _load_model_config()
    deep_params = config.get("deep_model_params", {})
    serial_safe = config.get("serial_safe", False)
    rf_n_jobs = 1 if serial_safe else -1
    xgb_n_jobs = 1 if serial_safe else -1
    return {
        "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=rf_n_jobs),
        "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=rf_n_jobs),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
        "xgboost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=xgb_n_jobs),
        "mlp": TabularMLPRegressor(
            hidden_dims=tuple(deep_params.get("mlp", {}).get("hidden_dims", [64, 32])),
            learning_rate=deep_params.get("mlp", {}).get("learning_rate", 0.001),
            epochs=deep_params.get("mlp", {}).get("epochs", 25),
            batch_size=deep_params.get("mlp", {}).get("batch_size", 128),
            seed=deep_params.get("mlp", {}).get("seed", 42),
        ),
        "entity_embedding_nn": EntityEmbeddingRegressor(
            categorical_cols=tuple(deep_params.get("entity_embedding_nn", {}).get("categorical_cols", ["store_id", "sku_id"])),
            hidden_dims=tuple(deep_params.get("entity_embedding_nn", {}).get("hidden_dims", [64, 32])),
            learning_rate=deep_params.get("entity_embedding_nn", {}).get("learning_rate", 0.001),
            epochs=deep_params.get("entity_embedding_nn", {}).get("epochs", 25),
            batch_size=deep_params.get("entity_embedding_nn", {}).get("batch_size", 128),
            seed=deep_params.get("entity_embedding_nn", {}).get("seed", 42),
        ),
    }


def get_candidate_models(model_config: dict | None = None, candidate_model_names: list[str] | None = None) -> dict:
    config = model_config or _load_model_config()
    requested_names = candidate_model_names or config.get("candidate_models", DEFAULT_MODEL_CONFIG["candidate_models"])
    all_models = _build_all_models(config)
    return {name: all_models[name] for name in requested_names if name in all_models}


def fit_and_score_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_config: dict | None = None,
    candidate_model_names: list[str] | None = None,
) -> list[ModelResult]:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    results: list[ModelResult] = []
    for name, model in get_candidate_models(model_config=model_config, candidate_model_names=candidate_model_names).items():
        model.fit(X_train, y_train)
        preds = pd.Series(model.predict(X_test), index=test_df.index, name="forecast_units")
        metrics = compute_metrics(y_test, preds)
        results.append(ModelResult(name, model, metrics, preds))
    return results
