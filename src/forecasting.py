from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


@dataclass
class ModelResult:
    model_name: str
    model_obj: object
    metrics: dict
    preds: pd.Series


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


def get_candidate_models() -> dict:
    return {
        "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
        "xgboost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    }


def fit_and_score_models(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> list[ModelResult]:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    results: list[ModelResult] = []
    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        preds = pd.Series(model.predict(X_test), index=test_df.index, name="forecast_units")
        metrics = compute_metrics(y_test, preds)
        results.append(ModelResult(name, model, metrics, preds))
    return results
