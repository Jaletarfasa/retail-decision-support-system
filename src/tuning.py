from __future__ import annotations

import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import pandas as pd


def tune_xgboost(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str], target_col: str, n_trials: int = 10) -> dict:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return mean_absolute_error(y_valid, model.predict(X_valid))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
