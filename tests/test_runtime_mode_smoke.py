from __future__ import annotations

from src.orchestrator import _resolve_runtime_settings


def test_demo_runtime_settings_apply_expected_overrides() -> None:
    app_cfg = {
        "runtime_mode": "demo",
        "demo_data": {"n_stores": 8, "n_skus": 12, "n_sites": 6, "n_days": 45, "budget_limit": 30000.0},
    }
    model_cfg = {
        "test_days": 30,
        "candidate_models": ["elasticnet", "random_forest"],
        "serial_safe": False,
        "demo_overrides": {
            "test_days": 14,
            "serial_safe": True,
            "candidate_models": ["elasticnet", "mlp"],
        },
    }
    monitoring_cfg = {
        "monitored_features": ["lag_7"],
        "p_threshold": 0.05,
        "wmape_degradation_threshold": 0.05,
        "min_drifted_features": 1,
        "min_sample_size": 2,
        "demo_overrides": {"min_sample_size": 5},
    }

    runtime_mode, data_cfg, runtime_model_cfg, runtime_monitoring_cfg = _resolve_runtime_settings(
        app_cfg, model_cfg, monitoring_cfg
    )

    assert runtime_mode == "demo"
    assert data_cfg["n_stores"] == 8
    assert runtime_model_cfg["serial_safe"] is True
    assert runtime_model_cfg["candidate_models"] == ["elasticnet", "mlp"]
    assert runtime_monitoring_cfg["min_sample_size"] == 5


def test_standard_runtime_settings_preserve_defaults() -> None:
    runtime_mode, data_cfg, runtime_model_cfg, runtime_monitoring_cfg = _resolve_runtime_settings(
        {"runtime_mode": "standard"},
        {"test_days": 30, "candidate_models": ["elasticnet"]},
        {"min_sample_size": 2},
    )

    assert runtime_mode == "standard"
    assert data_cfg["n_stores"] == 30
    assert runtime_model_cfg["candidate_models"] == ["elasticnet"]
    assert runtime_monitoring_cfg["min_sample_size"] == 2
