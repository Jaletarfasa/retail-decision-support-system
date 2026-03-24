# Portfolio Overview

## Project purpose

This project is a retail decision support system built for forecasting-led operational decisions. It demonstrates how synthetic retail data can flow through model evaluation, inventory recommendations, site prioritization, monitoring, reporting, explainability, and lightweight MCP-style orchestration scaffolding in one cohesive demo.

## Architecture layers

### 1. Data and feature pipeline

- Source data generation and validation live in `src/data_generation.py` and `src/validation.py`
- Forecast features such as lags, rolling windows, assortment health, traffic, and site context are built in `src/feature_engineering.py`
- `src/orchestrator.py` coordinates the end-to-end batch run and persists outputs to `artifacts/runs/`

### 2. Forecast model layer

- Classical candidates include ElasticNet, Random Forest, Extra Trees, HistGradientBoosting, and XGBoost
- Deep-learning candidates include a tabular MLP and an entity-embedding neural network in `src/deep_models.py`
- All candidates are evaluated through a shared metric pipeline in `src/forecasting.py`
- Champion and challenger selection is performed in `src/model_selection.py`

### 3. Decision-support outputs

- Promotion summary
- Assortment health
- Inventory reorder recommendations
- Site scoring and constrained optimization
- Drift monitoring and retraining review
- Executive summary and dashboard-ready tables

These capabilities remain in focused modules under `src/` and are stitched together by the orchestrator.

### 4. MCP-style orchestration scaffolding

The project now includes lightweight structured orchestration components:

- `src/schemas.py` for request/response and controller state types
- `src/tool_registry.py` for wrapping existing business capabilities as tools
- `src/agent_controller.py` for simple invocation and small multi-step chaining

This is intentionally lightweight: it demonstrates tool-oriented orchestration without replacing the existing batch pipeline.

### 5. Dashboard and explainers

- `app/streamlit_app.py` renders the artifact-backed dashboard
- `assets/animations/` contains lightweight SVG explainers for system flow, tree interactions, deep models, and MCP-style orchestration
- `app/explainers.py` provides a small manifest/loader for these assets

## Classical + deep model support

The portfolio story is deliberately metric-based rather than hype-based:

- Classical models remain first-class candidates
- Deep models are added as challengers using the same evaluation loop
- Model choice is driven by MAE, RMSE, WMAPE, and Bias instead of assumptions about complexity

This makes the demo stronger for review because it shows disciplined comparison, not just feature accumulation.

## How demo mode supports the reviewer experience

Portfolio reviewers usually want a fast, reliable walkthrough. Demo mode exists for exactly that:

- smaller synthetic datasets
- faster deep-model settings
- serial-safe model execution where practical
- quicker artifact generation for the dashboard
- cleaner handling of small-sample drift checks

The easiest entrypoint is:

```powershell
.\run_demo_pipeline.ps1
```

That launcher sets the demo runtime mode and runs the existing orchestrator without changing the underlying architecture.

## Recommended reviewer path

1. Run the tests
2. Launch the demo pipeline
3. Open the Streamlit dashboard
4. Walk through executive outputs first
5. Finish with the explainer section and the orchestration scaffolding in `src/`
