# Retail Decision Support System

This project is a modular retail decision support demo covering forecasting, model comparison, inventory actions, site prioritization, monitoring, explainers, and lightweight MCP-style orchestration scaffolding.

## What is implemented

- Classical forecasting candidates and metric-based model selection
- PyTorch tabular deep-learning candidates: MLP and entity embedding network
- Inventory, promotion, site, monitoring, and reporting outputs
- Structured schemas, tool registry, and a lightweight controller
- Streamlit dashboard with explainer assets
- Smoke-test baseline for demo stability

## Runtime modes

- `standard`: current default behavior
- `demo`: smaller synthetic data, serial-safe settings, and faster runtime for portfolio walkthroughs

Set the mode in [`config/app_config.yaml`](C:\Users\S.Tarfasa\Desktop\Data Scientist_2026\retail_decision_support_full_version\config\app_config.yaml) by changing `runtime_mode`.

## Quick start

See [`docs/demo_runbook.md`](C:\Users\S.Tarfasa\Desktop\Data Scientist_2026\retail_decision_support_full_version\docs\demo_runbook.md) for the happy-path install, test, demo pipeline, and dashboard steps.
