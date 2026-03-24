# Demo Runbook

## 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 2. Run tests

In this environment, disable global pytest plugin autoload for a clean run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests -q
```

## 3. Run the demo pipeline

Open [`config/app_config.yaml`](C:\Users\S.Tarfasa\Desktop\Data Scientist_2026\retail_decision_support_full_version\config\app_config.yaml) and set:

```yaml
runtime_mode: demo
```

Then run:

```powershell
python -m src.orchestrator
```

Demo mode uses:

- smaller synthetic datasets
- faster deep-model settings
- serial-safe classical model settings where applicable
- explicit small-sample handling in drift monitoring

To return to the full default path, switch `runtime_mode` back to `standard`.

## 4. Launch the dashboard

```powershell
streamlit run app/streamlit_app.py
```

Then select the newest run from `artifacts/runs/`.

## 5. Happy-path demo flow

Walk reviewers through:

1. Executive snapshot and model comparison
2. Inventory, site, and monitoring outputs
3. Model and system explainers
4. MCP-style orchestration scaffolding in `src/`
