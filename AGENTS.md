# AGENTS.md

## Project purpose
Retail decision support system for forecasting, inventory actions, promotion insights, site prioritization, monitoring, retrieval, and explainability.

## Non-negotiable architecture rules
- Preserve separation of concerns.
- Do not move files unless explicitly asked.
- Do not create duplicate modules when extending an existing one is appropriate.
- Keep training/model logic inside src/.
- Keep UI logic inside pp/.
- Keep tests inside 	ests/.
- Keep generated outputs in outputs/ or rtifacts/, never in src/.
- Do not store large generated binaries in source folders.
- Prefer modifying existing modules over creating parallel versions with similar names.
- Keep public function names stable unless a refactor is explicitly requested.

## Folder ownership
- src/: core business logic, data prep, modeling, orchestration, monitoring
- pp/: Streamlit app and UI components only
- 	ests/: unit/integration/smoke tests
- config/: configuration files
- docs/: project documentation
- 
otebooks/: exploratory work only, not production logic
- outputs/: generated reports/tables/plots
- rtifacts/: model artifacts and run artifacts

## Model requirements
- Maintain current classical candidate models.
- Add deep learning candidates:
  1. MLP for tabular forecasting
  2. Entity-embedding neural network for categorical-rich retail data
- Evaluate all candidate models under the same metrics:
  - MAE
  - RMSE
  - WMAPE
  - Bias
- Do not assume deep learning is superior; selection must be metric-based.

## Explainability / animation requirements
- Add animated explainers for:
  1. end-to-end system flow
  2. tree model interaction learning
  3. MLP model flow
  4. embedding model flow
  5. MCP-style orchestration flow
- Keep animation assets under ssets/animations/ unless otherwise instructed.
- Do not embed large binary animation files into src/.

## MCP-style orchestration requirements
- Add tool-oriented orchestration gradually.
- Preferred modules:
  - src/tool_registry.py
  - src/agent_controller.py
  - src/schemas.py
- Tool inputs/outputs should be structured and deterministic where possible.

## Testing rules
- Every new modeling or orchestration feature should include at least one test.
- Do not break existing pipeline entry points.
- Prefer smoke tests for orchestration and unit tests for model helpers.

## Safety / scope
- Never delete user data or generated outputs unless explicitly instructed.
- Never overwrite working baseline scripts without preserving a recoverable path via Git.
