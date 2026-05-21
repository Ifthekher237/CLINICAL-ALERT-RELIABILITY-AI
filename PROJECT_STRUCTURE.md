# Project Structure

This document is a quick navigation guide for `clinical-alert-reliability-ai`.

## Top-Level Folders

- `src/` contains the core simulation, modeling, monitoring, alert, LLM, database, and workflow modules.
- `api/` contains the read-only FastAPI backend used for local demo access to generated outputs.
- `dashboard/` contains the Streamlit dashboard for browsing simulated metrics, alerts, monitoring results, explanations, and recommendations.
- `data/` contains simulated and processed demo artifacts. The curated outputs in `data/processed/` are part of the portfolio demo.
- `models/` contains saved local model artifacts used by the simulated alert pipeline.
- `reports/` contains the final evaluation report and markdown/report-generation utilities.
- `knowledge_base/` contains local safety, workflow, escalation, and limitation rules for the RAG engine.
- `tests/` contains unit, integration, API, dashboard, safety, reliability, and repository-structure tests.
- `notebooks/` contains exploratory notebooks from the project workflow.
- `config/` contains lightweight settings and alert-rule configuration files.

## Source Module Map

- `src/data/` simulates patient vitals, preprocesses data, and builds engineered features.
- `src/models/` contains baseline risk modeling, anomaly detection, time-series risk logic, and model-update simulation.
- `src/alerts/` generates alerts, applies guardrails, audits alert quality, and reduces alert fatigue.
- `src/workflow/` simulates clinician response behavior and summarizes workflow responses.
- `src/monitoring/` handles reliability monitoring, drift detection, metrics, outcome evaluation, failure modes, and scenario tests.
- `src/rl/` contains the simulation-only threshold agent.
- `src/llm/` contains the local LLM client, explanation generator, RAG engine, and workflow action recommender.
- `src/database/` contains the lightweight SQLite demo database layer.

## Important Outputs

- `data/processed/project_metrics_summary.json`
- `data/processed/project_metrics_table.csv`
- `reports/evaluation_results.md`
- `data/processed/outcome_effectiveness_summary.json`
- `data/processed/failure_mode_summary.json`
- `data/processed/scenario_test_summary.json`
- `data/processed/clinical_alert_reliability.db`

## Demo Entry Points

- FastAPI app: `api/main.py`
- Streamlit dashboard: `dashboard/app.py`
- Final report generator: `reports/evaluation_report_generator.py`
- Project setup script: `setup_project.sh`

## Testing

Run the full test suite with:

```bash
.venv/bin/python -m pytest -q
```

The tests cover module behavior, generated artifact consistency, safety constraints, API resilience, dashboard loading, and repository packaging checks.

## Safety Boundary

All data and outputs are simulated. This project is for healthcare AI systems engineering practice and portfolio review only. It is not clinically validated and is not intended for real patient care.
