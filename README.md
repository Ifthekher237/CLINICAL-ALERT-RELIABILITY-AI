# Self-Monitoring AI System for Clinical Alert Reliability

A deployment-aware healthcare AI engineering prototype that simulates continuous patient monitoring, generates alert signals, audits alert quality, reduces alert fatigue, monitors reliability and drift, and produces explainable workflow recommendations. It also includes a local SQLite demo database, a read-only FastAPI backend, and a Streamlit dashboard for portfolio demonstration.

This project is designed as a serious university portfolio project for healthcare AI systems engineering. It focuses on reliability, safety boundaries, workflow pressure, and monitoring behavior around AI-generated alerts rather than presenting a model score in isolation.

## Important Disclaimer

This is a research and engineering prototype built with simulated data. It is **not** a clinically validated medical device, is **not** intended for real patient use, and must not be interpreted as evidence of real-world medical safety or effectiveness.

All outputs, reports, explanations, recommendations, and dashboard views are simulation-only engineering artifacts for portfolio demonstration and learning.

## Problem Statement

Hospitals and monitoring environments can experience alert fatigue when too many alerts are noisy, repeated, poorly prioritized, or hard to interpret. Even when an AI model performs well during development, the surrounding alert system can become unreliable under workflow pressure, noisy sensors, missing data, data drift, delayed response, or poor integration into review workflows.

This project explores a practical engineering question:

> How can an AI alert system monitor its own reliability before deployment-style use?

Instead of focusing only on prediction accuracy, the project simulates an end-to-end alert reliability system that checks alert quality, workflow behavior, safety constraints, drift, failure modes, and scenario-level stress cases.

## What This Project Does

- Simulates patient vital-sign monitoring data.
- Preprocesses and engineers time-series patient features.
- Trains baseline future deterioration risk models.
- Detects unusual patient vital-sign patterns with anomaly detection.
- Builds patient-specific time-series risk logic.
- Generates structured simulated alerts.
- Applies safety guardrails around alert handling.
- Audits alerts for actionability, repetition, urgency, and likely noise.
- Reduces alert fatigue while preserving critical alerts.
- Simulates clinician workflow responses.
- Monitors alert-system reliability over time.
- Detects data, alert, workflow, and reliability drift.
- Simulates feedback-based model update decisions.
- Tests a conservative contextual-bandit threshold agent in simulation.
- Provides a local LLM client with safe fallback behavior.
- Generates safe alert explanations.
- Uses local RAG over internal safety and workflow rules.
- Recommends workflow actions without giving medical advice.
- Evaluates simulated alert-to-outcome associations.
- Simulates deployment-style failure modes.
- Runs scenario tests across stable, deteriorating, noisy, missing-data, critical, and overload situations.
- Exposes outputs through FastAPI.
- Shows the system outputs in a Streamlit dashboard.
- Generates final metrics and evaluation reports.

## System Architecture

```text
Simulated vitals -> preprocessing -> risk/anomaly/time-series models -> alert generation -> guardrails -> auditing -> fatigue reduction -> workflow simulation -> reliability/drift monitoring -> outcome/failure/scenario evaluation -> dashboard/report/API
```

```text
Simulated vitals
      |
      v
Preprocessing + feature engineering
      |
      v
Risk model + anomaly detection + time-series risk logic
      |
      v
Alert generation
      |
      v
Safety guardrails
      |
      v
Alert auditing
      |
      v
Alert fatigue reduction
      |
      v
Workflow simulation
      |
      v
Reliability monitoring + drift detection
      |
      v
Outcome evaluation + failure simulation + scenario testing
      |
      v
Metrics + reports + FastAPI + Streamlit dashboard
```

## AI and Engineering Techniques Used

- Supervised machine learning
- Logistic regression and random forest baselines
- Future-horizon target construction
- Anomaly detection with Isolation Forest
- Time-series rolling risk logic
- Rule-based safety guardrails
- Alert auditing and confidence scoring
- Alert fatigue reduction
- Workflow simulation
- Reliability monitoring
- Drift detection
- Feedback-based model update simulation
- Contextual-bandit threshold simulation
- Local LLM client with safe fallback mode
- Rule-based LLM explanation safety filters
- RAG over local safety and workflow rules
- Workflow action recommendation
- SQLite demo database layer
- FastAPI backend
- Streamlit dashboard
- Centralized project metrics
- Outcome association evaluation
- Failure-mode simulation
- Scenario testing
- Pytest-based reliability validation

## Project Structure

```text
clinical-alert-reliability-ai/
├── api/                 # FastAPI backend routes
├── dashboard/           # Streamlit dashboard
├── data/                # Simulated and processed artifacts
├── knowledge_base/      # Local safety, workflow, and limitation rules
├── models/              # Saved model artifacts
├── notebooks/           # Experiment notebooks
├── reports/             # Evaluation reports and report generator
├── src/                 # Core source modules
│   ├── alerts/          # Alert generation, guardrails, auditing, fatigue reduction
│   ├── data/            # Simulation, preprocessing, feature engineering
│   ├── database/        # SQLite database layer
│   ├── llm/             # LLM client, explanations, RAG, action recommendations
│   ├── models/          # Risk, anomaly, time-series, model update logic
│   ├── monitoring/      # Reliability, drift, metrics, outcome/failure/scenario evaluation
│   ├── rl/              # Threshold-agent simulation
│   └── workflow/        # Clinician workflow simulation and response tracking
├── tests/               # Unit, integration, safety, API, dashboard, reliability tests
├── README.md
└── requirements.txt
```

## How to Run

Create and activate a virtual environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use the lightweight setup helper:

```bash
bash setup_project.sh
```

## Run Full Demo

Run the complete simulated demo flow:

```bash
.venv/bin/python run_full_demo.py
```

The runner skips stages whose expected outputs already exist unless `--force` is provided. It does not start local servers automatically. To launch the demo surfaces separately:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
.venv/bin/python -m uvicorn api.main:app --reload
```

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

Run the FastAPI backend:

```bash
.venv/bin/python -m uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/dashboard-summary
```

Run the Streamlit dashboard:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Using `.venv/bin/python -m ...` is recommended because direct console scripts can break if the virtual environment path changes.

## Main Outputs

Important generated artifacts include:

- `data/processed/project_metrics_summary.json`
- `data/processed/project_metrics_table.csv`
- `reports/evaluation_results.md`
- `data/processed/outcome_effectiveness_summary.json`
- `data/processed/failure_mode_summary.json`
- `data/processed/scenario_test_summary.json`
- `data/processed/generated_alerts.csv`
- `data/processed/fatigue_reduced_alerts.csv`
- `data/processed/clinician_response_logs.csv`
- `data/processed/reliability_monitoring_results.csv`
- `data/processed/drift_detection_results.csv`
- `data/processed/action_recommendations.csv`

## Dashboard Preview

Screenshots can be added here later.

The current dashboard includes sections for overview metrics, patient vitals, alerts, auditing, fatigue reduction, workflow simulation, reliability monitoring, drift detection, explanations, action recommendations, and system limitations.

## Evaluation Summary

The final report in `reports/evaluation_results.md` summarizes the system as a simulated engineering evaluation.

Current evaluation themes include:

- Alert fatigue reduction was simulated and measured.
- Critical alert preservation was checked.
- Reliability and drift were monitored over time.
- Workflow response behavior was simulated.
- Outcome association was evaluated from synthetic outputs.
- Failure-mode simulation stress-tested noisy data, missing data, overload, delayed responses, repeated low-value alerts, confidence degradation, and drift.
- Scenario testing checked stable, deteriorating, critical, noisy, missing-data, repeated-alert, and overload situations.

These results are useful for engineering review, but they are not evidence of real-world medical performance.

## Safety and Ethics

This project is intentionally safety-bounded:

- Uses simulated data only.
- Contains no real patient data.
- Does not claim clinical validation.
- Keeps human review central for safety-sensitive alerts.
- Constrains LLM outputs to support explanations and workflow context.
- Preserves critical alerts during fatigue reduction.
- Treats drift, uncertainty, and unsafe-review states as review triggers.
- Keeps all recommendations workflow-oriented and simulation-only.

For a fuller discussion, see [reports/limitations_and_ethics.md](reports/limitations_and_ethics.md).

## Limitations

- The dataset is synthetic.
- Workflow assumptions are simplified.
- There is no real hospital integration.
- There is no prospective clinical trial.
- There is no real clinician feedback loop.
- Outcome labels are simulated.
- Failure modes are approximations of engineering risks.
- Scenario tests are not real operational validation.
- LLM/RAG explanations are support text only.
- Dashboard and API outputs are local demo views over simulated artifacts.

## Future Work

- Tune alert severity distribution and threshold behavior.
- Integrate public de-identified datasets where appropriate.
- Improve dashboard visuals and interaction polish.
- Add CI later for automated test execution.
- Add PostgreSQL later for more realistic persistence experiments.
- Add model cards and data cards.
- Add deeper calibration analysis.
- Expand drift and reliability monitoring with more baselines.
- Test with clinical collaborators in a safe non-clinical setting.
- Improve documentation around assumptions and known failure modes.

## Portfolio Positioning

This project demonstrates healthcare AI engineering beyond model training. It shows reliability monitoring, safety-aware ML design, workflow simulation, explainability, local RAG, API/dashboard development, database persistence, testing maturity, and deployment-readiness thinking in a simulated environment.

The main portfolio message is:

> Healthcare AI systems need monitoring, auditing, workflow awareness, and safety boundaries around the model, not only a trained model artifact.
