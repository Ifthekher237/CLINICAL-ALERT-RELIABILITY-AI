# Project Overview

## Purpose

`clinical-alert-reliability-ai` is a simulated healthcare AI systems engineering project. It explores how an alerting system can be evaluated beyond model accuracy by checking alert quality, workflow burden, drift, reliability, safety constraints, and deployment-style failure conditions.

The project is designed for portfolio and educational review. It uses synthetic/demo artifacts only and does not claim clinical validation or real-world patient benefit.

## System Workflow

```text
Simulated patient vitals
  -> preprocessing and feature engineering
  -> supervised risk model, anomaly detection, and time-series risk logic
  -> structured alert generation
  -> safety guardrails
  -> alert auditing
  -> alert fatigue reduction
  -> simulated clinician workflow response
  -> reliability and drift monitoring
  -> outcome, failure-mode, and scenario evaluation
  -> metrics, report, API, and dashboard
```

## Safety-First Design

The project treats safety as an engineering constraint around the model rather than a single model score. The alert pipeline includes:

- Guardrails that prevent unsafe handling of critical or uncertain alerts.
- Audit labels for actionability, fatigue risk, urgency, likely noise, and escalation need.
- Fatigue-reduction logic that keeps rows for auditability and preserves critical alerts.
- Human-review flags for safety-sensitive, uncertain, degraded, or severe-drift conditions.
- Documentation and tests that scan for unsafe claims and simulation-only boundaries.

## Monitoring and Reliability Focus

The reliability layer evaluates whether the simulated alert system itself is behaving consistently. It tracks alert volume, ignored or delayed responses, false-alert signals, critical preservation, response quality, and system reliability over time windows.

The drift detector compares earlier and later windows for patient vitals, alert behavior, response behavior, alert volume, and reliability score. Severe drift does not trigger automatic deployment changes; it triggers review recommendations.

## Deployment-Thinking Components

This repository includes several modules that are common in deployment-aware AI systems engineering:

- SQLite demo database for local storage of simulated outputs.
- FastAPI backend for read-only access to generated artifacts.
- Streamlit dashboard for reviewer-friendly exploration.
- Metrics module that centralizes project-level summaries.
- Outcome-effectiveness evaluator for simulated alert-to-workflow association.
- Failure-mode simulator for noisy data, missing data, overload, delayed response, confidence degradation, and drift.
- Scenario tester for stable, deteriorating, critical, noisy, missing-data, repeated-alert, and workload-overload cases.
- Simulation-only model update and threshold-agent modules that recommend review rather than automatic deployment.

## LLM and RAG Components

The LLM components are constrained to support text. The local LLM client can use Ollama if available and falls back safely when no local model is running. The RAG engine retrieves internal project rules from `knowledge_base/`, including safety rules, escalation rules, workflow rules, and system limitations.

The action recommender suggests workflow actions such as review, escalation, monitoring, grouping, or manual verification. It does not provide medical advice and does not replace human judgment.

## Dashboard and API Overview

The FastAPI backend in `api/` exposes local JSON endpoints for alerts, monitoring outputs, dashboard summary metrics, and rule-based alert explanations.

The Streamlit app in `dashboard/` presents:

- Overview metrics
- Patient vitals simulation
- Active alerts
- Alert auditing and fatigue reduction
- Clinician workflow simulation
- Reliability monitoring
- Drift detection
- LLM explanations
- Action recommendations
- System limitations

Both are local demo surfaces over simulated artifacts.

## Limitations

- Data is synthetic.
- Workflow behavior is simulated.
- Outcome labels are synthetic and incomplete.
- No real hospital system is integrated.
- No prospective study or clinical trial has been run.
- No real clinician feedback loop is included.
- LLM/RAG text is constrained support content only.
- Results are useful for engineering review, not clinical validation.

## Future Work

- Add model cards and data cards.
- Improve calibration analysis and threshold sensitivity review.
- Integrate public de-identified datasets where appropriate.
- Add CI for automated test execution.
- Add PostgreSQL later for more realistic persistence experiments.
- Improve dashboard visuals and reviewer navigation.
- Collaborate with clinical reviewers in a safe, non-clinical setting to evaluate assumptions and workflow realism.
