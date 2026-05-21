# Final Demo Guide

This guide explains how to run and present `clinical-alert-reliability-ai` as a polished simulated healthcare AI systems engineering portfolio project.

## 1. Project Demo Overview

The final demo connects the full local pipeline:

```text
Simulated vitals -> preprocessing/features -> risk/anomaly/time-series scoring
-> alerts -> guardrails -> auditing -> fatigue reduction -> workflow simulation
-> reliability/drift monitoring -> model-update and RL simulation
-> LLM-safe explanations -> RAG -> action recommendations
-> metrics -> outcome/failure/scenario evaluation -> final report
```

The project is simulation-only. It uses synthetic/demo artifacts to demonstrate reliability-aware engineering around an AI alert system.

## 2. What the Demo Proves

The demo shows that the repository is organized enough to run as one reproducible workflow. It demonstrates:

- End-to-end simulated patient monitoring data flow.
- Alert generation, guardrails, auditing, and fatigue reduction.
- Preservation of critical alerts in the simulated fatigue-reduction layer.
- Workflow response simulation and response summary tracking.
- Reliability monitoring and drift detection over time.
- Safe model-update and threshold-agent simulations that do not deploy changes automatically.
- LLM-safe explanation generation with fallback behavior.
- RAG retrieval over local safety and workflow rules.
- Workflow-oriented action recommendations.
- Outcome-effectiveness, failure-mode, and scenario-level evaluation.
- Final metrics and report generation.

It does not prove clinical effectiveness, real patient safety, or deployment readiness.

## 3. Full Demo Command

```bash
.venv/bin/python run_full_demo.py
```

By default, the runner skips stages when their expected output files already exist. To regenerate all demo artifacts, use:

```bash
.venv/bin/python run_full_demo.py --force
```

For a quick packaging/demo check that avoids running pytest inside the demo:

```bash
.venv/bin/python run_full_demo.py --skip-tests
```

## 4. Test Command

```bash
.venv/bin/python -m pytest -q
```

## 5. FastAPI Command

```bash
.venv/bin/python -m uvicorn api.main:app --reload
```

Useful local pages:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/dashboard-summary`

## 6. Streamlit Command

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

## 7. Suggested Demo Walkthrough

1. Show `README.md` and point out the simulation-only disclaimer.
2. Run the test suite:

   ```bash
   .venv/bin/python -m pytest -q
   ```

3. Run the final demo flow:

   ```bash
   .venv/bin/python run_full_demo.py --skip-tests
   ```

4. Open the Streamlit dashboard and walk through overview metrics, active alerts, reliability, drift, explanations, and action recommendations.
5. Open FastAPI docs at `http://127.0.0.1:8000/docs` and show the read-only API structure.
6. Open `reports/evaluation_results.md` and show the final engineering evaluation.
7. Close by explaining limitations and future work.

## 8. What to Explain During Presentation

- Alert fatigue can make monitoring workflows harder to manage.
- The project focuses on a self-monitoring alert system, not just a model score.
- Safety guardrails keep critical and uncertain alerts review-centered.
- Alert auditing and fatigue reduction reduce repeated low-value burden while preserving audit trails.
- Reliability monitoring and drift detection check whether the system is becoming less trustworthy over time.
- Outcome, failure-mode, and scenario testing add deployment-style engineering thinking.
- LLM and RAG components are constrained to support explanations and workflow context.
- The whole project remains simulated and is not clinically validated.

## 9. Troubleshooting

### Direct `uvicorn` or `streamlit` command does not work

Use module execution through the virtual environment:

```bash
.venv/bin/python -m uvicorn api.main:app --reload
.venv/bin/python -m streamlit run dashboard/app.py
```

### Missing package

Reinstall dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Or rerun setup:

```bash
bash setup_project.sh
```

### Missing output file

Run the full demo without `--skip-tests` for a complete check:

```bash
.venv/bin/python run_full_demo.py
```

If you intentionally want to regenerate outputs:

```bash
.venv/bin/python run_full_demo.py --force
```

### Tests fail after local edits

Run the focused failing test first, then the full suite:

```bash
.venv/bin/python -m pytest tests/test_final_demo_flow.py -q
.venv/bin/python -m pytest -q
```

## 10. Safety Disclaimer

This is a simulated research and engineering prototype. It uses synthetic/demo data only, is not clinically validated, is not a medical device, and must not be used for real patient care. All dashboard, API, report, explanation, and recommendation outputs are engineering artifacts for local demonstration and review.
