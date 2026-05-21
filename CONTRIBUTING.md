# Contributing

Thanks for reviewing or improving this project. `clinical-alert-reliability-ai` is a simulated healthcare AI engineering portfolio project, so contributions should keep the code understandable, reproducible, and safety-aware.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can also run:

```bash
bash setup_project.sh
```

The setup script installs dependencies only. It does not retrain models or regenerate demo outputs.

## Testing

Run the full test suite before submitting changes:

```bash
.venv/bin/python -m pytest -q
```

## Coding Expectations

- Keep modules small, readable, and beginner-explainable.
- Prefer transparent rule-based logic for safety and workflow modules.
- Avoid changing generated outputs unless the task explicitly requires it.
- Add focused tests for new behavior.
- Keep file names, output columns, and function signatures consistent with the roadmap.
- Use simulated-data language in comments, docs, and reports.

## Safety Boundary

This repository uses simulated data only. It is not clinically validated, not a medical device, and not intended for real patient use. Do not add claims that imply real-world medical safety, effectiveness, or deployment readiness.

LLM, RAG, and action-recommendation features should remain constrained to explaining simulated system outputs and workflow review context. They must not present themselves as clinical decision-makers.

## Pull Request Guidance

When proposing changes, include:

- What changed and why.
- Which files or modules were touched.
- Test command and result.
- Any assumptions about simulated data or generated artifacts.
- Any safety or limitation notes reviewers should know.

Keep pull requests scoped. A small, well-tested change is preferred over a broad rewrite.
