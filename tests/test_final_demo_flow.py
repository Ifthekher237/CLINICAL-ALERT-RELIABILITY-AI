from __future__ import annotations

from pathlib import Path

import run_full_demo


ROOT = Path(__file__).resolve().parents[1]


def test_final_demo_files_exist() -> None:
    assert (ROOT / "run_full_demo.py").is_file()
    assert (ROOT / "DEMO_GUIDE.md").is_file()


def test_demo_guide_includes_required_commands() -> None:
    guide = (ROOT / "DEMO_GUIDE.md").read_text(encoding="utf-8")

    required_commands = [
        ".venv/bin/python run_full_demo.py",
        ".venv/bin/python -m pytest -q",
        ".venv/bin/python -m uvicorn api.main:app --reload",
        ".venv/bin/python -m streamlit run dashboard/app.py",
    ]

    for command in required_commands:
        assert command in guide


def test_run_full_demo_has_ordered_pipeline_stages() -> None:
    stage_names = [stage.name for stage in run_full_demo.DEMO_STAGES]

    expected_stage_names = [
        "simulate patient data",
        "preprocess / feature engineering",
        "train or load baseline risk models",
        "generate anomaly scores",
        "generate time-series risk scores",
        "generate raw alerts",
        "apply safety guardrails",
        "audit alerts",
        "reduce alert fatigue",
        "simulate clinician responses",
        "track workflow response summary",
        "monitor reliability",
        "detect drift",
        "simulate online model update / model registry",
        "run RL threshold simulation",
        "generate LLM-safe explanations",
        "run RAG retrieval layer",
        "generate action recommendations",
        "calculate project metrics",
        "evaluate outcome effectiveness",
        "simulate failure modes",
        "run scenario tests",
        "regenerate final evaluation report",
        "confirm FastAPI and Streamlit launch commands",
    ]

    assert stage_names == expected_stage_names
    assert [stage.stage_number for stage in run_full_demo.DEMO_STAGES] == list(range(1, 25))


def test_run_full_demo_does_not_auto_start_api_or_dashboard() -> None:
    runnable_text = "\n".join(
        " ".join(stage.subprocess_args or ()) for stage in run_full_demo.DEMO_STAGES
    )
    command_labels = "\n".join(stage.command_label for stage in run_full_demo.DEMO_STAGES)

    assert "uvicorn" not in runnable_text
    assert "streamlit" not in runnable_text
    assert run_full_demo.FASTAPI_COMMAND in run_full_demo.print_launch_commands.__globals__.values()
    assert run_full_demo.STREAMLIT_COMMAND in run_full_demo.print_launch_commands.__globals__.values()
    assert "uvicorn api.main:app" not in command_labels
    assert "streamlit run dashboard/app.py" not in command_labels


def test_output_verification_function_exists_and_reports_status() -> None:
    verification = run_full_demo.verify_important_outputs(
        ("README.md", "data/processed/not_a_real_demo_output.csv")
    )

    assert verification["README.md"] is True
    assert verification["data/processed/not_a_real_demo_output.csv"] is False


def test_important_outputs_are_declared() -> None:
    required_outputs = {
        "data/processed/generated_alerts.csv",
        "data/processed/fatigue_reduced_alerts.csv",
        "data/processed/clinician_response_summary.json",
        "data/processed/reliability_summary.json",
        "data/processed/drift_summary.json",
        "data/processed/alert_explanations.csv",
        "data/processed/action_recommendations.csv",
        "data/processed/project_metrics_summary.json",
        "data/processed/outcome_effectiveness_summary.json",
        "data/processed/failure_mode_summary.json",
        "data/processed/scenario_test_summary.json",
        "reports/evaluation_results.md",
    }

    assert required_outputs.issubset(set(run_full_demo.IMPORTANT_OUTPUTS))


def test_simulation_only_disclaimer_appears() -> None:
    guide = (ROOT / "DEMO_GUIDE.md").read_text(encoding="utf-8").lower()
    runner = (ROOT / "run_full_demo.py").read_text(encoding="utf-8").lower()

    assert "simulation-only" in guide
    assert "not clinically validated" in guide
    assert "simulation-only" in runner
    assert "not clinically validated" in runner


def test_no_unsafe_clinical_wording_in_demo_docs() -> None:
    unsafe_phrases = [
        "clinically proven",
        "safe for real patients",
        "treatment should",
        "diagnosis is",
        "prescribe",
        "replace clinician judgment",
    ]
    combined_text = "\n".join(
        [
            (ROOT / "DEMO_GUIDE.md").read_text(encoding="utf-8"),
            (ROOT / "run_full_demo.py").read_text(encoding="utf-8"),
        ]
    ).lower()

    for phrase in unsafe_phrases:
        assert phrase not in combined_text


def test_readme_contains_full_demo_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Run Full Demo" in readme
    assert ".venv/bin/python run_full_demo.py" in readme
    assert ".venv/bin/python -m streamlit run dashboard/app.py" in readme
    assert ".venv/bin/python -m uvicorn api.main:app --reload" in readme
