"""Final demo orchestration for the clinical alert reliability prototype.

This runner connects the completed project stages into one reproducible local
demo flow. It skips stages when their curated outputs already exist unless
``--force`` is provided. The project remains simulation-only: this script does
not start servers automatically, call external APIs, or claim clinical validity.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

SIMULATION_ONLY_NOTE = (
    "Simulation-only healthcare AI engineering demo. Not clinically validated, "
    "not for real patient use, and not a medical device."
)

FASTAPI_COMMAND = ".venv/bin/python -m uvicorn api.main:app --reload"
STREAMLIT_COMMAND = ".venv/bin/python -m streamlit run dashboard/app.py"
TEST_COMMAND = ".venv/bin/python -m pytest -q"


@dataclass(frozen=True)
class DemoStage:
    """One runnable stage in the final portfolio demo flow."""

    stage_number: int
    name: str
    command_label: str
    outputs: tuple[str, ...] = field(default_factory=tuple)
    function_path: str | None = None
    subprocess_args: tuple[str, ...] | None = None
    note: str = ""


DEMO_STAGES: tuple[DemoStage, ...] = (
    DemoStage(
        1,
        "simulate patient data",
        ".venv/bin/python src/data/simulator.py",
        ("data/simulated/patient_monitoring.csv",),
        function_path="src.data.simulator:generate_patient_monitoring_data",
    ),
    DemoStage(
        2,
        "preprocess / feature engineering",
        ".venv/bin/python src/data/preprocessing.py",
        ("data/processed/processed_data.csv",),
        function_path="src.data.preprocessing:prepare_modeling_data",
    ),
    DemoStage(
        3,
        "train or load baseline risk models",
        ".venv/bin/python src/models/risk_model.py",
        (
            "models/logistic_regression.pkl",
            "models/random_forest.pkl",
            "models/scaler.pkl",
        ),
        function_path="src.models.risk_model:train_and_evaluate_models",
    ),
    DemoStage(
        4,
        "generate anomaly scores",
        ".venv/bin/python src/models/anomaly_model.py",
        ("models/isolation_forest_anomaly.pkl",),
        subprocess_args=("src/models/anomaly_model.py",),
    ),
    DemoStage(
        5,
        "generate time-series risk scores",
        ".venv/bin/python src/models/timeseries_model.py",
        ("data/processed/timeseries_risk_scored.csv",),
        subprocess_args=("src/models/timeseries_model.py",),
    ),
    DemoStage(
        6,
        "generate raw alerts",
        ".venv/bin/python src/alerts/alert_generator.py",
        ("data/processed/generated_alerts.csv",),
        subprocess_args=("src/alerts/alert_generator.py",),
    ),
    DemoStage(
        7,
        "apply safety guardrails",
        ".venv/bin/python src/alerts/safety_guardrails.py",
        ("data/processed/guardrail_reviewed_alerts.csv",),
        subprocess_args=("src/alerts/safety_guardrails.py",),
    ),
    DemoStage(
        8,
        "audit alerts",
        ".venv/bin/python src/alerts/alert_auditor.py",
        ("data/processed/audited_alerts.csv",),
        subprocess_args=("src/alerts/alert_auditor.py",),
    ),
    DemoStage(
        9,
        "reduce alert fatigue",
        ".venv/bin/python src/alerts/fatigue_reducer.py",
        ("data/processed/fatigue_reduced_alerts.csv",),
        subprocess_args=("src/alerts/fatigue_reducer.py",),
    ),
    DemoStage(
        10,
        "simulate clinician responses",
        ".venv/bin/python src/workflow/clinician_simulator.py",
        ("data/processed/clinician_response_logs.csv",),
        subprocess_args=("src/workflow/clinician_simulator.py",),
    ),
    DemoStage(
        11,
        "track workflow response summary",
        ".venv/bin/python src/workflow/response_tracker.py",
        ("data/processed/clinician_response_summary.json",),
        subprocess_args=("src/workflow/response_tracker.py",),
    ),
    DemoStage(
        12,
        "monitor reliability",
        ".venv/bin/python src/monitoring/reliability_monitor.py",
        (
            "data/processed/reliability_monitoring_results.csv",
            "data/processed/reliability_summary.json",
        ),
        subprocess_args=("src/monitoring/reliability_monitor.py",),
    ),
    DemoStage(
        13,
        "detect drift",
        ".venv/bin/python src/monitoring/drift_detector.py",
        (
            "data/processed/drift_detection_results.csv",
            "data/processed/drift_summary.json",
        ),
        subprocess_args=("src/monitoring/drift_detector.py",),
    ),
    DemoStage(
        14,
        "simulate online model update / model registry",
        ".venv/bin/python src/models/model_registry.py",
        (
            "data/processed/model_update_simulation_results.csv",
            "data/processed/model_version_registry.json",
            "data/processed/threshold_update_summary.json",
        ),
        subprocess_args=("src/models/model_registry.py",),
    ),
    DemoStage(
        15,
        "run RL threshold simulation",
        ".venv/bin/python src/rl/threshold_agent.py",
        (
            "data/processed/rl_threshold_simulation_results.csv",
            "data/processed/rl_threshold_policy_summary.json",
        ),
        subprocess_args=("src/rl/threshold_agent.py",),
    ),
    DemoStage(
        16,
        "generate LLM-safe explanations",
        ".venv/bin/python src/llm/explanation_generator.py",
        ("data/processed/alert_explanations.csv",),
        subprocess_args=("src/llm/explanation_generator.py",),
    ),
    DemoStage(
        17,
        "run RAG retrieval layer",
        ".venv/bin/python src/llm/rag_engine.py",
        (
            "knowledge_base/safety_rules.md",
            "knowledge_base/alert_escalation_rules.md",
            "knowledge_base/workflow_rules.md",
            "knowledge_base/system_limitations.md",
        ),
        subprocess_args=("src/llm/rag_engine.py",),
    ),
    DemoStage(
        18,
        "generate action recommendations",
        ".venv/bin/python src/llm/action_recommender.py",
        ("data/processed/action_recommendations.csv",),
        subprocess_args=("src/llm/action_recommender.py",),
    ),
    DemoStage(
        19,
        "calculate project metrics",
        ".venv/bin/python src/monitoring/metrics.py",
        (
            "data/processed/project_metrics_summary.json",
            "data/processed/project_metrics_table.csv",
        ),
        subprocess_args=("src/monitoring/metrics.py",),
    ),
    DemoStage(
        20,
        "evaluate outcome effectiveness",
        ".venv/bin/python src/monitoring/outcome_evaluator.py",
        (
            "data/processed/outcome_effectiveness_results.csv",
            "data/processed/outcome_effectiveness_summary.json",
        ),
        subprocess_args=("src/monitoring/outcome_evaluator.py",),
    ),
    DemoStage(
        21,
        "simulate failure modes",
        ".venv/bin/python src/monitoring/failure_mode_simulator.py",
        (
            "data/processed/failure_mode_results.csv",
            "data/processed/failure_mode_summary.json",
        ),
        subprocess_args=("src/monitoring/failure_mode_simulator.py",),
    ),
    DemoStage(
        22,
        "run scenario tests",
        ".venv/bin/python src/monitoring/scenario_tester.py",
        (
            "data/processed/scenario_test_results.csv",
            "data/processed/scenario_test_summary.json",
        ),
        subprocess_args=("src/monitoring/scenario_tester.py",),
    ),
    DemoStage(
        23,
        "regenerate final evaluation report",
        ".venv/bin/python reports/evaluation_report_generator.py",
        ("reports/evaluation_results.md",),
        subprocess_args=("reports/evaluation_report_generator.py",),
    ),
    DemoStage(
        24,
        "confirm FastAPI and Streamlit launch commands",
        "print local launch commands",
        note="Servers are not started automatically by the demo runner.",
    ),
)

IMPORTANT_OUTPUTS: tuple[str, ...] = (
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
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse final demo command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the final simulated clinical alert reliability demo flow."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run every pipeline stage even when expected outputs already exist.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the final pytest command.",
    )
    parser.add_argument(
        "--skip-dashboard-note",
        action="store_true",
        help="Do not print FastAPI and Streamlit launch commands at the end.",
    )
    return parser.parse_args(argv)


def run_full_demo(
    force: bool = False,
    skip_tests: bool = False,
    skip_dashboard_note: bool = False,
) -> dict[str, bool]:
    """Run the final demo flow and return important output verification status."""
    print("Clinical Alert Reliability AI - final demo flow")
    print(SIMULATION_ONLY_NOTE)
    print()

    for stage in DEMO_STAGES:
        run_demo_stage(stage, force=force)

    if not skip_tests:
        print("\nRunning final test suite")
        run_subprocess_command((PYTHON_EXE, "-m", "pytest", "-q"))

    verification = verify_important_outputs()
    print_output_verification(verification)

    if not skip_dashboard_note:
        print_launch_commands()

    missing_outputs = [path for path, exists in verification.items() if not exists]
    if missing_outputs:
        raise RuntimeError(
            "Demo flow completed but required outputs are missing: "
            + ", ".join(missing_outputs)
        )

    print("\nFinal demo flow complete.")
    return verification


def run_demo_stage(stage: DemoStage, force: bool = False) -> None:
    """Run one stage unless all expected outputs already exist."""
    prefix = f"[{stage.stage_number:02d}/24]"
    print(f"{prefix} {stage.name}")

    if stage.note:
        print(f"  Note: {stage.note}")

    if stage.stage_number == 24:
        print(f"  FastAPI: {FASTAPI_COMMAND}")
        print(f"  Streamlit: {STREAMLIT_COMMAND}")
        return

    if not force and stage.outputs and outputs_exist(stage.outputs):
        print("  Skipped: expected output files already exist.")
        return

    print(f"  Running: {stage.command_label}")
    if stage.function_path:
        call_function_path(stage.function_path)
    elif stage.subprocess_args:
        run_subprocess_command((PYTHON_EXE, *stage.subprocess_args))
    else:
        print("  No runnable command required for this stage.")


def outputs_exist(outputs: tuple[str, ...]) -> bool:
    """Return True when every expected output for a stage exists."""
    return all((PROJECT_ROOT / output).exists() for output in outputs)


def call_function_path(function_path: str) -> Any:
    """Import and call a no-argument pipeline function by 'module:function' path."""
    module_name, function_name = function_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    function: Callable[[], Any] = getattr(module, function_name)
    return function()


def run_subprocess_command(command: tuple[str, ...]) -> None:
    """Run a local Python command and fail clearly if it exits unsuccessfully."""
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def verify_important_outputs(
    required_outputs: tuple[str, ...] = IMPORTANT_OUTPUTS,
) -> dict[str, bool]:
    """Check that the key demo outputs exist after the final flow."""
    return {output: (PROJECT_ROOT / output).exists() for output in required_outputs}


def print_output_verification(verification: dict[str, bool]) -> None:
    """Print a compact pass/missing table for important outputs."""
    print("\nImportant output verification")
    for output, exists in verification.items():
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {output}")


def print_launch_commands() -> None:
    """Print local server commands without starting the servers."""
    print("\nLocal demo launch commands")
    print(f"  FastAPI:   {FASTAPI_COMMAND}")
    print(f"  Streamlit: {STREAMLIT_COMMAND}")
    print("  API docs:  http://127.0.0.1:8000/docs")
    print("  Dashboard: http://localhost:8501")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the final demo runner."""
    args = parse_args(argv)
    run_full_demo(
        force=args.force,
        skip_tests=args.skip_tests,
        skip_dashboard_note=args.skip_dashboard_note,
    )


if __name__ == "__main__":
    main()
