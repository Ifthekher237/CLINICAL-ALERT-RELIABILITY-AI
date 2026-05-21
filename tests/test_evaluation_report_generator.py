"""Focused tests for Step 24 evaluation report generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reports import evaluation_report_generator as generator


REQUIRED_HEADINGS = [
    "## 1. Project Overview",
    "## 2. Dataset Summary",
    "## 3. Alert Generation Summary",
    "## 4. Alert Auditing and Fatigue Reduction",
    "## 5. Workflow Simulation Summary",
    "## 6. Reliability Monitoring Summary",
    "## 7. Drift Detection Summary",
    "## 8. Model Update and RL Threshold Simulation",
    "## 9. LLM/RAG/Action Recommendation Summary",
    "## 10. Outcome Effectiveness Evaluation",
    "## 11. Failure Mode Simulation",
    "## 12. Scenario Testing Summary",
    "## 13. Real-World Deployment Readiness Discussion",
    "## 14. Key Findings",
    "## 15. Limitations",
    "## 16. Conclusion",
]


def test_safe_load_json_handles_missing_file(tmp_path: Path) -> None:
    loaded = generator.safe_load_json(str(tmp_path / "missing.json"))

    assert loaded == {}


def test_safe_load_csv_handles_missing_file(tmp_path: Path) -> None:
    loaded = generator.safe_load_csv(str(tmp_path / "missing.csv"))

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.empty


def test_formatting_functions_work() -> None:
    assert generator.format_number(1445) == "1,445"
    assert generator.format_number(0.8967) == "0.8967"
    assert generator.format_percent(0.1719) == "17.19%"
    assert generator.get_nested_metric({"dataset": {"total_patients": 5}}, "dataset", "total_patients") == 5
    assert generator.get_nested_metric({}, "dataset", "total_patients", 0) == 0


def test_report_generation_returns_markdown_string(tmp_path: Path) -> None:
    output_path = tmp_path / "evaluation_results.md"
    report = generator.generate_evaluation_report(str(output_path))

    assert isinstance(report, str)
    assert report.startswith("# Clinical Alert Reliability AI")
    assert output_path.exists()


def test_report_includes_all_required_headings(tmp_path: Path) -> None:
    report = generator.generate_evaluation_report(str(tmp_path / "report.md"))

    for heading in REQUIRED_HEADINGS:
        assert heading in report


def test_new_section_functions_exist() -> None:
    for function_name in [
        "generate_outcome_effectiveness_section",
        "generate_failure_mode_section",
        "generate_scenario_testing_section",
        "generate_deployment_readiness_discussion_section",
    ]:
        assert hasattr(generator, function_name)
        assert callable(getattr(generator, function_name))


def test_report_includes_new_step_24d_sections(tmp_path: Path) -> None:
    report = generator.generate_evaluation_report(str(tmp_path / "report.md"))

    assert "Outcome Effectiveness Evaluation" in report
    assert "Failure Mode Simulation" in report
    assert "Scenario Testing Summary" in report
    assert "Real-World Deployment Readiness Discussion" in report
    assert "simulated associations only" in report


def test_report_includes_simulation_only_disclaimer(tmp_path: Path) -> None:
    report = generator.generate_evaluation_report(str(tmp_path / "report.md")).lower()

    assert "simulated healthcare ai engineering prototype" in report
    assert "not clinically validated" in report
    assert "must not be used for real patient care" in report


def test_report_does_not_include_unsafe_clinical_claims(tmp_path: Path) -> None:
    report = generator.generate_evaluation_report(str(tmp_path / "report.md")).lower()

    unsafe_positive_claims = [
        "is clinically validated",
        "safe for real patient care",
        "safe for real patients",
        "clinically proven",
        "diagnoses patients",
        "diagnose",
        "diagnosis",
        "recommend treatment",
        "recommends treatment",
        "treatment",
        "replaces clinicians",
        "approved medical device",
        "real-world deployment ready",
    ]

    for phrase in unsafe_positive_claims:
        assert phrase not in report


def test_output_file_is_saved(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "evaluation_results.md"
    report = generator.generate_evaluation_report(str(output_path))

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == report


def test_missing_optional_files_do_not_crash_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_paths = {
        key: str(tmp_path / f"missing_{key}.csv")
        for key in generator.DATA_PATHS
    }
    missing_paths["metrics_summary"] = str(tmp_path / "missing_metrics.json")
    missing_paths["clinician_response_summary"] = str(tmp_path / "missing_response.json")
    missing_paths["reliability_summary"] = str(tmp_path / "missing_reliability.json")
    missing_paths["drift_summary"] = str(tmp_path / "missing_drift.json")
    missing_paths["rl_policy_summary"] = str(tmp_path / "missing_rl.json")
    missing_paths["outcome_effectiveness_summary"] = str(tmp_path / "missing_outcome.json")
    missing_paths["failure_mode_summary"] = str(tmp_path / "missing_failure.json")
    missing_paths["scenario_test_summary"] = str(tmp_path / "missing_scenario.json")
    monkeypatch.setattr(generator, "DATA_PATHS", missing_paths)

    report = generator.generate_evaluation_report(str(tmp_path / "report.md"))

    assert "# Clinical Alert Reliability AI" in report
    for heading in REQUIRED_HEADINGS:
        assert heading in report
