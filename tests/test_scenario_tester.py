"""Focused tests for Step 24C scenario testing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.monitoring import scenario_tester


REQUIRED_SUMMARY_KEYS = {
    "total_scenarios",
    "scenario_distribution",
    "overall_status_distribution",
    "safety_check_distribution",
    "average_reliability_score",
    "average_ignored_alert_rate",
    "average_delayed_alert_rate",
    "average_outcome_effectiveness_score",
    "failure_mode_trigger_rate",
    "human_review_required_rate",
    "passed_safety_checks",
    "warning_safety_checks",
    "failed_safety_checks",
    "simulation_only_note",
}


def _sample_context() -> dict[str, float | int]:
    return {
        "total_patients": 5,
        "missing_data_rate": 0.03,
        "total_raw_alerts": 100,
        "critical_alert_count": 20,
        "critical_preservation_rate": 1.0,
        "active_alerts_after_reduction": 82,
        "total_reduced_alerts": 18,
        "grouped_alert_count": 12,
        "delayed_alert_count": 2,
        "downgraded_alert_count": 4,
        "ignored_alert_rate": 0.05,
        "delayed_alert_rate": 0.06,
        "average_reliability_score": 0.90,
        "severe_drift_count": 5,
        "average_drift_score": 0.40,
        "average_outcome_effectiveness_score": 0.66,
        "unsafe_failure_count": 3,
        "low_alert_count": 20,
        "medium_alert_count": 30,
        "high_alert_count": 30,
        "noisy_sensor_spike_events": 6,
        "missing_patient_data_events": 5,
        "alert_overload_events": 2,
        "repeated_low_value_alerts_events": 12,
        "delayed_response_failure_events": 10,
        "model_confidence_drop_events": 4,
        "data_distribution_shift_events": 7,
    }


def test_all_seven_scenarios_are_generated() -> None:
    results = scenario_tester.build_scenario_results_table(_sample_context())

    assert len(results) == 7
    assert set(results["scenario_name"]) == set(scenario_tester.SCENARIO_NAMES)


def test_required_columns_exist() -> None:
    results = scenario_tester.build_scenario_results_table(_sample_context())

    assert set(scenario_tester.REQUIRED_OUTPUT_COLUMNS).issubset(results.columns)


def test_valid_categories_statuses_and_drift_levels_only() -> None:
    results = scenario_tester.build_scenario_results_table(_sample_context())

    assert set(results["scenario_category"]).issubset(scenario_tester.VALID_SCENARIO_CATEGORIES)
    assert set(results["safety_check_status"]).issubset(scenario_tester.VALID_SAFETY_CHECK_STATUSES)
    assert set(results["overall_scenario_status"]).issubset(scenario_tester.VALID_OVERALL_SCENARIO_STATUSES)
    assert set(results["drift_risk_level"]).issubset(scenario_tester.VALID_DRIFT_RISK_LEVELS)


def test_scores_and_rates_are_between_zero_and_one() -> None:
    results = scenario_tester.build_scenario_results_table(_sample_context())

    for column in [
        "ignored_alert_rate",
        "delayed_alert_rate",
        "reliability_score",
        "outcome_effectiveness_score",
    ]:
        assert pd.to_numeric(results[column], errors="coerce").between(0, 1).all()


def test_human_review_required_is_boolean() -> None:
    results = scenario_tester.build_scenario_results_table(_sample_context())

    assert results["human_review_required"].map(type).eq(bool).all()


def test_sudden_critical_event_requires_human_review() -> None:
    results = scenario_tester.build_scenario_results_table(_sample_context())
    critical_row = results[results["scenario_name"] == "sudden_critical_event"].iloc[0]

    assert critical_row["critical_alert_count"] > 0
    assert critical_row["human_review_required"] is True
    assert critical_row["safety_check_status"] in {"warning", "failed"}


def test_scenario_summaries_do_not_contain_treatment_or_diagnosis_wording() -> None:
    results = scenario_tester.build_scenario_results_table(_sample_context())
    combined_text = " ".join(
        results["scenario_summary"].astype(str).tolist()
        + results["simulation_note"].astype(str).tolist()
    ).lower()

    for unsafe_phrase in [
        "diagnosis",
        "diagnose",
        "recommend treatment",
        "prescribe",
        "clinically validated",
        "safe for real patient",
    ]:
        assert unsafe_phrase not in combined_text


def test_output_csv_and_json_are_saved(tmp_path: Path) -> None:
    results = scenario_tester.build_scenario_results_table(_sample_context())
    summary = scenario_tester.calculate_scenario_summary_metrics(results)
    csv_path = scenario_tester.save_scenario_results(results, str(tmp_path / "scenario_test_results.csv"))
    json_path = scenario_tester.save_scenario_summary(summary, str(tmp_path / "scenario_test_summary.json"))

    assert csv_path.exists()
    assert json_path.exists()
    assert set(scenario_tester.REQUIRED_OUTPUT_COLUMNS).issubset(pd.read_csv(csv_path).columns)
    with json_path.open("r", encoding="utf-8") as file:
        saved_summary = json.load(file)
    assert REQUIRED_SUMMARY_KEYS.issubset(saved_summary.keys())


def test_missing_files_do_not_crash_pipeline(tmp_path: Path) -> None:
    results = scenario_tester.run_scenario_testing_pipeline(
        raw_alerts_path=str(tmp_path / "missing_raw.csv"),
        fatigue_path=str(tmp_path / "missing_fatigue.csv"),
        audited_path=str(tmp_path / "missing_audited.csv"),
        response_path=str(tmp_path / "missing_response.csv"),
        reliability_path=str(tmp_path / "missing_reliability.csv"),
        drift_path=str(tmp_path / "missing_drift.csv"),
        outcome_results_path=str(tmp_path / "missing_outcome_results.csv"),
        outcome_summary_path=str(tmp_path / "missing_outcome_summary.json"),
        failure_results_path=str(tmp_path / "missing_failure_results.csv"),
        failure_summary_path=str(tmp_path / "missing_failure_summary.json"),
        metrics_path=str(tmp_path / "missing_metrics.json"),
        output_path=str(tmp_path / "scenario_test_results.csv"),
        summary_path=str(tmp_path / "scenario_test_summary.json"),
    )

    assert len(results) == 7
    assert set(results["scenario_name"]) == set(scenario_tester.SCENARIO_NAMES)


def test_pipeline_runs_successfully(tmp_path: Path) -> None:
    results_path = tmp_path / "scenario_test_results.csv"
    summary_path = tmp_path / "scenario_test_summary.json"

    results = scenario_tester.run_scenario_testing_pipeline(
        output_path=str(results_path),
        summary_path=str(summary_path),
    )

    assert results_path.exists()
    assert summary_path.exists()
    assert len(results) == 7
    assert set(scenario_tester.REQUIRED_OUTPUT_COLUMNS).issubset(results.columns)
