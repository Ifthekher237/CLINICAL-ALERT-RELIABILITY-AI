"""Project-wide pipeline consistency checks for Step 25."""

from __future__ import annotations

import pandas as pd

from testing_utils import assert_unique, bool_series, load_csv


def test_alert_ids_remain_unique_across_alert_outputs() -> None:
    for path in [
        "data/processed/generated_alerts.csv",
        "data/processed/audited_alerts.csv",
        "data/processed/fatigue_reduced_alerts.csv",
        "data/processed/clinician_response_logs.csv",
        "data/processed/outcome_effectiveness_results.csv",
        "data/processed/failure_mode_results.csv",
        "data/processed/scenario_test_results.csv",
    ]:
        df = load_csv(path)
        key = {
            "data/processed/clinician_response_logs.csv": "response_id",
            "data/processed/outcome_effectiveness_results.csv": "outcome_eval_id",
            "data/processed/failure_mode_results.csv": "failure_event_id",
            "data/processed/scenario_test_results.csv": "scenario_test_id",
        }.get(path, "alert_id")
        assert_unique(df, key)


def test_patient_ids_are_consistent_between_alerts_and_responses() -> None:
    alerts = load_csv("data/processed/fatigue_reduced_alerts.csv")
    responses = load_csv("data/processed/clinician_response_logs.csv")
    simulated = load_csv("data/simulated/patient_monitoring.csv")

    simulated_patients = set(simulated["patient_id"].astype(str))
    assert set(alerts["patient_id"].astype(str)).issubset(simulated_patients)
    assert set(responses["patient_id"].astype(str)).issubset(simulated_patients)

    merged = alerts[["alert_id", "patient_id"]].merge(
        responses[["alert_id", "patient_id"]],
        on="alert_id",
        suffixes=("_alert", "_response"),
    )
    assert (merged["patient_id_alert"].astype(str) == merged["patient_id_response"].astype(str)).all()


def test_timestamps_are_parseable_across_pipeline_outputs() -> None:
    for path, timestamp_column in [
        ("data/simulated/patient_monitoring.csv", "timestamp"),
        ("data/processed/generated_alerts.csv", "timestamp"),
        ("data/processed/fatigue_reduced_alerts.csv", "timestamp"),
        ("data/processed/clinician_response_logs.csv", "timestamp"),
        ("data/processed/scenario_test_results.csv", "timestamp"),
    ]:
        df = load_csv(path)
        parsed = pd.to_datetime(df[timestamp_column], errors="coerce")
        assert parsed.notna().all(), f"Unparseable timestamps in {path}"


def test_fatigue_reduction_never_increases_alert_count() -> None:
    raw_alerts = load_csv("data/processed/generated_alerts.csv")
    fatigue = load_csv("data/processed/fatigue_reduced_alerts.csv")

    assert len(fatigue) <= len(raw_alerts)
    assert set(fatigue["alert_id"]).issubset(set(raw_alerts["alert_id"]))


def test_critical_alerts_remain_preserved_after_fatigue_reduction() -> None:
    fatigue = load_csv("data/processed/fatigue_reduced_alerts.csv")
    critical = fatigue[bool_series(fatigue["critical_flag"])]

    assert not critical.empty
    assert critical["final_alert_status"].astype(str).str.lower().eq("active").all()
    assert critical["fatigue_action"].astype(str).str.lower().isin(["retain", "escalate_pattern"]).all()


def test_outcome_evaluation_rows_align_with_alerts() -> None:
    alerts = load_csv("data/processed/fatigue_reduced_alerts.csv")
    outcomes = load_csv("data/processed/outcome_effectiveness_results.csv")

    assert len(outcomes) == len(alerts)
    assert set(outcomes["alert_id"]).issubset(set(alerts["alert_id"]))


def test_scenario_testing_references_valid_statuses() -> None:
    scenarios = load_csv("data/processed/scenario_test_results.csv")

    assert set(scenarios["safety_check_status"]).issubset({"passed", "warning", "failed"})
    assert set(scenarios["overall_scenario_status"]).issubset(
        {"stable", "monitored", "degraded", "unsafe_review_required"}
    )
    assert set(scenarios["drift_risk_level"]).issubset({"low", "medium", "high", "severe"})
