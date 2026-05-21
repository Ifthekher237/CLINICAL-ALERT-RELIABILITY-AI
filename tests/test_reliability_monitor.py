"""Focused tests for Step 12 reliability monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.monitoring import reliability_monitor


def _sample_alerts_and_responses() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create compact alert/response data with a critical safety case."""
    alerts = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-P0001-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "low",
                "critical_flag": False,
                "safety_priority": "routine",
                "escalation_recommendation": "no_escalation",
                "final_alert_status": "grouped",
            },
            {
                "alert_id": "ALERT-P0001-002",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:30:00",
                "severity": "medium",
                "critical_flag": False,
                "safety_priority": "review",
                "escalation_recommendation": "clinician_review",
                "final_alert_status": "active",
            },
            {
                "alert_id": "ALERT-P0002-001",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 10:00:00",
                "severity": "critical",
                "critical_flag": True,
                "safety_priority": "immediate",
                "escalation_recommendation": "immediate_escalation",
                "final_alert_status": "active",
            },
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "response_id": "RESP-P0001-001",
                "alert_id": "ALERT-P0001-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "low",
                "final_alert_status": "grouped",
                "fatigue_action": "group_repeated",
                "simulated_response": "marked_false",
                "response_time_minutes": 55.0,
                "response_reason": "Marked false in simulation",
                "clinician_burden_score": 0.6,
                "perceived_alert_usefulness": 0.2,
                "workflow_stage": "closed",
                "escalation_completed": False,
                "response_simulation_note": "Simulated response only.",
            },
            {
                "response_id": "RESP-P0001-002",
                "alert_id": "ALERT-P0001-002",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:30:00",
                "severity": "medium",
                "final_alert_status": "active",
                "fatigue_action": "retain",
                "simulated_response": "delayed",
                "response_time_minutes": 80.0,
                "response_reason": "Delayed in simulation",
                "clinician_burden_score": 0.7,
                "perceived_alert_usefulness": 0.4,
                "workflow_stage": "triage_queue",
                "escalation_completed": False,
                "response_simulation_note": "Simulated response only.",
            },
            {
                "response_id": "RESP-P0002-001",
                "alert_id": "ALERT-P0002-001",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 10:00:00",
                "severity": "critical",
                "final_alert_status": "active",
                "fatigue_action": "retain",
                "simulated_response": "escalated",
                "response_time_minutes": 3.0,
                "response_reason": "Escalated in simulation",
                "clinician_burden_score": 0.5,
                "perceived_alert_usefulness": 0.9,
                "workflow_stage": "escalated_review",
                "escalation_completed": True,
                "response_simulation_note": "Simulated response only.",
            },
        ]
    )
    return alerts, responses


def test_alerts_and_response_logs_can_be_loaded() -> None:
    alerts = reliability_monitor.load_fatigue_reduced_alerts(
        "data/processed/fatigue_reduced_alerts.csv"
    )
    responses = reliability_monitor.load_response_logs(
        "data/processed/clinician_response_logs.csv"
    )

    assert not alerts.empty
    assert not responses.empty


def test_merge_keeps_required_fields() -> None:
    alerts, responses = _sample_alerts_and_responses()
    merged = reliability_monitor.merge_alerts_and_responses(alerts, responses)

    assert len(merged) == len(alerts)
    assert {"alert_id", "simulated_response", "response_time_minutes"}.issubset(merged.columns)


def test_monitoring_windows_are_created() -> None:
    alerts, responses = _sample_alerts_and_responses()
    merged = reliability_monitor.merge_alerts_and_responses(alerts, responses)
    windowed = reliability_monitor.create_monitoring_windows(merged, window_minutes=120)

    assert "monitoring_window_id" in windowed.columns
    assert "window_start" in windowed.columns
    assert windowed["monitoring_window_id"].nunique() >= 1


def test_reliability_output_columns_scores_and_values_are_valid() -> None:
    alerts, responses = _sample_alerts_and_responses()
    results = reliability_monitor.monitor_reliability(alerts, responses)

    assert set(reliability_monitor.REQUIRED_OUTPUT_COLUMNS).issubset(results.columns)
    assert results["reliability_score"].between(0, 1).all()
    assert set(results["reliability_status"]).issubset(
        reliability_monitor.VALID_RELIABILITY_STATUSES
    )
    assert set(results["review_recommendation"]).issubset(
        reliability_monitor.VALID_REVIEW_RECOMMENDATIONS
    )


def test_ignored_critical_alert_forces_unsafe_review_required() -> None:
    alerts, responses = _sample_alerts_and_responses()
    responses.loc[responses["alert_id"] == "ALERT-P0002-001", "simulated_response"] = "ignored"
    results = reliability_monitor.monitor_reliability(alerts, responses)
    critical_window = results[results["critical_alerts"] > 0].iloc[0]

    assert critical_window["reliability_status"] == "unsafe_review_required"
    assert critical_window["review_recommendation"] == "urgent_human_review"


def test_results_csv_is_saved(tmp_path: Path) -> None:
    alerts, responses = _sample_alerts_and_responses()
    results = reliability_monitor.monitor_reliability(alerts, responses)
    output_path = reliability_monitor.save_reliability_results(
        results,
        tmp_path / "reliability_monitoring_results.csv",
    )

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(reliability_monitor.REQUIRED_OUTPUT_COLUMNS).issubset(saved.columns)


def test_summary_json_is_saved(tmp_path: Path) -> None:
    summary = {
        "total_monitoring_windows": 2,
        "average_reliability_score": 0.75,
        "simulation_note": "Simulated reliability metrics only.",
    }
    output_path = reliability_monitor.save_reliability_summary(
        summary,
        tmp_path / "reliability_summary.json",
    )

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["total_monitoring_windows"] == 2


def test_pipeline_saves_results_and_summary(tmp_path: Path) -> None:
    results_path = tmp_path / "reliability_monitoring_results.csv"
    summary_path = tmp_path / "reliability_summary.json"
    results = reliability_monitor.run_reliability_monitoring_pipeline(
        alerts_path="data/processed/fatigue_reduced_alerts.csv",
        responses_path="data/processed/clinician_response_logs.csv",
        results_path=results_path,
        summary_path=summary_path,
    )

    assert results_path.exists()
    assert summary_path.exists()
    assert not results.empty
