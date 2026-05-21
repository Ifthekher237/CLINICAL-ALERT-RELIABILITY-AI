"""Focused tests for Step 11 workflow simulation and response tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.workflow import clinician_simulator, response_tracker


def _sample_fatigue_reduced_alerts() -> pd.DataFrame:
    """Create compact fatigue-reduced alerts with critical and repeated cases."""
    base = {
        "source_model": "random_forest",
        "recommended_review_time": "review within 60 minutes",
        "guardrail_decision": "allow",
        "guardrail_action": "Allow alert for downstream logging and audit",
        "guardrail_reason": "Low severity alert is allowed and retained for later audit.",
        "requires_human_review": False,
        "safety_priority": "routine",
        "actionability_score": 0.45,
        "fatigue_risk_score": 0.70,
        "urgency_score": 0.30,
        "false_positive_likelihood": 0.55,
        "confidence_score": 0.55,
        "audit_status": "repeated_low_value",
        "escalation_recommendation": "monitor",
        "audit_reason": "Repeated recent alerts increase fatigue risk",
        "recent_alert_count": 1,
        "recent_same_pattern_count": 1,
        "recent_max_risk_score": 0.28,
        "recent_max_severity_rank": 1,
        "repeated_alert_pattern": True,
        "worsening_repeated_pattern": False,
        "fatigue_action": "group_repeated",
        "fatigue_reason": "Repeated low/medium alert grouped with recent similar alerts.",
        "original_alert_retained": False,
        "grouped_alert_count": 2,
        "fatigue_reduction_safe": True,
        "final_alert_status": "grouped",
    }
    return pd.DataFrame(
        [
            {
                **base,
                "alert_id": "ALERT-P0001-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "low",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.28,
                "trigger_reason": "Elevated baseline future deterioration risk",
                "critical_flag": False,
            },
            {
                **base,
                "alert_id": "ALERT-P0001-002",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:15:00",
                "severity": "medium",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.34,
                "trigger_reason": "Elevated baseline future deterioration risk",
                "critical_flag": False,
                "fatigue_action": "delay_non_critical",
                "final_alert_status": "delayed",
            },
            {
                **base,
                "alert_id": "ALERT-P0002-001",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 02:00:00",
                "severity": "critical",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.95,
                "trigger_reason": "Critical simulated vital-sign pattern",
                "critical_flag": True,
                "guardrail_decision": "escalate",
                "guardrail_action": "Preserve and escalate critical alert",
                "guardrail_reason": "Critical alerts must not be downgraded or ignored.",
                "requires_human_review": True,
                "safety_priority": "immediate",
                "actionability_score": 0.95,
                "fatigue_risk_score": 0.20,
                "false_positive_likelihood": 0.05,
                "audit_status": "high_priority",
                "escalation_recommendation": "immediate_escalation",
                "fatigue_action": "retain",
                "original_alert_retained": True,
                "grouped_alert_count": 1,
                "final_alert_status": "active",
            },
        ]
    )


def test_fatigue_reduced_alerts_can_be_loaded() -> None:
    alerts = clinician_simulator.load_fatigue_reduced_alerts(
        "data/processed/fatigue_reduced_alerts.csv"
    )

    assert not alerts.empty
    assert set(clinician_simulator.REQUIRED_INPUT_COLUMNS).issubset(alerts.columns)


def test_response_log_output_contains_required_columns_and_valid_values() -> None:
    responses = clinician_simulator.simulate_clinician_responses(
        _sample_fatigue_reduced_alerts(),
        seed=7,
    )

    assert set(clinician_simulator.REQUIRED_RESPONSE_COLUMNS).issubset(responses.columns)
    assert set(responses["simulated_response"]).issubset(clinician_simulator.VALID_RESPONSES)
    assert set(responses["workflow_stage"]).issubset(clinician_simulator.VALID_WORKFLOW_STAGES)
    assert pd.api.types.is_numeric_dtype(responses["response_time_minutes"])
    assert (responses["response_time_minutes"] >= 0).all()


def test_critical_immediate_alerts_are_not_ignored() -> None:
    responses = clinician_simulator.simulate_clinician_responses(
        _sample_fatigue_reduced_alerts(),
        seed=42,
    )
    critical_response = responses[responses["alert_id"] == "ALERT-P0002-001"].iloc[0]

    assert critical_response["simulated_response"] != "ignored"
    assert critical_response["workflow_stage"] in {"escalated_review", "clinician_review"}


def test_output_csv_is_saved(tmp_path: Path) -> None:
    responses = clinician_simulator.simulate_clinician_responses(
        _sample_fatigue_reduced_alerts(),
        seed=5,
    )
    output_path = clinician_simulator.save_response_logs(
        responses,
        tmp_path / "clinician_response_logs.csv",
    )

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(clinician_simulator.REQUIRED_RESPONSE_COLUMNS).issubset(saved.columns)


def test_reproducibility_with_seed() -> None:
    alerts = _sample_fatigue_reduced_alerts()
    first = clinician_simulator.simulate_clinician_responses(alerts, seed=123)
    second = clinician_simulator.simulate_clinician_responses(alerts, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_response_summary_contains_rates() -> None:
    responses = clinician_simulator.simulate_clinician_responses(
        _sample_fatigue_reduced_alerts(),
        seed=11,
    )
    summary = response_tracker.calculate_response_summary(responses)

    assert {
        "ignored_alert_rate",
        "delayed_alert_rate",
        "escalation_rate",
        "average_response_time_minutes",
        "response_by_severity",
    }.issubset(summary.keys())
    assert 0 <= summary["ignored_alert_rate"] <= 1
    assert 0 <= summary["delayed_alert_rate"] <= 1
    assert 0 <= summary["escalation_rate"] <= 1


def test_response_by_severity_returns_dataframe() -> None:
    responses = clinician_simulator.simulate_clinician_responses(
        _sample_fatigue_reduced_alerts(),
        seed=13,
    )
    by_severity = response_tracker.calculate_response_by_severity(responses)

    assert not by_severity.empty
    assert {
        "severity",
        "count",
        "ignored_rate",
        "delayed_rate",
        "escalation_rate",
        "average_response_time_minutes",
    }.issubset(by_severity.columns)


def test_summary_json_is_saved(tmp_path: Path) -> None:
    responses = clinician_simulator.simulate_clinician_responses(
        _sample_fatigue_reduced_alerts(),
        seed=17,
    )
    summary = response_tracker.calculate_response_summary(responses)
    output_path = response_tracker.save_response_summary(
        summary,
        tmp_path / "clinician_response_summary.json",
    )

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert "ignored_alert_rate" in saved
