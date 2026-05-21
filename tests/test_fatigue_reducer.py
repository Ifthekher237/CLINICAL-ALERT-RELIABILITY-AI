"""Focused tests for Step 10 alert fatigue reduction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.alerts import fatigue_reducer


def _sample_audited_alerts() -> pd.DataFrame:
    """Create audited alerts with protected and reducible cases."""
    base = {
        "source_model": "random_forest",
        "recommended_review_time": "review within 60 minutes",
        "guardrail_action": "Allow alert for downstream logging and audit",
        "guardrail_reason": "Low severity alert is allowed and retained for later audit.",
        "requires_human_review": False,
        "safety_priority": "routine",
        "actionability_score": 0.45,
        "fatigue_risk_score": 0.72,
        "urgency_score": 0.30,
        "false_positive_likelihood": 0.58,
        "confidence_score": 0.55,
        "audit_status": "repeated_low_value",
        "escalation_recommendation": "monitor",
        "audit_reason": "Repeated recent alerts increase fatigue risk",
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
                "guardrail_decision": "allow",
            },
            {
                **base,
                "alert_id": "ALERT-P0001-002",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:10:00",
                "severity": "low",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.30,
                "trigger_reason": "Elevated baseline future deterioration risk",
                "critical_flag": False,
                "guardrail_decision": "allow",
            },
            {
                **base,
                "alert_id": "ALERT-P0001-003",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:20:00",
                "severity": "medium",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.32,
                "trigger_reason": "Elevated baseline future deterioration risk",
                "critical_flag": False,
                "guardrail_decision": "allow",
            },
            {
                **base,
                "alert_id": "ALERT-P0002-001",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 08:05:00",
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
                "urgency_score": 1.0,
                "false_positive_likelihood": 0.05,
                "confidence_score": 0.92,
                "audit_status": "high_priority",
                "escalation_recommendation": "immediate_escalation",
                "audit_reason": "High priority because urgency is elevated",
            },
            {
                **base,
                "alert_id": "ALERT-P0003-001",
                "patient_id": "P0003",
                "timestamp": "2026-01-01 08:15:00",
                "severity": "high",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.76,
                "trigger_reason": "Multiple sustained abnormal vital signs",
                "critical_flag": False,
                "guardrail_decision": "escalate",
                "requires_human_review": True,
                "safety_priority": "urgent",
                "audit_status": "high_priority",
                "escalation_recommendation": "urgent_review",
            },
        ]
    )


def test_audited_alerts_can_be_loaded() -> None:
    alerts = fatigue_reducer.load_audited_alerts("data/processed/audited_alerts.csv")

    assert not alerts.empty
    assert set(fatigue_reducer.REQUIRED_INPUT_COLUMNS).issubset(alerts.columns)


def test_validate_fatigue_input_schema_requires_columns() -> None:
    alerts = _sample_audited_alerts().drop(columns=["audit_reason"])

    with pytest.raises(ValueError, match="missing required columns"):
        fatigue_reducer.validate_fatigue_input_schema(alerts)


def test_required_output_columns_and_allowed_values() -> None:
    reduced = fatigue_reducer.apply_fatigue_reduction(_sample_audited_alerts())

    assert set(fatigue_reducer.FATIGUE_COLUMNS).issubset(reduced.columns)
    assert set(reduced["fatigue_action"].unique()).issubset(
        fatigue_reducer.ALLOWED_FATIGUE_ACTIONS
    )
    assert set(reduced["final_alert_status"].unique()).issubset(
        fatigue_reducer.ALLOWED_FINAL_ALERT_STATUSES
    )


def test_critical_alerts_remain_active() -> None:
    reduced = fatigue_reducer.apply_fatigue_reduction(_sample_audited_alerts())
    critical = reduced[reduced["severity"] == "critical"].iloc[0]

    assert critical["fatigue_action"] == "retain"
    assert critical["final_alert_status"] == "active"
    assert critical["original_alert_retained"] is True or critical["original_alert_retained"] == True


def test_critical_flag_and_immediate_escalation_remain_active() -> None:
    alerts = _sample_audited_alerts()
    alerts.loc[0, "critical_flag"] = True
    alerts.loc[0, "safety_priority"] = "immediate"
    alerts.loc[0, "escalation_recommendation"] = "immediate_escalation"
    reduced = fatigue_reducer.apply_fatigue_reduction(alerts)
    protected = reduced[reduced["alert_id"] == "ALERT-P0001-001"].iloc[0]

    assert protected["fatigue_action"] == "retain"
    assert protected["final_alert_status"] == "active"


def test_repeated_low_medium_alerts_can_be_reduced() -> None:
    reduced = fatigue_reducer.apply_fatigue_reduction(_sample_audited_alerts())
    repeated = reduced[reduced["patient_id"] == "P0001"]

    assert repeated["fatigue_action"].isin(
        ["group_repeated", "delay_non_critical", "downgrade_priority"]
    ).any()
    assert repeated["final_alert_status"].isin(
        ["grouped", "delayed", "priority_downgraded"]
    ).any()


def test_no_alerts_are_physically_removed() -> None:
    alerts = _sample_audited_alerts()
    reduced = fatigue_reducer.apply_fatigue_reduction(alerts)

    assert len(reduced) == len(alerts)
    assert set(reduced["alert_id"]) == set(alerts["alert_id"])


def test_fatigue_metrics_return_expected_keys() -> None:
    alerts = _sample_audited_alerts()
    reduced = fatigue_reducer.apply_fatigue_reduction(alerts)
    metrics = fatigue_reducer.calculate_fatigue_metrics(alerts, reduced)

    assert {
        "total_original_alerts",
        "total_active_after_reduction",
        "grouped_alerts",
        "delayed_alerts",
        "downgraded_alerts",
        "escalated_patterns",
        "alert_reduction_rate",
        "critical_alerts_preserved",
        "critical_preservation_rate",
        "repeated_alert_reduction",
    }.issubset(metrics.keys())
    assert metrics["critical_alerts_preserved"] is True


def test_output_csv_is_saved(tmp_path: Path) -> None:
    reduced = fatigue_reducer.apply_fatigue_reduction(_sample_audited_alerts())
    output_path = fatigue_reducer.save_fatigue_reduced_alerts(
        reduced,
        tmp_path / "fatigue_reduced_alerts.csv",
    )

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(fatigue_reducer.FATIGUE_COLUMNS).issubset(saved.columns)


def test_pipeline_saves_fatigue_reduced_alerts(tmp_path: Path) -> None:
    output_path = tmp_path / "fatigue_reduced_alerts.csv"
    reduced = fatigue_reducer.run_fatigue_reduction_pipeline(
        input_path="data/processed/audited_alerts.csv",
        output_path=output_path,
    )

    assert output_path.exists()
    assert not reduced.empty
    assert set(fatigue_reducer.FATIGUE_COLUMNS).issubset(reduced.columns)
