"""Focused tests for Step 9 alert auditing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.alerts import alert_auditor


def _sample_reviewed_alerts() -> pd.DataFrame:
    """Create compact guardrail-reviewed alerts with repeated patient history."""
    base = {
        "recommended_review_time": "review within 60 minutes",
        "guardrail_action": "Allow alert for downstream logging and audit",
        "guardrail_reason": "Low severity alert is allowed and retained for later audit.",
        "requires_human_review": False,
        "safety_priority": "routine",
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
                "source_model": "random_forest",
                "critical_flag": False,
                "guardrail_decision": "allow",
            },
            {
                **base,
                "alert_id": "ALERT-P0002-001",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 08:02:00",
                "severity": "low",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.29,
                "trigger_reason": "Elevated baseline future deterioration risk",
                "source_model": "random_forest",
                "critical_flag": False,
                "guardrail_decision": "allow",
            },
            {
                **base,
                "alert_id": "ALERT-P0001-002",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:20:00",
                "severity": "medium",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.42,
                "trigger_reason": "Elevated baseline future deterioration risk",
                "source_model": "random_forest",
                "critical_flag": False,
                "guardrail_decision": "allow",
            },
            {
                **base,
                "alert_id": "ALERT-P0003-001",
                "patient_id": "P0003",
                "timestamp": "2026-01-01 08:30:00",
                "severity": "critical",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.93,
                "trigger_reason": "Critical simulated vital-sign pattern",
                "source_model": "random_forest+time_series_rules",
                "recommended_review_time": "immediate review",
                "critical_flag": True,
                "guardrail_decision": "escalate",
                "guardrail_action": "Preserve and escalate critical alert",
                "guardrail_reason": "Critical alerts must not be downgraded or ignored.",
                "requires_human_review": True,
                "safety_priority": "immediate",
            },
        ]
    )


def test_input_alerts_can_be_loaded() -> None:
    alerts = alert_auditor.load_guardrail_reviewed_alerts(
        "data/processed/guardrail_reviewed_alerts.csv"
    )

    assert not alerts.empty
    assert set(alert_auditor.REQUIRED_INPUT_COLUMNS).issubset(alerts.columns)


def test_required_input_columns_are_validated() -> None:
    alerts = _sample_reviewed_alerts().drop(columns=["guardrail_reason"])

    with pytest.raises(ValueError, match="missing required columns"):
        alert_auditor.validate_audit_input_schema(alerts)


def test_audit_output_columns_statuses_and_recommendations_are_valid() -> None:
    audited = alert_auditor.audit_alerts(_sample_reviewed_alerts())

    assert set(alert_auditor.AUDIT_COLUMNS).issubset(audited.columns)
    assert set(audited["audit_status"].unique()).issubset(alert_auditor.VALID_AUDIT_STATUSES)
    assert set(audited["escalation_recommendation"].unique()).issubset(
        alert_auditor.VALID_ESCALATION_RECOMMENDATIONS
    )


def test_scores_are_numeric_and_between_zero_and_one() -> None:
    audited = alert_auditor.audit_alerts(_sample_reviewed_alerts())
    score_columns = [
        "actionability_score",
        "fatigue_risk_score",
        "urgency_score",
        "false_positive_likelihood",
        "confidence_score",
    ]

    for column in score_columns:
        assert pd.api.types.is_numeric_dtype(audited[column])
        assert audited[column].between(0, 1).all()


def test_patient_specific_history_uses_past_alerts_only() -> None:
    audited = alert_auditor.audit_alerts(_sample_reviewed_alerts())
    p1_first = audited[audited["alert_id"] == "ALERT-P0001-001"].iloc[0]
    p1_second = audited[audited["alert_id"] == "ALERT-P0001-002"].iloc[0]
    p2_first = audited[audited["alert_id"] == "ALERT-P0002-001"].iloc[0]

    assert p1_first["fatigue_risk_score"] == 0
    assert p2_first["fatigue_risk_score"] == 0
    assert p1_second["fatigue_risk_score"] > 0


def test_output_csv_is_saved(tmp_path: Path) -> None:
    audited = alert_auditor.audit_alerts(_sample_reviewed_alerts())
    output_path = alert_auditor.save_audited_alerts(
        audited,
        tmp_path / "audited_alerts.csv",
    )

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(alert_auditor.AUDIT_COLUMNS).issubset(saved.columns)


def test_no_suppress_decision_is_created() -> None:
    audited = alert_auditor.audit_alerts(_sample_reviewed_alerts())

    assert "suppress" not in set(audited["audit_status"])
    assert "suppress" not in set(audited["escalation_recommendation"])


def test_pipeline_saves_audited_alerts(tmp_path: Path) -> None:
    output_path = tmp_path / "audited_alerts.csv"
    audited = alert_auditor.run_alert_audit_pipeline(
        input_path="data/processed/guardrail_reviewed_alerts.csv",
        output_path=output_path,
    )

    assert output_path.exists()
    assert not audited.empty
    assert set(alert_auditor.AUDIT_COLUMNS).issubset(audited.columns)
