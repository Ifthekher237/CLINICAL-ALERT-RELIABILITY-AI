"""Focused tests for Step 8 safety guardrails."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.alerts import safety_guardrails


def _sample_alerts() -> pd.DataFrame:
    """Create a compact set of generated alerts for guardrail tests."""
    return pd.DataFrame(
        [
            {
                "alert_id": "ALERT-P0001-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "critical",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.92,
                "trigger_reason": "Critical simulated vital-sign pattern",
                "source_model": "random_forest+time_series_rules",
                "recommended_review_time": "immediate review",
                "critical_flag": True,
            },
            {
                "alert_id": "ALERT-P0002-001",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 08:05:00",
                "severity": "high",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.64,
                "trigger_reason": "Elevated combined simulated alert risk",
                "source_model": "random_forest",
                "recommended_review_time": "review within 10 minutes",
                "critical_flag": False,
            },
            {
                "alert_id": "ALERT-P0003-001",
                "patient_id": "P0003",
                "timestamp": "2026-01-01 08:10:00",
                "severity": "medium",
                "alert_type": "anomaly_detected",
                "risk_score": 0.45,
                "trigger_reason": "Unusual oxygen saturation pattern",
                "source_model": "isolation_forest+time_series_rules",
                "recommended_review_time": "review within 30 minutes",
                "critical_flag": False,
            },
            {
                "alert_id": "ALERT-P0004-001",
                "patient_id": "P0004",
                "timestamp": "2026-01-01 08:15:00",
                "severity": "low",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.28,
                "trigger_reason": "Elevated baseline future deterioration risk",
                "source_model": "random_forest",
                "recommended_review_time": "review within 60 minutes",
                "critical_flag": False,
            },
        ]
    )


def test_generated_alerts_can_be_loaded() -> None:
    alerts = safety_guardrails.load_generated_alerts("data/processed/generated_alerts.csv")

    assert not alerts.empty
    assert set(safety_guardrails.REQUIRED_INPUT_COLUMNS).issubset(alerts.columns)


def test_validate_alert_schema_requires_input_columns() -> None:
    alerts = _sample_alerts().drop(columns=["trigger_reason"])

    with pytest.raises(ValueError, match="missing required columns"):
        safety_guardrails.validate_alert_schema(alerts)


def test_required_output_columns_and_valid_values_are_created() -> None:
    reviewed = safety_guardrails.apply_safety_guardrails(_sample_alerts())

    assert set(safety_guardrails.REQUIRED_OUTPUT_COLUMNS).issubset(reviewed.columns)
    assert set(reviewed["guardrail_decision"].unique()).issubset(
        safety_guardrails.VALID_GUARDRAIL_DECISIONS
    )
    assert set(reviewed["safety_priority"].unique()).issubset(
        safety_guardrails.VALID_SAFETY_PRIORITIES
    )
    assert "suppress" not in set(reviewed["guardrail_decision"])


def test_critical_alerts_are_escalated_and_immediate() -> None:
    reviewed = safety_guardrails.apply_safety_guardrails(_sample_alerts())
    critical = reviewed[reviewed["severity"] == "critical"].iloc[0]

    assert critical["guardrail_decision"] == "escalate"
    assert critical["requires_human_review"] is True or critical["requires_human_review"] == True
    assert critical["safety_priority"] == "immediate"


def test_critical_flag_true_alerts_require_immediate_priority() -> None:
    alerts = _sample_alerts()
    alerts.loc[1, "severity"] = "high"
    alerts.loc[1, "critical_flag"] = True
    reviewed = safety_guardrails.apply_safety_guardrails(alerts)
    critical_flagged = reviewed[reviewed["alert_id"] == "ALERT-P0002-001"].iloc[0]

    assert critical_flagged["guardrail_decision"] == "escalate"
    assert critical_flagged["safety_priority"] == "immediate"


def test_missing_required_row_fields_trigger_manual_verification() -> None:
    row = _sample_alerts().iloc[0].copy()
    row["trigger_reason"] = ""
    decision = safety_guardrails.apply_guardrail_to_alert(row)

    assert decision["guardrail_decision"] == "manual_verification_required"
    assert decision["requires_human_review"] is True
    assert "trigger_reason" in decision["guardrail_reason"]


def test_output_csv_is_saved(tmp_path: Path) -> None:
    reviewed = safety_guardrails.apply_safety_guardrails(_sample_alerts())
    output_path = safety_guardrails.save_guardrail_reviewed_alerts(
        reviewed,
        tmp_path / "guardrail_reviewed_alerts.csv",
    )

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(safety_guardrails.REQUIRED_OUTPUT_COLUMNS).issubset(saved.columns)


def test_pipeline_saves_reviewed_alerts(tmp_path: Path) -> None:
    output_path = tmp_path / "guardrail_reviewed_alerts.csv"
    reviewed = safety_guardrails.run_safety_guardrail_pipeline(
        input_path="data/processed/generated_alerts.csv",
        output_path=output_path,
    )

    assert output_path.exists()
    assert not reviewed.empty
    assert "suppress" not in set(reviewed["guardrail_decision"])
