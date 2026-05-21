"""Focused tests for Step 24A simulated outcome effectiveness evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.monitoring import outcome_evaluator


REQUIRED_SUMMARY_KEYS = {
    "total_evaluated_alerts",
    "improved_count",
    "unchanged_count",
    "worsened_count",
    "unknown_count",
    "useful_alert_rate",
    "useless_alert_rate",
    "action_to_outcome_success_rate",
    "average_outcome_effectiveness_score",
    "average_delayed_response_impact_score",
    "simulation_only_note",
}


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alerts = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-1",
                "patient_id": "P001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "critical",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.95,
            },
            {
                "alert_id": "ALERT-2",
                "patient_id": "P002",
                "timestamp": "2026-01-01 09:00:00",
                "severity": "low",
                "alert_type": "anomaly_detected",
                "risk_score": 0.25,
            },
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-1",
                "simulated_response": "escalated",
                "response_time_minutes": 4.0,
                "perceived_alert_usefulness": 0.9,
                "workflow_stage": "escalated_review",
                "escalation_completed": True,
            },
            {
                "alert_id": "ALERT-2",
                "simulated_response": "ignored",
                "response_time_minutes": 90.0,
                "perceived_alert_usefulness": 0.2,
                "workflow_stage": "closed",
                "escalation_completed": False,
            },
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-1",
                "recommended_action": "immediate_human_review",
                "action_priority": "immediate",
            },
            {
                "alert_id": "ALERT-2",
                "recommended_action": "no_action_beyond_logging",
                "action_priority": "low",
            },
        ]
    )
    patients = pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "timestamp": "2026-01-01 08:00:00",
                "patient_outcome_after_alert": "improved",
                "outcome_severity_change": -1.2,
            },
            {
                "patient_id": "P002",
                "timestamp": "2026-01-01 09:00:00",
                "patient_outcome_after_alert": "worsened",
                "outcome_severity_change": 1.5,
            },
        ]
    )
    return alerts, responses, actions, patients


def test_missing_files_do_not_crash(tmp_path: Path) -> None:
    assert outcome_evaluator.safe_load_csv(str(tmp_path / "missing.csv")).empty
    assert outcome_evaluator.safe_load_json(str(tmp_path / "missing.json")) == {}


def test_merge_outcome_context_works() -> None:
    alerts, responses, actions, patients = _sample_inputs()
    merged = outcome_evaluator.merge_outcome_context(alerts, responses, actions, patients)

    assert len(merged) == 2
    assert "simulated_response" in merged.columns
    assert "recommended_action" in merged.columns
    assert "outcome_severity_change" in merged.columns


def test_outcome_labels_are_valid() -> None:
    alerts, responses, actions, patients = _sample_inputs()
    results = outcome_evaluator.evaluate_alert_outcomes(alerts, responses, actions, patients)

    assert set(results["outcome_label"]).issubset(outcome_evaluator.VALID_OUTCOME_LABELS)
    assert results.loc[0, "outcome_label"] == "improved"
    assert results.loc[1, "outcome_label"] == "worsened"


def test_scores_are_between_zero_and_one() -> None:
    alerts, responses, actions, patients = _sample_inputs()
    results = outcome_evaluator.evaluate_alert_outcomes(alerts, responses, actions, patients)

    assert results["outcome_effectiveness_score"].between(0, 1).all()
    assert results["delayed_response_impact_score"].between(0, 1).all()


def test_boolean_outputs_are_boolean() -> None:
    alerts, responses, actions, patients = _sample_inputs()
    results = outcome_evaluator.evaluate_alert_outcomes(alerts, responses, actions, patients)

    assert results["timely_response"].map(type).eq(bool).all()
    assert results["action_taken"].map(type).eq(bool).all()
    assert results["alert_useful"].map(type).eq(bool).all()


def test_summary_has_required_keys() -> None:
    alerts, responses, actions, patients = _sample_inputs()
    results = outcome_evaluator.evaluate_alert_outcomes(alerts, responses, actions, patients)
    summary = outcome_evaluator.calculate_outcome_summary(results)

    assert REQUIRED_SUMMARY_KEYS.issubset(summary.keys())
    assert summary["total_evaluated_alerts"] == 2
    assert 0 <= summary["useful_alert_rate"] <= 1


def test_csv_and_json_outputs_are_saved(tmp_path: Path) -> None:
    alerts, responses, actions, patients = _sample_inputs()
    results = outcome_evaluator.evaluate_alert_outcomes(alerts, responses, actions, patients)
    summary = outcome_evaluator.calculate_outcome_summary(results)

    csv_path = outcome_evaluator.save_outcome_results(
        results,
        str(tmp_path / "outcome_effectiveness_results.csv"),
    )
    json_path = outcome_evaluator.save_outcome_summary(
        summary,
        str(tmp_path / "outcome_effectiveness_summary.json"),
    )

    assert csv_path.exists()
    assert json_path.exists()
    assert set(outcome_evaluator.REQUIRED_OUTPUT_COLUMNS).issubset(pd.read_csv(csv_path).columns)
    with json_path.open("r", encoding="utf-8") as file:
        saved_summary = json.load(file)
    assert REQUIRED_SUMMARY_KEYS.issubset(saved_summary.keys())


def test_pipeline_saves_outputs(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    summary_path = tmp_path / "summary.json"

    results = outcome_evaluator.run_outcome_evaluation_pipeline(
        results_path=str(results_path),
        summary_path=str(summary_path),
    )

    assert results_path.exists()
    assert summary_path.exists()
    assert set(outcome_evaluator.REQUIRED_OUTPUT_COLUMNS).issubset(results.columns)


def test_no_positive_clinical_claims_are_made() -> None:
    alerts, responses, actions, patients = _sample_inputs()
    results = outcome_evaluator.evaluate_alert_outcomes(alerts, responses, actions, patients)
    combined_text = " ".join(
        results["evaluation_reason"].astype(str).tolist()
        + results["simulation_only_note"].astype(str).tolist()
    ).lower()

    unsafe_claims = [
        "proves clinical effectiveness",
        "real patient benefit",
        "safe for patient care",
        "medical device evidence",
        "caused improvement",
    ]
    for claim in unsafe_claims:
        assert claim not in combined_text
