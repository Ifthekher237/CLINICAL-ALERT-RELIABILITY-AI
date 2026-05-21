"""Focused tests for Step 21 safe action recommender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.llm.action_recommender import (
    ALLOWED_ACTION_PRIORITIES,
    ALLOWED_CONFIDENCE_LEVELS,
    ALLOWED_RECOMMENDED_ACTIONS,
    REQUIRED_OUTPUT_COLUMNS,
    generate_action_recommendations,
    load_alert_explanations,
    load_audited_alerts,
    load_fatigue_reduced_alerts,
    load_response_logs,
    merge_recommendation_context,
    recommend_action_for_alert,
    retrieve_relevant_rules,
    run_action_recommendation_pipeline,
    save_action_recommendations,
)


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alerts = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-CRITICAL",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "critical",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.95,
                "trigger_reason": "Critical simulated instability",
                "critical_flag": True,
                "requires_human_review": True,
                "safety_priority": "immediate",
                "guardrail_decision": "escalate",
                "fatigue_action": "retain",
                "final_alert_status": "active",
                "grouped_alert_count": 1,
            },
            {
                "alert_id": "ALERT-LOW",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 09:00:00",
                "severity": "low",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.22,
                "trigger_reason": "Repeated low-value alert",
                "critical_flag": False,
                "requires_human_review": False,
                "safety_priority": "routine",
                "guardrail_decision": "allow",
                "fatigue_action": "group_repeated",
                "final_alert_status": "grouped",
                "grouped_alert_count": 4,
                "actionability_score": 0.30,
                "fatigue_risk_score": 0.75,
            },
        ]
    )
    audited = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-CRITICAL",
                "audit_status": "high_priority",
                "actionability_score": 0.90,
                "fatigue_risk_score": 0.10,
                "urgency_score": 1.0,
                "false_positive_likelihood": 0.05,
                "confidence_score": 0.90,
                "escalation_recommendation": "immediate_escalation",
                "audit_reason": "Critical simulated alert",
            },
            {
                "alert_id": "ALERT-LOW",
                "audit_status": "repeated_low_value",
                "actionability_score": 0.30,
                "fatigue_risk_score": 0.75,
                "urgency_score": 0.20,
                "false_positive_likelihood": 0.55,
                "confidence_score": 0.45,
                "escalation_recommendation": "monitor",
                "audit_reason": "Repeated low value alert",
            },
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-CRITICAL",
                "simulated_response": "escalated",
                "response_time_minutes": 2.0,
                "response_reason": "Escalated in simulation",
                "workflow_stage": "escalated_review",
                "escalation_completed": True,
            },
            {
                "alert_id": "ALERT-LOW",
                "simulated_response": "delayed",
                "response_time_minutes": 60.0,
                "response_reason": "Delayed grouped alert",
                "workflow_stage": "triage_queue",
                "escalation_completed": False,
            },
        ]
    )
    explanations = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-CRITICAL",
                "explanation_text": "Simulation-only critical alert explanation.",
                "uncertainty_note": "Human review required.",
                "safety_note": "Simulation only.",
            },
            {
                "alert_id": "ALERT-LOW",
                "explanation_text": "Simulation-only low alert explanation.",
                "uncertainty_note": "Monitor uncertainty.",
                "safety_note": "Simulation only.",
            },
        ]
    )
    return alerts, audited, responses, explanations


def test_input_csvs_can_be_loaded() -> None:
    alerts = load_fatigue_reduced_alerts("data/processed/fatigue_reduced_alerts.csv")
    audited = load_audited_alerts("data/processed/audited_alerts.csv")
    responses = load_response_logs("data/processed/clinician_response_logs.csv")
    explanations = load_alert_explanations("data/processed/alert_explanations.csv")

    assert not alerts.empty
    assert not audited.empty
    assert not responses.empty
    assert not explanations.empty


def test_recommendation_context_merges_correctly() -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    merged = merge_recommendation_context(alerts, audited, responses, explanations)

    assert len(merged) == 2
    assert "simulated_response" in merged.columns
    assert "explanation_text" in merged.columns
    assert merged.loc[0, "audit_status"] == "high_priority"


def test_output_dataframe_has_required_columns() -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    recommendations = generate_action_recommendations(
        alerts,
        audited,
        responses,
        explanations,
        limit=None,
        use_rag=False,
    )

    assert set(REQUIRED_OUTPUT_COLUMNS).issubset(recommendations.columns)
    assert len(recommendations) == 2


def test_recommended_action_values_are_valid() -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    recommendations = generate_action_recommendations(
        alerts,
        audited,
        responses,
        explanations,
        use_rag=False,
    )

    assert set(recommendations["recommended_action"]).issubset(ALLOWED_RECOMMENDED_ACTIONS)
    assert set(recommendations["action_priority"]).issubset(ALLOWED_ACTION_PRIORITIES)
    assert set(recommendations["confidence_level"]).issubset(ALLOWED_CONFIDENCE_LEVELS)


def test_critical_alerts_always_get_immediate_human_review() -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    row = merge_recommendation_context(alerts, audited, responses, explanations).iloc[0]
    recommendation = recommend_action_for_alert(row, rag_context={"rag_sources": "safety_rules.md"})

    assert recommendation["recommended_action"] == "immediate_human_review"
    assert recommendation["action_priority"] == "immediate"
    assert recommendation["human_review_required"] is True


def test_repeated_low_value_alert_can_be_grouped() -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    row = merge_recommendation_context(alerts, audited, responses, explanations).iloc[1]
    recommendation = recommend_action_for_alert(row, rag_context={"rag_sources": "workflow_rules.md"})

    assert recommendation["recommended_action"] == "group_repeated_low_value_alert"
    assert recommendation["action_priority"] == "low"


def test_no_recommendation_contains_treatment_or_diagnosis_wording() -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    recommendations = generate_action_recommendations(
        alerts,
        audited,
        responses,
        explanations,
        use_rag=False,
    )
    combined_text = " ".join(
        recommendations["action_reason"].astype(str).tolist()
        + recommendations["safety_note"].astype(str).tolist()
    ).lower()

    assert "diagnosis" not in combined_text
    assert "treatment" not in combined_text
    assert "prescribe" not in combined_text
    assert "the patient has" not in combined_text


def test_rag_source_filenames_are_included_when_enabled() -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    recommendations = generate_action_recommendations(
        alerts,
        audited,
        responses,
        explanations,
        limit=1,
        use_rag=True,
    )

    assert recommendations.iloc[0]["rag_sources"]
    assert ".md" in recommendations.iloc[0]["rag_sources"]


def test_retrieve_relevant_rules_returns_sources() -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    row = merge_recommendation_context(alerts, audited, responses, explanations).iloc[0]
    rag_context = retrieve_relevant_rules(row)

    assert "sources" in rag_context
    assert rag_context["sources"]
    assert rag_context["rag_sources"]


def test_output_csv_is_saved(tmp_path: Path) -> None:
    alerts, audited, responses, explanations = _sample_inputs()
    recommendations = generate_action_recommendations(
        alerts,
        audited,
        responses,
        explanations,
        use_rag=False,
    )
    output_path = save_action_recommendations(
        recommendations,
        tmp_path / "action_recommendations.csv",
    )

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(REQUIRED_OUTPUT_COLUMNS).issubset(saved.columns)


def test_pipeline_runs_without_internet_paid_apis_ollama_or_real_patient_data(tmp_path: Path) -> None:
    output_path = tmp_path / "action_recommendations.csv"
    recommendations = run_action_recommendation_pipeline(
        alerts_path="data/processed/fatigue_reduced_alerts.csv",
        audited_path="data/processed/audited_alerts.csv",
        response_path="data/processed/clinician_response_logs.csv",
        explanations_path="data/processed/alert_explanations.csv",
        output_path=output_path,
        limit=5,
        use_rag=True,
    )

    assert output_path.exists()
    assert len(recommendations) == 5
    assert set(recommendations["recommended_action"]).issubset(ALLOWED_RECOMMENDED_ACTIONS)
