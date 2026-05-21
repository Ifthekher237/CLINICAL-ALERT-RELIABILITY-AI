"""Focused tests for Step 19 safe alert explanation generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.llm.explanation_generator import (
    REQUIRED_OUTPUT_COLUMNS,
    build_alert_explanation_prompt,
    generate_alert_explanations,
    generate_explanation_for_alert,
    generate_rule_based_explanation,
    load_audited_alerts,
    load_fatigue_reduced_alerts,
    load_response_logs,
    merge_alert_context,
    run_explanation_generation_pipeline,
    sanitize_explanation_text,
    save_alert_explanations,
)
from src.llm.llm_client import LLMClient, LLMResponse


def _sample_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alerts = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "high",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.82,
                "trigger_reason": "Multiple sustained abnormal vital signs",
                "critical_flag": False,
                "requires_human_review": True,
                "safety_priority": "urgent",
                "fatigue_action": "retain",
                "final_alert_status": "active",
            }
        ]
    )
    audited = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-001",
                "audit_status": "high_priority",
                "actionability_score": 0.85,
                "fatigue_risk_score": 0.20,
                "urgency_score": 0.90,
                "false_positive_likelihood": 0.10,
                "confidence_score": 0.88,
                "escalation_recommendation": "urgent_review",
                "audit_reason": "High-priority simulated alert",
            }
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-001",
                "simulated_response": "escalated",
                "response_time_minutes": 5.0,
                "response_reason": "Escalated in simulation",
                "workflow_stage": "escalated_review",
                "escalation_completed": True,
            }
        ]
    )
    return alerts, audited, responses


class UnsafeFakeClient:
    def generate(self, prompt: str, system_prompt=None, max_tokens: int = 300) -> LLMResponse:
        return LLMResponse(
            prompt=prompt,
            response_text="The patient has pneumonia and treatment should start now.",
            model_name="fake",
            backend="ollama",
            success=True,
            fallback_used=False,
            safety_note="fake",
            error_message=None,
        )


class SafeFakeClient:
    def generate(self, prompt: str, system_prompt=None, max_tokens: int = 300) -> LLMResponse:
        return LLMResponse(
            prompt=prompt,
            response_text=(
                "This simulated alert was generated from prototype system signals. "
                "It explains uncertainty and keeps human review in the loop."
            ),
            model_name="fake",
            backend="ollama",
            success=True,
            fallback_used=False,
            safety_note="fake",
            error_message=None,
        )


def test_input_csvs_can_be_loaded() -> None:
    alerts = load_fatigue_reduced_alerts("data/processed/fatigue_reduced_alerts.csv")
    audited = load_audited_alerts("data/processed/audited_alerts.csv")
    responses = load_response_logs("data/processed/clinician_response_logs.csv")

    assert not alerts.empty
    assert not audited.empty
    assert not responses.empty


def test_alert_audit_response_context_merges_correctly() -> None:
    alerts, audited, responses = _sample_context()
    merged = merge_alert_context(alerts, audited, responses)

    assert len(merged) == 1
    assert merged.iloc[0]["alert_id"] == "ALERT-001"
    assert merged.iloc[0]["audit_status"] == "high_priority"
    assert merged.iloc[0]["simulated_response"] == "escalated"


def test_prompt_includes_safety_constraints() -> None:
    alerts, audited, responses = _sample_context()
    row = merge_alert_context(alerts, audited, responses).iloc[0]
    prompt = build_alert_explanation_prompt(row).lower()

    assert "do not diagnose" in prompt
    assert "do not recommend treatment" in prompt
    assert "human review" in prompt
    assert "simulated" in prompt


def test_rule_based_explanation_contains_simulation_safety_note() -> None:
    alerts, audited, responses = _sample_context()
    row = merge_alert_context(alerts, audited, responses).iloc[0]
    explanation = generate_rule_based_explanation(row).lower()

    assert "simulated" in explanation
    assert "human review" in explanation
    assert "does not diagnose" in explanation
    assert "replace clinician review" in explanation


def test_generated_dataframe_has_required_columns() -> None:
    alerts, audited, responses = _sample_context()
    explanations = generate_alert_explanations(
        alerts,
        audited,
        responses,
        limit=1,
        use_llm=False,
    )

    assert set(REQUIRED_OUTPUT_COLUMNS).issubset(explanations.columns)
    assert len(explanations) == 1


def test_fallback_mode_works_without_ollama() -> None:
    alerts, audited, responses = _sample_context()
    row = merge_alert_context(alerts, audited, responses).iloc[0]
    client = LLMClient(base_url="http://127.0.0.1:9", timeout_seconds=1)
    explanation = generate_explanation_for_alert(row, llm_client=client, use_llm=True)

    assert explanation["fallback_used"] is True
    assert "simulated" in explanation["explanation_text"].lower()
    assert "human review" in explanation["explanation_text"].lower()


def test_unsafe_explanation_text_is_sanitized_or_replaced() -> None:
    unsafe = "Diagnosis is sepsis. The patient has an infection. Treatment should begin."
    sanitized = sanitize_explanation_text(unsafe).lower()

    assert "diagnosis is" not in sanitized
    assert "the patient has" not in sanitized
    assert "treatment should" not in sanitized
    assert "human review" in sanitized


def test_unsafe_llm_output_uses_rule_based_explanation() -> None:
    alerts, audited, responses = _sample_context()
    row = merge_alert_context(alerts, audited, responses).iloc[0]
    explanation = generate_explanation_for_alert(
        row,
        llm_client=UnsafeFakeClient(),
        use_llm=True,
    )

    text = explanation["explanation_text"].lower()
    assert explanation["fallback_used"] is True
    assert "the patient has" not in text
    assert "treatment should" not in text
    assert "multiple sustained abnormal vital signs" in text


def test_safe_llm_output_can_be_used() -> None:
    alerts, audited, responses = _sample_context()
    row = merge_alert_context(alerts, audited, responses).iloc[0]
    explanation = generate_explanation_for_alert(
        row,
        llm_client=SafeFakeClient(),
        use_llm=True,
    )

    assert explanation["fallback_used"] is False
    assert explanation["llm_backend"] == "ollama"


def test_explanation_output_does_not_contain_unsafe_claim_phrases() -> None:
    alerts, audited, responses = _sample_context()
    explanations = generate_alert_explanations(
        alerts,
        audited,
        responses,
        limit=1,
        use_llm=False,
    )
    text = explanations.iloc[0]["explanation_text"].lower()

    assert "diagnosis is" not in text
    assert "treatment should" not in text
    assert "doctor is not needed" not in text
    assert "ignore clinician review" not in text


def test_output_csv_is_saved(tmp_path: Path) -> None:
    alerts, audited, responses = _sample_context()
    explanations = generate_alert_explanations(
        alerts,
        audited,
        responses,
        limit=1,
        use_llm=False,
    )
    output_path = save_alert_explanations(explanations, tmp_path / "alert_explanations.csv")

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(REQUIRED_OUTPUT_COLUMNS).issubset(saved.columns)


def test_pipeline_saves_output_without_requiring_ollama(tmp_path: Path) -> None:
    output_path = tmp_path / "alert_explanations.csv"
    explanations = run_explanation_generation_pipeline(
        alerts_path="data/processed/fatigue_reduced_alerts.csv",
        audited_path="data/processed/audited_alerts.csv",
        response_path="data/processed/clinician_response_logs.csv",
        reliability_summary_path="data/processed/reliability_summary.json",
        drift_summary_path="data/processed/drift_summary.json",
        output_path=output_path,
        limit=3,
        use_llm=False,
    )

    assert output_path.exists()
    assert len(explanations) == 3
    assert set(REQUIRED_OUTPUT_COLUMNS).issubset(explanations.columns)
