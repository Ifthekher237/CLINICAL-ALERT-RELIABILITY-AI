"""Safety wording and review-gating checks for Step 25."""

from __future__ import annotations

from pathlib import Path

from testing_utils import bool_series, load_csv


UNSAFE_WORDING = [
    "prescribe",
    "treatment should",
    "medically proven",
    "clinically proven",
    "safe for patients",
    "safe for real patients",
]


def test_no_unsafe_wording_appears_in_user_facing_artifacts() -> None:
    text_paths = [
        Path("reports/evaluation_results.md"),
        Path("data/processed/alert_explanations.csv"),
        Path("data/processed/action_recommendations.csv"),
        Path("data/processed/outcome_effectiveness_results.csv"),
        Path("data/processed/failure_mode_results.csv"),
        Path("data/processed/scenario_test_results.csv"),
    ]

    for path in text_paths:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in UNSAFE_WORDING:
            assert phrase not in text, f"Unsafe wording `{phrase}` found in {path}"


def test_medical_claim_words_are_only_used_as_negated_disclaimers() -> None:
    text_paths = [
        Path("reports/evaluation_results.md"),
        Path("data/processed/alert_explanations.csv"),
        Path("data/processed/action_recommendations.csv"),
    ]
    allowed_contexts = [
        "not a diagnosis",
        "does not diagnose",
        "does not diagnose, recommend treatment",
        "not treatment advice",
        "does not recommend treatment",
        "cannot provide medical advice",
        "not medical advice",
    ]

    for path in text_paths:
        for line in path.read_text(encoding="utf-8").lower().splitlines():
            if any(term in line for term in ["diagnosis", "diagnose", "treatment"]):
                assert any(context in line for context in allowed_contexts), (
                    f"Clinical wording without clear negation in {path}: {line[:160]}"
                )


def test_simulation_only_disclaimers_appear_in_safety_sensitive_outputs() -> None:
    required_paths = [
        "data/processed/outcome_effectiveness_results.csv",
        "data/processed/failure_mode_results.csv",
        "data/processed/scenario_test_results.csv",
        "reports/evaluation_results.md",
    ]

    for path in required_paths:
        text = Path(path).read_text(encoding="utf-8").lower()
        assert "simulation-only" in text or "simulated" in text


def test_critical_alerts_require_review_or_escalation() -> None:
    fatigue = load_csv("data/processed/fatigue_reduced_alerts.csv")
    critical = fatigue[bool_series(fatigue["critical_flag"])]

    assert not critical.empty
    assert critical["safety_priority"].astype(str).str.lower().eq("immediate").all()
    assert critical["escalation_recommendation"].astype(str).str.lower().eq("immediate_escalation").all()
    assert bool_series(critical["requires_human_review"]).all()


def test_unsafe_review_required_scenarios_require_human_review() -> None:
    scenarios = load_csv("data/processed/scenario_test_results.csv")
    unsafe = scenarios[scenarios["overall_scenario_status"].astype(str) == "unsafe_review_required"]

    if not unsafe.empty:
        assert bool_series(unsafe["human_review_required"]).all()


def test_failure_events_with_unsafe_status_require_human_review() -> None:
    failures = load_csv("data/processed/failure_mode_results.csv")
    unsafe = failures[failures["safety_status"].astype(str) == "unsafe_review_required"]

    assert not unsafe.empty
    assert bool_series(unsafe["requires_human_review"]).all()


def test_no_alert_suppression_decision_was_created() -> None:
    checked_columns = {
        "data/processed/guardrail_reviewed_alerts.csv": ["guardrail_decision", "guardrail_action"],
        "data/processed/fatigue_reduced_alerts.csv": ["fatigue_action", "final_alert_status"],
        "data/processed/audited_alerts.csv": ["audit_status", "escalation_recommendation"],
    }

    for path, columns in checked_columns.items():
        df = load_csv(path)
        for column in columns:
            assert column in df.columns
            assert not df[column].fillna("").astype(str).str.lower().str.contains("suppress").any()
