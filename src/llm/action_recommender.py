"""Safe workflow action recommender for simulated alerts.

Step 21 recommends non-clinical workflow actions such as review, escalation,
monitoring, grouping, and manual verification. It does not diagnose, recommend
treatment, replace clinicians, or act as a medical decision-maker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.rag_engine import RAGEngine, create_rag_engine


DEFAULT_ALERTS_PATH = Path("data/processed/fatigue_reduced_alerts.csv")
DEFAULT_AUDITED_PATH = Path("data/processed/audited_alerts.csv")
DEFAULT_RESPONSE_PATH = Path("data/processed/clinician_response_logs.csv")
DEFAULT_EXPLANATIONS_PATH = Path("data/processed/alert_explanations.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/action_recommendations.csv")

SAFETY_NOTE = (
    "Simulation-only workflow recommendation for project review; not medical advice "
    "and not a substitute for clinician judgment."
)

ALLOWED_RECOMMENDED_ACTIONS = {
    "immediate_human_review",
    "urgent_clinician_review",
    "nurse_review",
    "monitor_trend",
    "group_repeated_low_value_alert",
    "request_manual_verification",
    "flag_sensor_reliability_issue",
    "no_action_beyond_logging",
}

ALLOWED_ACTION_PRIORITIES = {
    "immediate",
    "urgent",
    "routine",
    "low",
}

ALLOWED_CONFIDENCE_LEVELS = {
    "high",
    "medium",
    "low",
}

REQUIRED_OUTPUT_COLUMNS = [
    "recommendation_id",
    "alert_id",
    "patient_id",
    "severity",
    "alert_type",
    "recommended_action",
    "action_priority",
    "action_reason",
    "confidence_level",
    "human_review_required",
    "rag_sources",
    "safety_note",
    "generated_at",
]

UNSAFE_WORDING = [
    "diagnosis",
    "diagnose",
    "treatment",
    "prescribe",
    "the patient has",
    "disease",
    "cure",
]


def load_fatigue_reduced_alerts(path: str | Path = DEFAULT_ALERTS_PATH) -> pd.DataFrame:
    """Load fatigue-reduced alerts from Step 10."""
    return _load_csv(path, "fatigue-reduced alerts")


def load_audited_alerts(path: str | Path = DEFAULT_AUDITED_PATH) -> pd.DataFrame:
    """Load audited alerts from Step 9."""
    return _load_csv(path, "audited alerts")


def load_response_logs(path: str | Path = DEFAULT_RESPONSE_PATH) -> pd.DataFrame:
    """Load simulated clinician response logs from Step 11."""
    return _load_csv(path, "clinician response logs")


def load_alert_explanations(path: str | Path = DEFAULT_EXPLANATIONS_PATH) -> pd.DataFrame:
    """Load generated alert explanations from Step 19."""
    return _load_csv(path, "alert explanations")


def merge_recommendation_context(
    alerts_df: pd.DataFrame,
    audited_df: pd.DataFrame,
    response_df: pd.DataFrame,
    explanations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge alert, audit, response, and explanation context by alert_id."""
    for label, df in {
        "alerts_df": alerts_df,
        "audited_df": audited_df,
        "response_df": response_df,
        "explanations_df": explanations_df,
    }.items():
        if "alert_id" not in df.columns:
            raise ValueError(f"{label} must contain alert_id.")

    alert_columns = [
        column
        for column in [
            "alert_id",
            "patient_id",
            "timestamp",
            "severity",
            "alert_type",
            "risk_score",
            "trigger_reason",
            "critical_flag",
            "guardrail_decision",
            "guardrail_reason",
            "requires_human_review",
            "safety_priority",
            "actionability_score",
            "fatigue_risk_score",
            "false_positive_likelihood",
            "audit_status",
            "escalation_recommendation",
            "audit_reason",
            "recent_alert_count",
            "recent_same_pattern_count",
            "repeated_alert_pattern",
            "fatigue_action",
            "final_alert_status",
            "grouped_alert_count",
        ]
        if column in alerts_df.columns
    ]
    audit_columns = [
        column
        for column in [
            "alert_id",
            "audit_status",
            "actionability_score",
            "fatigue_risk_score",
            "urgency_score",
            "false_positive_likelihood",
            "confidence_score",
            "escalation_recommendation",
            "audit_reason",
        ]
        if column in audited_df.columns
    ]
    response_columns = [
        column
        for column in [
            "alert_id",
            "simulated_response",
            "response_time_minutes",
            "response_reason",
            "workflow_stage",
            "escalation_completed",
        ]
        if column in response_df.columns
    ]
    explanation_columns = [
        column
        for column in [
            "alert_id",
            "explanation_text",
            "uncertainty_note",
            "safety_note",
        ]
        if column in explanations_df.columns
    ]

    merged = alerts_df[alert_columns].merge(
        audited_df[audit_columns],
        on="alert_id",
        how="left",
        suffixes=("", "_audit"),
        validate="one_to_one",
    )
    merged = merged.merge(
        response_df[response_columns],
        on="alert_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        explanations_df[explanation_columns],
        on="alert_id",
        how="left",
        validate="one_to_one",
    )
    return merged.reset_index(drop=True)


def retrieve_relevant_rules(
    row: pd.Series,
    rag_engine: RAGEngine | None = None,
) -> dict[str, Any]:
    """Retrieve local project rules for recommendation grounding."""
    query = " ".join(
        str(value)
        for value in [
            _value(row, "severity"),
            _value(row, "alert_type"),
            _value(row, "trigger_reason"),
            _value(row, "audit_status"),
            _value(row, "final_alert_status"),
            _value(row, "simulated_response"),
            "human review workflow safety critical manual verification",
        ]
        if value is not None
    )
    try:
        engine = rag_engine or create_rag_engine()
        contexts = engine.retrieve(query, top_k=3)
        sources = engine.get_source_summary(contexts)
        return {
            "query": query,
            "contexts": contexts,
            "sources": sources,
            "rag_sources": ";".join(source["source_file"] for source in sources),
        }
    except Exception:
        return {
            "query": query,
            "contexts": [],
            "sources": [],
            "rag_sources": "",
        }


def recommend_action_for_alert(
    row: pd.Series,
    rag_context: dict | None = None,
) -> dict[str, Any]:
    """Recommend one safe workflow action for a simulated alert."""
    severity = _severity(row)
    safety_priority = _text(row, "safety_priority").lower()
    escalation = _text(row, "escalation_recommendation").lower()
    alert_type = _text(row, "alert_type").lower()
    trigger_reason = _text(row, "trigger_reason").lower()
    audit_status = _text(row, "audit_status").lower()
    guardrail_decision = _text(row, "guardrail_decision").lower()
    final_status = _text(row, "final_alert_status").lower()
    fatigue_action = _text(row, "fatigue_action").lower()

    critical = (
        severity == "critical"
        or _bool_value(row, "critical_flag")
        or safety_priority == "immediate"
        or escalation == "immediate_escalation"
    )
    high_or_urgent = severity == "high" or safety_priority == "urgent" or escalation == "urgent_review"
    sensor_issue = any(
        phrase in trigger_reason
        for phrase in ["sensor", "noise", "missing", "unstable signal", "manual verification"]
    )
    anomaly_or_uncertain = (
        "anomaly" in alert_type
        or "uncertain" in trigger_reason
        or guardrail_decision == "manual_verification_required"
        or _bool_value(row, "requires_human_review")
    )
    repeated_low_value = (
        severity == "low"
        and final_status in {"grouped", "priority_downgraded", "delayed"}
        and fatigue_action in {"group_repeated", "downgrade_priority", "delay_non_critical"}
    )

    if critical:
        action = "immediate_human_review"
        priority = "immediate"
        human_review_required = True
    elif high_or_urgent:
        action = "urgent_clinician_review"
        priority = "urgent"
        human_review_required = True
    elif sensor_issue:
        action = "flag_sensor_reliability_issue"
        priority = "urgent" if severity in {"high", "critical"} else "routine"
        human_review_required = True
    elif anomaly_or_uncertain:
        action = "request_manual_verification"
        priority = "urgent" if severity in {"high", "critical"} else "routine"
        human_review_required = True
    elif repeated_low_value:
        action = "group_repeated_low_value_alert"
        priority = "low"
        human_review_required = False
    elif severity == "medium" and audit_status in {"review_needed", "high_priority"}:
        if _numeric(row, "fatigue_risk_score") >= 0.60:
            action = "monitor_trend"
        else:
            action = "nurse_review"
        priority = "routine"
        human_review_required = action == "nurse_review"
    elif severity == "low" and _numeric(row, "actionability_score") < 0.45:
        action = "no_action_beyond_logging"
        priority = "low"
        human_review_required = False
    else:
        action = "monitor_trend"
        priority = "routine"
        human_review_required = bool(audit_status in {"review_needed", "high_priority"})

    confidence = calculate_action_confidence(row, rag_context)
    reason = generate_action_reason(row, action, rag_context)
    reason = _sanitize_reason(reason)

    return {
        "recommendation_id": _record_id("ACTION", _value(row, "alert_id")),
        "alert_id": _value(row, "alert_id"),
        "patient_id": _value(row, "patient_id"),
        "severity": _value(row, "severity"),
        "alert_type": _value(row, "alert_type"),
        "recommended_action": action,
        "action_priority": priority,
        "action_reason": reason,
        "confidence_level": confidence,
        "human_review_required": bool(human_review_required),
        "rag_sources": (rag_context or {}).get("rag_sources", ""),
        "safety_note": SAFETY_NOTE,
        "generated_at": _timestamp_now(),
    }


def calculate_action_confidence(
    row: pd.Series,
    rag_context: dict | None = None,
) -> str:
    """Estimate confidence in the workflow recommendation."""
    score = 0.45
    severity = _severity(row)
    if severity in {"critical", "high"}:
        score += 0.20
    if _bool_value(row, "critical_flag") or _text(row, "safety_priority").lower() in {"immediate", "urgent"}:
        score += 0.20
    if _numeric(row, "confidence_score") >= 0.75:
        score += 0.10
    if _text(row, "trigger_reason"):
        score += 0.05
    if rag_context and rag_context.get("sources"):
        score += 0.10
    if _numeric(row, "false_positive_likelihood") >= 0.60:
        score -= 0.20
    if _text(row, "simulated_response").lower() in {"marked_false", "ignored"}:
        score -= 0.10

    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def generate_action_reason(
    row: pd.Series,
    recommended_action: str,
    rag_context: dict | None = None,
) -> str:
    """Generate a readable reason for a workflow recommendation."""
    severity = _value(row, "severity", "unknown")
    trigger_reason = _value(row, "trigger_reason", "no trigger reason recorded")
    audit_status = _value(row, "audit_status", "no audit status")
    final_status = _value(row, "final_alert_status", "no fatigue status")
    simulated_response = _value(row, "simulated_response", "no simulated response")
    source_text = ""
    if rag_context and rag_context.get("sources"):
        source_files = ", ".join(source["source_file"] for source in rag_context["sources"])
        source_text = f" Local project rules referenced: {source_files}."

    reason_templates = {
        "immediate_human_review": (
            f"The alert is {severity} or safety-sensitive, so the simulated workflow should preserve immediate human review. "
            f"Trigger: {trigger_reason}."
        ),
        "urgent_clinician_review": (
            f"The alert is {severity} or urgent in the safety workflow, so it should be routed for urgent clinician review in simulation. "
            f"Audit status: {audit_status}."
        ),
        "nurse_review": (
            f"The alert has review-relevant evidence but is not immediate, so nurse review is a reasonable simulated workflow step. "
            f"Audit status: {audit_status}; fatigue status: {final_status}."
        ),
        "monitor_trend": (
            f"The alert can be monitored in trend context because it is not currently immediate and should remain visible for simulated follow-up. "
            f"Trigger: {trigger_reason}."
        ),
        "group_repeated_low_value_alert": (
            f"The alert appears low severity and already fatigue-reviewed as repeated or lower-priority, so grouping preserves an audit trail while reducing simulated burden."
        ),
        "request_manual_verification": (
            f"The alert has anomaly, uncertainty, guardrail, or manual-verification signals, so simulated manual verification is appropriate before further workflow interpretation."
        ),
        "flag_sensor_reliability_issue": (
            f"The trigger or context suggests possible sensor/noise/missing-data concern, so the simulated workflow should flag sensor reliability for verification."
        ),
        "no_action_beyond_logging": (
            f"The alert appears low-priority in simulation, so logging without additional workflow action is enough unless later monitoring changes."
        ),
    }
    return (
        reason_templates.get(recommended_action, "Recommendation generated from simulated workflow rules.")
        + f" Simulated response: {simulated_response}."
        + source_text
        + " This is simulation-only workflow guidance."
    )


def generate_action_recommendations(
    alerts_df: pd.DataFrame,
    audited_df: pd.DataFrame,
    response_df: pd.DataFrame,
    explanations_df: pd.DataFrame,
    limit: int | None = 50,
    use_rag: bool = True,
) -> pd.DataFrame:
    """Generate workflow action recommendations for a batch of simulated alerts."""
    merged = merge_recommendation_context(alerts_df, audited_df, response_df, explanations_df)
    if limit is not None:
        merged = merged.head(max(int(limit), 0))

    rag_engine = create_rag_engine() if use_rag else None
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        rag_context = retrieve_relevant_rules(row, rag_engine) if use_rag else {"rag_sources": ""}
        records.append(recommend_action_for_alert(row, rag_context=rag_context))
    return pd.DataFrame(records, columns=REQUIRED_OUTPUT_COLUMNS)


def save_action_recommendations(df: pd.DataFrame, path: str | Path) -> Path:
    """Save action recommendations to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_action_recommendation_pipeline(
    alerts_path: str | Path = DEFAULT_ALERTS_PATH,
    audited_path: str | Path = DEFAULT_AUDITED_PATH,
    response_path: str | Path = DEFAULT_RESPONSE_PATH,
    explanations_path: str | Path = DEFAULT_EXPLANATIONS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = 50,
    use_rag: bool = True,
) -> pd.DataFrame:
    """Run Step 21 action recommendation generation end to end."""
    alerts_df = load_fatigue_reduced_alerts(alerts_path)
    audited_df = load_audited_alerts(audited_path)
    response_df = load_response_logs(response_path)
    explanations_df = load_alert_explanations(explanations_path)
    recommendations = generate_action_recommendations(
        alerts_df,
        audited_df,
        response_df,
        explanations_df,
        limit=limit,
        use_rag=use_rag,
    )
    saved_path = save_action_recommendations(recommendations, output_path)
    recommendations.attrs["output_path"] = str(saved_path)
    return recommendations


def recommend_action() -> pd.DataFrame:
    """Compatibility wrapper for older placeholder imports."""
    return run_action_recommendation_pipeline()


def _load_csv(path: str | Path, label: str) -> pd.DataFrame:
    input_path = _resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{label} file not found: {input_path}")
    return pd.read_csv(input_path)


def _severity(row: pd.Series) -> str:
    return _text(row, "severity").lower()


def _text(row: pd.Series, column: str) -> str:
    value = _value(row, column, "")
    return "" if value is None else str(value)


def _value(row: pd.Series, column: str, default: Any = None) -> Any:
    value = row.get(column, default)
    if value is None or pd.isna(value):
        return default
    return value


def _numeric(row: pd.Series, column: str) -> float:
    value = _value(row, column, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_value(row: pd.Series, column: str) -> bool:
    value = _value(row, column, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _sanitize_reason(reason: str) -> str:
    lowered = reason.lower()
    if any(phrase in lowered for phrase in UNSAFE_WORDING):
        return (
            "Recommendation reason was constrained to simulation-only workflow language. "
            "Use project safety rules, preserve uncertainty, and keep human review for safety-sensitive alerts. "
            "This is project workflow guidance only."
        )
    return " ".join(reason.split())


def _record_id(prefix: str, alert_id: Any) -> str:
    safe_alert_id = "".join(
        character if character.isalnum() else "-"
        for character in str(alert_id or "unknown")
    ).strip("-")
    return f"{prefix}-{safe_alert_id}"


def _timestamp_now() -> str:
    return str(pd.Timestamp.now("UTC").floor("s").tz_localize(None))


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


if __name__ == "__main__":
    result_df = run_action_recommendation_pipeline()
    print(f"Total recommendations generated: {len(result_df)}")
    if not result_df.empty:
        print("Action distribution:")
        print(result_df["recommended_action"].value_counts().to_string())
        print("\nPriority distribution:")
        print(result_df["action_priority"].value_counts().to_string())
        print("\nFirst few recommendations:")
        preview_columns = [
            "alert_id",
            "severity",
            "recommended_action",
            "action_priority",
            "confidence_level",
            "rag_sources",
        ]
        print(result_df[preview_columns].head().to_string(index=False))
    print(f"\nSaved action recommendations to {result_df.attrs.get('output_path', DEFAULT_OUTPUT_PATH)}")
