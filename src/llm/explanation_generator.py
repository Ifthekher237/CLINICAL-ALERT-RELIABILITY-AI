"""Safe alert explanation generator for simulated clinical-alert outputs.

Step 19 creates concise explanations from existing alert, audit, fatigue, and
workflow artifacts. The generator may use the Step 18 LLM client, but it always
keeps a rule-based fallback. It does not diagnose, recommend treatment, replace
clinician review, use RAG, or act as a medical device.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.llm_client import LLMClient, LLMResponse, create_llm_client


DEFAULT_ALERTS_PATH = Path("data/processed/fatigue_reduced_alerts.csv")
DEFAULT_AUDITED_PATH = Path("data/processed/audited_alerts.csv")
DEFAULT_RESPONSE_PATH = Path("data/processed/clinician_response_logs.csv")
DEFAULT_RELIABILITY_SUMMARY_PATH = Path("data/processed/reliability_summary.json")
DEFAULT_DRIFT_SUMMARY_PATH = Path("data/processed/drift_summary.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/alert_explanations.csv")

SAFETY_NOTE = (
    "Simulation-only support explanation. Not a diagnosis, not treatment advice, "
    "and not a substitute for clinician review."
)
UNCERTAINTY_NOTE = (
    "This explanation is based on simulated system outputs and may be incomplete; "
    "uncertain or safety-sensitive alerts require human review."
)

UNSAFE_PHRASES = [
    "diagnosis is",
    "treatment should",
    "the patient has",
    "doctor is not needed",
    "ignore clinician review",
    "ignore clinical review",
    "no clinician review needed",
    "prescribe",
]

REQUIRED_OUTPUT_COLUMNS = [
    "explanation_id",
    "alert_id",
    "patient_id",
    "severity",
    "alert_type",
    "trigger_reason",
    "audit_status",
    "final_alert_status",
    "simulated_response",
    "explanation_text",
    "uncertainty_note",
    "safety_note",
    "llm_backend",
    "fallback_used",
    "generated_at",
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


def merge_alert_context(
    alerts_df: pd.DataFrame,
    audited_df: pd.DataFrame,
    response_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge alert, audit, fatigue, and response context by alert_id."""
    if "alert_id" not in alerts_df.columns:
        raise ValueError("alerts_df must include alert_id.")
    if "alert_id" not in audited_df.columns:
        raise ValueError("audited_df must include alert_id.")
    if "alert_id" not in response_df.columns:
        raise ValueError("response_df must include alert_id.")

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
            "source_model",
            "recommended_review_time",
            "critical_flag",
            "guardrail_decision",
            "guardrail_reason",
            "requires_human_review",
            "safety_priority",
            "fatigue_action",
            "fatigue_reason",
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

    merged = alerts_df[alert_columns].merge(
        audited_df[audit_columns],
        on="alert_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        response_df[response_columns],
        on="alert_id",
        how="left",
        validate="one_to_one",
    )
    return merged.reset_index(drop=True)


def build_alert_explanation_prompt(
    row: pd.Series,
    reliability_summary: dict | None = None,
    drift_summary: dict | None = None,
) -> str:
    """Build a concise safety-constrained prompt for one alert."""
    reliability_summary = reliability_summary or {}
    drift_summary = drift_summary or {}
    return (
        "Create a concise explanation for a simulated clinical-alert system output.\n"
        "Safety constraints: do not diagnose, do not recommend treatment, do not say "
        "clinician review is unnecessary, and do not make clinical validity claims.\n"
        "Explain only why the simulated alert was generated and why human review may be needed.\n\n"
        f"Alert ID: {_value(row, 'alert_id')}\n"
        f"Patient ID: {_value(row, 'patient_id')}\n"
        f"Severity: {_value(row, 'severity')}\n"
        f"Alert type: {_value(row, 'alert_type')}\n"
        f"Trigger reason: {_value(row, 'trigger_reason')}\n"
        f"Risk score: {_value(row, 'risk_score')}\n"
        f"Audit status: {_value(row, 'audit_status')}\n"
        f"Audit reason: {_value(row, 'audit_reason')}\n"
        f"Fatigue status: {_value(row, 'final_alert_status')}\n"
        f"Fatigue action: {_value(row, 'fatigue_action')}\n"
        f"Simulated response: {_value(row, 'simulated_response')}\n"
        f"Reliability summary: {_compact_summary(reliability_summary)}\n"
        f"Drift summary: {_compact_summary(drift_summary)}\n\n"
        "Return one readable paragraph with a simulation-only safety note."
    )


def generate_rule_based_explanation(
    row: pd.Series,
    reliability_summary: dict | None = None,
    drift_summary: dict | None = None,
) -> str:
    """Generate a useful explanation without relying on an LLM."""
    reliability_summary = reliability_summary or {}
    drift_summary = drift_summary or {}
    severity = _value(row, "severity", "unknown")
    alert_type = _value(row, "alert_type", "unknown alert type")
    trigger_reason = _value(row, "trigger_reason", "the system detected a simulated risk signal")
    audit_status = _value(row, "audit_status", "not audited")
    final_status = _value(row, "final_alert_status", "not fatigue-reviewed")
    fatigue_action = _value(row, "fatigue_action", "no fatigue action recorded")
    simulated_response = _value(row, "simulated_response", "no simulated response recorded")
    review_reason = _review_reason(row)

    parts = [
        f"This simulated {severity} alert was generated as {alert_type}.",
        f"The main trigger was: {trigger_reason}.",
        f"It may need human review because {review_reason}.",
        f"The audit layer labeled it as {audit_status}, and fatigue review left it as {final_status} with action {fatigue_action}.",
        f"The simulated workflow response was {simulated_response}.",
    ]

    if reliability_summary:
        average_reliability = reliability_summary.get("average_reliability_score")
        windows_requiring_review = reliability_summary.get("windows_requiring_review")
        parts.append(
            "Reliability monitoring provides context only: "
            f"average reliability score {average_reliability}, windows requiring review {windows_requiring_review}."
        )
    if drift_summary:
        severe_drift_count = drift_summary.get("severe_drift_count")
        checks_requiring_review = drift_summary.get("checks_requiring_review")
        parts.append(
            "Drift monitoring provides uncertainty context only: "
            f"severe drift checks {severe_drift_count}, checks requiring review {checks_requiring_review}."
        )

    parts.append(
        "This is simulation-only support text and does not diagnose, recommend treatment, or replace clinician review."
    )
    return sanitize_explanation_text(" ".join(parts))


def generate_explanation_for_alert(
    row: pd.Series,
    llm_client: LLMClient | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Generate one safe explanation record for an alert row."""
    reliability_summary = row.attrs.get("reliability_summary", {})
    drift_summary = row.attrs.get("drift_summary", {})
    rule_based = generate_rule_based_explanation(row, reliability_summary, drift_summary)
    llm_backend = "rule_based"
    fallback_used = True
    explanation_text = rule_based

    if use_llm:
        client = llm_client or create_llm_client()
        prompt = build_alert_explanation_prompt(row, reliability_summary, drift_summary)
        response = client.generate(prompt, max_tokens=220)
        llm_backend = response.backend
        fallback_used = bool(response.fallback_used)
        raw_candidate = response.response_text
        candidate = sanitize_explanation_text(raw_candidate)
        if response.success and not _contains_unsafe_phrase(raw_candidate) and _is_safe_explanation(candidate):
            explanation_text = candidate
            fallback_used = False
        else:
            # Fallback text from the client is safe but generic; alert-specific rule text is more useful.
            explanation_text = rule_based
            fallback_used = True
    else:
        explanation_text = rule_based

    explanation_text = sanitize_explanation_text(explanation_text)
    if not _is_safe_explanation(explanation_text):
        explanation_text = rule_based
        fallback_used = True

    return {
        "explanation_id": _record_id("EXPLAIN", _value(row, "alert_id")),
        "alert_id": _value(row, "alert_id"),
        "patient_id": _value(row, "patient_id"),
        "severity": _value(row, "severity"),
        "alert_type": _value(row, "alert_type"),
        "trigger_reason": _value(row, "trigger_reason"),
        "audit_status": _value(row, "audit_status"),
        "final_alert_status": _value(row, "final_alert_status"),
        "simulated_response": _value(row, "simulated_response"),
        "explanation_text": explanation_text,
        "uncertainty_note": UNCERTAINTY_NOTE,
        "safety_note": SAFETY_NOTE,
        "llm_backend": llm_backend,
        "fallback_used": bool(fallback_used),
        "generated_at": _timestamp_now(),
    }


def generate_alert_explanations(
    alerts_df: pd.DataFrame,
    audited_df: pd.DataFrame,
    response_df: pd.DataFrame,
    limit: int | None = 50,
    use_llm: bool = True,
) -> pd.DataFrame:
    """Generate explanation records for a batch of alerts."""
    merged = merge_alert_context(alerts_df, audited_df, response_df)
    if limit is not None:
        merged = merged.head(max(int(limit), 0))

    reliability_summary = alerts_df.attrs.get("reliability_summary", {})
    drift_summary = alerts_df.attrs.get("drift_summary", {})
    llm_client = create_llm_client() if use_llm else None
    llm_available = bool(use_llm and llm_client and llm_client.is_available())
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        row.attrs["reliability_summary"] = reliability_summary
        row.attrs["drift_summary"] = drift_summary
        records.append(
            generate_explanation_for_alert(
                row,
                llm_client=llm_client,
                use_llm=llm_available,
            )
        )
    return pd.DataFrame(records, columns=REQUIRED_OUTPUT_COLUMNS)


def save_alert_explanations(df: pd.DataFrame, path: str | Path) -> Path:
    """Save generated explanations to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_explanation_generation_pipeline(
    alerts_path: str | Path = DEFAULT_ALERTS_PATH,
    audited_path: str | Path = DEFAULT_AUDITED_PATH,
    response_path: str | Path = DEFAULT_RESPONSE_PATH,
    reliability_summary_path: str | Path = DEFAULT_RELIABILITY_SUMMARY_PATH,
    drift_summary_path: str | Path = DEFAULT_DRIFT_SUMMARY_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = 50,
    use_llm: bool = True,
) -> pd.DataFrame:
    """Run Step 19 explanation generation end to end."""
    alerts_df = load_fatigue_reduced_alerts(alerts_path)
    audited_df = load_audited_alerts(audited_path)
    response_df = load_response_logs(response_path)
    reliability_summary = _load_optional_json(reliability_summary_path)
    drift_summary = _load_optional_json(drift_summary_path)

    alerts_df.attrs["reliability_summary"] = reliability_summary
    alerts_df.attrs["drift_summary"] = drift_summary
    explanations = generate_alert_explanations(
        alerts_df,
        audited_df,
        response_df,
        limit=limit,
        use_llm=use_llm,
    )
    saved_path = save_alert_explanations(explanations, output_path)
    explanations.attrs["output_path"] = str(saved_path)
    return explanations


def sanitize_explanation_text(text: str) -> str:
    """Remove or replace unsafe explanation text."""
    if not isinstance(text, str) or not text.strip():
        return (
            "This simulated alert requires a cautious rule-based explanation. "
            "It does not diagnose, recommend treatment, or replace clinician review."
        )
    lowered = text.lower()
    if _contains_unsafe_phrase(text):
        return (
            "This simulated alert explanation was replaced because unsafe clinical wording "
            "was detected. The alert should be interpreted only as a prototype system output, "
            "with uncertainty and human review preserved. It does not diagnose, recommend "
            "treatment, or replace clinician review."
        )
    return " ".join(text.split())


def generate_alert_explanation() -> pd.DataFrame:
    """Compatibility wrapper for older placeholder imports."""
    return run_explanation_generation_pipeline()


def _load_csv(path: str | Path, label: str) -> pd.DataFrame:
    input_path = _resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{label} file not found: {input_path}")
    return pd.read_csv(input_path)


def _load_optional_json(path: str | Path) -> dict[str, Any]:
    json_path = _resolve_project_path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _is_safe_explanation(text: str) -> bool:
    lowered = text.lower()
    if _contains_unsafe_phrase(text):
        return False
    safety_markers = [
        "simulation",
        "simulated",
        "human review",
        "clinician review",
        "not a diagnosis",
        "does not diagnose",
    ]
    return any(marker in lowered for marker in safety_markers)


def _contains_unsafe_phrase(text: str) -> bool:
    lowered = str(text).lower()
    return any(phrase in lowered for phrase in UNSAFE_PHRASES)


def _review_reason(row: pd.Series) -> str:
    severity = str(_value(row, "severity", "")).lower()
    critical = _coerce_bool(_value(row, "critical_flag"))
    requires_review = _coerce_bool(_value(row, "requires_human_review"))
    audit_status = str(_value(row, "audit_status", "")).lower()
    safety_priority = str(_value(row, "safety_priority", "")).lower()
    if critical or severity in {"high", "critical"} or safety_priority in {"urgent", "immediate"}:
        return "the alert is safety-sensitive or higher severity in the simulation"
    if requires_review or audit_status in {"review_needed", "high_priority"}:
        return "the audit or guardrail layer marked it for review"
    if _value(row, "simulated_response"):
        return "the simulated workflow recorded a response that should be interpreted by a human reviewer"
    return "the prototype output has uncertainty and should not be interpreted automatically"


def _compact_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return "not available"
    keys = [
        "average_reliability_score",
        "windows_requiring_review",
        "average_drift_score",
        "severe_drift_count",
        "checks_requiring_review",
    ]
    selected = {key: summary.get(key) for key in keys if key in summary}
    return json.dumps(selected, sort_keys=True) if selected else "available"


def _value(row: pd.Series, column: str, default: Any = None) -> Any:
    value = row.get(column, default)
    if value is None or pd.isna(value):
        return default
    return value


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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
    result_df = run_explanation_generation_pipeline()
    print(f"Total explanations generated: {len(result_df)}")
    if not result_df.empty:
        print("Backend distribution:")
        print(result_df["llm_backend"].value_counts().to_string())
        print("Fallback distribution:")
        print(result_df["fallback_used"].value_counts().to_string())
        preview_columns = [
            "alert_id",
            "severity",
            "alert_type",
            "llm_backend",
            "fallback_used",
            "explanation_text",
        ]
        print("\nFirst few explanations:")
        print(result_df[preview_columns].head().to_string(index=False))
    print(f"\nSaved explanations to {result_df.attrs.get('output_path', DEFAULT_OUTPUT_PATH)}")
