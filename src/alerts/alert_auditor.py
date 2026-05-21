"""Clinical-style alert auditing for simulated alert reliability experiments.

Step 9 evaluates generated and guardrail-reviewed alerts for usefulness,
actionability, urgency, repetition, likely noise, alert burden, and escalation
need. This module is transparent and rule-based. It does not suppress alerts,
and it is not a validated clinical decision system.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_INPUT_PATH = Path("data/processed/guardrail_reviewed_alerts.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/audited_alerts.csv")
RECENT_ALERT_WINDOW_MINUTES = 60

REQUIRED_INPUT_COLUMNS = [
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
    "guardrail_action",
    "guardrail_reason",
    "requires_human_review",
    "safety_priority",
]

AUDIT_COLUMNS = [
    "audit_status",
    "actionability_score",
    "fatigue_risk_score",
    "urgency_score",
    "false_positive_likelihood",
    "confidence_score",
    "escalation_recommendation",
    "audit_reason",
]

VALID_AUDIT_STATUSES = {
    "useful",
    "review_needed",
    "likely_noise",
    "repeated_low_value",
    "high_priority",
}

VALID_ESCALATION_RECOMMENDATIONS = {
    "no_escalation",
    "monitor",
    "clinician_review",
    "urgent_review",
    "immediate_escalation",
}

WEAK_TRIGGER_REASONS = {
    "",
    "unknown",
    "none",
    "nan",
    "elevated combined simulated alert risk",
}


def load_guardrail_reviewed_alerts(path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load guardrail-reviewed alerts from Step 8."""
    input_path = _resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Guardrail-reviewed alerts file not found: {input_path}")
    alerts_df = pd.read_csv(input_path)
    validate_audit_input_schema(alerts_df)
    return alerts_df


def validate_audit_input_schema(df: pd.DataFrame) -> None:
    """Validate the Step 8 alert schema required for auditing."""
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Audit input is missing required columns: {missing_columns}")


def calculate_actionability_score(
    row: pd.Series | dict[str, Any],
    patient_history: pd.DataFrame | None = None,
) -> float:
    """Estimate whether an alert gives clear, useful next-step information."""
    risk = _risk_score(row)
    severity_component = _severity_weight(row)
    trigger_clarity = 0.0 if _has_weak_trigger_reason(row) else 1.0
    source_strength = _source_strength(row)
    guardrail_component = 1.0 if _requires_human_review(row) else 0.65

    score = (
        0.25 * risk
        + 0.25 * severity_component
        + 0.25 * trigger_clarity
        + 0.15 * source_strength
        + 0.10 * guardrail_component
    )

    recent_alerts = _recent_alerts_for_row(row, patient_history)
    if _is_repeated_low_medium_alert(row, recent_alerts):
        score -= 0.15
    if _critical_flag(row):
        score = max(score, 0.90)

    return _round_score(score)


def calculate_fatigue_risk_score(
    row: pd.Series | dict[str, Any],
    recent_alerts: pd.DataFrame | None = None,
) -> float:
    """Estimate repeated-alert burden using only prior alerts for one patient."""
    if recent_alerts is None or recent_alerts.empty:
        return 0.0

    recent_count = len(recent_alerts)
    same_type_count = int((recent_alerts["alert_type"].astype(str) == str(_row_value(row, "alert_type", ""))).sum())
    low_medium_count = int(
        recent_alerts["severity"].astype(str).str.lower().isin(["low", "medium"]).sum()
    )
    same_reason_count = int(
        (
            recent_alerts["trigger_reason"].astype(str).str.lower()
            == str(_row_value(row, "trigger_reason", "")).lower()
        ).sum()
    )

    score = (
        min(recent_count / 6.0, 1.0) * 0.35
        + min(same_type_count / 4.0, 1.0) * 0.30
        + min(low_medium_count / 5.0, 1.0) * 0.20
        + min(same_reason_count / 3.0, 1.0) * 0.15
    )

    if _severity(row) == "critical" or _critical_flag(row):
        score = min(score, 0.50)
    return _round_score(score)


def calculate_urgency_score(row: pd.Series | dict[str, Any]) -> float:
    """Estimate alert urgency from severity, guardrail priority, and risk."""
    severity_base = {
        "low": 0.20,
        "medium": 0.45,
        "high": 0.75,
        "critical": 1.00,
    }.get(_severity(row), 0.20)
    priority_boost = {
        "routine": 0.00,
        "review": 0.10,
        "urgent": 0.18,
        "immediate": 0.25,
    }.get(str(_row_value(row, "safety_priority", "routine")).lower(), 0.0)
    score = max(severity_base, _risk_score(row)) + priority_boost
    if _critical_flag(row):
        score = 1.0
    return _round_score(score)


def estimate_false_positive_likelihood(
    row: pd.Series | dict[str, Any],
    recent_alerts: pd.DataFrame | None = None,
) -> float:
    """Estimate likely noise or weak evidence without suppressing the alert."""
    weak_reason = _has_weak_trigger_reason(row)
    risk = _risk_score(row)
    anomaly_only = _is_anomaly_only_signal(row)
    repeated_low_value = _is_repeated_low_medium_alert(row, recent_alerts)

    score = 0.10
    if weak_reason:
        score += 0.25
    if risk < 0.35:
        score += 0.20
    elif risk < 0.50:
        score += 0.10
    if anomaly_only:
        score += 0.20
    if repeated_low_value:
        score += 0.20
    if _severity(row) in {"high", "critical"}:
        score -= 0.12
    if _critical_flag(row):
        score = min(score, 0.20)
    return _round_score(score)


def calculate_confidence_score(row: pd.Series | dict[str, Any]) -> float:
    """Estimate confidence from evidence strength and guardrail support."""
    score = 0.35
    if not _has_weak_trigger_reason(row):
        score += 0.20
    score += _risk_score(row) * 0.20
    score += _source_strength(row) * 0.15
    if str(_row_value(row, "guardrail_decision", "")).lower() in {"escalate", "allow_with_review"}:
        score += 0.08
    if str(_row_value(row, "guardrail_decision", "")).lower() == "manual_verification_required":
        score -= 0.08
    if _critical_flag(row):
        score = max(score, 0.85)
    return _round_score(score)


def assign_audit_status(row: pd.Series | dict[str, Any]) -> str:
    """Assign a reviewer-friendly audit status."""
    urgency = _numeric(row, "urgency_score", 0.0)
    fatigue = _numeric(row, "fatigue_risk_score", 0.0)
    false_positive = _numeric(row, "false_positive_likelihood", 0.0)
    actionability = _numeric(row, "actionability_score", 0.0)
    requires_review = _requires_human_review(row)

    if _critical_flag(row) or _severity(row) == "critical" or urgency >= 0.85:
        return "high_priority"
    if fatigue >= 0.70 and actionability < 0.55:
        return "repeated_low_value"
    if false_positive >= 0.65 and actionability < 0.50:
        return "likely_noise"
    if requires_review or false_positive >= 0.45 or actionability < 0.55:
        return "review_needed"
    return "useful"


def assign_escalation_recommendation(row: pd.Series | dict[str, Any]) -> str:
    """Recommend review intensity without suppressing any alert."""
    status = str(_row_value(row, "audit_status", "")).lower()
    urgency = _numeric(row, "urgency_score", 0.0)
    requires_review = _requires_human_review(row)

    if _critical_flag(row) or status == "high_priority" or urgency >= 0.90:
        return "immediate_escalation"
    if urgency >= 0.70 or _severity(row) == "high":
        return "urgent_review"
    if requires_review or status == "review_needed":
        return "clinician_review"
    if status in {"likely_noise", "repeated_low_value"}:
        return "monitor"
    return "no_escalation"


def generate_audit_reason(row: pd.Series | dict[str, Any]) -> str:
    """Generate a compact, human-readable audit explanation."""
    status = str(_row_value(row, "audit_status", "")).lower()
    reasons: list[str] = []

    if status == "high_priority":
        reasons.append("High priority because severity, critical flag, or urgency is elevated")
    if _numeric(row, "fatigue_risk_score", 0.0) >= 0.60:
        reasons.append("Repeated recent alerts increase fatigue risk")
    if _numeric(row, "false_positive_likelihood", 0.0) >= 0.55:
        reasons.append("Weak or noisy evidence increases false-positive likelihood")
    if _numeric(row, "actionability_score", 0.0) >= 0.70:
        reasons.append("Clear trigger and strong risk signal improve actionability")
    if _requires_human_review(row):
        reasons.append("Guardrails require human review")
    if not reasons:
        reasons.append("Alert has adequate evidence and routine audit profile")

    return "; ".join(reasons[:3])


def audit_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Audit guardrail-reviewed alerts using current and past patient history only."""
    validate_audit_input_schema(alerts_df)
    audited = alerts_df.copy()
    audited["timestamp"] = pd.to_datetime(audited["timestamp"], errors="coerce")
    audited = audited.sort_values(["patient_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for _, patient_alerts in audited.groupby("patient_id", sort=False):
        patient_history = pd.DataFrame(columns=audited.columns)
        for _, row in patient_alerts.iterrows():
            recent_alerts = _recent_alerts_for_row(row, patient_history)
            row_dict = row.to_dict()
            row_dict["actionability_score"] = calculate_actionability_score(row, patient_history)
            row_dict["fatigue_risk_score"] = calculate_fatigue_risk_score(row, recent_alerts)
            row_dict["urgency_score"] = calculate_urgency_score(row)
            row_dict["false_positive_likelihood"] = estimate_false_positive_likelihood(
                row,
                recent_alerts,
            )
            row_dict["confidence_score"] = calculate_confidence_score(row)
            row_dict["audit_status"] = assign_audit_status(row_dict)
            row_dict["escalation_recommendation"] = assign_escalation_recommendation(row_dict)
            row_dict["audit_reason"] = generate_audit_reason(row_dict)
            records.append(row_dict)

            patient_history = pd.concat([patient_history, row.to_frame().T], ignore_index=True)

    audited_df = pd.DataFrame(records)
    audited_df["requires_human_review"] = audited_df["requires_human_review"].apply(_coerce_bool)
    audited_df["critical_flag"] = audited_df["critical_flag"].apply(_coerce_bool)
    for column in [
        "actionability_score",
        "fatigue_risk_score",
        "urgency_score",
        "false_positive_likelihood",
        "confidence_score",
    ]:
        audited_df[column] = pd.to_numeric(audited_df[column], errors="coerce").fillna(0.0)

    return audited_df


def save_audited_alerts(df: pd.DataFrame, path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Save audited alerts to CSV for later roadmap steps."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_alert_audit_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the Step 9 alert auditing workflow end to end."""
    alerts_df = load_guardrail_reviewed_alerts(input_path)
    audited_df = audit_alerts(alerts_df)
    saved_path = save_audited_alerts(audited_df, output_path)
    audited_df.attrs["output_path"] = str(saved_path)
    return audited_df


def audit_alert() -> pd.DataFrame:
    """Backward-compatible wrapper for the Step 9 audit pipeline."""
    return run_alert_audit_pipeline()


def _recent_alerts_for_row(
    row: pd.Series | dict[str, Any],
    patient_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return previous alerts for the same patient within the recent window."""
    if patient_history is None or patient_history.empty:
        return pd.DataFrame()

    current_time = pd.to_datetime(_row_value(row, "timestamp", pd.NaT), errors="coerce")
    if pd.isna(current_time):
        return patient_history.copy()

    patient_id = _row_value(row, "patient_id", None)
    history = patient_history[patient_history["patient_id"] == patient_id].copy()
    if history.empty:
        return history

    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
    window_start = current_time - pd.Timedelta(minutes=RECENT_ALERT_WINDOW_MINUTES)
    return history[(history["timestamp"] < current_time) & (history["timestamp"] >= window_start)]


def _is_repeated_low_medium_alert(
    row: pd.Series | dict[str, Any],
    recent_alerts: pd.DataFrame | None,
) -> bool:
    """Detect recent repeated low/medium alerts for the same patient."""
    if recent_alerts is None or recent_alerts.empty:
        return False
    severity = _severity(row)
    if severity not in {"low", "medium"}:
        return False
    same_type_count = int(
        (recent_alerts["alert_type"].astype(str) == str(_row_value(row, "alert_type", ""))).sum()
    )
    low_medium_count = int(
        recent_alerts["severity"].astype(str).str.lower().isin(["low", "medium"]).sum()
    )
    return same_type_count >= 2 or low_medium_count >= 3


def _is_anomaly_only_signal(row: pd.Series | dict[str, Any]) -> bool:
    """Identify alerts dominated by anomaly evidence without stronger support."""
    alert_type = str(_row_value(row, "alert_type", "")).lower()
    source_model = str(_row_value(row, "source_model", "")).lower()
    has_anomaly = "anomaly" in alert_type or "isolation_forest" in source_model
    has_time_series = "time_series" in source_model
    return bool(has_anomaly and not has_time_series and _risk_score(row) < 0.50)


def _source_strength(row: pd.Series | dict[str, Any]) -> float:
    """Estimate source evidence strength from contributing model names."""
    source_model = str(_row_value(row, "source_model", "")).lower()
    count = 0
    for source in ["random_forest", "logistic_regression", "isolation_forest", "time_series_rules"]:
        if source in source_model:
            count += 1
    return min(count / 3.0, 1.0)


def _severity(row: pd.Series | dict[str, Any]) -> str:
    """Read normalized severity."""
    return str(_row_value(row, "severity", "low")).strip().lower()


def _severity_weight(row: pd.Series | dict[str, Any]) -> float:
    """Map severity to a numeric score."""
    return {
        "low": 0.20,
        "medium": 0.45,
        "high": 0.75,
        "critical": 1.00,
    }.get(_severity(row), 0.20)


def _risk_score(row: pd.Series | dict[str, Any]) -> float:
    """Read the alert risk score safely."""
    return _numeric(row, "risk_score", 0.0)


def _critical_flag(row: pd.Series | dict[str, Any]) -> bool:
    """Read critical flag from bool or CSV-style string values."""
    return _coerce_bool(_row_value(row, "critical_flag", False))


def _requires_human_review(row: pd.Series | dict[str, Any]) -> bool:
    """Read human-review requirement from bool or CSV-style string values."""
    return _coerce_bool(_row_value(row, "requires_human_review", False))


def _has_weak_trigger_reason(row: pd.Series | dict[str, Any]) -> bool:
    """Detect weak or generic trigger explanations."""
    reason = str(_row_value(row, "trigger_reason", "")).strip().lower()
    return reason in WEAK_TRIGGER_REASONS or len(reason) < 12


def _numeric(row: pd.Series | dict[str, Any], key: str, default: float) -> float:
    """Read a numeric value safely."""
    value = _row_value(row, key, default)
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(value: Any) -> bool:
    """Coerce bool-like CSV values."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _round_score(value: float) -> float:
    """Clip a score to [0, 1] and round for stable CSV output."""
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    """Read a value from a row-like object with fallback."""
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return _project_root() / path_obj


def _project_root() -> Path:
    """Return the repository root for this project."""
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    audited_alerts = run_alert_audit_pipeline()
    output_path = audited_alerts.attrs.get("output_path", str(_resolve_project_path(DEFAULT_OUTPUT_PATH)))

    print("Step 9 alert audit complete")
    print(f"Total alerts audited: {len(audited_alerts)}")
    print("Audit status distribution:")
    print(audited_alerts["audit_status"].value_counts().to_dict())
    print("Escalation recommendation distribution:")
    print(audited_alerts["escalation_recommendation"].value_counts().to_dict())
    print(f"Average actionability score: {audited_alerts['actionability_score'].mean():.4f}")
    print(f"Average fatigue risk score: {audited_alerts['fatigue_risk_score'].mean():.4f}")
    print(f"Saved audited alerts to: {output_path}")
    print("First few audited alerts:")
    print(audited_alerts.head())
