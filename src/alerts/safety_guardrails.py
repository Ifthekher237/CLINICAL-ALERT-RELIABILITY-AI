"""Safety guardrails for simulated alert handling.

Step 8 reviews generated alert records before later auditing and fatigue
reduction. The rules are safety-first and transparent, but they are still part
of a simulated research prototype. They are not clinical validation and must not
be used for real patient care.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_ALERT_INPUT_PATH = Path("data/processed/generated_alerts.csv")
DEFAULT_GUARDRAIL_OUTPUT_PATH = Path("data/processed/guardrail_reviewed_alerts.csv")

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
]

REQUIRED_OUTPUT_COLUMNS = [
    *REQUIRED_INPUT_COLUMNS,
    "guardrail_decision",
    "guardrail_action",
    "guardrail_reason",
    "requires_human_review",
    "safety_priority",
]

VALID_GUARDRAIL_DECISIONS = {
    "allow",
    "allow_with_review",
    "escalate",
    "manual_verification_required",
}

VALID_SAFETY_PRIORITIES = {"routine", "review", "urgent", "immediate"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}

IMPORTANT_FIELD_COLUMNS = [
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "alert_type",
    "trigger_reason",
    "source_model",
]

WEAK_TRIGGER_REASONS = {
    "",
    "elevated combined simulated alert risk",
    "unknown",
    "none",
    "nan",
}


def load_generated_alerts(path: str | Path = DEFAULT_ALERT_INPUT_PATH) -> pd.DataFrame:
    """Load generated simulated alerts from CSV."""
    alerts_path = _resolve_project_path(path)
    if not alerts_path.exists():
        raise FileNotFoundError(f"Generated alerts file not found: {alerts_path}")
    alerts_df = pd.read_csv(alerts_path)
    validate_alert_schema(alerts_df)
    return alerts_df


def validate_alert_schema(df: pd.DataFrame) -> None:
    """Validate that generated alerts contain the Step 7 alert schema."""
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Generated alerts are missing required columns: {missing_columns}")


def apply_guardrail_to_alert(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Apply transparent safety-first guardrail rules to one alert row."""
    severity = _severity(row)
    risk_score = _risk_score(row)
    critical_flag = _critical_flag(row)
    weak_reason = _has_weak_trigger_reason(row)
    missing_fields = _missing_important_fields(row)
    anomaly_or_unstable = _has_anomaly_or_unstable_signal(row)

    if missing_fields:
        priority = "immediate" if critical_flag or severity == "critical" else "review"
        return {
            "guardrail_decision": "manual_verification_required",
            "guardrail_action": "Hold alert for manual verification before downstream handling",
            "guardrail_reason": (
                "Manual verification required because important alert fields are missing: "
                + ", ".join(missing_fields)
            ),
            "requires_human_review": True,
            "safety_priority": priority,
        }

    if critical_flag or severity == "critical":
        return {
            "guardrail_decision": "escalate",
            "guardrail_action": "Preserve and escalate critical alert",
            "guardrail_reason": (
                "Critical alerts and critical_flag=True alerts must not be downgraded, "
                "ignored, or suppressed in this prototype."
            ),
            "requires_human_review": True,
            "safety_priority": "immediate",
        }

    if anomaly_or_unstable:
        priority = "urgent" if severity == "high" else "review"
        return {
            "guardrail_decision": "manual_verification_required",
            "guardrail_action": "Route alert for manual verification of anomaly or instability signal",
            "guardrail_reason": (
                "Alert is linked to anomaly, instability, or noisy-signal evidence and "
                "should be manually verified before later workflow steps."
            ),
            "requires_human_review": True,
            "safety_priority": priority,
        }

    if severity == "high":
        if weak_reason or risk_score < 0.70:
            return {
                "guardrail_decision": "allow_with_review",
                "guardrail_action": "Allow alert but require human review",
                "guardrail_reason": (
                    "High severity alert has uncertain or borderline support, so it "
                    "requires human review before later handling."
                ),
                "requires_human_review": True,
                "safety_priority": "urgent",
            }
        return {
            "guardrail_decision": "escalate",
            "guardrail_action": "Escalate high severity alert for timely review",
            "guardrail_reason": "High severity alert has a clear trigger reason and elevated risk score.",
            "requires_human_review": True,
            "safety_priority": "urgent",
        }

    if severity == "medium":
        if weak_reason or risk_score >= 0.50:
            return {
                "guardrail_decision": "allow_with_review",
                "guardrail_action": "Allow medium alert with review requirement",
                "guardrail_reason": (
                    "Medium severity alert is allowed, but risk score or trigger clarity "
                    "requires human review."
                ),
                "requires_human_review": True,
                "safety_priority": "review",
            }
        return {
            "guardrail_decision": "allow",
            "guardrail_action": "Allow alert for downstream logging and audit",
            "guardrail_reason": "Medium severity alert has adequate trigger detail and routine risk.",
            "requires_human_review": False,
            "safety_priority": "routine",
        }

    return {
        "guardrail_decision": "allow",
        "guardrail_action": "Allow low severity alert for downstream logging and audit",
        "guardrail_reason": "Low severity alert is allowed and retained for later audit.",
        "requires_human_review": False,
        "safety_priority": "routine",
    }


def apply_safety_guardrails(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Step 8 guardrails to generated alerts without suppressing any rows."""
    validate_alert_schema(alerts_df)
    reviewed = alerts_df.copy().reset_index(drop=True)
    guardrail_rows = reviewed.apply(apply_guardrail_to_alert, axis=1, result_type="expand")
    reviewed = pd.concat([reviewed, guardrail_rows], axis=1)

    reviewed["critical_flag"] = reviewed["critical_flag"].apply(_coerce_bool)
    reviewed["requires_human_review"] = reviewed["requires_human_review"].astype(bool)
    reviewed = reviewed[REQUIRED_OUTPUT_COLUMNS]
    return reviewed


def save_guardrail_reviewed_alerts(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_GUARDRAIL_OUTPUT_PATH,
) -> Path:
    """Save guardrail-reviewed alerts to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_safety_guardrail_pipeline(
    input_path: str | Path = DEFAULT_ALERT_INPUT_PATH,
    output_path: str | Path = DEFAULT_GUARDRAIL_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the Step 8 safety guardrail workflow end to end."""
    alerts_df = load_generated_alerts(input_path)
    reviewed_df = apply_safety_guardrails(alerts_df)
    saved_path = save_guardrail_reviewed_alerts(reviewed_df, output_path)
    reviewed_df.attrs["output_path"] = str(saved_path)
    return reviewed_df


def _severity(row: pd.Series | dict[str, Any]) -> str:
    """Read and normalize alert severity."""
    severity = str(_row_value(row, "severity", "low")).strip().lower()
    return severity if severity in VALID_SEVERITIES else "low"


def _risk_score(row: pd.Series | dict[str, Any]) -> float:
    """Read risk score with a safe fallback."""
    value = _row_value(row, "risk_score", 0.0)
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _critical_flag(row: pd.Series | dict[str, Any]) -> bool:
    """Read critical flag from bool or CSV-style string values."""
    return _coerce_bool(_row_value(row, "critical_flag", False))


def _coerce_bool(value: Any) -> bool:
    """Coerce common CSV boolean encodings."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _has_weak_trigger_reason(row: pd.Series | dict[str, Any]) -> bool:
    """Check whether the trigger reason is too weak for high-risk automation."""
    reason = str(_row_value(row, "trigger_reason", "")).strip().lower()
    return reason in WEAK_TRIGGER_REASONS or len(reason) < 12


def _missing_important_fields(row: pd.Series | dict[str, Any]) -> list[str]:
    """Return required row fields that are empty or missing."""
    missing = []
    for column in IMPORTANT_FIELD_COLUMNS:
        value = _row_value(row, column, None)
        if value is None or pd.isna(value) or str(value).strip() == "":
            missing.append(column)
    return missing


def _has_anomaly_or_unstable_signal(row: pd.Series | dict[str, Any]) -> bool:
    """Identify alerts that should be manually verified before later handling."""
    alert_type = str(_row_value(row, "alert_type", "")).lower()
    source_model = str(_row_value(row, "source_model", "")).lower()
    reason = str(_row_value(row, "trigger_reason", "")).lower()
    unstable_keywords = (
        "anomaly",
        "isolation_forest",
        "unstable",
        "instability",
        "noisy",
        "sensor",
        "unusual",
    )
    combined = f"{alert_type} {source_model} {reason}"
    return any(keyword in combined for keyword in unstable_keywords)


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    """Read from a row-like object with a fallback."""
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
    reviewed_alerts = run_safety_guardrail_pipeline()
    output_path = reviewed_alerts.attrs.get(
        "output_path",
        str(_resolve_project_path(DEFAULT_GUARDRAIL_OUTPUT_PATH)),
    )

    print("Step 8 safety guardrail review complete")
    print(f"Total alerts reviewed: {len(reviewed_alerts)}")
    print("Guardrail decision distribution:")
    print(reviewed_alerts["guardrail_decision"].value_counts().to_dict())
    print("Safety priority distribution:")
    print(reviewed_alerts["safety_priority"].value_counts().to_dict())
    print(f"Alerts requiring human review: {int(reviewed_alerts['requires_human_review'].sum())}")
    print(f"Saved reviewed alerts to: {output_path}")
    print("First few guardrail-reviewed alerts:")
    print(reviewed_alerts.head())
