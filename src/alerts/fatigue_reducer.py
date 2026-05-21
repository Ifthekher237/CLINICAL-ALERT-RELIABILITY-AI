"""Alert fatigue reduction for simulated alert reliability experiments.

Step 10 reduces repeated low-value alert burden while preserving every critical
or safety-sensitive alert. The module never physically removes rows and never
suppresses critical alerts. It marks selected alerts as grouped, delayed, or
priority-downgraded so later workflow steps can inspect the full audit trail.

This is a research/engineering prototype, not a validated clinical alert system.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_INPUT_PATH = Path("data/processed/audited_alerts.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/fatigue_reduced_alerts.csv")
DEFAULT_REPETITION_WINDOW_MINUTES = 30

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
    "actionability_score",
    "fatigue_risk_score",
    "urgency_score",
    "false_positive_likelihood",
    "confidence_score",
    "audit_status",
    "escalation_recommendation",
    "audit_reason",
]

FATIGUE_COLUMNS = [
    "fatigue_action",
    "fatigue_reason",
    "original_alert_retained",
    "grouped_alert_count",
    "fatigue_reduction_safe",
    "final_alert_status",
]

ALLOWED_FATIGUE_ACTIONS = {
    "retain",
    "group_repeated",
    "downgrade_priority",
    "delay_non_critical",
    "escalate_pattern",
}

ALLOWED_FINAL_ALERT_STATUSES = {
    "active",
    "grouped",
    "delayed",
    "priority_downgraded",
    "escalated",
}

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def load_audited_alerts(path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load audited alerts from Step 9."""
    input_path = _resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audited alerts file not found: {input_path}")
    alerts_df = pd.read_csv(input_path)
    validate_fatigue_input_schema(alerts_df)
    return alerts_df


def validate_fatigue_input_schema(df: pd.DataFrame) -> None:
    """Validate the Step 9 schema needed for fatigue reduction."""
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Fatigue reduction input is missing required columns: {missing_columns}")


def identify_recent_repeated_alerts(
    alerts_df: pd.DataFrame,
    window_minutes: int = DEFAULT_REPETITION_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Add patient-local recent repetition features using only past alerts."""
    if window_minutes < 1:
        raise ValueError("window_minutes must be at least 1.")
    validate_fatigue_input_schema(alerts_df)

    working = alerts_df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.sort_values(["patient_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    repeated_counts: list[int] = []
    same_pattern_counts: list[int] = []
    recent_max_risks: list[float] = []
    recent_max_severity_ranks: list[int] = []

    for _, patient_alerts in working.groupby("patient_id", sort=False):
        history = pd.DataFrame(columns=working.columns)
        for _, row in patient_alerts.iterrows():
            recent = _recent_history(row, history, window_minutes)
            repeated_counts.append(len(recent))

            if recent.empty:
                same_pattern_counts.append(0)
                recent_max_risks.append(0.0)
                recent_max_severity_ranks.append(0)
            else:
                same_pattern = _same_pattern_mask(recent, row)
                same_pattern_counts.append(int(same_pattern.sum()))
                recent_max_risks.append(float(pd.to_numeric(recent["risk_score"], errors="coerce").fillna(0.0).max()))
                recent_max_severity_ranks.append(
                    int(recent["severity"].astype(str).str.lower().map(SEVERITY_RANK).fillna(0).max())
                )

            history = pd.concat([history, row.to_frame().T], ignore_index=True)

    working["recent_alert_count"] = repeated_counts
    working["recent_same_pattern_count"] = same_pattern_counts
    working["recent_max_risk_score"] = recent_max_risks
    working["recent_max_severity_rank"] = recent_max_severity_ranks
    working["repeated_alert_pattern"] = working["recent_same_pattern_count"] >= 1
    working["worsening_repeated_pattern"] = (
        (pd.to_numeric(working["risk_score"], errors="coerce").fillna(0.0) >= working["recent_max_risk_score"] + 0.10)
        | (working["severity"].astype(str).str.lower().map(SEVERITY_RANK).fillna(0) > working["recent_max_severity_rank"])
    ) & (working["recent_alert_count"] > 0)
    return working


def decide_fatigue_action(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Choose a safety-preserving fatigue action for one audited alert."""
    severity = _severity(row)
    safety_priority = str(_row_value(row, "safety_priority", "routine")).lower()
    escalation = str(_row_value(row, "escalation_recommendation", "no_escalation")).lower()
    audit_status = str(_row_value(row, "audit_status", "")).lower()
    repeated_count = int(_numeric(row, "recent_alert_count", 0))
    same_pattern_count = int(_numeric(row, "recent_same_pattern_count", 0))
    false_positive = _numeric(row, "false_positive_likelihood", 0.0)
    fatigue_risk = _numeric(row, "fatigue_risk_score", 0.0)
    risk_score = _numeric(row, "risk_score", 0.0)
    worsening = bool(_row_value(row, "worsening_repeated_pattern", False))

    if _is_safety_protected(row):
        return {
            "fatigue_action": "retain",
            "fatigue_reason": (
                "Safety-protected alert retained as active because it is critical, "
                "immediate priority, or marked for immediate escalation."
            ),
            "original_alert_retained": True,
            "grouped_alert_count": max(1, same_pattern_count + 1),
            "fatigue_reduction_safe": True,
            "final_alert_status": "active",
        }

    if severity == "high":
        if worsening or repeated_count >= 2:
            return {
                "fatigue_action": "escalate_pattern",
                "fatigue_reason": "Repeated high-priority pattern is escalated rather than grouped or delayed.",
                "original_alert_retained": True,
                "grouped_alert_count": max(1, same_pattern_count + 1),
                "fatigue_reduction_safe": True,
                "final_alert_status": "escalated",
            }
        return {
            "fatigue_action": "retain",
            "fatigue_reason": "High severity alert retained as active; high-risk alerts are not grouped away.",
            "original_alert_retained": True,
            "grouped_alert_count": max(1, same_pattern_count + 1),
            "fatigue_reduction_safe": True,
            "final_alert_status": "active",
        }

    if severity in {"low", "medium"} and worsening and repeated_count >= 1 and risk_score >= 0.45:
        return {
            "fatigue_action": "escalate_pattern",
            "fatigue_reason": "Repeated alert pattern shows worsening risk, so it is escalated instead of reduced.",
            "original_alert_retained": True,
            "grouped_alert_count": max(1, same_pattern_count + 1),
            "fatigue_reduction_safe": True,
            "final_alert_status": "escalated",
        }

    if (
        severity in {"low", "medium"}
        and audit_status == "repeated_low_value"
        and same_pattern_count >= 1
    ):
        return {
            "fatigue_action": "group_repeated",
            "fatigue_reason": "Repeated low/medium alert grouped with recent similar alerts for the same patient.",
            "original_alert_retained": False,
            "grouped_alert_count": same_pattern_count + 1,
            "fatigue_reduction_safe": True,
            "final_alert_status": "grouped",
        }

    if (
        severity in {"low", "medium"}
        and false_positive >= 0.55
        and safety_priority in {"routine", "review"}
    ):
        return {
            "fatigue_action": "delay_non_critical",
            "fatigue_reason": "Non-critical alert delayed because false-positive likelihood is high and safety priority is not urgent.",
            "original_alert_retained": False,
            "grouped_alert_count": max(1, same_pattern_count + 1),
            "fatigue_reduction_safe": True,
            "final_alert_status": "delayed",
        }

    if (
        severity in {"low", "medium"}
        and fatigue_risk >= 0.60
        and safety_priority in {"routine", "review"}
    ):
        return {
            "fatigue_action": "downgrade_priority",
            "fatigue_reason": "Priority downgraded for repeated non-critical alert with elevated fatigue risk.",
            "original_alert_retained": False,
            "grouped_alert_count": max(1, same_pattern_count + 1),
            "fatigue_reduction_safe": True,
            "final_alert_status": "priority_downgraded",
        }

    return {
        "fatigue_action": "retain",
        "fatigue_reason": "Alert retained because no safe fatigue-reduction rule applied.",
        "original_alert_retained": True,
        "grouped_alert_count": max(1, same_pattern_count + 1),
        "fatigue_reduction_safe": True,
        "final_alert_status": "active",
    }


def apply_fatigue_reduction(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Step 10 fatigue-reduction labels without removing alert rows."""
    validate_fatigue_input_schema(alerts_df)
    reduced = identify_recent_repeated_alerts(alerts_df)
    fatigue_decisions = reduced.apply(decide_fatigue_action, axis=1, result_type="expand")
    reduced = pd.concat([reduced, fatigue_decisions], axis=1)

    reduced["critical_flag"] = reduced["critical_flag"].apply(_coerce_bool)
    reduced["requires_human_review"] = reduced["requires_human_review"].apply(_coerce_bool)
    reduced["original_alert_retained"] = reduced["original_alert_retained"].astype(bool)
    reduced["fatigue_reduction_safe"] = reduced["fatigue_reduction_safe"].astype(bool)
    reduced["grouped_alert_count"] = pd.to_numeric(
        reduced["grouped_alert_count"],
        errors="coerce",
    ).fillna(1).astype(int)
    return reduced


def calculate_fatigue_metrics(
    original_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate burden-reduction and critical-preservation metrics."""
    total_original = int(len(original_df))
    active_after = int((reduced_df["final_alert_status"] == "active").sum())
    grouped = int((reduced_df["fatigue_action"] == "group_repeated").sum())
    delayed = int((reduced_df["fatigue_action"] == "delay_non_critical").sum())
    downgraded = int((reduced_df["fatigue_action"] == "downgrade_priority").sum())
    escalated = int((reduced_df["fatigue_action"] == "escalate_pattern").sum())
    reduced_burden = grouped + delayed + downgraded

    critical_original = original_df[
        original_df["severity"].astype(str).str.lower().eq("critical")
        | original_df["critical_flag"].apply(_coerce_bool)
        | original_df["safety_priority"].astype(str).str.lower().eq("immediate")
        | original_df["escalation_recommendation"].astype(str).str.lower().eq("immediate_escalation")
    ]
    critical_reduced = reduced_df.loc[critical_original.index.intersection(reduced_df.index)]
    critical_preserved = bool(
        critical_reduced.empty
        or (
            critical_reduced["final_alert_status"].isin(["active", "escalated"])
            & critical_reduced["fatigue_action"].isin(["retain", "escalate_pattern"])
        ).all()
    )
    critical_count = int(len(critical_original))
    critical_preservation_rate = 1.0 if critical_count == 0 else float(
        (
            critical_reduced["final_alert_status"].isin(["active", "escalated"])
            & critical_reduced["fatigue_action"].isin(["retain", "escalate_pattern"])
        ).sum()
        / critical_count
    )

    repeated_candidates = int(
        (
            original_df["audit_status"].astype(str).str.lower().eq("repeated_low_value")
            & original_df["severity"].astype(str).str.lower().isin(["low", "medium"])
        ).sum()
    )
    repeated_reduced = int(
        reduced_df["fatigue_action"].isin(
            ["group_repeated", "delay_non_critical", "downgrade_priority"]
        ).sum()
    )

    return {
        "total_original_alerts": total_original,
        "total_active_after_reduction": active_after,
        "grouped_alerts": grouped,
        "delayed_alerts": delayed,
        "downgraded_alerts": downgraded,
        "escalated_patterns": escalated,
        "alert_reduction_rate": round(reduced_burden / total_original, 4) if total_original else 0.0,
        "critical_alerts_preserved": critical_preserved,
        "critical_preservation_rate": round(critical_preservation_rate, 4),
        "repeated_alert_reduction": round(repeated_reduced / repeated_candidates, 4)
        if repeated_candidates
        else 0.0,
    }


def save_fatigue_reduced_alerts(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save fatigue-reviewed alerts to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_fatigue_reduction_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the Step 10 fatigue-reduction workflow end to end."""
    audited_df = load_audited_alerts(input_path)
    reduced_df = apply_fatigue_reduction(audited_df)
    saved_path = save_fatigue_reduced_alerts(reduced_df, output_path)
    reduced_df.attrs["output_path"] = str(saved_path)
    reduced_df.attrs["fatigue_metrics"] = calculate_fatigue_metrics(audited_df, reduced_df)
    return reduced_df


def reduce_alert_fatigue() -> pd.DataFrame:
    """Backward-compatible wrapper for the Step 10 fatigue pipeline."""
    return run_fatigue_reduction_pipeline()


def _recent_history(
    row: pd.Series | dict[str, Any],
    history: pd.DataFrame,
    window_minutes: int,
) -> pd.DataFrame:
    """Return previous alerts for the same patient within the configured window."""
    if history.empty:
        return history.copy()
    current_time = pd.to_datetime(_row_value(row, "timestamp", pd.NaT), errors="coerce")
    if pd.isna(current_time):
        return history[history["patient_id"] == _row_value(row, "patient_id", None)].copy()
    patient_history = history[history["patient_id"] == _row_value(row, "patient_id", None)].copy()
    if patient_history.empty:
        return patient_history
    patient_history["timestamp"] = pd.to_datetime(patient_history["timestamp"], errors="coerce")
    window_start = current_time - pd.Timedelta(minutes=window_minutes)
    return patient_history[
        (patient_history["timestamp"] < current_time)
        & (patient_history["timestamp"] >= window_start)
    ]


def _same_pattern_mask(recent_alerts: pd.DataFrame, row: pd.Series | dict[str, Any]) -> pd.Series:
    """Identify same-patient recent alerts with similar type/severity/reason."""
    current_type = str(_row_value(row, "alert_type", "")).lower()
    current_severity = _severity(row)
    current_reason_key = _reason_key(str(_row_value(row, "trigger_reason", "")))

    recent_type = recent_alerts["alert_type"].astype(str).str.lower()
    recent_severity = recent_alerts["severity"].astype(str).str.lower()
    recent_reason_key = recent_alerts["trigger_reason"].astype(str).map(_reason_key)
    return (recent_type == current_type) & (
        (recent_severity == current_severity) | (recent_reason_key == current_reason_key)
    )


def _reason_key(reason: str) -> str:
    """Create a simple comparable key from a trigger reason."""
    first_clause = reason.lower().split(";")[0].strip()
    return " ".join(first_clause.split()[:5])


def _is_safety_protected(row: pd.Series | dict[str, Any]) -> bool:
    """Return True for alerts that fatigue logic must never reduce."""
    return bool(
        _severity(row) == "critical"
        or _coerce_bool(_row_value(row, "critical_flag", False))
        or str(_row_value(row, "safety_priority", "")).lower() == "immediate"
        or str(_row_value(row, "escalation_recommendation", "")).lower()
        == "immediate_escalation"
    )


def _severity(row: pd.Series | dict[str, Any]) -> str:
    """Normalize severity values."""
    return str(_row_value(row, "severity", "low")).strip().lower()


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


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    """Read from a row-like object with fallback."""
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
    reduced_alerts = run_fatigue_reduction_pipeline()
    metrics = reduced_alerts.attrs.get("fatigue_metrics", {})
    output_path = reduced_alerts.attrs.get("output_path", str(_resolve_project_path(DEFAULT_OUTPUT_PATH)))

    print("Step 10 alert fatigue reduction complete")
    print(f"Total original alerts: {metrics.get('total_original_alerts', len(reduced_alerts))}")
    print(f"Active alerts after reduction: {metrics.get('total_active_after_reduction')}")
    print("Fatigue action distribution:")
    print(reduced_alerts["fatigue_action"].value_counts().to_dict())
    print("Final alert status distribution:")
    print(reduced_alerts["final_alert_status"].value_counts().to_dict())
    print(f"Alert reduction rate: {metrics.get('alert_reduction_rate', 0.0):.2%}")
    print(f"Critical preservation rate: {metrics.get('critical_preservation_rate', 0.0):.2%}")
    print(f"Saved fatigue-reviewed alerts to: {output_path}")
    print("First few fatigue-reviewed alerts:")
    print(reduced_alerts.head())
