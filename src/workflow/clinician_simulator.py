"""Simulated clinician workflow responses for alert reliability experiments.

Step 11 models how a care team might respond to fatigue-reduced alerts in a
hospital-like monitoring workflow. This is a transparent simulation for an
engineering portfolio project. It is not a real hospital workflow and must not
be used for patient care.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_INPUT_PATH = Path("data/processed/fatigue_reduced_alerts.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/clinician_response_logs.csv")
RECENT_ALERT_WINDOW_MINUTES = 60

VALID_RESPONSES = {
    "accepted",
    "ignored",
    "delayed",
    "escalated",
    "marked_false",
    "marked_useful",
}

VALID_WORKFLOW_STAGES = {
    "triage_queue",
    "nurse_review",
    "clinician_review",
    "escalated_review",
    "closed",
}

REQUIRED_INPUT_COLUMNS = [
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "risk_score",
    "critical_flag",
    "safety_priority",
    "actionability_score",
    "fatigue_risk_score",
    "false_positive_likelihood",
    "escalation_recommendation",
    "fatigue_action",
    "final_alert_status",
    "grouped_alert_count",
]

REQUIRED_RESPONSE_COLUMNS = [
    "response_id",
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "final_alert_status",
    "fatigue_action",
    "simulated_response",
    "response_time_minutes",
    "response_reason",
    "clinician_burden_score",
    "perceived_alert_usefulness",
    "workflow_stage",
    "escalation_completed",
    "response_simulation_note",
]


def load_fatigue_reduced_alerts(path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load fatigue-reduced alerts from Step 10."""
    input_path = _resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Fatigue-reduced alerts file not found: {input_path}")
    alerts_df = pd.read_csv(input_path)
    _validate_input_schema(alerts_df)
    return alerts_df


def calculate_clinician_burden_score(
    row: pd.Series | dict[str, Any],
    recent_alerts: pd.DataFrame | None = None,
) -> float:
    """Estimate simulated clinician burden from alert context and recent load."""
    fatigue = _numeric(row, "fatigue_risk_score", 0.0)
    actionability = _numeric(row, "actionability_score", 0.5)
    grouped_count = min(_numeric(row, "grouped_alert_count", 1.0) / 5.0, 1.0)
    reliability_penalty = 1.0 - _numeric(row, "system_reliability_score", 0.75)
    recent_count = 0 if recent_alerts is None else len(recent_alerts)
    recent_component = min(recent_count / 8.0, 1.0)

    status = str(_row_value(row, "final_alert_status", "active")).lower()
    status_component = {
        "active": 0.35,
        "grouped": 0.16,
        "delayed": 0.12,
        "priority_downgraded": 0.18,
        "escalated": 0.40,
    }.get(status, 0.25)

    time_component = 0.10 if _is_night_hour(row) else 0.02
    score = (
        0.30 * fatigue
        + 0.22 * recent_component
        + 0.18 * status_component
        + 0.12 * grouped_count
        + 0.10 * reliability_penalty
        + time_component
        - 0.12 * actionability
    )
    if _is_safety_sensitive(row):
        score += 0.08
    return _round_score(score)


def simulate_response_type(
    row: pd.Series | dict[str, Any],
    burden_score: float,
    random_state: np.random.Generator | int | None = None,
) -> str:
    """Simulate a response type using readable weighted rules and a seed."""
    rng = _as_rng(random_state)
    severity = _severity(row)
    status = str(_row_value(row, "final_alert_status", "active")).lower()
    actionability = _numeric(row, "actionability_score", 0.5)
    false_positive = _numeric(row, "false_positive_likelihood", 0.0)

    if _is_immediate_or_critical(row):
        return _weighted_choice(
            rng,
            {
                "escalated": 0.72,
                "accepted": 0.18,
                "marked_useful": 0.10,
            },
        )

    if severity == "high" or str(_row_value(row, "safety_priority", "")).lower() == "urgent":
        weights = {
            "escalated": 0.45,
            "accepted": 0.25,
            "marked_useful": 0.15,
            "delayed": 0.10 + 0.10 * burden_score,
            "ignored": 0.03,
            "marked_false": 0.02 + 0.08 * false_positive,
        }
        return _weighted_choice(rng, weights)

    if status in {"grouped", "delayed", "priority_downgraded"}:
        weights = {
            "delayed": 0.32 + 0.25 * burden_score,
            "ignored": 0.20 + 0.20 * burden_score,
            "marked_false": 0.15 + 0.25 * false_positive,
            "accepted": 0.15 + 0.20 * actionability,
            "marked_useful": 0.08 + 0.12 * actionability,
            "escalated": 0.03,
        }
        return _weighted_choice(rng, weights)

    weights = {
        "accepted": 0.24 + 0.30 * actionability,
        "marked_useful": 0.10 + 0.20 * actionability,
        "delayed": 0.16 + 0.22 * burden_score,
        "ignored": 0.10 + 0.20 * burden_score + 0.15 * false_positive,
        "marked_false": 0.05 + 0.25 * false_positive,
        "escalated": 0.04 + 0.18 * _numeric(row, "risk_score", 0.0),
    }
    return _weighted_choice(rng, weights)


def simulate_response_time(
    row: pd.Series | dict[str, Any],
    response_type: str,
    burden_score: float,
    random_state: np.random.Generator | int | None = None,
) -> float:
    """Simulate response time in minutes from severity, burden, and time of day."""
    rng = _as_rng(random_state)
    severity = _severity(row)
    base_time = {
        "critical": 3.0,
        "high": 8.0,
        "medium": 22.0,
        "low": 45.0,
    }.get(severity, 35.0)
    response_adjustment = {
        "escalated": 0.55,
        "accepted": 0.85,
        "marked_useful": 0.90,
        "delayed": 1.80,
        "ignored": 2.20,
        "marked_false": 1.25,
    }.get(response_type, 1.0)
    status_adjustment = {
        "active": 1.0,
        "grouped": 1.30,
        "delayed": 1.60,
        "priority_downgraded": 1.35,
        "escalated": 0.65,
    }.get(str(_row_value(row, "final_alert_status", "active")).lower(), 1.0)
    night_adjustment = 1.25 if _is_night_hour(row) else 0.90
    jitter = rng.normal(1.0, 0.08)

    minutes = base_time * response_adjustment * status_adjustment * night_adjustment
    minutes *= 1.0 + 0.55 * burden_score
    minutes *= max(0.75, float(jitter))

    if _is_immediate_or_critical(row):
        minutes = min(minutes, 8.0)
    return round(max(0.0, float(minutes)), 2)


def generate_response_reason(
    row: pd.Series | dict[str, Any],
    response_type: str,
    burden_score: float,
) -> str:
    """Explain why the simulated workflow produced a response type."""
    if response_type == "escalated":
        if _is_immediate_or_critical(row):
            return "Critical or immediate-priority alert was escalated quickly."
        return "High urgency or worsening alert pattern led to escalation."
    if response_type == "accepted":
        return "Alert had enough actionability and risk evidence to be accepted for review."
    if response_type == "marked_useful":
        return "Clear trigger reason and useful evidence made the alert helpful in simulation."
    if response_type == "marked_false":
        return "High false-positive likelihood or weak evidence led to a simulated false-alert mark."
    if response_type == "ignored":
        return "Lower-priority repeated alert was ignored under simulated alert burden."
    if response_type == "delayed":
        return "Response delayed because burden, time of day, or reduced alert status slowed review."
    return "Response generated by simulated workflow rules."


def assign_workflow_stage(response_type: str, row: pd.Series | dict[str, Any] | None = None) -> str:
    """Map response type and alert context to a workflow stage."""
    if response_type == "escalated":
        return "escalated_review"
    if response_type in {"accepted", "marked_useful"}:
        if row is not None and _severity(row) in {"high", "critical"}:
            return "clinician_review"
        return "nurse_review"
    if response_type in {"ignored", "marked_false"}:
        return "closed"
    return "triage_queue"


def simulate_clinician_responses(alerts_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate response logs using current alert data and past patient history."""
    _validate_input_schema(alerts_df)
    rng = np.random.default_rng(seed)
    alerts = alerts_df.copy()
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], errors="coerce")
    alerts = alerts.sort_values(["patient_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    response_rows: list[dict[str, Any]] = []
    for _, patient_alerts in alerts.groupby("patient_id", sort=False):
        patient_history = pd.DataFrame(columns=alerts.columns)
        for _, row in patient_alerts.iterrows():
            recent_alerts = _recent_alerts(row, patient_history)
            burden_score = calculate_clinician_burden_score(row, recent_alerts)
            response_type = simulate_response_type(row, burden_score, rng)
            response_time = simulate_response_time(row, response_type, burden_score, rng)
            stage = assign_workflow_stage(response_type, row)
            usefulness = _perceived_usefulness(row, response_type, burden_score)
            response_rows.append(
                {
                    "response_id": _make_response_id(row, len(response_rows) + 1),
                    "alert_id": row["alert_id"],
                    "patient_id": row["patient_id"],
                    "timestamp": row["timestamp"],
                    "severity": _severity(row),
                    "final_alert_status": str(_row_value(row, "final_alert_status", "active")),
                    "fatigue_action": str(_row_value(row, "fatigue_action", "retain")),
                    "simulated_response": response_type,
                    "response_time_minutes": response_time,
                    "response_reason": generate_response_reason(row, response_type, burden_score),
                    "clinician_burden_score": burden_score,
                    "perceived_alert_usefulness": usefulness,
                    "workflow_stage": stage,
                    "escalation_completed": bool(response_type == "escalated"),
                    "response_simulation_note": (
                        "Simulated response only; not a real clinical workflow or care decision."
                    ),
                }
            )
            patient_history = pd.concat([patient_history, row.to_frame().T], ignore_index=True)

    response_df = pd.DataFrame(response_rows, columns=REQUIRED_RESPONSE_COLUMNS)
    response_df["response_time_minutes"] = pd.to_numeric(
        response_df["response_time_minutes"],
        errors="coerce",
    ).fillna(0.0)
    response_df["clinician_burden_score"] = pd.to_numeric(
        response_df["clinician_burden_score"],
        errors="coerce",
    ).fillna(0.0)
    response_df["perceived_alert_usefulness"] = pd.to_numeric(
        response_df["perceived_alert_usefulness"],
        errors="coerce",
    ).fillna(0.0)
    response_df["escalation_completed"] = response_df["escalation_completed"].astype(bool)
    return response_df


def save_response_logs(df: pd.DataFrame, path: str | Path) -> Path:
    """Save simulated clinician response logs."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_clinician_simulation_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the Step 11 clinician response simulation end to end."""
    alerts_df = load_fatigue_reduced_alerts(input_path)
    responses_df = simulate_clinician_responses(alerts_df, seed=42)
    saved_path = save_response_logs(responses_df, output_path)
    responses_df.attrs["output_path"] = str(saved_path)
    return responses_df


def simulate_clinician_response() -> pd.DataFrame:
    """Backward-compatible wrapper for Step 11 simulation."""
    return run_clinician_simulation_pipeline()


def _validate_input_schema(df: pd.DataFrame) -> None:
    """Validate required fatigue-reduced alert columns."""
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Fatigue-reduced alerts are missing required columns: {missing}")


def _recent_alerts(row: pd.Series | dict[str, Any], history: pd.DataFrame) -> pd.DataFrame:
    """Return previous alerts for the same patient within one hour."""
    if history.empty:
        return history.copy()
    current_time = pd.to_datetime(_row_value(row, "timestamp", pd.NaT), errors="coerce")
    if pd.isna(current_time):
        return history[history["patient_id"] == _row_value(row, "patient_id", None)].copy()
    patient_history = history[history["patient_id"] == _row_value(row, "patient_id", None)].copy()
    patient_history["timestamp"] = pd.to_datetime(patient_history["timestamp"], errors="coerce")
    window_start = current_time - pd.Timedelta(minutes=RECENT_ALERT_WINDOW_MINUTES)
    return patient_history[
        (patient_history["timestamp"] < current_time)
        & (patient_history["timestamp"] >= window_start)
    ]


def _perceived_usefulness(
    row: pd.Series | dict[str, Any],
    response_type: str,
    burden_score: float,
) -> float:
    """Estimate simulated perceived alert usefulness."""
    actionability = _numeric(row, "actionability_score", 0.5)
    false_positive = _numeric(row, "false_positive_likelihood", 0.0)
    risk = _numeric(row, "risk_score", 0.0)
    usefulness = 0.45 * actionability + 0.25 * risk + 0.20 * (1.0 - false_positive)
    usefulness -= 0.15 * burden_score
    if response_type in {"marked_useful", "accepted", "escalated"}:
        usefulness += 0.12
    if response_type in {"ignored", "marked_false"}:
        usefulness -= 0.20
    if _is_immediate_or_critical(row):
        usefulness = max(usefulness, 0.75)
    return _round_score(usefulness)


def _weighted_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Choose a key from non-negative weights."""
    labels = list(weights)
    raw = np.array([max(float(weights[label]), 0.0) for label in labels], dtype=float)
    if raw.sum() <= 0:
        return labels[0]
    probabilities = raw / raw.sum()
    return str(rng.choice(labels, p=probabilities))


def _as_rng(random_state: np.random.Generator | int | None) -> np.random.Generator:
    """Return a numpy random generator from supported inputs."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _is_immediate_or_critical(row: pd.Series | dict[str, Any]) -> bool:
    """Return True for alerts that should not be ignored in simulation."""
    return bool(
        _severity(row) == "critical"
        or _coerce_bool(_row_value(row, "critical_flag", False))
        or str(_row_value(row, "safety_priority", "")).lower() == "immediate"
        or str(_row_value(row, "escalation_recommendation", "")).lower()
        == "immediate_escalation"
    )


def _is_safety_sensitive(row: pd.Series | dict[str, Any]) -> bool:
    """Return True for alerts with review or safety-sensitive status."""
    return bool(
        _is_immediate_or_critical(row)
        or str(_row_value(row, "safety_priority", "")).lower() in {"urgent", "review"}
        or _coerce_bool(_row_value(row, "requires_human_review", False))
    )


def _is_night_hour(row: pd.Series | dict[str, Any]) -> bool:
    """Return True for nighttime simulated staffing context."""
    timestamp = pd.to_datetime(_row_value(row, "timestamp", pd.NaT), errors="coerce")
    if pd.isna(timestamp):
        return False
    return bool(timestamp.hour < 7 or timestamp.hour >= 19)


def _make_response_id(row: pd.Series, sequence_number: int) -> str:
    """Create a deterministic response id."""
    alert_id = str(row["alert_id"]).replace("ALERT", "RESP")
    return f"{alert_id}-{sequence_number:06d}"


def _severity(row: pd.Series | dict[str, Any]) -> str:
    """Normalize severity."""
    return str(_row_value(row, "severity", "low")).strip().lower()


def _numeric(row: pd.Series | dict[str, Any], key: str, default: float) -> float:
    """Read numeric row values safely."""
    value = _row_value(row, key, default)
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(value: Any) -> bool:
    """Coerce bool-like values from CSV."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _round_score(value: float) -> float:
    """Clip and round a score."""
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    """Read a row-like value with fallback."""
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
    response_logs = run_clinician_simulation_pipeline()
    output_path = response_logs.attrs.get("output_path", str(_resolve_project_path(DEFAULT_OUTPUT_PATH)))

    print("Step 11 clinician workflow simulation complete")
    print(f"Total responses simulated: {len(response_logs)}")
    print("Simulated response distribution:")
    print(response_logs["simulated_response"].value_counts().to_dict())
    print(f"Average response time: {response_logs['response_time_minutes'].mean():.2f} minutes")
    print(f"Saved response logs to: {output_path}")
    print("First few response rows:")
    print(response_logs.head())
