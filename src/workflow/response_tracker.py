"""Response tracking summaries for simulated clinician workflow logs.

Step 11 summarizes response behavior after the clinician simulation layer. These
metrics are for prototype reliability experiments only and are not real clinical
operations metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_RESPONSE_LOG_PATH = Path("data/processed/clinician_response_logs.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/clinician_response_summary.json")

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


def load_response_logs(path: str | Path = DEFAULT_RESPONSE_LOG_PATH) -> pd.DataFrame:
    """Load simulated clinician response logs."""
    response_path = _resolve_project_path(path)
    if not response_path.exists():
        raise FileNotFoundError(f"Response log file not found: {response_path}")
    df = pd.read_csv(response_path)
    _validate_response_schema(df)
    return df


def calculate_response_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate core workflow response summary metrics."""
    _validate_response_schema(df)
    by_severity = calculate_response_by_severity(df)
    return {
        "total_responses": int(len(df)),
        "ignored_alert_rate": calculate_ignored_alert_rate(df),
        "delayed_alert_rate": calculate_delayed_alert_rate(df),
        "escalation_rate": calculate_escalation_rate(df),
        "average_response_time_minutes": calculate_average_response_time(df),
        "response_distribution": _value_counts_dict(df, "simulated_response"),
        "workflow_stage_distribution": _value_counts_dict(df, "workflow_stage"),
        "average_clinician_burden_score": _mean_numeric(df, "clinician_burden_score"),
        "average_perceived_alert_usefulness": _mean_numeric(df, "perceived_alert_usefulness"),
        "response_by_severity": by_severity.to_dict(orient="records"),
        "simulation_note": "Simulated workflow metrics only; not real clinical operations.",
    }


def calculate_ignored_alert_rate(df: pd.DataFrame) -> float:
    """Return proportion of simulated responses that ignored alerts."""
    if len(df) == 0:
        return 0.0
    return round(float((df["simulated_response"] == "ignored").mean()), 4)


def calculate_delayed_alert_rate(df: pd.DataFrame) -> float:
    """Return proportion of simulated responses that delayed alerts."""
    if len(df) == 0:
        return 0.0
    return round(float((df["simulated_response"] == "delayed").mean()), 4)


def calculate_escalation_rate(df: pd.DataFrame) -> float:
    """Return proportion of simulated responses that escalated alerts."""
    if len(df) == 0:
        return 0.0
    return round(float((df["simulated_response"] == "escalated").mean()), 4)


def calculate_average_response_time(df: pd.DataFrame) -> float:
    """Return average simulated response time in minutes."""
    return _mean_numeric(df, "response_time_minutes")


def calculate_response_by_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize response metrics by alert severity."""
    _validate_response_schema(df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "severity",
                "count",
                "ignored_rate",
                "delayed_rate",
                "escalation_rate",
                "average_response_time_minutes",
            ]
        )

    grouped_rows = []
    for severity, group in df.groupby("severity", sort=True):
        grouped_rows.append(
            {
                "severity": severity,
                "count": int(len(group)),
                "ignored_rate": calculate_ignored_alert_rate(group),
                "delayed_rate": calculate_delayed_alert_rate(group),
                "escalation_rate": calculate_escalation_rate(group),
                "average_response_time_minutes": calculate_average_response_time(group),
            }
        )
    return pd.DataFrame(grouped_rows)


def save_response_summary(summary: dict[str, Any], path: str | Path) -> Path:
    """Save response summary metrics to JSON."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def track_response() -> dict[str, Any]:
    """Backward-compatible wrapper for response summary tracking."""
    df = load_response_logs()
    summary = calculate_response_summary(df)
    save_response_summary(summary, DEFAULT_SUMMARY_PATH)
    return summary


def _validate_response_schema(df: pd.DataFrame) -> None:
    """Validate required response-log columns."""
    missing = [column for column in REQUIRED_RESPONSE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Response logs are missing required columns: {missing}")


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Return JSON-friendly value counts for a column."""
    return {str(key): int(value) for key, value in df[column].value_counts().to_dict().items()}


def _mean_numeric(df: pd.DataFrame, column: str) -> float:
    """Return rounded mean for a numeric column."""
    if df.empty:
        return 0.0
    return round(float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean()), 4)


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
    response_logs = load_response_logs()
    response_summary = calculate_response_summary(response_logs)
    output_path = save_response_summary(response_summary, DEFAULT_SUMMARY_PATH)

    print("Step 11 response summary complete")
    print(response_summary)
    print(f"Saved response summary to: {output_path}")
