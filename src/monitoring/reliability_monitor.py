"""Meta-AI reliability monitoring for simulated alert-system behavior.

Step 12 checks whether the prototype alert system appears reliable over time.
It uses transparent time-window metrics over generated alerts and simulated
clinician responses. This is not drift detection, online learning, clinical
validation, or a medical-device safety monitor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_ALERTS_PATH = Path("data/processed/fatigue_reduced_alerts.csv")
DEFAULT_RESPONSES_PATH = Path("data/processed/clinician_response_logs.csv")
DEFAULT_RESULTS_PATH = Path("data/processed/reliability_monitoring_results.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/reliability_summary.json")
DEFAULT_WINDOW_MINUTES = 120

REQUIRED_ALERT_COLUMNS = [
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "critical_flag",
    "safety_priority",
    "escalation_recommendation",
    "final_alert_status",
]

REQUIRED_RESPONSE_COLUMNS = [
    "alert_id",
    "simulated_response",
    "response_time_minutes",
    "workflow_stage",
    "escalation_completed",
]

REQUIRED_OUTPUT_COLUMNS = [
    "monitoring_window_id",
    "window_start",
    "window_end",
    "total_alerts",
    "active_alerts",
    "critical_alerts",
    "ignored_alert_rate",
    "delayed_alert_rate",
    "average_response_time_minutes",
    "false_alert_rate",
    "alert_volume_score",
    "response_quality_score",
    "safety_preservation_score",
    "reliability_score",
    "reliability_status",
    "reliability_warning",
    "review_recommendation",
]

VALID_RELIABILITY_STATUSES = {
    "stable",
    "watch",
    "degraded",
    "unsafe_review_required",
}

VALID_REVIEW_RECOMMENDATIONS = {
    "no_action_needed",
    "monitor_next_window",
    "review_thresholds",
    "review_workflow_burden",
    "urgent_human_review",
}


def load_fatigue_reduced_alerts(path: str | Path = DEFAULT_ALERTS_PATH) -> pd.DataFrame:
    """Load fatigue-reduced alerts from Step 10."""
    alerts_path = _resolve_project_path(path)
    if not alerts_path.exists():
        raise FileNotFoundError(f"Fatigue-reduced alerts file not found: {alerts_path}")
    alerts_df = pd.read_csv(alerts_path)
    _validate_columns(alerts_df, REQUIRED_ALERT_COLUMNS, "fatigue-reduced alerts")
    return alerts_df


def load_response_logs(path: str | Path = DEFAULT_RESPONSES_PATH) -> pd.DataFrame:
    """Load simulated clinician response logs from Step 11."""
    responses_path = _resolve_project_path(path)
    if not responses_path.exists():
        raise FileNotFoundError(f"Response logs file not found: {responses_path}")
    responses_df = pd.read_csv(responses_path)
    _validate_columns(responses_df, REQUIRED_RESPONSE_COLUMNS, "response logs")
    return responses_df


def merge_alerts_and_responses(
    alerts_df: pd.DataFrame,
    responses_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge fatigue-reduced alerts with simulated workflow responses."""
    _validate_columns(alerts_df, REQUIRED_ALERT_COLUMNS, "fatigue-reduced alerts")
    _validate_columns(responses_df, REQUIRED_RESPONSE_COLUMNS, "response logs")

    response_columns = [
        "alert_id",
        "simulated_response",
        "response_time_minutes",
        "workflow_stage",
        "escalation_completed",
    ]
    merged = alerts_df.merge(
        responses_df[response_columns],
        on="alert_id",
        how="left",
        validate="one_to_one",
    )
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged["simulated_response"] = merged["simulated_response"].fillna("missing_response")
    merged["response_time_minutes"] = pd.to_numeric(
        merged["response_time_minutes"],
        errors="coerce",
    ).fillna(0.0)
    merged["critical_flag"] = merged["critical_flag"].apply(_coerce_bool)
    return merged.sort_values(["timestamp", "patient_id", "alert_id"], kind="mergesort").reset_index(drop=True)


def create_monitoring_windows(
    df: pd.DataFrame,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Assign fixed-width monitoring windows to merged alert/response rows."""
    if window_minutes < 1:
        raise ValueError("window_minutes must be at least 1.")
    if df.empty:
        result = df.copy()
        result["monitoring_window_id"] = pd.Series(dtype=str)
        result["window_start"] = pd.Series(dtype="datetime64[ns]")
        result["window_end"] = pd.Series(dtype="datetime64[ns]")
        return result

    windowed = df.copy()
    windowed["timestamp"] = pd.to_datetime(windowed["timestamp"], errors="coerce")
    min_timestamp = windowed["timestamp"].dropna().min()
    if pd.isna(min_timestamp):
        min_timestamp = pd.Timestamp("1970-01-01 00:00:00")
    window_delta = pd.Timedelta(minutes=window_minutes)
    elapsed = (windowed["timestamp"].fillna(min_timestamp) - min_timestamp) / window_delta
    window_index = np.floor(elapsed).astype(int)

    windowed["_window_index"] = window_index
    windowed["window_start"] = min_timestamp + windowed["_window_index"] * window_delta
    windowed["window_end"] = windowed["window_start"] + window_delta
    windowed["monitoring_window_id"] = windowed["_window_index"].apply(
        lambda value: f"WINDOW-{int(value) + 1:03d}"
    )
    return windowed.drop(columns=["_window_index"])


def calculate_false_alert_rate(window_df: pd.DataFrame) -> float:
    """Calculate proportion of alerts marked false in the simulated response."""
    if window_df.empty:
        return 0.0
    return _round_rate((window_df["simulated_response"] == "marked_false").mean())


def calculate_alert_volume_score(window_df: pd.DataFrame) -> float:
    """Score alert volume burden for a monitoring window."""
    total_alerts = len(window_df)
    if total_alerts <= 25:
        return 1.0
    if total_alerts <= 50:
        return _round_rate(1.0 - ((total_alerts - 25) / 25.0) * 0.35)
    if total_alerts <= 75:
        return _round_rate(0.65 - ((total_alerts - 50) / 25.0) * 0.35)
    return 0.20


def calculate_response_quality_score(window_df: pd.DataFrame) -> float:
    """Score response quality from ignored, delayed, and slow responses."""
    if window_df.empty:
        return 1.0
    ignored_rate = _rate(window_df["simulated_response"] == "ignored")
    delayed_rate = _rate(window_df["simulated_response"] == "delayed")
    average_response_time = pd.to_numeric(
        window_df["response_time_minutes"],
        errors="coerce",
    ).fillna(0.0).mean()
    time_penalty = min(float(average_response_time) / 90.0, 1.0)
    score = 1.0 - (0.45 * ignored_rate + 0.35 * delayed_rate + 0.20 * time_penalty)
    return _round_rate(score)


def calculate_safety_preservation_score(window_df: pd.DataFrame) -> float:
    """Score whether critical/immediate alerts were preserved in responses."""
    protected = _protected_alert_mask(window_df)
    protected_df = window_df[protected]
    if protected_df.empty:
        return 1.0

    ignored_rate = _rate(protected_df["simulated_response"] == "ignored")
    delayed_or_false_rate = _rate(
        protected_df["simulated_response"].isin(["delayed", "marked_false"])
    )
    non_escalated_critical_rate = _rate(
        (protected_df["severity"].astype(str).str.lower() == "critical")
        & (~protected_df["simulated_response"].isin(["escalated", "accepted", "marked_useful"]))
    )
    score = 1.0 - (
        1.00 * ignored_rate
        + 0.45 * delayed_or_false_rate
        + 0.30 * non_escalated_critical_rate
    )
    return _round_rate(score)


def calculate_reliability_score(metrics: dict[str, Any]) -> float:
    """Combine transparent reliability components into one score."""
    score = (
        0.40 * float(metrics["safety_preservation_score"])
        + 0.30 * float(metrics["response_quality_score"])
        + 0.20 * float(metrics["alert_volume_score"])
        + 0.10 * (1.0 - float(metrics["false_alert_rate"]))
    )
    return _round_rate(score)


def assign_reliability_status(score: float, metrics: dict[str, Any]) -> str:
    """Assign reliability status from score and safety-sensitive conditions."""
    if bool(metrics.get("critical_or_immediate_ignored", False)):
        return "unsafe_review_required"
    if float(metrics.get("safety_preservation_score", 1.0)) < 0.70:
        return "unsafe_review_required"
    if score < 0.50:
        return "degraded"
    if score < 0.72:
        return "watch"
    return "stable"


def generate_reliability_warning(status: str, metrics: dict[str, Any]) -> str:
    """Generate a human-readable reliability warning."""
    if status == "unsafe_review_required":
        return "Unsafe review required: critical or immediate-priority alert handling was not preserved."
    if status == "degraded":
        return "Reliability degraded: response quality, alert volume, or false-alert burden needs review."
    if status == "watch":
        if float(metrics.get("alert_volume_score", 1.0)) < 0.60:
            return "Watch: alert volume is high enough to create workflow burden."
        if float(metrics.get("ignored_alert_rate", 0.0)) > 0.15:
            return "Watch: ignored alert rate is elevated in this window."
        if float(metrics.get("delayed_alert_rate", 0.0)) > 0.20:
            return "Watch: delayed alert rate is elevated in this window."
        return "Watch: reliability score is below the stable range."
    return "Stable: no reliability warning for this simulated window."


def generate_review_recommendation(status: str, metrics: dict[str, Any]) -> str:
    """Recommend review action for a monitoring window."""
    if status == "unsafe_review_required":
        return "urgent_human_review"
    if float(metrics.get("alert_volume_score", 1.0)) < 0.55 or float(metrics.get("delayed_alert_rate", 0.0)) > 0.25:
        return "review_workflow_burden"
    if float(metrics.get("false_alert_rate", 0.0)) > 0.20 or status == "degraded":
        return "review_thresholds"
    if status == "watch":
        return "monitor_next_window"
    return "no_action_needed"


def monitor_reliability(
    alerts_df: pd.DataFrame,
    responses_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate reliability metrics for each monitoring window."""
    merged = merge_alerts_and_responses(alerts_df, responses_df)
    windowed = create_monitoring_windows(merged)
    records: list[dict[str, Any]] = []

    for window_id, window_df in windowed.groupby("monitoring_window_id", sort=True):
        ignored_rate = _rate(window_df["simulated_response"] == "ignored")
        delayed_rate = _rate(window_df["simulated_response"] == "delayed")
        avg_response = _round_metric(
            pd.to_numeric(window_df["response_time_minutes"], errors="coerce").fillna(0.0).mean()
        )
        false_rate = calculate_false_alert_rate(window_df)
        volume_score = calculate_alert_volume_score(window_df)
        response_score = calculate_response_quality_score(window_df)
        safety_score = calculate_safety_preservation_score(window_df)
        protected = _protected_alert_mask(window_df)
        critical_or_immediate_ignored = bool(
            (protected & (window_df["simulated_response"] == "ignored")).any()
        )

        metrics = {
            "ignored_alert_rate": ignored_rate,
            "delayed_alert_rate": delayed_rate,
            "false_alert_rate": false_rate,
            "alert_volume_score": volume_score,
            "response_quality_score": response_score,
            "safety_preservation_score": safety_score,
            "critical_or_immediate_ignored": critical_or_immediate_ignored,
        }
        reliability_score = calculate_reliability_score(metrics)
        status = assign_reliability_status(reliability_score, metrics)
        warning = generate_reliability_warning(status, metrics)
        recommendation = generate_review_recommendation(status, metrics)

        records.append(
            {
                "monitoring_window_id": window_id,
                "window_start": window_df["window_start"].iloc[0],
                "window_end": window_df["window_end"].iloc[0],
                "total_alerts": int(len(window_df)),
                "active_alerts": int((window_df["final_alert_status"] == "active").sum()),
                "critical_alerts": int(_protected_alert_mask(window_df).sum()),
                "ignored_alert_rate": ignored_rate,
                "delayed_alert_rate": delayed_rate,
                "average_response_time_minutes": avg_response,
                "false_alert_rate": false_rate,
                "alert_volume_score": volume_score,
                "response_quality_score": response_score,
                "safety_preservation_score": safety_score,
                "reliability_score": reliability_score,
                "reliability_status": status,
                "reliability_warning": warning,
                "review_recommendation": recommendation,
            }
        )

    return pd.DataFrame(records, columns=REQUIRED_OUTPUT_COLUMNS)


def save_reliability_results(df: pd.DataFrame, path: str | Path) -> Path:
    """Save per-window reliability monitoring results to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_reliability_summary(summary: dict[str, Any], path: str | Path) -> Path:
    """Save reliability summary to JSON."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def run_reliability_monitoring_pipeline(
    alerts_path: str | Path = DEFAULT_ALERTS_PATH,
    responses_path: str | Path = DEFAULT_RESPONSES_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> pd.DataFrame:
    """Run the Step 12 reliability monitoring workflow end to end."""
    alerts_df = load_fatigue_reduced_alerts(alerts_path)
    responses_df = load_response_logs(responses_path)
    results_df = monitor_reliability(alerts_df, responses_df)
    results_saved_path = save_reliability_results(results_df, results_path)
    summary = _build_reliability_summary(results_df)
    summary_saved_path = save_reliability_summary(summary, summary_path)
    results_df.attrs["results_path"] = str(results_saved_path)
    results_df.attrs["summary_path"] = str(summary_saved_path)
    results_df.attrs["summary"] = summary
    return results_df


def _build_reliability_summary(results_df: pd.DataFrame) -> dict[str, Any]:
    """Build JSON-friendly aggregate reliability summary."""
    if results_df.empty:
        return {
            "total_monitoring_windows": 0,
            "average_reliability_score": 0.0,
            "reliability_status_distribution": {},
            "review_recommendation_distribution": {},
            "unsafe_windows": 0,
            "simulation_note": "Simulated reliability metrics only; not clinical validation.",
        }
    return {
        "total_monitoring_windows": int(len(results_df)),
        "average_reliability_score": _round_metric(results_df["reliability_score"].mean()),
        "minimum_reliability_score": _round_metric(results_df["reliability_score"].min()),
        "reliability_status_distribution": _value_counts(results_df, "reliability_status"),
        "review_recommendation_distribution": _value_counts(results_df, "review_recommendation"),
        "unsafe_windows": int((results_df["reliability_status"] == "unsafe_review_required").sum()),
        "windows_requiring_review": int(
            (results_df["review_recommendation"] != "no_action_needed").sum()
        ),
        "simulation_note": "Simulated reliability metrics only; not clinical validation.",
    }


def _protected_alert_mask(df: pd.DataFrame) -> pd.Series:
    """Return mask for critical/immediate safety-sensitive alerts."""
    return (
        df["severity"].astype(str).str.lower().eq("critical")
        | df["critical_flag"].apply(_coerce_bool)
        | df["safety_priority"].astype(str).str.lower().eq("immediate")
        | df["escalation_recommendation"].astype(str).str.lower().eq("immediate_escalation")
    )


def _validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    """Raise a clear schema error for missing columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _rate(mask: pd.Series) -> float:
    """Return rounded true-rate for a boolean mask."""
    if len(mask) == 0:
        return 0.0
    return _round_rate(mask.mean())


def _round_rate(value: float) -> float:
    """Clip a rate to [0, 1] and round."""
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def _round_metric(value: float) -> float:
    """Round a metric for stable CSV/JSON output."""
    if pd.isna(value):
        return 0.0
    return round(float(value), 4)


def _coerce_bool(value: Any) -> bool:
    """Coerce bool-like CSV values."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Return JSON-friendly value counts."""
    return {str(key): int(value) for key, value in df[column].value_counts().to_dict().items()}


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
    reliability_results = run_reliability_monitoring_pipeline()
    summary = reliability_results.attrs.get("summary", _build_reliability_summary(reliability_results))
    warnings = reliability_results[
        reliability_results["review_recommendation"] != "no_action_needed"
    ][["monitoring_window_id", "reliability_status", "reliability_warning", "review_recommendation"]]

    print("Step 12 reliability monitoring complete")
    print(f"Total monitoring windows: {len(reliability_results)}")
    print("Reliability status distribution:")
    print(reliability_results["reliability_status"].value_counts().to_dict())
    print(f"Average reliability score: {summary.get('average_reliability_score', 0.0):.4f}")
    print("Warnings and recommendations:")
    print(warnings.head(10))
    print(f"Saved results to: {reliability_results.attrs.get('results_path')}")
    print(f"Saved summary to: {reliability_results.attrs.get('summary_path')}")
