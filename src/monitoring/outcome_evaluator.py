"""Simulated outcome effectiveness evaluation for alert reliability.

Step 24A evaluates associations across the simulated chain:
alert -> simulated response -> workflow recommendation -> simulated outcome.

This module is a research/engineering prototype only. It does not measure real
clinical effectiveness, patient benefit, diagnosis quality, or medical safety.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_RESULTS_PATH = Path("data/processed/outcome_effectiveness_results.csv")
OUTPUT_SUMMARY_PATH = Path("data/processed/outcome_effectiveness_summary.json")
SIMULATION_ONLY_NOTE = (
    "Simulation-only outcome association; not evidence of clinical effectiveness, "
    "patient benefit, or medical-device safety."
)

VALID_OUTCOME_LABELS = {"improved", "unchanged", "worsened", "unknown"}
REQUIRED_OUTPUT_COLUMNS = [
    "outcome_eval_id",
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "alert_type",
    "simulated_response",
    "response_time_minutes",
    "recommended_action",
    "outcome_label",
    "outcome_severity_change",
    "timely_response",
    "action_taken",
    "alert_useful",
    "outcome_effectiveness_score",
    "delayed_response_impact_score",
    "evaluation_reason",
    "simulation_only_note",
]


def safe_load_csv(path: str) -> pd.DataFrame:
    """Load a CSV artifact safely, returning an empty dataframe if unavailable."""
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def safe_load_json(path: str) -> dict[str, Any]:
    """Load a JSON artifact safely, returning an empty dictionary if unavailable."""
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def merge_outcome_context(
    alerts_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge alerts with simulated responses, recommendations, and outcome fields."""
    if alerts_df.empty:
        return pd.DataFrame()

    merged = alerts_df.copy()
    merged = _normalize_timestamp_column(merged, "timestamp")

    response_columns = [
        column
        for column in [
            "alert_id",
            "simulated_response",
            "response_time_minutes",
            "response_reason",
            "clinician_burden_score",
            "perceived_alert_usefulness",
            "workflow_stage",
            "escalation_completed",
        ]
        if column in responses_df.columns
    ]
    if response_columns and "alert_id" in response_columns:
        merged = merged.merge(
            responses_df[response_columns],
            on="alert_id",
            how="left",
            suffixes=("", "_response"),
        )

    action_columns = [
        column
        for column in [
            "alert_id",
            "recommended_action",
            "action_priority",
            "human_review_required",
            "confidence_level",
            "rag_sources",
        ]
        if column in actions_df.columns
    ]
    if action_columns and "alert_id" in action_columns:
        merged = merged.merge(
            actions_df[action_columns],
            on="alert_id",
            how="left",
            suffixes=("", "_action"),
        )

    patient_columns = [
        column
        for column in [
            "patient_id",
            "timestamp",
            "patient_outcome_after_alert",
            "outcome_timestamp",
            "outcome_severity_change",
        ]
        if column in patient_df.columns
    ]
    if {"patient_id", "timestamp"}.issubset(patient_columns):
        patient_outcomes = _normalize_timestamp_column(patient_df[patient_columns].copy(), "timestamp")
        merged = merged.merge(
            patient_outcomes,
            on=["patient_id", "timestamp"],
            how="left",
            suffixes=("", "_patient"),
        )

    return merged


def classify_outcome(row: pd.Series) -> str:
    """Classify simulated outcome from severity change or provided outcome label."""
    severity_change = _safe_float(row.get("outcome_severity_change"), default=None)
    if severity_change is not None:
        if severity_change < 0:
            return "improved"
        if severity_change > 0:
            return "worsened"
        return "unchanged"

    existing_label = str(row.get("patient_outcome_after_alert", "")).strip().lower()
    if existing_label in VALID_OUTCOME_LABELS:
        return existing_label
    return "unknown"


def calculate_timely_response(row: pd.Series) -> bool:
    """Check whether the simulated response time met severity-based expectations."""
    response_time = _safe_float(row.get("response_time_minutes"), default=None)
    if response_time is None:
        return False
    severity = str(row.get("severity", "")).lower()
    threshold = _severity_time_threshold_minutes(severity)
    return response_time <= threshold


def calculate_action_taken(row: pd.Series) -> bool:
    """Estimate whether the simulated workflow shows action beyond passive logging."""
    response = str(row.get("simulated_response", "")).strip().lower()
    action = str(row.get("recommended_action", "")).strip().lower()
    workflow_stage = str(row.get("workflow_stage", "")).strip().lower()
    escalation_completed = _coerce_bool(row.get("escalation_completed"))

    if response in {"accepted", "escalated", "marked_useful"}:
        return True
    if escalation_completed:
        return True
    if workflow_stage in {"nurse_review", "clinician_review", "escalated_review"} and response not in {
        "ignored",
        "marked_false",
    }:
        return True
    return action in {
        "immediate_human_review",
        "urgent_clinician_review",
        "nurse_review",
        "monitor_trend",
        "request_manual_verification",
        "flag_sensor_reliability_issue",
    } and response not in {"ignored", "marked_false"}


def calculate_alert_usefulness(row: pd.Series) -> bool:
    """Estimate simulated alert usefulness from response, action, timeliness, and outcome."""
    response = str(row.get("simulated_response", "")).strip().lower()
    if response == "marked_false":
        return False
    if response in {"accepted", "escalated", "marked_useful"}:
        return True

    perceived_usefulness = _safe_float(row.get("perceived_alert_usefulness"), default=0.0)
    severity = str(row.get("severity", "")).strip().lower()
    outcome_label = str(row.get("outcome_label", classify_outcome(row))).lower()
    timely = _coerce_bool(row.get("timely_response")) or calculate_timely_response(row)
    action_taken = _coerce_bool(row.get("action_taken")) or calculate_action_taken(row)

    if perceived_usefulness >= 0.6:
        return True
    if severity in {"critical", "high"} and timely and action_taken:
        return True
    return outcome_label in {"improved", "unchanged"} and action_taken and timely


def calculate_outcome_effectiveness_score(row: pd.Series) -> float:
    """Calculate a transparent simulated effectiveness score between 0 and 1."""
    outcome_label = str(row.get("outcome_label", classify_outcome(row))).lower()
    base_scores = {
        "improved": 0.75,
        "unchanged": 0.55,
        "worsened": 0.25,
        "unknown": 0.40,
    }
    score = base_scores.get(outcome_label, 0.40)

    timely = _coerce_bool(row.get("timely_response")) or calculate_timely_response(row)
    action_taken = _coerce_bool(row.get("action_taken")) or calculate_action_taken(row)
    alert_useful = _coerce_bool(row.get("alert_useful"))
    response = str(row.get("simulated_response", "")).strip().lower()
    severity = str(row.get("severity", "")).strip().lower()

    if timely:
        score += 0.10
    if action_taken:
        score += 0.12
    if alert_useful:
        score += 0.10
    if severity in {"critical", "high"} and timely and action_taken:
        score += 0.05
    if response == "ignored":
        score -= 0.20
    elif response == "delayed":
        score -= 0.12
    elif response == "marked_false":
        score -= 0.15

    score -= 0.15 * calculate_delayed_response_impact_score(row)
    return _clip_score(score)


def calculate_delayed_response_impact_score(row: pd.Series) -> float:
    """Estimate simulated negative impact from delayed or ignored responses."""
    response = str(row.get("simulated_response", "")).strip().lower()
    response_time = _safe_float(row.get("response_time_minutes"), default=None)
    severity = str(row.get("severity", "")).strip().lower()
    threshold = _severity_time_threshold_minutes(severity)

    score = 0.0
    if response == "ignored":
        score += 0.45
    elif response == "delayed":
        score += 0.30
    elif response == "marked_false":
        score += 0.15

    if response_time is None:
        score += 0.20
    elif response_time > threshold:
        score += min((response_time - threshold) / max(threshold, 1), 1.0) * 0.45

    if severity in {"critical", "high"} and score > 0:
        score += 0.15
    return _clip_score(score)


def generate_evaluation_reason(row: pd.Series) -> str:
    """Generate a concise, non-clinical explanation for the simulated evaluation."""
    outcome = str(row.get("outcome_label", classify_outcome(row)))
    response = str(row.get("simulated_response", "unknown"))
    action = str(row.get("recommended_action", "not_available"))
    timely = "timely" if _coerce_bool(row.get("timely_response")) else "not timely"
    useful = "useful" if _coerce_bool(row.get("alert_useful")) else "less useful"
    return (
        f"Simulated association: outcome was classified as {outcome}; response "
        f"was {response}; recommended workflow action was {action}; response was "
        f"{timely}; alert was assessed as {useful}. This is not clinical effectiveness evidence."
    )


def evaluate_alert_outcomes(
    alerts_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate simulated outcome effectiveness for alert records."""
    context = merge_outcome_context(alerts_df, responses_df, actions_df, patient_df)
    if context.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    results = context.copy()
    results["outcome_eval_id"] = [
        f"OUTCOME-EVAL-{index + 1:06d}" for index in range(len(results))
    ]
    results["simulated_response"] = results.get("simulated_response", "unknown")
    results["recommended_action"] = results.get("recommended_action", "not_available")
    results["outcome_label"] = results.apply(classify_outcome, axis=1)
    results["timely_response"] = results.apply(calculate_timely_response, axis=1)
    results["action_taken"] = results.apply(calculate_action_taken, axis=1)
    results["alert_useful"] = results.apply(calculate_alert_usefulness, axis=1)
    results["outcome_effectiveness_score"] = results.apply(
        calculate_outcome_effectiveness_score,
        axis=1,
    )
    results["delayed_response_impact_score"] = results.apply(
        calculate_delayed_response_impact_score,
        axis=1,
    )
    results["evaluation_reason"] = results.apply(generate_evaluation_reason, axis=1)
    results["simulation_only_note"] = SIMULATION_ONLY_NOTE

    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in results.columns:
            results[column] = pd.NA

    return results[REQUIRED_OUTPUT_COLUMNS].copy()


def calculate_outcome_summary(results_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize simulated outcome effectiveness results."""
    total = int(len(results_df))
    if results_df.empty:
        return {
            "total_evaluated_alerts": 0,
            "improved_count": 0,
            "unchanged_count": 0,
            "worsened_count": 0,
            "unknown_count": 0,
            "useful_alert_rate": 0.0,
            "useless_alert_rate": 0.0,
            "action_to_outcome_success_rate": 0.0,
            "average_outcome_effectiveness_score": 0.0,
            "average_delayed_response_impact_score": 0.0,
            "simulation_only_note": SIMULATION_ONLY_NOTE,
        }

    useful_count = int(_bool_series(results_df["alert_useful"]).sum())
    action_taken = _bool_series(results_df["action_taken"])
    successful_outcomes = results_df["outcome_label"].isin(["improved", "unchanged"])
    action_success_count = int((action_taken & successful_outcomes).sum())
    action_count = int(action_taken.sum())

    return {
        "total_evaluated_alerts": total,
        "improved_count": _label_count(results_df, "improved"),
        "unchanged_count": _label_count(results_df, "unchanged"),
        "worsened_count": _label_count(results_df, "worsened"),
        "unknown_count": _label_count(results_df, "unknown"),
        "useful_alert_rate": _safe_rate(useful_count, total),
        "useless_alert_rate": _safe_rate(total - useful_count, total),
        "action_to_outcome_success_rate": _safe_rate(action_success_count, action_count),
        "average_outcome_effectiveness_score": _mean(results_df, "outcome_effectiveness_score"),
        "average_delayed_response_impact_score": _mean(results_df, "delayed_response_impact_score"),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


def save_outcome_results(df: pd.DataFrame, path: str) -> Path:
    """Save detailed outcome effectiveness results."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_outcome_summary(summary: dict[str, Any], path: str) -> Path:
    """Save outcome effectiveness summary JSON."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def run_outcome_evaluation_pipeline(
    patient_path: str = "data/simulated/patient_monitoring.csv",
    generated_alerts_path: str = "data/processed/generated_alerts.csv",
    fatigue_alerts_path: str = "data/processed/fatigue_reduced_alerts.csv",
    response_path: str = "data/processed/clinician_response_logs.csv",
    action_path: str = "data/processed/action_recommendations.csv",
    metrics_path: str = "data/processed/project_metrics_summary.json",
    results_path: str = str(OUTPUT_RESULTS_PATH),
    summary_path: str = str(OUTPUT_SUMMARY_PATH),
) -> pd.DataFrame:
    """Run the full simulated outcome effectiveness evaluation pipeline."""
    patient_df = safe_load_csv(patient_path)
    generated_alerts_df = safe_load_csv(generated_alerts_path)
    fatigue_alerts_df = safe_load_csv(fatigue_alerts_path)
    responses_df = safe_load_csv(response_path)
    actions_df = safe_load_csv(action_path)
    _ = safe_load_json(metrics_path)

    alerts_df = _choose_alert_source(generated_alerts_df, fatigue_alerts_df)
    results_df = evaluate_alert_outcomes(alerts_df, responses_df, actions_df, patient_df)
    summary = calculate_outcome_summary(results_df)

    save_outcome_results(results_df, results_path)
    save_outcome_summary(summary, summary_path)
    return results_df


def evaluate_simulated_outcomes() -> pd.DataFrame:
    """Compatibility wrapper for the earlier placeholder function."""
    return run_outcome_evaluation_pipeline()


def _choose_alert_source(
    generated_alerts_df: pd.DataFrame,
    fatigue_alerts_df: pd.DataFrame,
) -> pd.DataFrame:
    if fatigue_alerts_df.empty:
        return generated_alerts_df.copy()
    if generated_alerts_df.empty:
        return fatigue_alerts_df.copy()

    base = fatigue_alerts_df.copy()
    missing_columns = [
        column
        for column in generated_alerts_df.columns
        if column not in base.columns and column != "alert_id"
    ]
    if not missing_columns:
        return base
    return base.merge(
        generated_alerts_df[["alert_id", *missing_columns]],
        on="alert_id",
        how="left",
    )


def _normalize_timestamp_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    df[column] = pd.to_datetime(df[column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _severity_time_threshold_minutes(severity: str) -> float:
    return {
        "critical": 10.0,
        "high": 15.0,
        "medium": 30.0,
        "low": 60.0,
    }.get(str(severity).lower(), 60.0)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    text_true = series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
    numeric_true = pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return text_true | numeric_true


def _clip_score(value: Any) -> float:
    numeric = _safe_float(value, default=0.0)
    if numeric is None:
        numeric = 0.0
    return round(max(0.0, min(1.0, numeric)), 4)


def _safe_rate(numerator: Any, denominator: Any) -> float:
    denominator_float = _safe_float(denominator, default=0.0) or 0.0
    if denominator_float <= 0:
        return 0.0
    numerator_float = _safe_float(numerator, default=0.0) or 0.0
    return round(numerator_float / denominator_float, 4)


def _mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return round(float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean()), 4)


def _label_count(df: pd.DataFrame, label: str) -> int:
    if df.empty or "outcome_label" not in df.columns:
        return 0
    return int(df["outcome_label"].astype(str).str.lower().eq(label).sum())


if __name__ == "__main__":
    results = run_outcome_evaluation_pipeline()
    summary = calculate_outcome_summary(results)
    print(f"Total evaluated alerts: {summary['total_evaluated_alerts']}")
    print("Outcome label distribution:")
    if not results.empty and "outcome_label" in results.columns:
        print(results["outcome_label"].value_counts().to_string())
    print(f"Useful alert rate: {summary['useful_alert_rate']:.4f}")
    print(f"Average effectiveness score: {summary['average_outcome_effectiveness_score']:.4f}")
    print(f"Saved results to {_resolve_project_path(OUTPUT_RESULTS_PATH)}")
    print(f"Saved summary to {_resolve_project_path(OUTPUT_SUMMARY_PATH)}")
