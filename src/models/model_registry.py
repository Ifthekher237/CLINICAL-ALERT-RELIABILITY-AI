"""Simulation-only model registry and incremental update logic.

Step 14 shows how feedback, reliability monitoring, and drift detection could
inform a human-reviewed threshold/calibration update. It does not retrain,
deploy, overwrite model artifacts, or make clinical claims. The output is a
local simulation record for a healthcare AI engineering portfolio project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_AUDITED_PATH = Path("data/processed/audited_alerts.csv")
DEFAULT_FATIGUE_PATH = Path("data/processed/fatigue_reduced_alerts.csv")
DEFAULT_RESPONSE_PATH = Path("data/processed/clinician_response_logs.csv")
DEFAULT_RELIABILITY_PATH = Path("data/processed/reliability_monitoring_results.csv")
DEFAULT_DRIFT_PATH = Path("data/processed/drift_detection_results.csv")
DEFAULT_RESULTS_PATH = Path("data/processed/model_update_simulation_results.csv")
DEFAULT_REGISTRY_PATH = Path("data/processed/model_version_registry.json")
DEFAULT_THRESHOLD_SUMMARY_PATH = Path("data/processed/threshold_update_summary.json")

DEFAULT_CURRENT_THRESHOLD = 0.65
MIN_THRESHOLD = 0.40
MAX_THRESHOLD = 0.90

REQUIRED_AUDITED_COLUMNS = [
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "risk_score",
    "actionability_score",
    "false_positive_likelihood",
    "audit_status",
    "escalation_recommendation",
]

REQUIRED_FATIGUE_COLUMNS = [
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "critical_flag",
    "fatigue_action",
    "final_alert_status",
]

REQUIRED_RESPONSE_COLUMNS = [
    "response_id",
    "alert_id",
    "patient_id",
    "timestamp",
    "simulated_response",
    "response_time_minutes",
]

REQUIRED_RELIABILITY_COLUMNS = [
    "monitoring_window_id",
    "window_start",
    "window_end",
    "ignored_alert_rate",
    "delayed_alert_rate",
    "safety_preservation_score",
    "reliability_score",
    "reliability_status",
    "review_recommendation",
]

REQUIRED_DRIFT_COLUMNS = [
    "drift_window_id",
    "window_start",
    "window_end",
    "drift_type",
    "monitored_feature",
    "drift_score",
    "drift_status",
    "recalibration_recommendation",
    "requires_review",
]

REQUIRED_SIMULATION_COLUMNS = [
    "update_id",
    "previous_model_version",
    "proposed_model_version",
    "update_timestamp",
    "feedback_window_start",
    "feedback_window_end",
    "false_alert_rate",
    "ignored_alert_rate",
    "delayed_alert_rate",
    "useful_alert_rate",
    "average_drift_score",
    "severe_drift_count",
    "current_risk_threshold",
    "proposed_risk_threshold",
    "threshold_change",
    "update_reason",
    "expected_effect",
    "deployment_recommendation",
    "human_review_required",
]

VALID_DEPLOYMENT_RECOMMENDATIONS = {
    "no_update_needed",
    "monitor_more_data",
    "threshold_review_recommended",
    "calibration_review_recommended",
    "retraining_review_recommended",
}


def load_audited_alerts(path: str | Path = DEFAULT_AUDITED_PATH) -> pd.DataFrame:
    """Load Step 9 audited alerts."""
    df = _load_csv(path, "audited alerts")
    _validate_columns(df, REQUIRED_AUDITED_COLUMNS, "audited alerts")
    return df


def load_fatigue_reduced_alerts(path: str | Path = DEFAULT_FATIGUE_PATH) -> pd.DataFrame:
    """Load Step 10 fatigue-reduced alerts."""
    df = _load_csv(path, "fatigue-reduced alerts")
    _validate_columns(df, REQUIRED_FATIGUE_COLUMNS, "fatigue-reduced alerts")
    return df


def load_response_logs(path: str | Path = DEFAULT_RESPONSE_PATH) -> pd.DataFrame:
    """Load Step 11 simulated clinician response logs."""
    df = _load_csv(path, "response logs")
    _validate_columns(df, REQUIRED_RESPONSE_COLUMNS, "response logs")
    return df


def load_reliability_results(path: str | Path = DEFAULT_RELIABILITY_PATH) -> pd.DataFrame:
    """Load Step 12 reliability monitoring results."""
    df = _load_csv(path, "reliability results")
    _validate_columns(df, REQUIRED_RELIABILITY_COLUMNS, "reliability results")
    return df


def load_drift_results(path: str | Path = DEFAULT_DRIFT_PATH) -> pd.DataFrame:
    """Load Step 13 drift detection results."""
    df = _load_csv(path, "drift results")
    _validate_columns(df, REQUIRED_DRIFT_COLUMNS, "drift results")
    return df


def collect_feedback_signals(
    audited_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
    response_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
) -> dict[str, Any]:
    """Collect feedback signals for a human-reviewed update simulation."""
    _validate_columns(audited_df, REQUIRED_AUDITED_COLUMNS, "audited alerts")
    _validate_columns(fatigue_df, REQUIRED_FATIGUE_COLUMNS, "fatigue-reduced alerts")
    _validate_columns(response_df, REQUIRED_RESPONSE_COLUMNS, "response logs")
    _validate_columns(drift_df, REQUIRED_DRIFT_COLUMNS, "drift results")
    _validate_columns(reliability_df, REQUIRED_RELIABILITY_COLUMNS, "reliability results")

    response_count = max(len(response_df), 1)
    audited_count = max(len(audited_df), 1)
    fatigue_count = max(len(fatigue_df), 1)

    simulated_response = response_df["simulated_response"].astype(str).str.lower()
    false_alert_rate = _round_rate((simulated_response == "marked_false").mean())
    ignored_alert_rate = _round_rate((simulated_response == "ignored").mean())
    delayed_alert_rate = _round_rate((simulated_response == "delayed").mean())
    useful_alert_rate = _round_rate(
        simulated_response.isin(["marked_useful", "accepted", "escalated"]).mean()
    )

    false_positive_likelihood = pd.to_numeric(
        audited_df["false_positive_likelihood"],
        errors="coerce",
    ).fillna(0.0)
    actionability = pd.to_numeric(
        audited_df["actionability_score"],
        errors="coerce",
    ).fillna(0.0)
    drift_score = pd.to_numeric(drift_df["drift_score"], errors="coerce").fillna(0.0)
    reliability_score = pd.to_numeric(
        reliability_df["reliability_score"],
        errors="coerce",
    ).fillna(0.0)
    safety_score = pd.to_numeric(
        reliability_df["safety_preservation_score"],
        errors="coerce",
    ).fillna(1.0)

    feedback_start, feedback_end = _feedback_window(
        [audited_df, fatigue_df, response_df, drift_df, reliability_df]
    )
    severe_drift_count = int((drift_df["drift_status"].astype(str) == "severe_shift").sum())
    drift_review_count = int(drift_df["requires_review"].apply(_coerce_bool).sum())
    workflow_burden_windows = int(
        reliability_df["review_recommendation"].astype(str).isin(
            ["review_workflow_burden", "review_thresholds"]
        ).sum()
    )
    unstable_reliability_windows = int(
        reliability_df["reliability_status"].astype(str).isin(
            ["watch", "degraded", "unsafe_review_required"]
        ).sum()
    )

    feedback = {
        "feedback_window_start": str(feedback_start),
        "feedback_window_end": str(feedback_end),
        "total_audited_alerts": int(len(audited_df)),
        "total_fatigue_reduced_alerts": int(len(fatigue_df)),
        "total_response_logs": int(len(response_df)),
        "total_drift_checks": int(len(drift_df)),
        "total_reliability_windows": int(len(reliability_df)),
        "false_alert_rate": false_alert_rate,
        "ignored_alert_rate": ignored_alert_rate,
        "delayed_alert_rate": delayed_alert_rate,
        "useful_alert_rate": useful_alert_rate,
        "marked_false_count": int((simulated_response == "marked_false").sum()),
        "ignored_alert_count": int((simulated_response == "ignored").sum()),
        "delayed_alert_count": int((simulated_response == "delayed").sum()),
        "useful_alert_count": int(
            simulated_response.isin(["marked_useful", "accepted", "escalated"]).sum()
        ),
        "high_false_positive_likelihood_rate": _round_rate(
            (false_positive_likelihood >= 0.60).sum() / audited_count
        ),
        "low_actionability_rate": _round_rate((actionability < 0.45).sum() / audited_count),
        "average_false_positive_likelihood": _round_rate(false_positive_likelihood.mean()),
        "average_actionability_score": _round_rate(actionability.mean()),
        "grouped_or_delayed_alert_rate": _round_rate(
            fatigue_df["final_alert_status"].astype(str).str.lower().isin(["grouped", "delayed"]).sum()
            / fatigue_count
        ),
        "critical_alert_rate": _round_rate(
            fatigue_df["critical_flag"].apply(_coerce_bool).sum() / fatigue_count
        ),
        "average_drift_score": _round_rate(drift_score.mean()),
        "maximum_drift_score": _round_rate(drift_score.max()),
        "severe_drift_count": severe_drift_count,
        "drift_review_count": drift_review_count,
        "average_reliability_score": _round_rate(reliability_score.mean()),
        "minimum_reliability_score": _round_rate(reliability_score.min()),
        "average_safety_preservation_score": _round_rate(safety_score.mean()),
        "critical_preservation_strong": bool(safety_score.min() >= 0.95),
        "workflow_burden_windows": workflow_burden_windows,
        "unstable_reliability_windows": unstable_reliability_windows,
        "simulation_note": "Feedback signals are simulated and require human review before any real model change.",
    }

    # Keep explicit denominators visible for reviewers.
    feedback["response_count_denominator"] = int(response_count)
    feedback["audited_count_denominator"] = int(audited_count)
    return feedback


def calculate_threshold_adjustment(
    feedback: dict[str, Any],
    current_threshold: float = DEFAULT_CURRENT_THRESHOLD,
) -> dict[str, Any]:
    """Suggest a small bounded threshold update without deployment."""
    current_threshold = _clamp_threshold(current_threshold)
    false_burden = (
        0.45 * float(feedback.get("false_alert_rate", 0.0))
        + 0.25 * float(feedback.get("ignored_alert_rate", 0.0))
        + 0.20 * float(feedback.get("high_false_positive_likelihood_rate", 0.0))
        + 0.10 * float(feedback.get("low_actionability_rate", 0.0))
    )
    useful_signal = float(feedback.get("useful_alert_rate", 0.0))
    delayed_rate = float(feedback.get("delayed_alert_rate", 0.0))
    severe_drift_count = int(feedback.get("severe_drift_count", 0))
    critical_preservation_strong = bool(feedback.get("critical_preservation_strong", False))

    if severe_drift_count > 0:
        raw_change = 0.0
        direction = "hold_for_retraining_review"
        rationale = "Severe drift is present, so a threshold-only update should not be simulated as sufficient."
    elif false_burden >= 0.35 and useful_signal < 0.75:
        raw_change = 0.07
        direction = "raise_threshold"
        rationale = "False-alert burden is high and useful response evidence is not dominant."
    elif false_burden >= 0.22:
        raw_change = 0.05
        direction = "raise_threshold"
        rationale = "False-alert or low-actionability burden supports a small threshold review."
    elif delayed_rate >= 0.18 and useful_signal >= 0.65 and critical_preservation_strong:
        raw_change = -0.03
        direction = "lower_threshold_slightly"
        rationale = "Useful alerts are present but delayed, so calibration review may preserve sensitivity."
    else:
        raw_change = 0.0
        direction = "keep_threshold"
        rationale = "Feedback is mixed or stable, so more monitoring is preferred over changing thresholds."

    proposed_threshold = _clamp_threshold(current_threshold + raw_change)
    threshold_change = _round_rate(proposed_threshold - current_threshold)

    return {
        "current_risk_threshold": current_threshold,
        "proposed_risk_threshold": proposed_threshold,
        "threshold_change": threshold_change,
        "threshold_direction": direction,
        "false_burden_score": _round_rate(false_burden),
        "bounded_threshold_min": MIN_THRESHOLD,
        "bounded_threshold_max": MAX_THRESHOLD,
        "rationale": rationale,
        "simulation_only": True,
        "human_review_required": bool(
            severe_drift_count > 0 or abs(threshold_change) >= 0.05
        ),
    }


def create_model_version_record(
    previous_version: str,
    proposed_version: str,
    feedback: dict[str, Any],
    threshold_update: dict[str, Any],
) -> dict[str, Any]:
    """Create version metadata without touching trained model files."""
    recommendation = generate_deployment_recommendation(feedback, threshold_update)
    human_review_required = _human_review_required(feedback, threshold_update, recommendation)
    return {
        "previous_model_version": previous_version,
        "proposed_model_version": proposed_version,
        "created_at": _update_timestamp(feedback),
        "status": "simulation_only_not_deployed",
        "deployment_recommendation": recommendation,
        "human_review_required": human_review_required,
        "current_risk_threshold": threshold_update["current_risk_threshold"],
        "proposed_risk_threshold": threshold_update["proposed_risk_threshold"],
        "threshold_change": threshold_update["threshold_change"],
        "feedback_window_start": feedback.get("feedback_window_start"),
        "feedback_window_end": feedback.get("feedback_window_end"),
        "average_drift_score": feedback.get("average_drift_score", 0.0),
        "severe_drift_count": feedback.get("severe_drift_count", 0),
        "simulation_note": "Version record only; no model artifact was replaced or deployed.",
    }


def compare_old_vs_proposed_configuration(
    feedback: dict[str, Any],
    threshold_update: dict[str, Any],
) -> dict[str, Any]:
    """Compare old and proposed threshold configuration in plain terms."""
    change = float(threshold_update.get("threshold_change", 0.0))
    if change > 0:
        expected_false_alert_burden = "slightly_lower"
        expected_sensitivity = "possibly_lower"
    elif change < 0:
        expected_false_alert_burden = "possibly_higher"
        expected_sensitivity = "slightly_higher"
    else:
        expected_false_alert_burden = "unchanged"
        expected_sensitivity = "unchanged"

    if int(feedback.get("severe_drift_count", 0)) > 0:
        confidence = "low_until_drift_review"
    elif float(feedback.get("average_reliability_score", 0.0)) >= 0.85:
        confidence = "moderate"
    else:
        confidence = "limited"

    return {
        "old_threshold": threshold_update["current_risk_threshold"],
        "proposed_threshold": threshold_update["proposed_risk_threshold"],
        "threshold_change": threshold_update["threshold_change"],
        "expected_false_alert_burden": expected_false_alert_burden,
        "expected_sensitivity": expected_sensitivity,
        "comparison_confidence": confidence,
        "critical_preservation_observed": bool(feedback.get("critical_preservation_strong", False)),
        "simulation_note": "Comparison is estimated from simulated feedback, not measured clinical performance.",
    }


def generate_update_reason(
    feedback: dict[str, Any],
    threshold_update: dict[str, Any],
) -> str:
    """Explain why the simulated update was recommended."""
    severe_drift_count = int(feedback.get("severe_drift_count", 0))
    if severe_drift_count > 0:
        return (
            f"Detected {severe_drift_count} severe drift checks, so the simulation recommends "
            "human retraining/calibration review instead of a direct threshold update."
        )
    if float(threshold_update.get("threshold_change", 0.0)) > 0:
        return (
            "False-alert, ignored-alert, or low-actionability feedback supports a small "
            "human-reviewed threshold increase."
        )
    if float(threshold_update.get("threshold_change", 0.0)) < 0:
        return (
            "Useful alert evidence with delayed responses supports a cautious calibration review "
            "rather than aggressive thresholding."
        )
    return "Feedback does not support a clear threshold change; continue collecting simulated evidence."


def generate_deployment_recommendation(
    feedback: dict[str, Any],
    threshold_update: dict[str, Any],
) -> str:
    """Recommend next action without deploying an updated model."""
    severe_drift_count = int(feedback.get("severe_drift_count", 0))
    average_drift = float(feedback.get("average_drift_score", 0.0))
    reliability = float(feedback.get("average_reliability_score", 0.0))
    workflow_burden_windows = int(feedback.get("workflow_burden_windows", 0))
    false_alert_rate = float(feedback.get("false_alert_rate", 0.0))
    threshold_change = abs(float(threshold_update.get("threshold_change", 0.0)))

    if severe_drift_count > 0 or average_drift >= 0.35:
        return "retraining_review_recommended"
    if threshold_change >= 0.05:
        return "threshold_review_recommended"
    if reliability >= 0.80 and workflow_burden_windows > 0:
        return "calibration_review_recommended"
    if false_alert_rate >= 0.12:
        return "threshold_review_recommended"
    if reliability < 0.70:
        return "monitor_more_data"
    return "no_update_needed"


def save_model_version_registry(registry: dict[str, Any], path: str | Path) -> Path:
    """Save model-version metadata JSON without modifying model artifacts."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)
    return output_path


def save_threshold_update_summary(summary: dict[str, Any], path: str | Path) -> Path:
    """Save threshold-update simulation summary JSON."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def run_model_update_simulation(
    audited_path: str | Path = DEFAULT_AUDITED_PATH,
    fatigue_path: str | Path = DEFAULT_FATIGUE_PATH,
    response_path: str | Path = DEFAULT_RESPONSE_PATH,
    reliability_path: str | Path = DEFAULT_RELIABILITY_PATH,
    drift_path: str | Path = DEFAULT_DRIFT_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    threshold_summary_path: str | Path = DEFAULT_THRESHOLD_SUMMARY_PATH,
) -> pd.DataFrame:
    """Run the simulation-only feedback and model-version update workflow."""
    audited_df = load_audited_alerts(audited_path)
    fatigue_df = load_fatigue_reduced_alerts(fatigue_path)
    response_df = load_response_logs(response_path)
    reliability_df = load_reliability_results(reliability_path)
    drift_df = load_drift_results(drift_path)

    feedback = collect_feedback_signals(
        audited_df,
        fatigue_df,
        response_df,
        drift_df,
        reliability_df,
    )
    threshold_update = calculate_threshold_adjustment(feedback, DEFAULT_CURRENT_THRESHOLD)
    previous_version = "risk_alert_model_v1.0.0"
    proposed_version = _next_patch_version(previous_version)
    version_record = create_model_version_record(
        previous_version,
        proposed_version,
        feedback,
        threshold_update,
    )
    comparison = compare_old_vs_proposed_configuration(feedback, threshold_update)
    recommendation = version_record["deployment_recommendation"]
    human_review_required = version_record["human_review_required"]

    result_record = {
        "update_id": f"MODEL-UPDATE-{_safe_timestamp_token(feedback.get('feedback_window_end'))}",
        "previous_model_version": previous_version,
        "proposed_model_version": proposed_version,
        "update_timestamp": version_record["created_at"],
        "feedback_window_start": feedback["feedback_window_start"],
        "feedback_window_end": feedback["feedback_window_end"],
        "false_alert_rate": feedback["false_alert_rate"],
        "ignored_alert_rate": feedback["ignored_alert_rate"],
        "delayed_alert_rate": feedback["delayed_alert_rate"],
        "useful_alert_rate": feedback["useful_alert_rate"],
        "average_drift_score": feedback["average_drift_score"],
        "severe_drift_count": feedback["severe_drift_count"],
        "current_risk_threshold": threshold_update["current_risk_threshold"],
        "proposed_risk_threshold": threshold_update["proposed_risk_threshold"],
        "threshold_change": threshold_update["threshold_change"],
        "update_reason": generate_update_reason(feedback, threshold_update),
        "expected_effect": _expected_effect(comparison, recommendation),
        "deployment_recommendation": recommendation,
        "human_review_required": human_review_required,
    }
    results_df = pd.DataFrame([result_record], columns=REQUIRED_SIMULATION_COLUMNS)

    results_saved_path = _save_results(results_df, results_path)
    registry = {
        "registry_name": "clinical_alert_reliability_simulated_registry",
        "registry_version": "1.0",
        "last_updated": version_record["created_at"],
        "simulation_only": True,
        "deployment_status": "not_deployed",
        "versions": [version_record],
        "note": "This registry stores simulated metadata only and does not replace model files.",
    }
    registry_saved_path = save_model_version_registry(registry, registry_path)
    threshold_summary = {
        "feedback_summary": feedback,
        "threshold_update": threshold_update,
        "old_vs_proposed_configuration": comparison,
        "deployment_recommendation": recommendation,
        "human_review_required": human_review_required,
        "simulation_note": "No model was retrained, deployed, or overwritten.",
    }
    summary_saved_path = save_threshold_update_summary(
        threshold_summary,
        threshold_summary_path,
    )

    results_df.attrs["feedback"] = feedback
    results_df.attrs["threshold_update"] = threshold_update
    results_df.attrs["registry"] = registry
    results_df.attrs["threshold_summary"] = threshold_summary
    results_df.attrs["results_path"] = str(results_saved_path)
    results_df.attrs["registry_path"] = str(registry_saved_path)
    results_df.attrs["threshold_summary_path"] = str(summary_saved_path)
    return results_df


class ModelRegistry:
    """Small compatibility wrapper around the Step 14 simulation workflow."""

    def register(self) -> pd.DataFrame:
        """Create a simulated registry entry; no model artifact is deployed."""
        return run_model_update_simulation()


def _load_csv(path: str | Path, label: str) -> pd.DataFrame:
    input_path = _resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{label} file not found: {input_path}")
    return pd.read_csv(input_path)


def _feedback_window(dataframes: list[pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for df in dataframes:
        for column in ["timestamp", "window_start", "window_end"]:
            if column in df.columns:
                values = pd.to_datetime(df[column], errors="coerce").dropna()
                if not values.empty:
                    starts.append(values.min())
                    ends.append(values.max())

    if not starts or not ends:
        now = pd.Timestamp.utcnow().floor("s").tz_localize(None)
        return now, now
    return min(starts), max(ends)


def _update_timestamp(feedback: dict[str, Any]) -> str:
    timestamp = pd.to_datetime(feedback.get("feedback_window_end"), errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.utcnow().floor("s").tz_localize(None)
    return str(timestamp)


def _human_review_required(
    feedback: dict[str, Any],
    threshold_update: dict[str, Any],
    recommendation: str,
) -> bool:
    return bool(
        recommendation == "retraining_review_recommended"
        or int(feedback.get("severe_drift_count", 0)) > 0
        or abs(float(threshold_update.get("threshold_change", 0.0))) >= 0.05
        or int(feedback.get("unstable_reliability_windows", 0)) > 0
    )


def _expected_effect(comparison: dict[str, Any], recommendation: str) -> str:
    if recommendation == "retraining_review_recommended":
        return "No automatic threshold deployment; review drift and calibration before any model update."
    return (
        "Expected false-alert burden: "
        f"{comparison['expected_false_alert_burden']}; expected sensitivity: "
        f"{comparison['expected_sensitivity']}; confidence: {comparison['comparison_confidence']}."
    )


def _next_patch_version(previous_version: str) -> str:
    prefix = previous_version.rsplit(".", maxsplit=1)[0]
    suffix = previous_version.rsplit(".", maxsplit=1)[-1]
    try:
        patch = int(suffix) + 1
    except ValueError:
        return f"{previous_version}.simulated_update"
    return f"{prefix}.{patch}-simulated"


def _safe_timestamp_token(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.utcnow().floor("s").tz_localize(None)
    return timestamp.strftime("%Y%m%d%H%M%S")


def _save_results(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _clamp_threshold(value: float) -> float:
    return _round_rate(min(max(float(value), MIN_THRESHOLD), MAX_THRESHOLD))


def _round_rate(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value), 4)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _project_root() / candidate


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    results = run_model_update_simulation()
    feedback = results.attrs["feedback"]
    threshold_update = results.attrs["threshold_update"]
    row = results.iloc[0]

    print("Feedback summary:")
    print(f"  false alert rate: {feedback['false_alert_rate']:.4f}")
    print(f"  ignored alert rate: {feedback['ignored_alert_rate']:.4f}")
    print(f"  delayed alert rate: {feedback['delayed_alert_rate']:.4f}")
    print(f"  useful alert rate: {feedback['useful_alert_rate']:.4f}")
    print(f"  average drift score: {feedback['average_drift_score']:.4f}")
    print(f"  severe drift count: {feedback['severe_drift_count']}")
    print(f"Current threshold: {threshold_update['current_risk_threshold']:.4f}")
    print(f"Proposed threshold: {threshold_update['proposed_risk_threshold']:.4f}")
    print(f"Threshold change: {threshold_update['threshold_change']:.4f}")
    print(f"Deployment recommendation: {row['deployment_recommendation']}")
    print(f"Human review required: {bool(row['human_review_required'])}")
    print(f"\nSaved simulation results to {results.attrs['results_path']}")
    print(f"Saved model registry to {results.attrs['registry_path']}")
    print(f"Saved threshold summary to {results.attrs['threshold_summary_path']}")
