"""Transparent drift detection for simulated alert reliability experiments.

Step 13 monitors whether patient data, alert behavior, simulated workflow
responses, and reliability scores are changing over time. The logic is a
research/engineering prototype for simulated data only. It is not clinical
validation, automatic retraining, online learning, or a medical-device monitor.
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


DEFAULT_PROCESSED_PATH = Path("data/processed/processed_data.csv")
DEFAULT_ALERTS_PATH = Path("data/processed/generated_alerts.csv")
DEFAULT_RESPONSE_PATH = Path("data/processed/clinician_response_logs.csv")
DEFAULT_RELIABILITY_PATH = Path("data/processed/reliability_monitoring_results.csv")
DEFAULT_RESULTS_PATH = Path("data/processed/drift_detection_results.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/drift_summary.json")
DEFAULT_WINDOW_MINUTES = 180

VITAL_FEATURES = [
    "heart_rate",
    "oxygen_saturation",
    "respiratory_rate",
    "instability_score",
]

REQUIRED_PROCESSED_COLUMNS = ["patient_id", "timestamp"]
REQUIRED_ALERT_COLUMNS = [
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "alert_type",
    "risk_score",
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
    "total_alerts",
    "ignored_alert_rate",
    "delayed_alert_rate",
    "reliability_score",
]

REQUIRED_OUTPUT_COLUMNS = [
    "drift_window_id",
    "window_start",
    "window_end",
    "drift_type",
    "monitored_feature",
    "baseline_value",
    "current_value",
    "drift_score",
    "drift_status",
    "drift_warning",
    "recalibration_recommendation",
    "requires_review",
]

VALID_DRIFT_TYPES = {
    "data_drift",
    "alert_distribution_drift",
    "response_behavior_drift",
    "reliability_drift",
}

VALID_DRIFT_STATUSES = {
    "stable",
    "mild_shift",
    "moderate_shift",
    "severe_shift",
}

VALID_RECALIBRATION_RECOMMENDATIONS = {
    "no_action_needed",
    "monitor_next_window",
    "review_thresholds",
    "review_alert_logic",
    "review_workflow_behavior",
    "retraining_review_recommended",
}


def load_processed_data(path: str | Path = DEFAULT_PROCESSED_PATH) -> pd.DataFrame:
    """Load Step 3/4 processed monitoring data."""
    processed_path = _resolve_project_path(path)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {processed_path}")
    df = pd.read_csv(processed_path)
    _validate_columns(df, REQUIRED_PROCESSED_COLUMNS, "processed data")
    return df


def load_generated_alerts(path: str | Path = DEFAULT_ALERTS_PATH) -> pd.DataFrame:
    """Load generated alerts from Step 7."""
    alerts_path = _resolve_project_path(path)
    if not alerts_path.exists():
        raise FileNotFoundError(f"Generated alerts file not found: {alerts_path}")
    df = pd.read_csv(alerts_path)
    _validate_columns(df, REQUIRED_ALERT_COLUMNS, "generated alerts")
    return df


def load_response_logs(path: str | Path = DEFAULT_RESPONSE_PATH) -> pd.DataFrame:
    """Load simulated clinician response logs from Step 11."""
    response_path = _resolve_project_path(path)
    if not response_path.exists():
        raise FileNotFoundError(f"Response logs file not found: {response_path}")
    df = pd.read_csv(response_path)
    _validate_columns(df, REQUIRED_RESPONSE_COLUMNS, "response logs")
    return df


def load_reliability_results(path: str | Path = DEFAULT_RELIABILITY_PATH) -> pd.DataFrame:
    """Load reliability monitoring results from Step 12."""
    reliability_path = _resolve_project_path(path)
    if not reliability_path.exists():
        raise FileNotFoundError(f"Reliability results file not found: {reliability_path}")
    df = pd.read_csv(reliability_path)
    _validate_columns(df, REQUIRED_RELIABILITY_COLUMNS, "reliability results")
    return df


def create_drift_windows(
    df: pd.DataFrame,
    timestamp_column: str,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> list[dict[str, Any]]:
    """Create ordered fixed-width drift windows from a timestamp column."""
    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column for drift windows: {timestamp_column}")
    if window_minutes < 1:
        raise ValueError("window_minutes must be at least 1.")

    if df.empty:
        return []

    windowed = df.copy()
    windowed[timestamp_column] = pd.to_datetime(windowed[timestamp_column], errors="coerce")
    windowed = windowed.dropna(subset=[timestamp_column])
    if windowed.empty:
        return []

    windowed = windowed.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)
    start = windowed[timestamp_column].min()
    delta = pd.Timedelta(minutes=window_minutes)
    window_index = np.floor((windowed[timestamp_column] - start) / delta).astype(int)
    windowed["_drift_window_index"] = window_index

    windows: list[dict[str, Any]] = []
    for index, group in windowed.groupby("_drift_window_index", sort=True):
        window_start = start + int(index) * delta
        window_end = window_start + delta
        windows.append(
            {
                "drift_window_id": f"DRIFT-WINDOW-{int(index) + 1:03d}",
                "window_start": window_start,
                "window_end": window_end,
                "data": group.drop(columns=["_drift_window_index"]).reset_index(drop=True),
            }
        )
    return windows


def calculate_population_stability_index(
    expected: Any,
    actual: Any,
    bins: int = 10,
) -> float:
    """Calculate PSI for numeric distributions using expected-data bins."""
    expected_values = _numeric_array(expected)
    actual_values = _numeric_array(actual)
    if expected_values.size == 0 or actual_values.size == 0:
        return 0.0

    unique_expected = np.unique(expected_values)
    if unique_expected.size < 2:
        baseline = float(unique_expected[0])
        if np.allclose(actual_values, baseline):
            return 0.0
        denominator = max(abs(baseline), 1.0)
        return _round_score(min(float(np.mean(np.abs(actual_values - baseline))) / denominator, 1.0))

    bins = max(2, int(bins))
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    breakpoints = np.unique(np.quantile(expected_values, quantiles))
    if breakpoints.size < 3:
        data_min = min(float(expected_values.min()), float(actual_values.min()))
        data_max = max(float(expected_values.max()), float(actual_values.max()))
        if np.isclose(data_min, data_max):
            return 0.0
        breakpoints = np.linspace(data_min, data_max, bins + 1)

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    expected_counts, _ = np.histogram(expected_values, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_values, bins=breakpoints)

    epsilon = 1e-6
    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), epsilon)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), epsilon)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return _round_score(max(float(psi), 0.0))


def calculate_distribution_shift(expected_series: Any, actual_series: Any) -> float:
    """Calculate categorical distribution shift using total variation distance."""
    expected_counts = pd.Series(expected_series).dropna().astype(str).value_counts(normalize=True)
    actual_counts = pd.Series(actual_series).dropna().astype(str).value_counts(normalize=True)
    if expected_counts.empty or actual_counts.empty:
        return 0.0

    categories = sorted(set(expected_counts.index).union(set(actual_counts.index)))
    difference = sum(abs(float(actual_counts.get(category, 0.0)) - float(expected_counts.get(category, 0.0))) for category in categories)
    return _round_score(0.5 * difference)


def detect_vital_sign_drift(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Detect patient vital-sign distribution drift with rolling PSI checks."""
    _validate_columns(processed_df, REQUIRED_PROCESSED_COLUMNS, "processed data")
    windows = create_drift_windows(processed_df, "timestamp")
    features = [feature for feature in VITAL_FEATURES if feature in processed_df.columns]
    records: list[dict[str, Any]] = []

    for index in range(1, len(windows)):
        baseline_df = _combine_prior_windows(windows, index)
        current_df = windows[index]["data"]
        for feature in features:
            score = calculate_population_stability_index(baseline_df[feature], current_df[feature])
            records.append(
                _build_drift_record(
                    windows[index],
                    drift_type="data_drift",
                    monitored_feature=feature,
                    baseline_value=_series_mean(baseline_df[feature]),
                    current_value=_series_mean(current_df[feature]),
                    drift_score=score,
                )
            )

    return _results_dataframe(records)


def detect_alert_distribution_drift(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Detect alert severity, type, risk, and volume drift."""
    _validate_columns(alerts_df, REQUIRED_ALERT_COLUMNS, "generated alerts")
    windows = create_drift_windows(alerts_df, "timestamp")
    records: list[dict[str, Any]] = []

    for index in range(1, len(windows)):
        baseline_df = _combine_prior_windows(windows, index)
        current_df = windows[index]["data"]
        records.extend(
            [
                _build_drift_record(
                    windows[index],
                    drift_type="alert_distribution_drift",
                    monitored_feature="severity_distribution",
                    baseline_value=_category_distribution(baseline_df["severity"]),
                    current_value=_category_distribution(current_df["severity"]),
                    drift_score=calculate_distribution_shift(
                        baseline_df["severity"],
                        current_df["severity"],
                    ),
                ),
                _build_drift_record(
                    windows[index],
                    drift_type="alert_distribution_drift",
                    monitored_feature="alert_type_distribution",
                    baseline_value=_category_distribution(baseline_df["alert_type"]),
                    current_value=_category_distribution(current_df["alert_type"]),
                    drift_score=calculate_distribution_shift(
                        baseline_df["alert_type"],
                        current_df["alert_type"],
                    ),
                ),
                _build_drift_record(
                    windows[index],
                    drift_type="alert_distribution_drift",
                    monitored_feature="alert_volume",
                    baseline_value=_mean_prior_window_count(windows, index),
                    current_value=float(len(current_df)),
                    drift_score=_relative_shift(
                        _mean_prior_window_count(windows, index),
                        float(len(current_df)),
                    ),
                ),
            ]
        )

        if "risk_score" in alerts_df.columns:
            records.append(
                _build_drift_record(
                    windows[index],
                    drift_type="alert_distribution_drift",
                    monitored_feature="alert_risk_score",
                    baseline_value=_series_mean(baseline_df["risk_score"]),
                    current_value=_series_mean(current_df["risk_score"]),
                    drift_score=calculate_population_stability_index(
                        baseline_df["risk_score"],
                        current_df["risk_score"],
                    ),
                )
            )

    return _results_dataframe(records)


def detect_response_behavior_drift(response_df: pd.DataFrame) -> pd.DataFrame:
    """Detect simulated clinician response behavior and response-time drift."""
    _validate_columns(response_df, REQUIRED_RESPONSE_COLUMNS, "response logs")
    windows = create_drift_windows(response_df, "timestamp")
    records: list[dict[str, Any]] = []

    for index in range(1, len(windows)):
        baseline_df = _combine_prior_windows(windows, index)
        current_df = windows[index]["data"]
        baseline_ignored = _response_rate(baseline_df, "ignored")
        current_ignored = _response_rate(current_df, "ignored")
        baseline_delayed = _response_rate(baseline_df, "delayed")
        current_delayed = _response_rate(current_df, "delayed")
        baseline_response_time = _series_mean(baseline_df["response_time_minutes"])
        current_response_time = _series_mean(current_df["response_time_minutes"])

        records.extend(
            [
                _build_drift_record(
                    windows[index],
                    drift_type="response_behavior_drift",
                    monitored_feature="simulated_response_distribution",
                    baseline_value=_category_distribution(baseline_df["simulated_response"]),
                    current_value=_category_distribution(current_df["simulated_response"]),
                    drift_score=calculate_distribution_shift(
                        baseline_df["simulated_response"],
                        current_df["simulated_response"],
                    ),
                ),
                _build_drift_record(
                    windows[index],
                    drift_type="response_behavior_drift",
                    monitored_feature="ignored_alert_rate",
                    baseline_value=baseline_ignored,
                    current_value=current_ignored,
                    drift_score=abs(current_ignored - baseline_ignored),
                ),
                _build_drift_record(
                    windows[index],
                    drift_type="response_behavior_drift",
                    monitored_feature="delayed_alert_rate",
                    baseline_value=baseline_delayed,
                    current_value=current_delayed,
                    drift_score=abs(current_delayed - baseline_delayed),
                ),
                _build_drift_record(
                    windows[index],
                    drift_type="response_behavior_drift",
                    monitored_feature="average_response_time_minutes",
                    baseline_value=baseline_response_time,
                    current_value=current_response_time,
                    drift_score=_relative_shift(baseline_response_time, current_response_time),
                ),
            ]
        )

    return _results_dataframe(records)


def detect_reliability_score_drift(reliability_df: pd.DataFrame) -> pd.DataFrame:
    """Detect reliability-score and reliability-window alert-volume drift."""
    _validate_columns(reliability_df, REQUIRED_RELIABILITY_COLUMNS, "reliability results")
    windows = create_drift_windows(reliability_df, "window_start")
    records: list[dict[str, Any]] = []

    for index in range(1, len(windows)):
        baseline_df = _combine_prior_windows(windows, index)
        current_df = windows[index]["data"]
        baseline_score = _series_mean(baseline_df["reliability_score"])
        current_score = _series_mean(current_df["reliability_score"])
        baseline_alerts = _series_mean(baseline_df["total_alerts"])
        current_alerts = _series_mean(current_df["total_alerts"])

        records.extend(
            [
                _build_drift_record(
                    windows[index],
                    drift_type="reliability_drift",
                    monitored_feature="reliability_score",
                    baseline_value=baseline_score,
                    current_value=current_score,
                    drift_score=abs(current_score - baseline_score),
                ),
                _build_drift_record(
                    windows[index],
                    drift_type="reliability_drift",
                    monitored_feature="reliability_alert_volume",
                    baseline_value=baseline_alerts,
                    current_value=current_alerts,
                    drift_score=_relative_shift(baseline_alerts, current_alerts),
                ),
            ]
        )

    return _results_dataframe(records)


def assign_drift_status(score: float) -> str:
    """Assign drift status using documented transparent thresholds."""
    score = max(float(score), 0.0)
    if score < 0.10:
        return "stable"
    if score < 0.20:
        return "mild_shift"
    if score < 0.35:
        return "moderate_shift"
    return "severe_shift"


def generate_drift_warning(drift_type: str, status: str, feature: str) -> str:
    """Generate human-readable drift warning text."""
    readable_type = drift_type.replace("_", " ")
    readable_feature = feature.replace("_", " ")
    if status == "stable":
        return f"Stable: no meaningful {readable_feature} shift detected in {readable_type}."
    if status == "mild_shift":
        return f"Mild shift: {readable_feature} changed slightly in {readable_type}; monitor the next window."
    if status == "moderate_shift":
        return f"Moderate shift: {readable_feature} changed enough to justify review of simulated thresholds or workflow assumptions."
    return f"Severe shift: {readable_feature} changed substantially and should be reviewed before relying on later simulated outputs."


def generate_recalibration_recommendation(drift_type: str, status: str) -> str:
    """Map drift status and type to a transparent review recommendation."""
    if status == "stable":
        return "no_action_needed"
    if status == "mild_shift":
        return "monitor_next_window"
    if drift_type == "data_drift":
        return "review_thresholds" if status == "moderate_shift" else "retraining_review_recommended"
    if drift_type == "alert_distribution_drift":
        return "review_alert_logic"
    if drift_type == "response_behavior_drift":
        return "review_workflow_behavior"
    if drift_type == "reliability_drift":
        return "review_thresholds" if status == "moderate_shift" else "retraining_review_recommended"
    return "monitor_next_window"


def combine_drift_results(
    vital_df: pd.DataFrame,
    alert_df: pd.DataFrame,
    response_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all Step 13 drift checks and apply review flags."""
    pieces = [vital_df, alert_df, response_df, reliability_df]
    non_empty = [piece for piece in pieces if piece is not None and not piece.empty]
    if not non_empty:
        return _results_dataframe([])

    combined = pd.concat(non_empty, ignore_index=True)
    combined = combined.reindex(columns=REQUIRED_OUTPUT_COLUMNS)
    combined["window_start"] = pd.to_datetime(combined["window_start"], errors="coerce")
    combined = combined.sort_values(
        ["window_start", "drift_type", "monitored_feature"],
        kind="mergesort",
    ).reset_index(drop=True)
    return _apply_review_rules(combined)


def save_drift_results(df: pd.DataFrame, path: str | Path) -> Path:
    """Save detailed drift results to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_drift_summary(summary: dict[str, Any], path: str | Path) -> Path:
    """Save JSON summary of drift monitoring results."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def run_drift_detection_pipeline(
    processed_path: str | Path = DEFAULT_PROCESSED_PATH,
    alerts_path: str | Path = DEFAULT_ALERTS_PATH,
    response_path: str | Path = DEFAULT_RESPONSE_PATH,
    reliability_path: str | Path = DEFAULT_RELIABILITY_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> pd.DataFrame:
    """Run the full Step 13 drift-detection workflow."""
    processed_df = load_processed_data(processed_path)
    alerts_df = load_generated_alerts(alerts_path)
    response_df = load_response_logs(response_path)
    reliability_df = load_reliability_results(reliability_path)

    vital_results = detect_vital_sign_drift(processed_df)
    alert_results = detect_alert_distribution_drift(alerts_df)
    response_results = detect_response_behavior_drift(response_df)
    reliability_results = detect_reliability_score_drift(reliability_df)
    drift_results = combine_drift_results(
        vital_results,
        alert_results,
        response_results,
        reliability_results,
    )

    results_saved_path = save_drift_results(drift_results, results_path)
    summary = _build_drift_summary(drift_results)
    summary_saved_path = save_drift_summary(summary, summary_path)
    drift_results.attrs["results_path"] = str(results_saved_path)
    drift_results.attrs["summary_path"] = str(summary_saved_path)
    drift_results.attrs["summary"] = summary
    return drift_results


def detect_drift() -> pd.DataFrame:
    """Compatibility wrapper for older placeholder imports."""
    return run_drift_detection_pipeline()


def _build_drift_record(
    window: dict[str, Any],
    drift_type: str,
    monitored_feature: str,
    baseline_value: Any,
    current_value: Any,
    drift_score: float,
) -> dict[str, Any]:
    """Build one normalized drift result row."""
    rounded_score = _round_score(drift_score)
    status = assign_drift_status(rounded_score)
    return {
        "drift_window_id": window["drift_window_id"],
        "window_start": window["window_start"],
        "window_end": window["window_end"],
        "drift_type": drift_type,
        "monitored_feature": monitored_feature,
        "baseline_value": _format_value(baseline_value),
        "current_value": _format_value(current_value),
        "drift_score": rounded_score,
        "drift_status": status,
        "drift_warning": generate_drift_warning(drift_type, status, monitored_feature),
        "recalibration_recommendation": generate_recalibration_recommendation(
            drift_type,
            status,
        ),
        "requires_review": _initial_review_flag(
            drift_type,
            monitored_feature,
            baseline_value,
            current_value,
            status,
        ),
    }


def _build_drift_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build JSON-friendly drift summary metrics."""
    if df.empty:
        return {
            "total_drift_checks": 0,
            "average_drift_score": 0.0,
            "severe_drift_count": 0,
            "monitored_feature_count": 0,
            "drift_type_distribution": {},
            "drift_status_distribution": {},
            "checks_requiring_review": 0,
            "simulation_note": "Simulated drift detection only; not clinical validation.",
        }

    return {
        "total_drift_checks": int(len(df)),
        "average_drift_score": _round_score(df["drift_score"].mean()),
        "maximum_drift_score": _round_score(df["drift_score"].max()),
        "severe_drift_count": int((df["drift_status"] == "severe_shift").sum()),
        "monitored_feature_count": int(df["monitored_feature"].nunique()),
        "drift_type_distribution": _value_counts(df, "drift_type"),
        "drift_status_distribution": _value_counts(df, "drift_status"),
        "checks_requiring_review": int(df["requires_review"].astype(bool).sum()),
        "simulation_note": "Simulated drift detection only; not clinical validation.",
    }


def _apply_review_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply severe/repeated-moderate/reliability-degradation review flags."""
    reviewed = df.copy()
    reviewed["requires_review"] = reviewed["requires_review"].astype(bool)
    reviewed.loc[reviewed["drift_status"] == "severe_shift", "requires_review"] = True

    for _, group in reviewed.groupby(["drift_type", "monitored_feature"], sort=False):
        ordered_index = group.sort_values("window_start", kind="mergesort").index
        moderate_or_worse = reviewed.loc[ordered_index, "drift_status"].isin(
            ["moderate_shift", "severe_shift"]
        )
        repeated = moderate_or_worse & moderate_or_worse.shift(fill_value=False)
        reviewed.loc[ordered_index[repeated.to_numpy()], "requires_review"] = True

    reviewed["requires_review"] = reviewed["requires_review"].astype(bool)
    return reviewed


def _initial_review_flag(
    drift_type: str,
    monitored_feature: str,
    baseline_value: Any,
    current_value: Any,
    status: str,
) -> bool:
    """Flag review-sensitive drift rows before repeated-shift checks."""
    if status == "severe_shift":
        return True
    if drift_type == "reliability_drift" and monitored_feature == "reliability_score":
        baseline_numeric = _safe_float(baseline_value)
        current_numeric = _safe_float(current_value)
        return current_numeric < baseline_numeric - 0.10
    return False


def _results_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a consistently shaped drift-results dataframe."""
    return pd.DataFrame(records, columns=REQUIRED_OUTPUT_COLUMNS)


def _combine_prior_windows(windows: list[dict[str, Any]], index: int) -> pd.DataFrame:
    """Combine all prior windows so current checks only compare against the past."""
    prior_frames = [window["data"] for window in windows[:index]]
    return pd.concat(prior_frames, ignore_index=True)


def _mean_prior_window_count(windows: list[dict[str, Any]], index: int) -> float:
    counts = [len(window["data"]) for window in windows[:index]]
    return _round_score(float(np.mean(counts))) if counts else 0.0


def _response_rate(df: pd.DataFrame, response_value: str) -> float:
    if df.empty:
        return 0.0
    return _round_score((df["simulated_response"].astype(str) == response_value).mean())


def _series_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return _round_score(float(numeric.mean()))


def _relative_shift(baseline_value: Any, current_value: Any) -> float:
    baseline = _safe_float(baseline_value)
    current = _safe_float(current_value)
    denominator = max(abs(baseline), 1.0)
    return _round_score(min(abs(current - baseline) / denominator, 1.0))


def _category_distribution(series: pd.Series) -> dict[str, float]:
    counts = series.dropna().astype(str).value_counts(normalize=True).sort_index()
    return {str(key): _round_score(value) for key, value in counts.items()}


def _numeric_array(values: Any) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return numeric.to_numpy(dtype=float)


def _format_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (np.integer, np.floating)):
        return _round_score(float(value))
    if isinstance(value, float):
        return _round_score(value)
    return value


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round_score(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return round(max(float(value), 0.0), 4)


def _validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in df[column].value_counts().items()}


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _project_root() / candidate


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    results = run_drift_detection_pipeline()
    summary = results.attrs.get("summary", _build_drift_summary(results))

    print(f"Total drift checks: {summary['total_drift_checks']}")
    print(f"Drift status distribution: {summary['drift_status_distribution']}")
    print(f"Average drift score: {summary['average_drift_score']:.4f}")
    print(f"Severe drift count: {summary['severe_drift_count']}")
    print("\nFirst few drift results:")
    display_columns = [
        "drift_window_id",
        "drift_type",
        "monitored_feature",
        "drift_score",
        "drift_status",
        "recalibration_recommendation",
        "requires_review",
    ]
    print(results[display_columns].head().to_string(index=False))
    print(f"\nSaved drift results to {results.attrs['results_path']}")
    print(f"Saved drift summary to {results.attrs['summary_path']}")
