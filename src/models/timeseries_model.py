"""Rule-based time-series risk logic for simulated monitoring data.

Step 6 looks for sustained deterioration patterns over time instead of reacting
to a single abnormal reading. This module is a simulated research prototype, not
a clinical diagnosis or patient-monitoring tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/processed/processed_data.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/timeseries_risk_scored.csv")

VITAL_COLUMNS = [
    "heart_rate",
    "oxygen_saturation",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
]

REQUIRED_COLUMNS = ["patient_id", "timestamp", *VITAL_COLUMNS]

LEAKAGE_COLUMNS = {
    "patient_condition_label",
    "deterioration_event",
    "patient_outcome_after_alert",
    "outcome_timestamp",
    "outcome_severity_change",
    "target_label",
    "future_deterioration_label",
}

RISK_LEVELS = {"normal", "low", "medium", "high"}

# Engineering thresholds for simulated data only. They are not clinical rules.
FALLBACK_VALUES = {
    "heart_rate": 75.0,
    "oxygen_saturation": 97.0,
    "systolic_bp": 120.0,
    "diastolic_bp": 76.0,
    "respiratory_rate": 16.0,
    "temperature": 36.8,
}


def load_processed_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the processed monitoring dataset for Step 6 scoring."""
    data_path = _resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {data_path}")
    return pd.read_csv(data_path)


def ensure_time_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records by patient and timestamp before patient-local calculations."""
    _validate_required_columns(df)
    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    ordered = ordered.sort_values(["patient_id", "timestamp"], kind="mergesort")
    ordered = ordered.reset_index(drop=True)

    for column in VITAL_COLUMNS:
        ordered[column] = _current_and_past_fill(
            ordered,
            column=column,
            fallback=FALLBACK_VALUES[column],
        )
    return ordered


def calculate_sustained_abnormality(
    df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Add patient-specific rolling abnormality signals using current/past rows."""
    if window < 2:
        raise ValueError("window must be at least 2 to represent sustained patterns.")

    scored = ensure_time_order(df)
    grouped = scored.groupby("patient_id", group_keys=False)

    scored["heart_rate_recent_mean"] = _rolling_mean(grouped, "heart_rate", window)
    scored["oxygen_saturation_recent_mean"] = _rolling_mean(
        grouped,
        "oxygen_saturation",
        window,
    )
    scored["respiratory_rate_recent_mean"] = _rolling_mean(
        grouped,
        "respiratory_rate",
        window,
    )
    scored["systolic_bp_recent_std"] = _rolling_std(grouped, "systolic_bp", window)
    scored["diastolic_bp_recent_std"] = _rolling_std(grouped, "diastolic_bp", window)

    scored["heart_rate_abnormal_recent_count"] = _rolling_count(
        grouped,
        scored["heart_rate"] >= 105.0,
        window,
    )
    scored["oxygen_low_recent_count"] = _rolling_count(
        grouped,
        scored["oxygen_saturation"] <= 94.0,
        window,
    )
    scored["respiratory_high_recent_count"] = _rolling_count(
        grouped,
        scored["respiratory_rate"] >= 22.0,
        window,
    )
    bp_outside_expected = (scored["systolic_bp"] <= 95.0) | (scored["systolic_bp"] >= 160.0)
    scored["bp_abnormal_recent_count"] = _rolling_count(grouped, bp_outside_expected, window)

    min_sustained_count = max(2, int(np.ceil(window * 0.60)))
    scored["sustained_high_heart_rate"] = (
        scored["heart_rate_abnormal_recent_count"] >= min_sustained_count
    )
    scored["sustained_oxygen_saturation_drop"] = (
        scored["oxygen_low_recent_count"] >= min_sustained_count
    )
    scored["sustained_high_respiratory_rate"] = (
        scored["respiratory_high_recent_count"] >= min_sustained_count
    )
    scored["blood_pressure_instability"] = (
        (scored["bp_abnormal_recent_count"] >= min_sustained_count)
        | (scored["systolic_bp_recent_std"] >= 12.0)
        | (scored["diastolic_bp_recent_std"] >= 8.0)
    )

    scored["sustained_abnormal_vital_count"] = (
        scored[
            [
                "sustained_high_heart_rate",
                "sustained_oxygen_saturation_drop",
                "sustained_high_respiratory_rate",
                "blood_pressure_instability",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )

    scored["recent_instability_mean"] = _optional_rolling_mean(
        grouped,
        scored,
        column="instability_score",
        window=window,
    )
    scored["recent_abnormal_value_mean"] = _optional_rolling_mean(
        grouped,
        scored,
        column="abnormal_value_count",
        window=window,
    )
    return scored


def calculate_deterioration_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Add current-vs-past deterioration trajectory signals per patient."""
    scored = ensure_time_order(df)
    grouped = scored.groupby("patient_id", group_keys=False)

    scored["heart_rate_change_3"] = _change_from_lag(grouped, scored, "heart_rate", lag=3)
    scored["oxygen_saturation_change_3"] = _change_from_lag(
        grouped,
        scored,
        "oxygen_saturation",
        lag=3,
    )
    scored["respiratory_rate_change_3"] = _change_from_lag(
        grouped,
        scored,
        "respiratory_rate",
        lag=3,
    )
    scored["systolic_bp_change_3"] = _change_from_lag(grouped, scored, "systolic_bp", lag=3)

    scored["worsening_heart_rate_trend"] = scored["heart_rate_change_3"] >= 8.0
    scored["worsening_oxygen_trend"] = scored["oxygen_saturation_change_3"] <= -2.0
    scored["worsening_respiratory_trend"] = scored["respiratory_rate_change_3"] >= 3.0
    scored["worsening_bp_trend"] = scored["systolic_bp_change_3"] <= -8.0

    scored["recent_worsening_trend_count"] = (
        scored[
            [
                "worsening_heart_rate_trend",
                "worsening_oxygen_trend",
                "worsening_respiratory_trend",
                "worsening_bp_trend",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    return scored


def calculate_time_series_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine sustained and trajectory signals into an explainable risk score."""
    scored = df.copy()
    if "sustained_abnormal_vital_count" not in scored.columns:
        scored = calculate_sustained_abnormality(scored)
    if "recent_worsening_trend_count" not in scored.columns:
        scored = calculate_deterioration_trajectory(scored)

    sustained_component = np.minimum(scored["sustained_abnormal_vital_count"] / 3.0, 1.0)
    trajectory_component = np.minimum(scored["recent_worsening_trend_count"] / 3.0, 1.0)
    instability_component = np.minimum(scored["recent_instability_mean"].clip(lower=0.0) / 1.2, 1.0)
    abnormal_count_component = np.minimum(scored["recent_abnormal_value_mean"].clip(lower=0.0) / 3.0, 1.0)
    oxygen_component = scored["sustained_oxygen_saturation_drop"].astype(float)
    respiratory_component = scored["sustained_high_respiratory_rate"].astype(float)
    bp_component = scored["blood_pressure_instability"].astype(float)

    score = (
        0.24 * sustained_component
        + 0.20 * trajectory_component
        + 0.18 * oxygen_component
        + 0.12 * respiratory_component
        + 0.10 * bp_component
        + 0.10 * instability_component
        + 0.06 * abnormal_count_component
    )
    scored["time_series_risk_score"] = np.clip(score, 0.0, 1.0).round(4)
    scored["time_series_risk_level"] = scored["time_series_risk_score"].apply(
        assign_time_series_risk_level
    )
    scored["time_series_risk_reason"] = scored.apply(
        generate_time_series_risk_reason,
        axis=1,
    )
    return scored


def assign_time_series_risk_level(score: float) -> str:
    """Map a numeric time-series score into a simple risk level."""
    numeric_score = float(score)
    if numeric_score < 0.15:
        return "normal"
    if numeric_score < 0.35:
        return "low"
    if numeric_score < 0.60:
        return "medium"
    return "high"


def generate_time_series_risk_reason(row: pd.Series | dict[str, Any]) -> str:
    """Return the most explainable sustained risk reason for one row."""
    score = float(_row_value(row, "time_series_risk_score", 0.0))
    if score < 0.15:
        return "No sustained deterioration pattern"

    sustained_count = int(_row_value(row, "sustained_abnormal_vital_count", 0))
    if sustained_count >= 2:
        return "Multiple sustained abnormal vital signs"
    if bool(_row_value(row, "sustained_oxygen_saturation_drop", False)) or float(
        _row_value(row, "oxygen_saturation_change_3", 0.0)
    ) <= -2.0:
        return "Sustained oxygen saturation decline"
    if bool(_row_value(row, "sustained_high_respiratory_rate", False)) or float(
        _row_value(row, "respiratory_rate_change_3", 0.0)
    ) >= 3.0:
        return "Increasing respiratory rate trend"
    if float(_row_value(row, "recent_instability_mean", 0.0)) >= 0.75:
        return "High instability score over recent window"
    if bool(_row_value(row, "blood_pressure_instability", False)):
        return "Blood pressure instability over recent window"
    if int(_row_value(row, "recent_worsening_trend_count", 0)) >= 1:
        return "Recent worsening trend across vital signs"
    return "No sustained deterioration pattern"


def run_timeseries_risk_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """Run the Step 6 time-series risk pipeline and save scored output."""
    df = load_processed_data(data_path)
    scored = ensure_time_order(df)
    scored = calculate_sustained_abnormality(scored)
    scored = calculate_deterioration_trajectory(scored)
    scored = calculate_time_series_risk_score(scored)

    output_path = _resolve_project_path(DEFAULT_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    scored.attrs["output_path"] = str(output_path)
    return scored


def calculate_time_series_risk() -> pd.DataFrame:
    """Backward-compatible wrapper for the Step 6 risk pipeline."""
    return run_timeseries_risk_pipeline()


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if processed data lacks required columns."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required time-series columns: {missing}")


def _current_and_past_fill(
    df: pd.DataFrame,
    column: str,
    fallback: float,
) -> pd.Series:
    """Coerce a signal and fill missing values without looking ahead."""
    numeric = pd.to_numeric(df[column], errors="coerce")
    filled = numeric.groupby(df["patient_id"], group_keys=False).ffill()
    return filled.fillna(fallback).astype(float)


def _rolling_mean(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    column: str,
    window: int,
) -> pd.Series:
    """Patient-local rolling mean using current and past rows."""
    return grouped[column].transform(lambda series: series.rolling(window, min_periods=1).mean())


def _rolling_std(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    column: str,
    window: int,
) -> pd.Series:
    """Patient-local rolling standard deviation using current and past rows."""
    return grouped[column].transform(
        lambda series: series.rolling(window, min_periods=2).std().fillna(0.0)
    )


def _rolling_count(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    condition: pd.Series,
    window: int,
) -> pd.Series:
    """Patient-local rolling count for a boolean condition."""
    condition_frame = pd.DataFrame(
        {
            "patient_id": grouped.obj["patient_id"],
            "_condition": condition.astype(int),
        },
        index=grouped.obj.index,
    )
    return condition_frame.groupby("patient_id", group_keys=False)["_condition"].transform(
        lambda series: series.rolling(window, min_periods=1).sum()
    )


def _optional_rolling_mean(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    df: pd.DataFrame,
    column: str,
    window: int,
) -> pd.Series:
    """Return a patient-local rolling mean for an optional numeric column."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    numeric = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    working = pd.DataFrame({"patient_id": df["patient_id"], column: numeric}, index=df.index)
    return working.groupby("patient_id", group_keys=False)[column].transform(
        lambda series: series.rolling(window, min_periods=1).mean()
    )


def _change_from_lag(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    df: pd.DataFrame,
    column: str,
    lag: int,
) -> pd.Series:
    """Calculate current-minus-past change without using future rows."""
    previous = grouped[column].shift(lag)
    return (df[column] - previous).fillna(0.0)


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    """Read a row value with a fallback."""
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
    scored_data = run_timeseries_risk_pipeline()
    output_path = scored_data.attrs.get("output_path", str(_resolve_project_path(DEFAULT_OUTPUT_PATH)))
    medium_high = scored_data[
        scored_data["time_series_risk_level"].isin(["medium", "high"])
    ]
    display_columns = [
        "patient_id",
        "timestamp",
        "time_series_risk_score",
        "time_series_risk_level",
        "time_series_risk_reason",
    ]

    print("Step 6 time-series risk scoring complete")
    print(f"Rows scored: {len(scored_data)}")
    print("Risk level distribution:")
    print(scored_data["time_series_risk_level"].value_counts().to_dict())
    print(f"Saved scored output to: {output_path}")
    print("First few medium/high risk rows:")
    print(medium_high[display_columns].head())
