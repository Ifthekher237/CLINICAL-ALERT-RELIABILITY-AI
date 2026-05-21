"""Preprocessing utilities for simulated patient-monitoring data.

The functions in this module clean the Step 2 simulator output without changing
the clinical-prototype boundary of the project. Missing values are imputed only
within each patient's own timeline, noisy readings are retained and flagged, and
time-series train/test splits are made by timestamp to avoid future leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VITAL_COLUMNS = [
    "heart_rate",
    "oxygen_saturation",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
]

NUMERIC_COLUMNS = [*VITAL_COLUMNS, "outcome_severity_change"]
DATETIME_COLUMNS = ["timestamp", "outcome_timestamp"]
BOOLEAN_COLUMNS = ["deterioration_event", "sensor_noise_flag", "missing_data_flag"]
CATEGORICAL_COLUMNS = [
    "patient_id",
    "patient_condition_label",
    "patient_outcome_after_alert",
]

REQUIRED_COLUMNS = [
    "patient_id",
    "timestamp",
    *VITAL_COLUMNS,
    "patient_condition_label",
    "deterioration_event",
    "sensor_noise_flag",
    "missing_data_flag",
    "patient_outcome_after_alert",
    "outcome_timestamp",
    "outcome_severity_change",
]

DEFAULT_RAW_PATH = Path("data/simulated/patient_monitoring.csv")
DEFAULT_PROCESSED_PATH = Path("data/processed/processed_data.csv")

# Broad engineering ranges used only for validation flags. They are not
# clinical thresholds and should not be presented as medical guidance.
VALIDATION_RANGES = {
    "heart_rate": (35.0, 220.0),
    "oxygen_saturation": (50.0, 100.0),
    "systolic_bp": (60.0, 230.0),
    "diastolic_bp": (30.0, 140.0),
    "respiratory_rate": (5.0, 55.0),
    "temperature": (33.0, 42.5),
}

FALLBACK_VALUES = {
    "heart_rate": 75.0,
    "oxygen_saturation": 97.0,
    "systolic_bp": 120.0,
    "diastolic_bp": 76.0,
    "respiratory_rate": 16.0,
    "temperature": 36.8,
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load simulated patient-monitoring data from CSV.

    Args:
        path: CSV file path. Relative paths are resolved from the project root.

    Returns:
        Raw pandas DataFrame with required columns present.
    """
    data_path = _resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    df = pd.read_csv(data_path)
    _validate_required_columns(df)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean simulated monitoring data for feature engineering.

    The imputation strategy is deliberately patient-local: forward fill within
    each patient, backward fill within that same patient for leading gaps, and
    finally neutral fallback values only if an entire patient-vital sequence is
    missing.
    """
    _validate_required_columns(df)

    cleaned = df.copy()
    cleaned = _fix_data_types(cleaned)
    cleaned = _sort_patient_timeline(cleaned)

    # Preserve missingness as a model feature before filling vitals.
    cleaned["missing_value_count_raw"] = cleaned[VITAL_COLUMNS].isna().sum(axis=1).astype(int)
    cleaned["had_missing_vitals_before_imputation"] = (
        cleaned["missing_value_count_raw"] > 0
    )

    cleaned = _add_duplicate_flags(cleaned)
    cleaned = _add_invalid_range_flags(cleaned)
    cleaned = _impute_vitals_within_patient(cleaned)
    cleaned = _add_noise_intensity(cleaned)

    validation_summary = validate_preprocessed_data(cleaned)
    cleaned.attrs["validation_summary"] = validation_summary
    return cleaned.reset_index(drop=True)


def train_test_split_time_series(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time so training rows never use future test rows.

    The cutoff is global by timestamp, which keeps every patient's earlier
    records in train and later records in test when enough timestamps exist.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a timestamp column.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    sorted_df = _sort_patient_timeline(_fix_data_types(df.copy()))
    unique_times = sorted_df["timestamp"].dropna().sort_values().unique()

    if len(unique_times) < 2:
        return sorted_df.reset_index(drop=True), sorted_df.iloc[0:0].copy()

    split_index = int(np.floor(len(unique_times) * (1.0 - test_size)))
    split_index = min(max(split_index, 1), len(unique_times) - 1)
    cutoff_time = unique_times[split_index - 1]

    train_df = sorted_df[sorted_df["timestamp"] <= cutoff_time].copy()
    test_df = sorted_df[sorted_df["timestamp"] > cutoff_time].copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_processed_data(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_PROCESSED_PATH,
) -> Path:
    """Save cleaned or feature-engineered data to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def prepare_modeling_data(
    input_path: str | Path = DEFAULT_RAW_PATH,
    processed_path: str | Path = DEFAULT_PROCESSED_PATH,
    test_size: float = 0.20,
    prediction_horizon_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the Step 3 pipeline and return clean, feature, train, and test data."""
    from src.data.feature_engineering import create_features

    raw_df = load_data(input_path)
    clean_df = preprocess_data(raw_df)
    feature_df = create_features(
        clean_df,
        prediction_horizon_minutes=prediction_horizon_minutes,
    )
    save_processed_data(feature_df, processed_path)
    train_df, test_df = train_test_split_time_series(feature_df, test_size=test_size)
    return clean_df, feature_df, train_df, test_df


def preprocess_monitoring_data(
    input_path: str | Path = DEFAULT_RAW_PATH,
    processed_path: str | Path = DEFAULT_PROCESSED_PATH,
    test_size: float = 0.20,
    prediction_horizon_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible Step 3 pipeline entry point."""
    return prepare_modeling_data(
        input_path=input_path,
        processed_path=processed_path,
        test_size=test_size,
        prediction_horizon_minutes=prediction_horizon_minutes,
    )


def validate_preprocessed_data(df: pd.DataFrame) -> dict[str, Any]:
    """Return a lightweight validation summary for cleaned data."""
    vital_null_count = int(df[VITAL_COLUMNS].isna().sum().sum())
    duplicate_patient_timestamp_count = int(
        df.duplicated(subset=["patient_id", "timestamp"]).sum()
    )
    invalid_range_count = int(df.get("invalid_vital_range_flag", pd.Series(dtype=bool)).sum())
    return {
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()) if "patient_id" in df else 0,
        "vital_null_count_after_imputation": vital_null_count,
        "duplicate_patient_timestamp_count": duplicate_patient_timestamp_count,
        "invalid_range_row_count": invalid_range_count,
    }


def _fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent types expected by downstream feature engineering."""
    for column in DATETIME_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    for column in BOOLEAN_COLUMNS:
        if column in df.columns:
            df[column] = _coerce_bool(df[column])

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("unknown").astype(str)

    return df


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean encodings to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"}).astype(bool)


def _sort_patient_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records by patient and timestamp for patient-wise time operations."""
    return df.sort_values(["patient_id", "timestamp"], kind="mergesort").reset_index(drop=True)


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if Step 2 schema columns are missing."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _add_duplicate_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag duplicate rows and duplicate patient-timestamp records."""
    df["duplicate_record_flag"] = df.duplicated(keep=False)
    df["duplicate_patient_timestamp_flag"] = df.duplicated(
        subset=["patient_id", "timestamp"],
        keep=False,
    )
    return df


def _add_invalid_range_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with vitals outside broad engineering validation ranges."""
    invalid_counts = pd.Series(0, index=df.index, dtype=int)
    for column, (lower, upper) in VALIDATION_RANGES.items():
        invalid_counts += (
            df[column].notna() & ((df[column] < lower) | (df[column] > upper))
        ).astype(int)

    df["invalid_vital_count"] = invalid_counts
    df["invalid_vital_range_flag"] = invalid_counts > 0
    return df


def _impute_vitals_within_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Fill vital missingness within patient timelines, never across patients."""
    imputed = df.copy()
    before_missing = imputed[VITAL_COLUMNS].isna()

    imputed[VITAL_COLUMNS] = imputed.groupby("patient_id", group_keys=False)[
        VITAL_COLUMNS
    ].transform(lambda group: group.ffill().bfill())

    fallback_used = pd.Series(False, index=imputed.index)
    for column, fallback_value in FALLBACK_VALUES.items():
        still_missing = imputed[column].isna()
        if bool(still_missing.any()):
            fallback_used = fallback_used | still_missing
            imputed.loc[still_missing, column] = fallback_value

    imputed["vital_imputation_count"] = before_missing.sum(axis=1).astype(int)
    imputed["fallback_imputation_flag"] = fallback_used
    return imputed


def _add_noise_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact signal describing rows that may contain sensor artifacts."""
    df["noise_intensity"] = (
        df["sensor_noise_flag"].astype(int)
        + df["invalid_vital_count"].astype(int)
        + df["duplicate_patient_timestamp_flag"].astype(int)
    ).astype(float)
    return df


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return _project_root() / path_obj


def _project_root() -> Path:
    """Return the repository root for this project."""
    return Path(__file__).resolve().parents[2]
