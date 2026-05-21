"""Feature engineering for simulated patient-monitoring data.

All time-series features are calculated within each patient and only use the
current or previous rows. That keeps the data suitable for future model training
without leaking information from future timestamps.
"""

from __future__ import annotations

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

ROLLING_COLUMNS = [
    "heart_rate",
    "oxygen_saturation",
    "systolic_bp",
    "respiratory_rate",
]

KEY_VITAL_COLUMNS = [
    "heart_rate",
    "oxygen_saturation",
    "systolic_bp",
    "respiratory_rate",
]

ROLLING_WINDOWS = (3, 5)
LAG_STEPS = (1, 2)
RATE_OF_CHANGE_WINDOWS = (1, 3)
DEFAULT_PREDICTION_HORIZON_MINUTES = 60

# Simplified engineering thresholds for abnormal-count features. They are not
# clinical guidance and should only be used for the simulated prototype.
ABNORMAL_THRESHOLDS = {
    "heart_rate": (50.0, 110.0),
    "oxygen_saturation": (94.0, np.inf),
    "systolic_bp": (90.0, 160.0),
    "diastolic_bp": (50.0, 100.0),
    "respiratory_rate": (10.0, 22.0),
    "temperature": (35.5, 38.0),
}

SEVERITY_RANK_MAP = {
    "normal": 0,
    "deteriorating": 1,
    "critical": 2,
}


def create_features(
    df: pd.DataFrame,
    prediction_horizon_minutes: int = DEFAULT_PREDICTION_HORIZON_MINUTES,
) -> pd.DataFrame:
    """Create model-ready time-series features from cleaned monitoring data."""
    if prediction_horizon_minutes < 1:
        raise ValueError("prediction_horizon_minutes must be at least 1.")

    required_columns = {"patient_id", "timestamp", *VITAL_COLUMNS}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns required for feature engineering: {missing_columns}")

    features = df.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce")
    features = features.sort_values(["patient_id", "timestamp"], kind="mergesort")
    features = features.reset_index(drop=True)

    features = _add_rolling_statistics(features)
    features = _add_trend_features(features)
    features = _add_rate_of_change_features(features)
    features = _add_instability_score(features)
    features = _add_abnormal_value_count(features)
    features = _add_missing_value_count(features)
    features = _add_time_features(features)
    features = _add_lag_features(features)
    features = _add_target_label(
        features,
        prediction_horizon_minutes=prediction_horizon_minutes,
    )
    features = _finalize_feature_frame(features)
    features.attrs["prediction_horizon_minutes"] = prediction_horizon_minutes
    return features


def build_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for Step 3 feature engineering."""
    return create_features(df)


def _add_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add patient-wise rolling means and standard deviations."""
    grouped = df.groupby("patient_id", group_keys=False)

    for column in ROLLING_COLUMNS:
        for window in ROLLING_WINDOWS:
            mean_name = f"{column}_rolling_mean_{window}"
            std_name = f"{column}_rolling_std_{window}"
            df[mean_name] = grouped[column].transform(
                lambda series: series.rolling(window=window, min_periods=1).mean()
            )
            df[std_name] = grouped[column].transform(
                lambda series: series.rolling(window=window, min_periods=2).std()
            )
            df[std_name] = df[std_name].fillna(0.0)

    return df


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add current-minus-previous trend features within each patient."""
    grouped = df.groupby("patient_id", group_keys=False)

    df["heart_rate_trend"] = grouped["heart_rate"].diff().fillna(0.0)
    df["oxygen_trend"] = grouped["oxygen_saturation"].diff().fillna(0.0)
    df["bp_trend"] = grouped["systolic_bp"].diff().fillna(0.0)
    df["respiration_trend"] = grouped["respiratory_rate"].diff().fillna(0.0)
    return df


def _add_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add percent-change features using only previous patient rows."""
    grouped = df.groupby("patient_id", group_keys=False)

    for column in KEY_VITAL_COLUMNS:
        for periods in RATE_OF_CHANGE_WINDOWS:
            feature_name = f"{column}_pct_change_{periods}"
            df[feature_name] = grouped[column].transform(
                lambda series, periods=periods: series.pct_change(periods=periods)
            )
            df[feature_name] = _clean_numeric_feature(df[feature_name])

    return df


def _add_instability_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine variability, drops, and fluctuations into one risk signal."""
    heart_rate_variability = df["heart_rate_rolling_std_5"] / 20.0
    oxygen_drop = (
        np.maximum(0.0, -df["oxygen_trend"]) / 5.0
        + np.maximum(0.0, 95.0 - df["oxygen_saturation"]) / 5.0
    )
    bp_fluctuation = df["systolic_bp_rolling_std_5"] / 25.0 + np.abs(df["bp_trend"]) / 30.0
    respiration_instability = (
        df["respiratory_rate_rolling_std_5"] / 5.0
        + np.maximum(0.0, df["respiratory_rate"] - 20.0) / 10.0
    )

    df["instability_score"] = (
        0.30 * heart_rate_variability
        + 0.30 * oxygen_drop
        + 0.25 * bp_fluctuation
        + 0.15 * respiration_instability
    ).round(4)
    return df


def _add_abnormal_value_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many vitals are outside simplified engineering thresholds."""
    abnormal_count = pd.Series(0, index=df.index, dtype=int)

    for column, (lower, upper) in ABNORMAL_THRESHOLDS.items():
        lower_abnormal = df[column] < lower
        upper_abnormal = df[column] > upper
        abnormal_count += (lower_abnormal | upper_abnormal).astype(int)

    df["abnormal_value_count"] = abnormal_count
    return df


def _add_missing_value_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level missingness count preserved from preprocessing."""
    if "missing_value_count_raw" in df.columns:
        df["missing_value_count"] = df["missing_value_count_raw"].fillna(0).astype(int)
    else:
        df["missing_value_count"] = df[VITAL_COLUMNS].isna().sum(axis=1).astype(int)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add timestamp-derived context features."""
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["is_daytime"] = df["hour_of_day"].between(7, 18, inclusive="both").astype(int)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous one-step and two-step vital values within each patient."""
    grouped = df.groupby("patient_id", group_keys=False)

    for column in KEY_VITAL_COLUMNS:
        for lag in LAG_STEPS:
            feature_name = f"{column}_lag_{lag}"
            shifted = grouped[column].shift(lag)
            # Fill the first lag rows with the current value. This avoids NaNs
            # without using future records or another patient's values.
            df[feature_name] = shifted.fillna(df[column])

    return df


def _add_target_label(
    df: pd.DataFrame,
    prediction_horizon_minutes: int,
) -> pd.DataFrame:
    """Create a future deterioration target for supervised models.

    ``future_deterioration_label`` is binary:
    0 means no worsening signal appears in the future horizon.
    1 means a deterioration event or worse condition appears after the current
    row and within the configured horizon.
    """
    labels = pd.Series(0, index=df.index, dtype=int)
    horizon = pd.Timedelta(minutes=prediction_horizon_minutes)

    if "patient_condition_label" in df.columns:
        condition = (
            df["patient_condition_label"]
            .astype(str)
            .str.lower()
            .map(SEVERITY_RANK_MAP)
            .fillna(0)
            .astype(int)
        )
    else:
        condition = pd.Series(0, index=df.index, dtype=int)

    if "deterioration_event" in df.columns:
        event_flag = df["deterioration_event"].astype(bool)
    else:
        event_flag = pd.Series(False, index=df.index, dtype=bool)

    working = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "condition_rank": condition,
            "deterioration_event": event_flag,
        },
        index=df.index,
    )

    for _, patient_rows in working.groupby(df["patient_id"], sort=False):
        patient_rows = patient_rows.sort_values("timestamp", kind="mergesort")
        timestamps = patient_rows["timestamp"]
        ranks = patient_rows["condition_rank"]
        events = patient_rows["deterioration_event"]

        for row_index, current_time in timestamps.items():
            if pd.isna(current_time):
                continue

            horizon_end = current_time + horizon
            future_mask = (timestamps > current_time) & (timestamps <= horizon_end)
            if not bool(future_mask.any()):
                continue

            current_rank = int(ranks.loc[row_index])
            future_worsening = bool((ranks.loc[future_mask] > current_rank).any())
            future_event = bool(events.loc[future_mask].any())
            labels.loc[row_index] = int(future_worsening or future_event)

    df["future_deterioration_label"] = labels.astype(int)
    # Backward-compatible alias for earlier Step 4 code and notebooks.
    df["target_label"] = df["future_deterioration_label"]
    return df


def _finalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean remaining numeric feature artifacts after transformations."""
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns
    for column in numeric_columns:
        if column == "outcome_severity_change":
            continue
        df[column] = _clean_numeric_feature(df[column])

    return df.reset_index(drop=True)


def _clean_numeric_feature(series: pd.Series) -> pd.Series:
    """Replace infinities and NaNs introduced by transformations."""
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
