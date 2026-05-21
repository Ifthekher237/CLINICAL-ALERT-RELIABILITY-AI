"""Isolation Forest anomaly detection for simulated monitoring data.

This module is part of a research and engineering prototype. It flags unusual
patterns in synthetic vital-sign data for reliability experiments only. It is
not a clinical diagnosis tool and must not be used for real patient monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


DEFAULT_DATA_PATH = Path("data/processed/processed_data.csv")
DEFAULT_MODEL_PATH = Path("models/isolation_forest_anomaly.pkl")

LEAKAGE_COLUMNS = {
    "patient_id",
    "timestamp",
    "outcome_timestamp",
    "patient_condition_label",
    "deterioration_event",
    "patient_outcome_after_alert",
    "outcome_severity_change",
    "target_label",
    "future_deterioration_label",
}

REQUIRED_SCORE_COLUMNS = [
    "anomaly_score",
    "anomaly_label",
    "anomaly_severity",
    "anomaly_reason",
]


class AnomalyDetectionModel:
    """Small wrapper around the Step 5 Isolation Forest functions."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.model: Pipeline | None = None
        self.feature_columns: list[str] = []

    def train(self, X: pd.DataFrame) -> "AnomalyDetectionModel":
        """Train the wrapped Isolation Forest pipeline."""
        self.model = train_isolation_forest(
            X,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.feature_columns = list(getattr(self.model, "anomaly_feature_columns_", []))
        return self

    def score(
        self,
        X: pd.DataFrame,
        original_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Score rows for unusual simulated vital-sign patterns."""
        if self.model is None:
            raise ValueError("AnomalyDetectionModel must be trained before scoring.")
        return score_anomalies(self.model, X, original_df=original_df)


def load_processed_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the processed monitoring dataset used by Step 5."""
    data_path = _resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {data_path}")
    return pd.read_csv(data_path)


def select_anomaly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select numeric, non-leakage features for anomaly detection."""
    candidate_columns = [
        column
        for column in df.select_dtypes(include=["number", "bool"]).columns
        if column not in LEAKAGE_COLUMNS
    ]
    if not candidate_columns:
        raise ValueError("No valid numeric anomaly features were found.")

    X = df[candidate_columns].copy()
    for column in X.select_dtypes(include=["bool"]).columns:
        X[column] = X[column].astype(int)

    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_columns = [column for column in X.columns if not X[column].isna().all()]
    if not valid_columns:
        raise ValueError("All candidate anomaly features were empty after cleaning.")

    X = X[valid_columns]
    return X, valid_columns


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """Train an Isolation Forest for unusual simulated vital-sign patterns."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be greater than 0 and less than 0.5.")
    if X.empty:
        raise ValueError("Training data for anomaly detection cannot be empty.")

    feature_columns = list(X.columns)
    X_model = _prepare_feature_matrix(X, feature_columns)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "isolation_forest",
                IsolationForest(
                    contamination=contamination,
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_model)
    model.anomaly_feature_columns_ = feature_columns
    model.anomaly_contamination_ = contamination
    return model


def score_anomalies(
    model: Pipeline,
    X: pd.DataFrame,
    original_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return anomaly scores, labels, severity, and simple reasons."""
    feature_columns = list(getattr(model, "anomaly_feature_columns_", list(X.columns)))
    X_model = _prepare_feature_matrix(X, feature_columns)

    if original_df is not None:
        if len(original_df) != len(X_model):
            raise ValueError("original_df must have the same number of rows as X.")
        scored = original_df.reset_index(drop=True).copy()
    else:
        scored = X_model.reset_index(drop=True).copy()

    decision_scores = model.decision_function(X_model)
    predictions = model.predict(X_model)
    anomaly_scores = -decision_scores
    anomaly_labels = (predictions == -1).astype(int)

    scored["anomaly_score"] = anomaly_scores.round(6)
    scored["anomaly_label"] = anomaly_labels.astype(int)
    scored["anomaly_severity"] = [
        "normal" if label == 0 else assign_anomaly_severity(score)
        for score, label in zip(anomaly_scores, anomaly_labels, strict=True)
    ]
    scored["anomaly_reason"] = scored.apply(generate_anomaly_reason, axis=1)
    return scored


def assign_anomaly_severity(anomaly_score: float) -> str:
    """Map higher anomaly scores to simple severity buckets."""
    score = float(anomaly_score)
    if score <= 0.0:
        return "normal"
    if score < 0.05:
        return "low"
    if score < 0.10:
        return "medium"
    return "high"


def generate_anomaly_reason(row: pd.Series | dict[str, Any]) -> str:
    """Explain the most visible simulated pattern behind an anomaly flag."""
    label = int(_row_value(row, "anomaly_label", 1))
    if label == 0:
        return "No unusual pattern detected"

    abnormal_count = float(_row_value(row, "abnormal_value_count", 0.0))
    instability_score = float(_row_value(row, "instability_score", 0.0))
    oxygen = float(_row_value(row, "oxygen_saturation", np.nan))
    oxygen_trend = float(_row_value(row, "oxygen_trend", 0.0))
    noise_intensity = float(_row_value(row, "noise_intensity", 0.0))

    if abnormal_count >= 2:
        return "Multiple abnormal vital signs"
    if pd.notna(oxygen) and (oxygen < 94.0 or oxygen_trend < -2.0):
        return "Unusual oxygen saturation pattern"
    if instability_score >= 0.75:
        return "High instability score"
    if noise_intensity >= 1:
        return "Unusual sensor or data quality pattern"
    return "Unusual combined vital-sign pattern"


def save_anomaly_model(
    model: Pipeline,
    path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    """Save the trained Step 5 anomaly model artifact."""
    model_path = _resolve_project_path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def run_anomaly_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    """Run the Step 5 anomaly-detection workflow end to end."""
    df = load_processed_data(data_path)
    X, feature_columns = select_anomaly_features(df)
    model = train_isolation_forest(X)
    scored_df = score_anomalies(model, X, original_df=df)
    model_path = save_anomaly_model(model)
    return {
        "model": model,
        "scored_data": scored_df,
        "feature_columns": feature_columns,
        "model_path": model_path,
    }


def _prepare_feature_matrix(
    X: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Align, coerce, and clean feature data before model use."""
    X_model = X.copy()
    for column in X_model.select_dtypes(include=["bool"]).columns:
        X_model[column] = X_model[column].astype(int)
    X_model = X_model.apply(pd.to_numeric, errors="coerce")
    X_model = X_model.replace([np.inf, -np.inf], np.nan)
    return X_model.reindex(columns=feature_columns, fill_value=np.nan)


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    """Read a value from a row-like object with a fallback."""
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
    result = run_anomaly_pipeline()
    scored = result["scored_data"]
    anomaly_count = int(scored["anomaly_label"].sum())
    anomaly_rate = anomaly_count / len(scored) if len(scored) else 0.0
    display_columns = [
        "patient_id",
        "timestamp",
        "anomaly_score",
        "anomaly_label",
        "anomaly_severity",
        "anomaly_reason",
    ]

    print("Step 5 anomaly detection complete")
    print(f"Rows scored: {len(scored)}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Anomaly rate: {anomaly_rate:.2%}")
    print(f"Model saved to: {result['model_path']}")
    print("First few anomaly rows:")
    print(scored.loc[scored["anomaly_label"] == 1, display_columns].head())
