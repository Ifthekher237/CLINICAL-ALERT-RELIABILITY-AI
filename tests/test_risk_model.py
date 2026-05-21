"""Focused tests for Step 4 future-risk modeling."""

from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import create_features
from src.models import risk_model


def _sample_monitoring_frame() -> pd.DataFrame:
    """Create a small deterministic monitoring frame with future deterioration."""
    start = pd.Timestamp("2026-01-01 08:00:00")
    rows = []

    trajectories = {
        "P0001": ["normal", "normal", "deteriorating", "critical", "critical", "critical"],
        "P0002": ["normal", "normal", "normal", "normal", "normal", "normal"],
        "P0003": ["normal", "normal", "normal", "normal", "deteriorating", "critical"],
    }
    vitals_by_condition = {
        "normal": {
            "heart_rate": 74.0,
            "oxygen_saturation": 97.0,
            "systolic_bp": 120.0,
            "diastolic_bp": 76.0,
            "respiratory_rate": 16.0,
            "temperature": 36.8,
        },
        "deteriorating": {
            "heart_rate": 98.0,
            "oxygen_saturation": 94.0,
            "systolic_bp": 106.0,
            "diastolic_bp": 68.0,
            "respiratory_rate": 22.0,
            "temperature": 37.5,
        },
        "critical": {
            "heart_rate": 122.0,
            "oxygen_saturation": 89.0,
            "systolic_bp": 90.0,
            "diastolic_bp": 58.0,
            "respiratory_rate": 29.0,
            "temperature": 38.4,
        },
    }

    for patient_id, conditions in trajectories.items():
        previous_condition = "normal"
        for step, condition in enumerate(conditions):
            timestamp = start + pd.Timedelta(minutes=15 * step)
            event = condition != previous_condition and condition != "normal"
            rows.append(
                {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    **vitals_by_condition[condition],
                    "patient_condition_label": condition,
                    "deterioration_event": event,
                    "sensor_noise_flag": False,
                    "missing_data_flag": False,
                    "patient_outcome_after_alert": "unknown",
                    "outcome_timestamp": pd.NaT,
                    "outcome_severity_change": pd.NA,
                }
            )
            previous_condition = condition

    return pd.DataFrame(rows)


def _feature_frame() -> pd.DataFrame:
    return create_features(_sample_monitoring_frame(), prediction_horizon_minutes=60)


def test_future_target_is_created_and_not_single_class() -> None:
    features = _feature_frame()

    assert "future_deterioration_label" in features.columns
    assert "target_label" in features.columns
    assert features["future_deterioration_label"].equals(features["target_label"])
    assert features["future_deterioration_label"].nunique() > 1


def test_prepare_features_removes_leakage_columns() -> None:
    features = _feature_frame()
    X, y, feature_columns = risk_model.prepare_features_and_target(features)

    leakage_columns = {
        "patient_id",
        "timestamp",
        "patient_condition_label",
        "deterioration_event",
        "patient_outcome_after_alert",
        "outcome_timestamp",
        "outcome_severity_change",
        "future_deterioration_label",
        "target_label",
    }
    assert leakage_columns.isdisjoint(X.columns)
    assert leakage_columns.isdisjoint(feature_columns)
    assert y.nunique() > 1


def test_prediction_output_shape_and_evaluation_keys() -> None:
    features = _feature_frame()
    X, y, _ = risk_model.prepare_features_and_target(features)
    model = risk_model.train_random_forest(X, y)

    predictions = risk_model.predict_risk(model, X.head(4))
    metrics = risk_model.evaluate_model(model, X, y)

    assert predictions.shape == (4,)
    assert {
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "confusion_matrix",
        "classification_report",
        "auroc",
    }.issubset(metrics.keys())


def test_model_files_are_saved(tmp_path) -> None:
    features = _feature_frame()
    X, y, _ = risk_model.prepare_features_and_target(features)
    logistic_model, scaler = risk_model.train_logistic_regression(X, y)
    random_forest_model = risk_model.train_random_forest(X, y)

    paths = risk_model.save_models(
        logistic_model=logistic_model,
        scaler=scaler,
        random_forest_model=random_forest_model,
        model_dir=tmp_path,
    )

    assert paths["logistic_regression"].exists()
    assert paths["random_forest"].exists()
    assert paths["scaler"].exists()
