"""Focused tests for Step 6 time-series risk logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import timeseries_model


def _small_unsorted_frame() -> pd.DataFrame:
    """Create two patients where rolling logic must not cross patient boundaries."""
    rows = [
        {
            "patient_id": "P0002",
            "timestamp": "2026-01-01 08:15:00",
            "heart_rate": 76,
            "oxygen_saturation": 97,
            "systolic_bp": 120,
            "diastolic_bp": 76,
            "respiratory_rate": 16,
            "temperature": 36.8,
            "target_label": 1,
            "future_deterioration_label": 1,
            "outcome_severity_change": 99,
        },
        {
            "patient_id": "P0001",
            "timestamp": "2026-01-01 08:00:00",
            "heart_rate": 120,
            "oxygen_saturation": 90,
            "systolic_bp": 88,
            "diastolic_bp": 54,
            "respiratory_rate": 28,
            "temperature": 38.0,
            "target_label": 0,
            "future_deterioration_label": 0,
            "outcome_severity_change": -99,
        },
        {
            "patient_id": "P0001",
            "timestamp": "2026-01-01 08:05:00",
            "heart_rate": 122,
            "oxygen_saturation": 89,
            "systolic_bp": 87,
            "diastolic_bp": 53,
            "respiratory_rate": 29,
            "temperature": 38.1,
            "target_label": 0,
            "future_deterioration_label": 0,
            "outcome_severity_change": -99,
        },
        {
            "patient_id": "P0002",
            "timestamp": "2026-01-01 08:00:00",
            "heart_rate": 74,
            "oxygen_saturation": 98,
            "systolic_bp": 121,
            "diastolic_bp": 77,
            "respiratory_rate": 15,
            "temperature": 36.7,
            "target_label": 1,
            "future_deterioration_label": 1,
            "outcome_severity_change": 99,
        },
    ]
    return pd.DataFrame(rows)


def test_data_loads_correctly() -> None:
    df = timeseries_model.load_processed_data("data/processed/processed_data.csv")

    assert not df.empty
    assert {"patient_id", "timestamp"}.issubset(df.columns)


def test_sorting_by_patient_and_timestamp() -> None:
    ordered = timeseries_model.ensure_time_order(_small_unsorted_frame())

    assert ordered["patient_id"].tolist() == ["P0001", "P0001", "P0002", "P0002"]
    assert ordered.groupby("patient_id")["timestamp"].is_monotonic_increasing.all()


def test_no_patient_mixing_in_sustained_abnormality() -> None:
    scored = timeseries_model.calculate_sustained_abnormality(
        _small_unsorted_frame(),
        window=3,
    )
    first_p2_row = scored[scored["patient_id"] == "P0002"].iloc[0]

    assert first_p2_row["heart_rate_abnormal_recent_count"] == 0
    assert first_p2_row["oxygen_low_recent_count"] == 0
    assert first_p2_row["sustained_abnormal_vital_count"] == 0


def test_required_output_columns_levels_and_numeric_score() -> None:
    df = timeseries_model.load_processed_data("data/processed/processed_data.csv")
    scored = timeseries_model.calculate_time_series_risk_score(df)

    required_columns = {
        "time_series_risk_score",
        "time_series_risk_level",
        "time_series_risk_reason",
    }
    assert required_columns.issubset(scored.columns)
    assert set(scored["time_series_risk_level"].unique()).issubset(timeseries_model.RISK_LEVELS)
    assert pd.api.types.is_numeric_dtype(scored["time_series_risk_score"])


def test_future_target_columns_do_not_affect_scoring() -> None:
    df = timeseries_model.load_processed_data("data/processed/processed_data.csv").head(60)
    baseline = timeseries_model.calculate_time_series_risk_score(df)

    modified = df.copy()
    for column in timeseries_model.LEAKAGE_COLUMNS:
        if column in modified.columns:
            modified[column] = 999
    rescored = timeseries_model.calculate_time_series_risk_score(modified)

    pd.testing.assert_series_equal(
        baseline["time_series_risk_score"],
        rescored["time_series_risk_score"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        baseline["time_series_risk_level"],
        rescored["time_series_risk_level"],
        check_names=False,
    )


def test_output_csv_is_saved(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "timeseries_risk_scored.csv"
    monkeypatch.setattr(timeseries_model, "DEFAULT_OUTPUT_PATH", output_path)

    scored = timeseries_model.run_timeseries_risk_pipeline("data/processed/processed_data.csv")

    assert output_path.exists()
    assert not scored.empty
    saved = pd.read_csv(output_path)
    assert "time_series_risk_score" in saved.columns
