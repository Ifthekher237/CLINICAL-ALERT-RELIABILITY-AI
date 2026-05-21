"""Focused tests for Step 13 drift detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring import drift_detector


def _sample_processed_data() -> pd.DataFrame:
    """Create compact processed rows with a later vital-sign shift."""
    return pd.DataFrame(
        [
            {
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "heart_rate": 72,
                "oxygen_saturation": 98,
                "respiratory_rate": 16,
                "instability_score": 0.10,
            },
            {
                "patient_id": "P0002",
                "timestamp": "2026-01-01 08:30:00",
                "heart_rate": 76,
                "oxygen_saturation": 97,
                "respiratory_rate": 17,
                "instability_score": 0.12,
            },
            {
                "patient_id": "P0001",
                "timestamp": "2026-01-01 12:00:00",
                "heart_rate": 110,
                "oxygen_saturation": 91,
                "respiratory_rate": 27,
                "instability_score": 0.65,
            },
            {
                "patient_id": "P0002",
                "timestamp": "2026-01-01 12:30:00",
                "heart_rate": 116,
                "oxygen_saturation": 90,
                "respiratory_rate": 29,
                "instability_score": 0.72,
            },
        ]
    )


def _sample_alerts() -> pd.DataFrame:
    """Create compact generated alert rows with severity and volume shifts."""
    return pd.DataFrame(
        [
            {
                "alert_id": "ALERT-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "low",
                "alert_type": "anomaly_detected",
                "risk_score": 0.25,
            },
            {
                "alert_id": "ALERT-002",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 08:30:00",
                "severity": "medium",
                "alert_type": "future_deterioration_risk",
                "risk_score": 0.45,
            },
            {
                "alert_id": "ALERT-003",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 12:00:00",
                "severity": "critical",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.95,
            },
            {
                "alert_id": "ALERT-004",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 12:15:00",
                "severity": "high",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.88,
            },
        ]
    )


def _sample_responses() -> pd.DataFrame:
    """Create compact response logs with later behavior shift."""
    return pd.DataFrame(
        [
            {
                "response_id": "RESP-001",
                "alert_id": "ALERT-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "simulated_response": "accepted",
                "response_time_minutes": 18,
            },
            {
                "response_id": "RESP-002",
                "alert_id": "ALERT-002",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 08:30:00",
                "simulated_response": "accepted",
                "response_time_minutes": 22,
            },
            {
                "response_id": "RESP-003",
                "alert_id": "ALERT-003",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 12:00:00",
                "simulated_response": "delayed",
                "response_time_minutes": 85,
            },
            {
                "response_id": "RESP-004",
                "alert_id": "ALERT-004",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 12:15:00",
                "simulated_response": "marked_false",
                "response_time_minutes": 75,
            },
        ]
    )


def _sample_reliability() -> pd.DataFrame:
    """Create compact reliability rows with later score decline."""
    return pd.DataFrame(
        [
            {
                "monitoring_window_id": "WINDOW-001",
                "window_start": "2026-01-01 08:00:00",
                "window_end": "2026-01-01 10:00:00",
                "total_alerts": 2,
                "ignored_alert_rate": 0.0,
                "delayed_alert_rate": 0.0,
                "reliability_score": 0.95,
            },
            {
                "monitoring_window_id": "WINDOW-002",
                "window_start": "2026-01-01 12:00:00",
                "window_end": "2026-01-01 14:00:00",
                "total_alerts": 7,
                "ignored_alert_rate": 0.2,
                "delayed_alert_rate": 0.3,
                "reliability_score": 0.70,
            },
        ]
    )


def test_all_input_files_can_be_loaded() -> None:
    processed = drift_detector.load_processed_data("data/processed/processed_data.csv")
    alerts = drift_detector.load_generated_alerts("data/processed/generated_alerts.csv")
    responses = drift_detector.load_response_logs("data/processed/clinician_response_logs.csv")
    reliability = drift_detector.load_reliability_results(
        "data/processed/reliability_monitoring_results.csv"
    )

    assert not processed.empty
    assert not alerts.empty
    assert not responses.empty
    assert not reliability.empty


def test_monitoring_windows_are_created() -> None:
    windows = drift_detector.create_drift_windows(
        _sample_processed_data(),
        timestamp_column="timestamp",
        window_minutes=180,
    )

    assert len(windows) == 2
    assert {"drift_window_id", "window_start", "window_end", "data"}.issubset(windows[0])


def test_psi_calculation_returns_numeric_value() -> None:
    score = drift_detector.calculate_population_stability_index(
        [70, 72, 74, 75, 77],
        [90, 92, 94, 96, 98],
    )

    assert isinstance(score, float)
    assert score >= 0


def test_distribution_shift_returns_numeric_value() -> None:
    score = drift_detector.calculate_distribution_shift(
        pd.Series(["low", "low", "medium"]),
        pd.Series(["high", "critical", "critical"]),
    )

    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_drift_result_dataframe_contains_required_columns() -> None:
    vital = drift_detector.detect_vital_sign_drift(_sample_processed_data())
    alert = drift_detector.detect_alert_distribution_drift(_sample_alerts())
    response = drift_detector.detect_response_behavior_drift(_sample_responses())
    reliability = drift_detector.detect_reliability_score_drift(_sample_reliability())
    results = drift_detector.combine_drift_results(vital, alert, response, reliability)

    assert not results.empty
    assert set(drift_detector.REQUIRED_OUTPUT_COLUMNS).issubset(results.columns)


def test_drift_scores_statuses_recommendations_and_review_flags_are_valid() -> None:
    results = drift_detector.combine_drift_results(
        drift_detector.detect_vital_sign_drift(_sample_processed_data()),
        drift_detector.detect_alert_distribution_drift(_sample_alerts()),
        drift_detector.detect_response_behavior_drift(_sample_responses()),
        drift_detector.detect_reliability_score_drift(_sample_reliability()),
    )

    assert pd.api.types.is_numeric_dtype(results["drift_score"])
    assert (results["drift_score"] >= 0).all()
    assert set(results["drift_status"]).issubset(drift_detector.VALID_DRIFT_STATUSES)
    assert set(results["recalibration_recommendation"]).issubset(
        drift_detector.VALID_RECALIBRATION_RECOMMENDATIONS
    )
    assert results["requires_review"].map(lambda value: isinstance(value, (bool, np.bool_))).all()


def test_reliability_score_degradation_requires_review() -> None:
    reliability = drift_detector.detect_reliability_score_drift(_sample_reliability())
    combined = drift_detector.combine_drift_results(
        pd.DataFrame(columns=drift_detector.REQUIRED_OUTPUT_COLUMNS),
        pd.DataFrame(columns=drift_detector.REQUIRED_OUTPUT_COLUMNS),
        pd.DataFrame(columns=drift_detector.REQUIRED_OUTPUT_COLUMNS),
        reliability,
    )
    reliability_score_row = combined[combined["monitored_feature"] == "reliability_score"].iloc[0]

    assert bool(reliability_score_row["requires_review"]) is True


def test_results_csv_is_saved(tmp_path: Path) -> None:
    results = drift_detector.combine_drift_results(
        drift_detector.detect_vital_sign_drift(_sample_processed_data()),
        drift_detector.detect_alert_distribution_drift(_sample_alerts()),
        drift_detector.detect_response_behavior_drift(_sample_responses()),
        drift_detector.detect_reliability_score_drift(_sample_reliability()),
    )
    output_path = drift_detector.save_drift_results(results, tmp_path / "drift_results.csv")

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(drift_detector.REQUIRED_OUTPUT_COLUMNS).issubset(saved.columns)


def test_summary_json_is_saved(tmp_path: Path) -> None:
    summary = {
        "total_drift_checks": 3,
        "average_drift_score": 0.12,
        "severe_drift_count": 1,
    }
    output_path = drift_detector.save_drift_summary(summary, tmp_path / "drift_summary.json")

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["total_drift_checks"] == 3


def test_pipeline_saves_results_and_summary(tmp_path: Path) -> None:
    results_path = tmp_path / "drift_detection_results.csv"
    summary_path = tmp_path / "drift_summary.json"
    results = drift_detector.run_drift_detection_pipeline(
        processed_path="data/processed/processed_data.csv",
        alerts_path="data/processed/generated_alerts.csv",
        response_path="data/processed/clinician_response_logs.csv",
        reliability_path="data/processed/reliability_monitoring_results.csv",
        results_path=results_path,
        summary_path=summary_path,
    )

    assert results_path.exists()
    assert summary_path.exists()
    assert not results.empty
