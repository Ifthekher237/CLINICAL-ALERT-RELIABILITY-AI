"""Focused tests for Step 7 simulated alert generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alerts import alert_generator


def _sample_scored_rows() -> pd.DataFrame:
    """Create rows with enough risk signal to generate deterministic alerts."""
    return pd.DataFrame(
        [
            {
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "heart_rate": 118.0,
                "oxygen_saturation": 91.0,
                "systolic_bp": 101.0,
                "respiratory_rate": 27.0,
                "time_series_risk_score": 0.62,
                "time_series_risk_reason": "Sustained oxygen saturation decline",
                "sustained_oxygen_saturation_drop": True,
                "sustained_high_respiratory_rate": True,
                "oxygen_saturation_change_3": -3.2,
                "respiratory_rate_change_3": 4.0,
                "abnormal_value_count": 2,
                "instability_score": 0.8,
                "baseline_future_deterioration_risk": 0.55,
                "anomaly_label": 1,
                "anomaly_score": 0.08,
                "anomaly_reason": "Unusual oxygen saturation pattern",
                "patient_condition_label": "normal",
                "deterioration_event": False,
                "patient_outcome_after_alert": "unknown",
                "outcome_timestamp": "",
                "outcome_severity_change": 0.0,
                "target_label": 1,
                "future_deterioration_label": 1,
            },
            {
                "patient_id": "P0002",
                "timestamp": "2026-01-01 08:05:00",
                "heart_rate": 76.0,
                "oxygen_saturation": 97.0,
                "systolic_bp": 122.0,
                "respiratory_rate": 16.0,
                "time_series_risk_score": 0.05,
                "time_series_risk_reason": "No sustained deterioration pattern",
                "sustained_oxygen_saturation_drop": False,
                "sustained_high_respiratory_rate": False,
                "oxygen_saturation_change_3": 0.0,
                "respiratory_rate_change_3": 0.0,
                "abnormal_value_count": 0,
                "instability_score": 0.1,
                "baseline_future_deterioration_risk": 0.05,
                "anomaly_label": 0,
                "anomaly_score": -0.02,
                "anomaly_reason": "No unusual pattern detected",
                "patient_condition_label": "critical",
                "deterioration_event": True,
                "patient_outcome_after_alert": "worsened",
                "outcome_timestamp": "2026-01-01 09:00:00",
                "outcome_severity_change": 99.0,
                "target_label": 0,
                "future_deterioration_label": 0,
            },
        ]
    )


def test_scored_data_can_be_loaded() -> None:
    df = alert_generator.load_scored_data("data/processed/timeseries_risk_scored.csv")

    assert not df.empty
    assert {"patient_id", "timestamp", "time_series_risk_score"}.issubset(df.columns)


def test_generated_alerts_contain_required_columns_and_valid_values() -> None:
    alerts = alert_generator.generate_alerts(_sample_scored_rows())

    assert not alerts.empty
    assert set(alert_generator.REQUIRED_ALERT_COLUMNS).issubset(alerts.columns)
    assert set(alerts["severity"].unique()).issubset(alert_generator.SEVERITY_LEVELS)
    assert alerts["recommended_review_time"].notna().all()
    assert pd.api.types.is_bool_dtype(alerts["critical_flag"])


def test_alert_ids_are_unique_and_alerts_generated_when_risk_present() -> None:
    alerts = alert_generator.generate_alerts(_sample_scored_rows())

    assert len(alerts) >= 1
    assert alerts["alert_id"].is_unique


def test_output_csv_is_saved(tmp_path: Path) -> None:
    alerts = alert_generator.generate_alerts(_sample_scored_rows())
    output_path = alert_generator.save_alerts(alerts, tmp_path / "generated_alerts.csv")

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(alert_generator.REQUIRED_ALERT_COLUMNS).issubset(saved.columns)


def test_future_labels_and_outcomes_do_not_change_alert_logic() -> None:
    base = _sample_scored_rows()
    mutated = base.copy()
    for column in alert_generator.LEAKAGE_COLUMNS:
        if column in mutated.columns:
            mutated[column] = "changed_future_or_outcome_value"

    base_alerts = alert_generator.generate_alerts(base)
    mutated_alerts = alert_generator.generate_alerts(mutated)

    comparison_columns = [
        "patient_id",
        "timestamp",
        "severity",
        "alert_type",
        "risk_score",
        "trigger_reason",
        "recommended_review_time",
        "critical_flag",
    ]
    pd.testing.assert_frame_equal(
        base_alerts[comparison_columns],
        mutated_alerts[comparison_columns],
    )


def test_pipeline_generates_and_saves_alerts(tmp_path: Path) -> None:
    output_path = tmp_path / "generated_alerts.csv"
    alerts = alert_generator.run_alert_generation_pipeline(
        scored_data_path="data/processed/timeseries_risk_scored.csv",
        output_path=output_path,
    )

    assert output_path.exists()
    assert not alerts.empty
    assert alerts["alert_id"].is_unique
