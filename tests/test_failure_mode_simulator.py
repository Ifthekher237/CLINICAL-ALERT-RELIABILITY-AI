"""Focused tests for Step 24B failure-mode simulation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.monitoring import failure_mode_simulator


REQUIRED_SUMMARY_KEYS = {
    "total_failure_events",
    "failure_mode_distribution",
    "severity_distribution",
    "safety_status_distribution",
    "average_alert_volume_impact",
    "average_clinician_burden_impact",
    "average_reliability_score_impact",
    "average_drift_risk_impact",
    "average_outcome_risk_impact",
    "unsafe_review_required_count",
    "human_review_required_rate",
    "simulation_only_note",
}


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_df = pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "timestamp": "2026-01-01 08:00:00",
                "heart_rate": 155,
                "oxygen_saturation": 79,
                "respiratory_rate": 38,
                "sensor_noise_flag": True,
                "missing_data_flag": False,
            },
            {
                "patient_id": "P002",
                "timestamp": "2026-01-01 08:05:00",
                "heart_rate": None,
                "oxygen_saturation": 95,
                "respiratory_rate": 18,
                "sensor_noise_flag": False,
                "missing_data_flag": True,
            },
        ]
    )
    generated_alerts_df = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-1",
                "patient_id": "P001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "critical",
                "alert_type": "multiple_risk_signals",
                "risk_score": 0.95,
            },
            {
                "alert_id": "ALERT-2",
                "patient_id": "P002",
                "timestamp": "2026-01-01 08:05:00",
                "severity": "medium",
                "alert_type": "anomaly_detected",
                "risk_score": 0.45,
            },
        ]
    )
    fatigue_df = generated_alerts_df.assign(
        fatigue_action=["retain", "group_repeated"],
        final_alert_status=["active", "grouped"],
        grouped_alert_count=[1, 4],
        fatigue_risk_score=[0.1, 0.8],
    )
    responses_df = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-1",
                "patient_id": "P001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "critical",
                "simulated_response": "delayed",
                "response_time_minutes": 45,
            },
            {
                "alert_id": "ALERT-2",
                "patient_id": "P002",
                "timestamp": "2026-01-01 08:05:00",
                "severity": "medium",
                "simulated_response": "ignored",
                "response_time_minutes": 80,
            },
        ]
    )
    reliability_df = pd.DataFrame(
        [
            {
                "window_start": "2026-01-01 08:00:00",
                "total_alerts": 42,
                "average_response_time_minutes": 45,
                "reliability_score": 0.72,
                "review_recommendation": "review_workflow_burden",
            }
        ]
    )
    drift_df = pd.DataFrame(
        [
            {
                "window_start": "2026-01-01 08:00:00",
                "drift_type": "data_drift",
                "monitored_feature": "oxygen_saturation",
                "drift_score": 0.72,
                "drift_status": "severe_shift",
                "requires_review": True,
            }
        ]
    )
    outcome_df = pd.DataFrame(
        [
            {
                "alert_id": "ALERT-1",
                "outcome_effectiveness_score": 0.4,
                "delayed_response_impact_score": 0.8,
            }
        ]
    )
    return patient_df, generated_alerts_df, fatigue_df, responses_df, reliability_df, drift_df, outcome_df


def test_missing_files_do_not_crash(tmp_path: Path) -> None:
    assert failure_mode_simulator.safe_load_csv(str(tmp_path / "missing.csv")).empty
    assert failure_mode_simulator.safe_load_json(str(tmp_path / "missing.json")) == {}

    results = failure_mode_simulator.run_failure_mode_pipeline(
        patient_path=str(tmp_path / "missing_patient.csv"),
        generated_alerts_path=str(tmp_path / "missing_alerts.csv"),
        fatigue_alerts_path=str(tmp_path / "missing_fatigue.csv"),
        response_path=str(tmp_path / "missing_responses.csv"),
        reliability_path=str(tmp_path / "missing_reliability.csv"),
        drift_path=str(tmp_path / "missing_drift.csv"),
        outcome_path=str(tmp_path / "missing_outcomes.csv"),
        metrics_path=str(tmp_path / "missing_metrics.json"),
        results_path=str(tmp_path / "results.csv"),
        summary_path=str(tmp_path / "summary.json"),
    )

    assert results.empty
    assert set(failure_mode_simulator.REQUIRED_OUTPUT_COLUMNS).issubset(results.columns)


def test_valid_failure_modes_only() -> None:
    results = _build_sample_results()

    assert set(results["failure_mode"]).issubset(failure_mode_simulator.VALID_FAILURE_MODES)
    assert failure_mode_simulator.VALID_FAILURE_MODES.issubset(set(results["failure_mode"]))


def test_valid_severity_labels_and_safety_statuses_only() -> None:
    results = _build_sample_results()

    assert set(results["severity_level"]).issubset(failure_mode_simulator.VALID_SEVERITY_LEVELS)
    assert set(results["safety_status"]).issubset(failure_mode_simulator.VALID_SAFETY_STATUSES)


def test_scores_stay_between_zero_and_one() -> None:
    results = _build_sample_results()
    score_columns = [
        "alert_volume_impact",
        "clinician_burden_impact",
        "reliability_score_impact",
        "drift_risk_impact",
        "outcome_risk_impact",
    ]

    for column in score_columns:
        assert pd.to_numeric(results[column], errors="coerce").between(0, 1).all()


def test_mitigation_recommendations_exist() -> None:
    results = _build_sample_results()

    assert results["mitigation_recommendation"].fillna("").astype(str).str.len().gt(0).all()


def test_critical_and_high_failures_require_review() -> None:
    results = _build_sample_results()
    high_or_critical = results["severity_level"].isin(["high", "critical"])

    assert high_or_critical.any()
    assert results.loc[high_or_critical, "requires_human_review"].map(bool).all()


def test_csv_output_and_json_summary_are_saved(tmp_path: Path) -> None:
    results = _build_sample_results()
    summary = failure_mode_simulator.calculate_failure_summary(results)
    csv_path = failure_mode_simulator.save_failure_results(results, str(tmp_path / "failure_mode_results.csv"))
    json_path = failure_mode_simulator.save_failure_summary(summary, str(tmp_path / "failure_mode_summary.json"))

    assert csv_path.exists()
    assert json_path.exists()
    assert set(failure_mode_simulator.REQUIRED_OUTPUT_COLUMNS).issubset(pd.read_csv(csv_path).columns)
    with json_path.open("r", encoding="utf-8") as file:
        saved_summary = json.load(file)
    assert REQUIRED_SUMMARY_KEYS.issubset(saved_summary.keys())


def test_summary_contains_required_keys() -> None:
    summary = failure_mode_simulator.calculate_failure_summary(_build_sample_results())

    assert REQUIRED_SUMMARY_KEYS.issubset(summary.keys())
    assert summary["total_failure_events"] > 0


def test_no_unsafe_clinical_wording_appears() -> None:
    results = _build_sample_results()
    combined_text = " ".join(
        results["mitigation_recommendation"].astype(str).tolist()
        + results["failure_simulation_note"].astype(str).tolist()
    ).lower()

    for unsafe_phrase in [
        "diagnose",
        "prescribe",
        "recommend treatment",
        "replace clinician",
        "safe for real patient care",
        "clinically validated",
    ]:
        assert unsafe_phrase not in combined_text


def test_pipeline_runs_successfully(tmp_path: Path) -> None:
    results_path = tmp_path / "failure_mode_results.csv"
    summary_path = tmp_path / "failure_mode_summary.json"

    results = failure_mode_simulator.run_failure_mode_pipeline(
        results_path=str(results_path),
        summary_path=str(summary_path),
    )

    assert results_path.exists()
    assert summary_path.exists()
    assert set(failure_mode_simulator.REQUIRED_OUTPUT_COLUMNS).issubset(results.columns)


def _build_sample_results() -> pd.DataFrame:
    (
        patient_df,
        generated_alerts_df,
        fatigue_df,
        responses_df,
        reliability_df,
        drift_df,
        outcome_df,
    ) = _sample_inputs()
    return failure_mode_simulator.build_failure_mode_table(
        patient_df=patient_df,
        generated_alerts_df=generated_alerts_df,
        fatigue_df=fatigue_df,
        responses_df=responses_df,
        reliability_df=reliability_df,
        drift_df=drift_df,
        outcome_df=outcome_df,
        metrics={},
    )
