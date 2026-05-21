"""Focused tests for Step 14 model-update simulation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models import model_registry


def _sample_audited_alerts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alert_id": "ALERT-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "low",
                "risk_score": 0.30,
                "actionability_score": 0.35,
                "false_positive_likelihood": 0.75,
                "audit_status": "likely_noise",
                "escalation_recommendation": "monitor",
            },
            {
                "alert_id": "ALERT-002",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 09:00:00",
                "severity": "high",
                "risk_score": 0.85,
                "actionability_score": 0.82,
                "false_positive_likelihood": 0.10,
                "audit_status": "high_priority",
                "escalation_recommendation": "urgent_review",
            },
        ]
    )


def _sample_fatigue_alerts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alert_id": "ALERT-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "severity": "low",
                "critical_flag": False,
                "fatigue_action": "group_repeated",
                "final_alert_status": "grouped",
            },
            {
                "alert_id": "ALERT-002",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 09:00:00",
                "severity": "high",
                "critical_flag": False,
                "fatigue_action": "retain",
                "final_alert_status": "active",
            },
        ]
    )


def _sample_responses() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "response_id": "RESP-001",
                "alert_id": "ALERT-001",
                "patient_id": "P0001",
                "timestamp": "2026-01-01 08:00:00",
                "simulated_response": "marked_false",
                "response_time_minutes": 50,
            },
            {
                "response_id": "RESP-002",
                "alert_id": "ALERT-002",
                "patient_id": "P0002",
                "timestamp": "2026-01-01 09:00:00",
                "simulated_response": "escalated",
                "response_time_minutes": 6,
            },
        ]
    )


def _sample_reliability() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "monitoring_window_id": "WINDOW-001",
                "window_start": "2026-01-01 08:00:00",
                "window_end": "2026-01-01 10:00:00",
                "ignored_alert_rate": 0.0,
                "delayed_alert_rate": 0.0,
                "safety_preservation_score": 1.0,
                "reliability_score": 0.92,
                "reliability_status": "stable",
                "review_recommendation": "review_workflow_burden",
            }
        ]
    )


def _sample_drift(severe: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "drift_window_id": "DRIFT-WINDOW-001",
                "window_start": "2026-01-01 08:00:00",
                "window_end": "2026-01-01 11:00:00",
                "drift_type": "data_drift",
                "monitored_feature": "heart_rate",
                "drift_score": 0.42 if severe else 0.08,
                "drift_status": "severe_shift" if severe else "stable",
                "recalibration_recommendation": (
                    "retraining_review_recommended" if severe else "no_action_needed"
                ),
                "requires_review": severe,
            }
        ]
    )


def test_input_files_can_be_loaded() -> None:
    audited = model_registry.load_audited_alerts("data/processed/audited_alerts.csv")
    fatigue = model_registry.load_fatigue_reduced_alerts(
        "data/processed/fatigue_reduced_alerts.csv"
    )
    responses = model_registry.load_response_logs(
        "data/processed/clinician_response_logs.csv"
    )
    reliability = model_registry.load_reliability_results(
        "data/processed/reliability_monitoring_results.csv"
    )
    drift = model_registry.load_drift_results("data/processed/drift_detection_results.csv")

    assert not audited.empty
    assert not fatigue.empty
    assert not responses.empty
    assert not reliability.empty
    assert not drift.empty


def test_feedback_signals_are_calculated() -> None:
    feedback = model_registry.collect_feedback_signals(
        _sample_audited_alerts(),
        _sample_fatigue_alerts(),
        _sample_responses(),
        _sample_drift(),
        _sample_reliability(),
    )

    assert feedback["false_alert_rate"] == 0.5
    assert feedback["useful_alert_rate"] == 0.5
    assert feedback["average_drift_score"] == 0.08
    assert feedback["critical_preservation_strong"] is True


def test_threshold_update_stays_between_bounds() -> None:
    feedback = model_registry.collect_feedback_signals(
        _sample_audited_alerts(),
        _sample_fatigue_alerts(),
        _sample_responses(),
        _sample_drift(),
        _sample_reliability(),
    )
    threshold_update = model_registry.calculate_threshold_adjustment(
        feedback,
        current_threshold=0.89,
    )

    assert 0.40 <= threshold_update["proposed_risk_threshold"] <= 0.90


def test_simulation_output_contains_required_columns_and_valid_values(tmp_path: Path) -> None:
    results = model_registry.run_model_update_simulation(
        audited_path="data/processed/audited_alerts.csv",
        fatigue_path="data/processed/fatigue_reduced_alerts.csv",
        response_path="data/processed/clinician_response_logs.csv",
        reliability_path="data/processed/reliability_monitoring_results.csv",
        drift_path="data/processed/drift_detection_results.csv",
        results_path=tmp_path / "model_update_simulation_results.csv",
        registry_path=tmp_path / "model_version_registry.json",
        threshold_summary_path=tmp_path / "threshold_update_summary.json",
    )

    assert set(model_registry.REQUIRED_SIMULATION_COLUMNS).issubset(results.columns)
    assert set(results["deployment_recommendation"]).issubset(
        model_registry.VALID_DEPLOYMENT_RECOMMENDATIONS
    )
    assert results["human_review_required"].map(lambda value: isinstance(value, bool)).all()


def test_model_registry_json_is_saved(tmp_path: Path) -> None:
    registry = {
        "registry_name": "test_registry",
        "simulation_only": True,
        "versions": [{"proposed_model_version": "risk_alert_model_v1.0.1-simulated"}],
    }
    output_path = model_registry.save_model_version_registry(
        registry,
        tmp_path / "model_version_registry.json",
    )

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["simulation_only"] is True


def test_threshold_summary_json_is_saved(tmp_path: Path) -> None:
    summary = {
        "threshold_update": {"current_risk_threshold": 0.65, "proposed_risk_threshold": 0.70},
        "simulation_note": "No deployment.",
    }
    output_path = model_registry.save_threshold_update_summary(
        summary,
        tmp_path / "threshold_update_summary.json",
    )

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["threshold_update"]["current_risk_threshold"] == 0.65


def test_existing_model_files_are_not_overwritten(tmp_path: Path) -> None:
    model_paths = [
        Path("models/logistic_regression.pkl"),
        Path("models/random_forest.pkl"),
        Path("models/scaler.pkl"),
    ]
    before = {
        path: path.stat().st_mtime_ns
        for path in model_paths
        if path.exists()
    }

    model_registry.run_model_update_simulation(
        audited_path="data/processed/audited_alerts.csv",
        fatigue_path="data/processed/fatigue_reduced_alerts.csv",
        response_path="data/processed/clinician_response_logs.csv",
        reliability_path="data/processed/reliability_monitoring_results.csv",
        drift_path="data/processed/drift_detection_results.csv",
        results_path=tmp_path / "model_update_simulation_results.csv",
        registry_path=tmp_path / "model_version_registry.json",
        threshold_summary_path=tmp_path / "threshold_update_summary.json",
    )

    after = {
        path: path.stat().st_mtime_ns
        for path in model_paths
        if path.exists()
    }
    assert after == before


def test_simulation_is_explainable_and_does_not_perform_real_deployment() -> None:
    feedback = model_registry.collect_feedback_signals(
        _sample_audited_alerts(),
        _sample_fatigue_alerts(),
        _sample_responses(),
        _sample_drift(severe=True),
        _sample_reliability(),
    )
    threshold_update = model_registry.calculate_threshold_adjustment(feedback)
    record = model_registry.create_model_version_record(
        "risk_alert_model_v1.0.0",
        "risk_alert_model_v1.0.1-simulated",
        feedback,
        threshold_update,
    )
    reason = model_registry.generate_update_reason(feedback, threshold_update)

    assert record["status"] == "simulation_only_not_deployed"
    assert record["deployment_recommendation"] == "retraining_review_recommended"
    assert record["human_review_required"] is True
    assert "human" in reason.lower()
