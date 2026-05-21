"""Focused tests for Step 22 centralized project metrics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.monitoring import metrics


REQUIRED_TOP_LEVEL_CATEGORIES = {
    "dataset",
    "alerts",
    "audit_fatigue",
    "workflow",
    "reliability",
    "drift",
    "model_update_rl",
    "llm_action",
    "database_demo_readiness",
    "simulation_only_note",
}


def _sample_frames() -> dict[str, pd.DataFrame]:
    processed = pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "deterioration_event": 0,
                "missing_data_flag": 0,
                "sensor_noise_flag": 1,
            },
            {
                "patient_id": "P001",
                "deterioration_event": 1,
                "missing_data_flag": 1,
                "sensor_noise_flag": 0,
            },
            {
                "patient_id": "P002",
                "deterioration_event": 0,
                "missing_data_flag": 0,
                "sensor_noise_flag": 0,
            },
        ]
    )
    raw_alerts = pd.DataFrame(
        [
            {"alert_id": "A1", "critical_flag": True},
            {"alert_id": "A2", "critical_flag": False},
            {"alert_id": "A3", "critical_flag": False},
        ]
    )
    audited = pd.DataFrame(
        [
            {
                "alert_id": "A1",
                "actionability_score": 0.9,
                "fatigue_risk_score": 0.2,
                "false_positive_likelihood": 0.1,
            },
            {
                "alert_id": "A2",
                "actionability_score": 0.4,
                "fatigue_risk_score": 0.8,
                "false_positive_likelihood": 0.6,
            },
        ]
    )
    fatigue = pd.DataFrame(
        [
            {
                "alert_id": "A1",
                "critical_flag": True,
                "final_alert_status": "active",
                "fatigue_action": "retain",
            },
            {
                "alert_id": "A2",
                "critical_flag": False,
                "final_alert_status": "grouped",
                "fatigue_action": "group_repeated",
            },
            {
                "alert_id": "A3",
                "critical_flag": False,
                "final_alert_status": "delayed",
                "fatigue_action": "delay_non_critical",
            },
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "simulated_response": "ignored",
                "response_time_minutes": 40.0,
                "clinician_burden_score": 0.7,
                "perceived_alert_usefulness": 0.3,
            },
            {
                "simulated_response": "escalated",
                "response_time_minutes": 5.0,
                "clinician_burden_score": 0.5,
                "perceived_alert_usefulness": 0.9,
            },
        ]
    )
    reliability = pd.DataFrame(
        [
            {
                "reliability_score": 0.92,
                "reliability_status": "stable",
                "review_recommendation": "no_action_needed",
            },
            {
                "reliability_score": 0.55,
                "reliability_status": "degraded",
                "review_recommendation": "review_workflow_burden",
            },
        ]
    )
    drift = pd.DataFrame(
        [
            {
                "drift_score": 0.08,
                "drift_status": "stable",
                "drift_type": "data_drift",
                "requires_review": False,
            },
            {
                "drift_score": 0.42,
                "drift_status": "severe_shift",
                "drift_type": "reliability_drift",
                "requires_review": True,
            },
        ]
    )
    model_update = pd.DataFrame(
        [
            {
                "current_risk_threshold": 0.65,
                "proposed_risk_threshold": 0.68,
                "threshold_change": 0.03,
                "deployment_recommendation": "threshold_review_recommended",
                "human_review_required": True,
            }
        ]
    )
    explanations = pd.DataFrame(
        [
            {"explanation_id": "E1", "fallback_used": True},
            {"explanation_id": "E2", "fallback_used": False},
        ]
    )
    recommendations = pd.DataFrame(
        [
            {
                "recommendation_id": "R1",
                "action_priority": "immediate",
                "rag_sources": "safety_rules.md",
            },
            {
                "recommendation_id": "R2",
                "action_priority": "urgent",
                "rag_sources": "",
            },
        ]
    )
    return {
        "processed": processed,
        "raw_alerts": raw_alerts,
        "audited": audited,
        "fatigue": fatigue,
        "responses": responses,
        "reliability": reliability,
        "drift": drift,
        "model_update": model_update,
        "explanations": explanations,
        "recommendations": recommendations,
    }


def test_safe_load_csv_handles_missing_file(tmp_path: Path) -> None:
    loaded = metrics.safe_load_csv(tmp_path / "missing.csv")

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.empty


def test_safe_load_json_handles_missing_file(tmp_path: Path) -> None:
    loaded = metrics.safe_load_json(tmp_path / "missing.json")

    assert loaded == {}


def test_safe_rate_avoids_divide_by_zero() -> None:
    assert metrics.safe_rate(5, 0) == 0.0
    assert metrics.safe_rate(1, 4) == 0.25


def test_each_category_function_returns_dictionary() -> None:
    frames = _sample_frames()

    assert isinstance(metrics.calculate_dataset_metrics(frames["processed"]), dict)
    assert isinstance(
        metrics.calculate_alert_metrics(
            frames["raw_alerts"],
            frames["audited"],
            frames["fatigue"],
        ),
        dict,
    )
    audit_fatigue = metrics.calculate_audit_fatigue_metrics(
        frames["audited"],
        frames["fatigue"],
    )
    assert isinstance(audit_fatigue, dict)
    assert audit_fatigue["grouped_alert_count"] == 1
    assert audit_fatigue["delayed_alert_count"] == 1
    assert isinstance(metrics.calculate_workflow_metrics(frames["responses"]), dict)
    assert isinstance(metrics.calculate_reliability_metrics(frames["reliability"]), dict)
    assert isinstance(metrics.calculate_drift_metrics(frames["drift"]), dict)
    assert isinstance(
        metrics.calculate_model_update_metrics(
            frames["model_update"],
            {"recommended_action": "keep_threshold", "safety_violation_count": 0},
        ),
        dict,
    )
    assert isinstance(
        metrics.calculate_llm_action_metrics(
            frames["explanations"],
            frames["recommendations"],
        ),
        dict,
    )


def test_compile_project_metrics_returns_required_categories(tmp_path: Path) -> None:
    frames = _sample_frames()
    compiled = metrics.compile_project_metrics(
        processed_df=frames["processed"],
        raw_alerts_df=frames["raw_alerts"],
        audited_df=frames["audited"],
        fatigue_df=frames["fatigue"],
        response_df=frames["responses"],
        response_summary={},
        reliability_df=frames["reliability"],
        reliability_summary={},
        drift_df=frames["drift"],
        drift_summary={},
        model_update_df=frames["model_update"],
        rl_summary={"recommended_threshold": 0.65},
        explanations_df=frames["explanations"],
        recommendations_df=frames["recommendations"],
        db_path=tmp_path / "missing.db",
    )

    assert REQUIRED_TOP_LEVEL_CATEGORIES.issubset(compiled.keys())
    assert compiled["simulation_only_note"]


def test_flatten_metrics_for_table_returns_expected_columns() -> None:
    metrics_dict = {
        "dataset": {"total_patients": 2},
        "alerts": {"total_raw_alerts": 3},
        "simulation_only_note": "Simulation only.",
    }
    flattened = metrics.flatten_metrics_for_table(metrics_dict)

    assert {"category", "metric", "value"}.issubset(flattened.columns)
    assert len(flattened) == 3


def test_database_readiness_metrics_return_expected_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE patients (patient_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO patients VALUES ('P001')")

    readiness = metrics.calculate_database_readiness_metrics(db_path)

    assert readiness["database_available"] is True
    assert readiness["database_table_count"] == 1
    assert readiness["database_total_rows_loaded"] == 1


def test_run_metrics_pipeline_saves_json_and_csv_outputs(tmp_path: Path) -> None:
    output_json = tmp_path / "project_metrics_summary.json"
    output_csv = tmp_path / "project_metrics_table.csv"

    result = metrics.run_metrics_pipeline(
        output_json_path=output_json,
        output_csv_path=output_csv,
    )

    assert output_json.exists()
    assert output_csv.exists()
    assert REQUIRED_TOP_LEVEL_CATEGORIES.issubset(result.keys())
    saved_table = pd.read_csv(output_csv)
    assert {"category", "metric", "value"}.issubset(saved_table.columns)


def test_missing_optional_files_do_not_crash_pipeline(tmp_path: Path) -> None:
    output_json = tmp_path / "missing_inputs_metrics.json"
    output_csv = tmp_path / "missing_inputs_metrics.csv"

    result = metrics.run_metrics_pipeline(
        processed_path=tmp_path / "missing_processed.csv",
        raw_alerts_path=tmp_path / "missing_alerts.csv",
        audited_path=tmp_path / "missing_audited.csv",
        fatigue_path=tmp_path / "missing_fatigue.csv",
        response_path=tmp_path / "missing_responses.csv",
        response_summary_path=tmp_path / "missing_response_summary.json",
        reliability_path=tmp_path / "missing_reliability.csv",
        reliability_summary_path=tmp_path / "missing_reliability_summary.json",
        drift_path=tmp_path / "missing_drift.csv",
        drift_summary_path=tmp_path / "missing_drift_summary.json",
        model_update_path=tmp_path / "missing_model_update.csv",
        rl_summary_path=tmp_path / "missing_rl_summary.json",
        explanations_path=tmp_path / "missing_explanations.csv",
        recommendations_path=tmp_path / "missing_recommendations.csv",
        db_path=tmp_path / "missing.db",
        output_json_path=output_json,
        output_csv_path=output_csv,
    )

    assert output_json.exists()
    assert output_csv.exists()
    assert result["dataset"]["total_patients"] == 0
    assert result["alerts"]["total_raw_alerts"] == 0


def test_saved_summary_contains_simulation_only_note(tmp_path: Path) -> None:
    summary = {"dataset": {}, "simulation_only_note": metrics.SIMULATION_ONLY_NOTE}
    output_path = metrics.save_metrics_summary(summary, tmp_path / "summary.json")

    with output_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)

    assert "simulation_only_note" in saved
    assert "clinical validation" in saved["simulation_only_note"]
