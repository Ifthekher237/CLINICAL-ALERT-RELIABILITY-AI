"""Data integrity and artifact schema checks for Step 25."""

from __future__ import annotations

import pandas as pd

from src.monitoring import failure_mode_simulator, outcome_evaluator, scenario_tester
from testing_utils import assert_between_zero_and_one, assert_columns, assert_unique, load_csv, load_json


def test_required_columns_exist_in_core_outputs() -> None:
    required_by_path = {
        "data/processed/generated_alerts.csv": {
            "alert_id",
            "patient_id",
            "timestamp",
            "severity",
            "alert_type",
            "risk_score",
            "critical_flag",
        },
        "data/processed/fatigue_reduced_alerts.csv": {
            "alert_id",
            "final_alert_status",
            "fatigue_action",
            "critical_flag",
        },
        "data/processed/reliability_monitoring_results.csv": {
            "monitoring_window_id",
            "reliability_score",
            "reliability_status",
        },
        "data/processed/drift_detection_results.csv": {
            "drift_window_id",
            "drift_score",
            "drift_status",
            "requires_review",
        },
    }

    for path, columns in required_by_path.items():
        assert_columns(load_csv(path), columns)


def test_no_duplicate_primary_keys_in_new_monitoring_outputs() -> None:
    for path, key in [
        ("data/processed/outcome_effectiveness_results.csv", "outcome_eval_id"),
        ("data/processed/failure_mode_results.csv", "failure_event_id"),
        ("data/processed/scenario_test_results.csv", "scenario_test_id"),
        ("data/processed/project_metrics_table.csv", "metric"),
    ]:
        df = load_csv(path)
        if path.endswith("project_metrics_table.csv"):
            assert not df.duplicated(["category", "metric"]).any()
        else:
            assert_unique(df, key)


def test_numeric_scores_remain_within_valid_ranges() -> None:
    assert_between_zero_and_one(
        load_csv("data/processed/fatigue_reduced_alerts.csv"),
        ["risk_score", "actionability_score", "fatigue_risk_score", "false_positive_likelihood"],
    )
    assert_between_zero_and_one(
        load_csv("data/processed/outcome_effectiveness_results.csv"),
        ["outcome_effectiveness_score", "delayed_response_impact_score"],
    )
    assert_between_zero_and_one(
        load_csv("data/processed/failure_mode_results.csv"),
        [
            "alert_volume_impact",
            "clinician_burden_impact",
            "reliability_score_impact",
            "drift_risk_impact",
            "outcome_risk_impact",
        ],
    )
    assert_between_zero_and_one(
        load_csv("data/processed/scenario_test_results.csv"),
        [
            "ignored_alert_rate",
            "delayed_alert_rate",
            "reliability_score",
            "outcome_effectiveness_score",
        ],
    )


def test_json_summaries_contain_required_keys() -> None:
    required_keys = {
        "data/processed/project_metrics_summary.json": {"dataset", "alerts", "workflow", "simulation_only_note"},
        "data/processed/outcome_effectiveness_summary.json": {
            "total_evaluated_alerts",
            "useful_alert_rate",
            "average_outcome_effectiveness_score",
            "simulation_only_note",
        },
        "data/processed/failure_mode_summary.json": {
            "total_failure_events",
            "failure_mode_distribution",
            "safety_status_distribution",
            "simulation_only_note",
        },
        "data/processed/scenario_test_summary.json": {
            "total_scenarios",
            "overall_status_distribution",
            "safety_check_distribution",
            "simulation_only_note",
        },
    }

    for path, keys in required_keys.items():
        assert keys.issubset(load_json(path).keys())


def test_safe_loaders_and_empty_inputs_handle_missing_data_gracefully(tmp_path) -> None:
    assert outcome_evaluator.safe_load_csv(str(tmp_path / "missing.csv")).empty
    assert failure_mode_simulator.safe_load_json(str(tmp_path / "missing.json")) == {}

    empty_outcomes = outcome_evaluator.evaluate_alert_outcomes(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert set(outcome_evaluator.REQUIRED_OUTPUT_COLUMNS).issubset(empty_outcomes.columns)

    empty_failures = failure_mode_simulator.build_failure_mode_table(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
    )
    assert set(failure_mode_simulator.REQUIRED_OUTPUT_COLUMNS).issubset(empty_failures.columns)

    empty_scenarios = scenario_tester.build_scenario_results_table({})
    assert len(empty_scenarios) == 7
    assert set(scenario_tester.REQUIRED_OUTPUT_COLUMNS).issubset(empty_scenarios.columns)
