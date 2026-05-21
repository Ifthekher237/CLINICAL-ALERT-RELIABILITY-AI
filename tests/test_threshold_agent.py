"""Focused tests for Step 15 simulation-only RL threshold agent."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.rl import threshold_agent


def _context_from_real_artifacts() -> dict:
    model_update_df = threshold_agent.load_model_update_results(
        "data/processed/model_update_simulation_results.csv"
    )
    threshold_summary = threshold_agent.load_threshold_summary(
        "data/processed/threshold_update_summary.json"
    )
    audited_df = threshold_agent.load_audited_alerts("data/processed/audited_alerts.csv")
    fatigue_df = threshold_agent.load_fatigue_reduced_alerts(
        "data/processed/fatigue_reduced_alerts.csv"
    )
    response_df = threshold_agent.load_response_logs(
        "data/processed/clinician_response_logs.csv"
    )
    drift_df = threshold_agent.load_drift_results(
        "data/processed/drift_detection_results.csv"
    )
    reliability_df = threshold_agent.load_reliability_results(
        "data/processed/reliability_monitoring_results.csv"
    )
    return threshold_agent.build_context_state(
        model_update_df,
        threshold_summary,
        audited_df,
        fatigue_df,
        response_df,
        drift_df,
        reliability_df,
    )


def test_input_files_can_be_loaded() -> None:
    model_update_df = threshold_agent.load_model_update_results(
        "data/processed/model_update_simulation_results.csv"
    )
    threshold_summary = threshold_agent.load_threshold_summary(
        "data/processed/threshold_update_summary.json"
    )
    audited_df = threshold_agent.load_audited_alerts("data/processed/audited_alerts.csv")
    fatigue_df = threshold_agent.load_fatigue_reduced_alerts(
        "data/processed/fatigue_reduced_alerts.csv"
    )
    response_df = threshold_agent.load_response_logs(
        "data/processed/clinician_response_logs.csv"
    )
    drift_df = threshold_agent.load_drift_results(
        "data/processed/drift_detection_results.csv"
    )
    reliability_df = threshold_agent.load_reliability_results(
        "data/processed/reliability_monitoring_results.csv"
    )

    assert not model_update_df.empty
    assert "threshold_update" in threshold_summary
    assert not audited_df.empty
    assert not fatigue_df.empty
    assert not response_df.empty
    assert not drift_df.empty
    assert not reliability_df.empty


def test_context_state_contains_required_metrics() -> None:
    context = _context_from_real_artifacts()
    required_metrics = {
        "false_alert_rate",
        "ignored_alert_rate",
        "delayed_alert_rate",
        "useful_alert_rate",
        "severe_drift_count",
        "average_drift_score",
        "reliability_score",
        "alert_reduction_rate",
        "critical_preservation_rate",
        "average_response_time_minutes",
        "human_review_required",
    }

    assert required_metrics.issubset(context)
    assert 0.40 <= context["current_threshold"] <= 0.90


def test_action_values_are_valid() -> None:
    context = _context_from_real_artifacts()
    action = threshold_agent.choose_action(context, epsilon=0.0, random_state=42)

    assert action in threshold_agent.ALLOWED_ACTIONS


def test_proposed_threshold_stays_between_bounds() -> None:
    context = _context_from_real_artifacts()
    for action in threshold_agent.ALLOWED_ACTIONS:
        effect = threshold_agent.simulate_threshold_effect(context, action)
        assert 0.40 <= effect["proposed_threshold"] <= 0.90


def test_simulation_results_contain_required_columns_and_rewards() -> None:
    context = _context_from_real_artifacts()
    simulation_df = threshold_agent.run_threshold_agent_simulation(
        context,
        episodes=15,
        seed=42,
    )

    assert set(threshold_agent.REQUIRED_SIMULATION_COLUMNS).issubset(simulation_df.columns)
    assert set(simulation_df["action"]).issubset(threshold_agent.ALLOWED_ACTIONS)
    assert pd.api.types.is_numeric_dtype(simulation_df["reward"])
    assert simulation_df["safety_violation_flag"].map(lambda value: isinstance(value, bool)).all()


def test_policy_summary_contains_required_keys_and_simulation_note() -> None:
    context = _context_from_real_artifacts()
    simulation_df = threshold_agent.run_threshold_agent_simulation(
        context,
        episodes=15,
        seed=42,
    )
    summary = threshold_agent.summarize_policy(simulation_df, context)

    assert threshold_agent.REQUIRED_POLICY_KEYS.issubset(summary)
    assert summary["recommended_action"] in threshold_agent.ALLOWED_ACTIONS
    assert 0.40 <= summary["recommended_threshold"] <= 0.90
    assert "simulation" in summary["simulation_only_note"].lower()


def test_output_csv_is_saved(tmp_path: Path) -> None:
    context = _context_from_real_artifacts()
    simulation_df = threshold_agent.run_threshold_agent_simulation(
        context,
        episodes=10,
        seed=42,
    )
    output_path = threshold_agent.save_simulation_results(
        simulation_df,
        tmp_path / "rl_threshold_simulation_results.csv",
    )

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert set(threshold_agent.REQUIRED_SIMULATION_COLUMNS).issubset(saved.columns)


def test_output_json_is_saved(tmp_path: Path) -> None:
    summary = {
        "recommended_action": "keep_threshold",
        "current_threshold": 0.65,
        "recommended_threshold": 0.65,
        "average_reward": 0.2,
        "best_reward": 0.3,
        "safety_violation_count": 0,
        "human_review_required": True,
        "reason": "Simulation-only review.",
        "simulation_only_note": "Simulation only.",
    }
    output_path = threshold_agent.save_policy_summary(
        summary,
        tmp_path / "rl_threshold_policy_summary.json",
    )

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["recommended_action"] == "keep_threshold"


def test_model_files_are_not_overwritten(tmp_path: Path) -> None:
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

    threshold_agent.run_rl_threshold_pipeline(
        model_update_path="data/processed/model_update_simulation_results.csv",
        threshold_summary_path="data/processed/threshold_update_summary.json",
        audited_path="data/processed/audited_alerts.csv",
        fatigue_path="data/processed/fatigue_reduced_alerts.csv",
        response_path="data/processed/clinician_response_logs.csv",
        drift_path="data/processed/drift_detection_results.csv",
        reliability_path="data/processed/reliability_monitoring_results.csv",
        results_path=tmp_path / "rl_threshold_simulation_results.csv",
        summary_path=tmp_path / "rl_threshold_policy_summary.json",
    )

    after = {
        path: path.stat().st_mtime_ns
        for path in model_paths
        if path.exists()
    }
    assert after == before


def test_pipeline_saves_outputs(tmp_path: Path) -> None:
    results_path = tmp_path / "rl_threshold_simulation_results.csv"
    summary_path = tmp_path / "rl_threshold_policy_summary.json"
    simulation_df = threshold_agent.run_rl_threshold_pipeline(
        model_update_path="data/processed/model_update_simulation_results.csv",
        threshold_summary_path="data/processed/threshold_update_summary.json",
        audited_path="data/processed/audited_alerts.csv",
        fatigue_path="data/processed/fatigue_reduced_alerts.csv",
        response_path="data/processed/clinician_response_logs.csv",
        drift_path="data/processed/drift_detection_results.csv",
        reliability_path="data/processed/reliability_monitoring_results.csv",
        results_path=results_path,
        summary_path=summary_path,
    )

    assert results_path.exists()
    assert summary_path.exists()
    assert not simulation_df.empty
    assert simulation_df.attrs["policy_summary"]["recommended_action"] in threshold_agent.ALLOWED_ACTIONS
