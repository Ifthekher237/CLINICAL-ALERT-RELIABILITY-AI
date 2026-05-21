"""Simulation-only RL threshold agent for alert-threshold experiments.

Step 15 uses a small contextual-bandit style simulation to compare threshold
actions. It never updates deployed thresholds, overwrites model artifacts, or
controls medical decisions. The output is a human-review-oriented experiment
for a simulated healthcare AI engineering prototype.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_MODEL_UPDATE_PATH = Path("data/processed/model_update_simulation_results.csv")
DEFAULT_THRESHOLD_SUMMARY_PATH = Path("data/processed/threshold_update_summary.json")
DEFAULT_AUDITED_PATH = Path("data/processed/audited_alerts.csv")
DEFAULT_FATIGUE_PATH = Path("data/processed/fatigue_reduced_alerts.csv")
DEFAULT_RESPONSE_PATH = Path("data/processed/clinician_response_logs.csv")
DEFAULT_DRIFT_PATH = Path("data/processed/drift_detection_results.csv")
DEFAULT_RELIABILITY_PATH = Path("data/processed/reliability_monitoring_results.csv")
DEFAULT_RESULTS_PATH = Path("data/processed/rl_threshold_simulation_results.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/rl_threshold_policy_summary.json")

MIN_THRESHOLD = 0.40
MAX_THRESHOLD = 0.90
THRESHOLD_STEP = 0.03

ALLOWED_ACTIONS = {
    "lower_threshold",
    "keep_threshold",
    "raise_threshold",
}

REQUIRED_MODEL_UPDATE_COLUMNS = [
    "current_risk_threshold",
    "proposed_risk_threshold",
    "false_alert_rate",
    "ignored_alert_rate",
    "delayed_alert_rate",
    "useful_alert_rate",
    "average_drift_score",
    "severe_drift_count",
    "human_review_required",
]

REQUIRED_AUDITED_COLUMNS = [
    "alert_id",
    "timestamp",
    "risk_score",
    "actionability_score",
    "false_positive_likelihood",
]

REQUIRED_FATIGUE_COLUMNS = [
    "alert_id",
    "timestamp",
    "critical_flag",
    "fatigue_action",
    "final_alert_status",
]

REQUIRED_RESPONSE_COLUMNS = [
    "response_id",
    "alert_id",
    "timestamp",
    "simulated_response",
    "response_time_minutes",
]

REQUIRED_DRIFT_COLUMNS = [
    "drift_score",
    "drift_status",
    "requires_review",
]

REQUIRED_RELIABILITY_COLUMNS = [
    "reliability_score",
    "safety_preservation_score",
    "ignored_alert_rate",
    "delayed_alert_rate",
    "average_response_time_minutes",
]

REQUIRED_SIMULATION_COLUMNS = [
    "episode",
    "current_threshold",
    "action",
    "proposed_threshold",
    "simulated_false_alert_rate",
    "simulated_ignored_rate",
    "simulated_delayed_rate",
    "simulated_useful_alert_rate",
    "simulated_critical_preservation_rate",
    "simulated_reliability_score",
    "reward",
    "safety_violation_flag",
    "simulation_note",
]

REQUIRED_POLICY_KEYS = {
    "recommended_action",
    "current_threshold",
    "recommended_threshold",
    "average_reward",
    "best_reward",
    "safety_violation_count",
    "human_review_required",
    "reason",
    "simulation_only_note",
}


def load_model_update_results(path: str | Path = DEFAULT_MODEL_UPDATE_PATH) -> pd.DataFrame:
    """Load Step 14 model-update simulation results."""
    df = _load_csv(path, "model update simulation results")
    _validate_columns(df, REQUIRED_MODEL_UPDATE_COLUMNS, "model update simulation results")
    return df


def load_threshold_summary(path: str | Path = DEFAULT_THRESHOLD_SUMMARY_PATH) -> dict[str, Any]:
    """Load Step 14 threshold update summary JSON."""
    input_path = _resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Threshold summary file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if not isinstance(summary, dict):
        raise ValueError("Threshold summary must be a JSON object.")
    if "threshold_update" not in summary or "feedback_summary" not in summary:
        raise ValueError("Threshold summary missing threshold_update or feedback_summary.")
    return summary


def load_audited_alerts(path: str | Path = DEFAULT_AUDITED_PATH) -> pd.DataFrame:
    """Load audited alerts from Step 9."""
    df = _load_csv(path, "audited alerts")
    _validate_columns(df, REQUIRED_AUDITED_COLUMNS, "audited alerts")
    return df


def load_fatigue_reduced_alerts(path: str | Path = DEFAULT_FATIGUE_PATH) -> pd.DataFrame:
    """Load fatigue-reduced alerts from Step 10."""
    df = _load_csv(path, "fatigue-reduced alerts")
    _validate_columns(df, REQUIRED_FATIGUE_COLUMNS, "fatigue-reduced alerts")
    return df


def load_response_logs(path: str | Path = DEFAULT_RESPONSE_PATH) -> pd.DataFrame:
    """Load simulated clinician response logs from Step 11."""
    df = _load_csv(path, "response logs")
    _validate_columns(df, REQUIRED_RESPONSE_COLUMNS, "response logs")
    return df


def load_drift_results(path: str | Path = DEFAULT_DRIFT_PATH) -> pd.DataFrame:
    """Load drift-detection results from Step 13."""
    df = _load_csv(path, "drift results")
    _validate_columns(df, REQUIRED_DRIFT_COLUMNS, "drift results")
    return df


def load_reliability_results(path: str | Path = DEFAULT_RELIABILITY_PATH) -> pd.DataFrame:
    """Load reliability-monitoring results from Step 12."""
    df = _load_csv(path, "reliability results")
    _validate_columns(df, REQUIRED_RELIABILITY_COLUMNS, "reliability results")
    return df


def build_context_state(
    model_update_df: pd.DataFrame,
    threshold_summary: dict[str, Any],
    audited_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
    response_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build the static context for the threshold-agent simulation."""
    _validate_columns(model_update_df, REQUIRED_MODEL_UPDATE_COLUMNS, "model update results")
    _validate_columns(audited_df, REQUIRED_AUDITED_COLUMNS, "audited alerts")
    _validate_columns(fatigue_df, REQUIRED_FATIGUE_COLUMNS, "fatigue-reduced alerts")
    _validate_columns(response_df, REQUIRED_RESPONSE_COLUMNS, "response logs")
    _validate_columns(drift_df, REQUIRED_DRIFT_COLUMNS, "drift results")
    _validate_columns(reliability_df, REQUIRED_RELIABILITY_COLUMNS, "reliability results")

    update_row = model_update_df.iloc[-1]
    feedback_summary = threshold_summary.get("feedback_summary", {})
    threshold_update = threshold_summary.get("threshold_update", {})

    response_values = response_df["simulated_response"].astype(str).str.lower()
    response_count = max(len(response_df), 1)
    false_alert_rate = _prefer_metric(
        update_row.get("false_alert_rate"),
        (response_values == "marked_false").sum() / response_count,
    )
    ignored_alert_rate = _prefer_metric(
        update_row.get("ignored_alert_rate"),
        (response_values == "ignored").sum() / response_count,
    )
    delayed_alert_rate = _prefer_metric(
        update_row.get("delayed_alert_rate"),
        (response_values == "delayed").sum() / response_count,
    )
    useful_alert_rate = _prefer_metric(
        update_row.get("useful_alert_rate"),
        response_values.isin(["marked_useful", "accepted", "escalated"]).sum() / response_count,
    )

    reduced_status = fatigue_df["final_alert_status"].astype(str).str.lower().isin(
        ["grouped", "delayed", "priority_downgraded"]
    )
    critical_mask = fatigue_df["critical_flag"].apply(_coerce_bool)
    critical_count = int(critical_mask.sum())
    if critical_count:
        critical_preservation_rate = (
            fatigue_df.loc[critical_mask, "final_alert_status"]
            .astype(str)
            .str.lower()
            .eq("active")
            .mean()
        )
    else:
        critical_preservation_rate = 1.0

    drift_scores = pd.to_numeric(drift_df["drift_score"], errors="coerce").fillna(0.0)
    reliability_scores = pd.to_numeric(
        reliability_df["reliability_score"],
        errors="coerce",
    ).fillna(0.0)
    safety_scores = pd.to_numeric(
        reliability_df["safety_preservation_score"],
        errors="coerce",
    ).fillna(1.0)
    response_times = pd.to_numeric(
        response_df["response_time_minutes"],
        errors="coerce",
    ).fillna(0.0)

    context = {
        "current_threshold": _clamp_threshold(
            threshold_update.get("current_risk_threshold", update_row["current_risk_threshold"])
        ),
        "step_14_proposed_threshold": _clamp_threshold(
            threshold_update.get("proposed_risk_threshold", update_row["proposed_risk_threshold"])
        ),
        "false_alert_rate": _round_rate(false_alert_rate),
        "ignored_alert_rate": _round_rate(ignored_alert_rate),
        "delayed_alert_rate": _round_rate(delayed_alert_rate),
        "useful_alert_rate": _round_rate(useful_alert_rate),
        "severe_drift_count": int(
            _prefer_metric(update_row.get("severe_drift_count"), (drift_df["drift_status"] == "severe_shift").sum())
        ),
        "average_drift_score": _round_rate(
            _prefer_metric(update_row.get("average_drift_score"), drift_scores.mean())
        ),
        "reliability_score": _round_rate(reliability_scores.mean()),
        "minimum_reliability_score": _round_rate(reliability_scores.min()),
        "alert_reduction_rate": _round_rate(reduced_status.sum() / max(len(fatigue_df), 1)),
        "critical_preservation_rate": _round_rate(
            min(float(critical_preservation_rate), float(safety_scores.min()))
        ),
        "average_response_time_minutes": _round_rate(response_times.mean()),
        "average_actionability_score": _round_rate(
            pd.to_numeric(audited_df["actionability_score"], errors="coerce").fillna(0.0).mean()
        ),
        "average_false_positive_likelihood": _round_rate(
            pd.to_numeric(
                audited_df["false_positive_likelihood"],
                errors="coerce",
            ).fillna(0.0).mean()
        ),
        "human_review_required": bool(
            _coerce_bool(update_row.get("human_review_required"))
            or _coerce_bool(threshold_summary.get("human_review_required"))
            or _coerce_bool(threshold_update.get("human_review_required"))
        ),
        "step_14_deployment_recommendation": str(
            threshold_summary.get("deployment_recommendation", update_row.get("deployment_recommendation", ""))
        ),
        "drift_review_rate": _round_rate(drift_df["requires_review"].apply(_coerce_bool).mean()),
        "total_episodes_recommended": 75,
        "simulation_only_note": (
            "Context is built from simulated monitoring artifacts; no real threshold is changed."
        ),
    }

    context["high_drift_or_review_context"] = bool(
        context["human_review_required"]
        or context["severe_drift_count"] > 0
        or context["average_drift_score"] >= 0.35
    )
    return context


def choose_action(
    context: dict[str, Any],
    epsilon: float,
    random_state: np.random.Generator | int | None = None,
) -> str:
    """Choose a threshold action with conservative epsilon-greedy logic."""
    rng = _as_rng(random_state)
    epsilon = min(max(float(epsilon), 0.0), 1.0)
    if rng.random() < epsilon:
        probabilities = _exploration_probabilities(context)
        return str(rng.choice(list(probabilities.keys()), p=list(probabilities.values())))

    scores = _action_scores(context)
    return max(scores, key=scores.get)


def simulate_threshold_effect(context: dict[str, Any], action: str) -> dict[str, Any]:
    """Simulate the metric effect of one small threshold action."""
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Invalid threshold action: {action}")

    current_threshold = float(context["current_threshold"])
    threshold_delta = {
        "lower_threshold": -THRESHOLD_STEP,
        "keep_threshold": 0.0,
        "raise_threshold": THRESHOLD_STEP,
    }[action]
    proposed_threshold = _clamp_threshold(current_threshold + threshold_delta)
    actual_delta = proposed_threshold - current_threshold

    false_rate = float(context["false_alert_rate"])
    ignored_rate = float(context["ignored_alert_rate"])
    delayed_rate = float(context["delayed_alert_rate"])
    useful_rate = float(context["useful_alert_rate"])
    critical_rate = float(context["critical_preservation_rate"])
    reliability = float(context["reliability_score"])
    burden = _workflow_burden(context)

    if action == "raise_threshold":
        simulated_false = false_rate - 0.018
        simulated_ignored = ignored_rate - 0.006
        simulated_delayed = delayed_rate - 0.005
        simulated_useful = useful_rate - (0.010 + 0.020 * min(useful_rate, 1.0))
        simulated_critical = critical_rate - (0.006 if critical_rate >= 0.995 else 0.020)
    elif action == "lower_threshold":
        simulated_false = false_rate + 0.025 + 0.020 * burden
        simulated_ignored = ignored_rate + 0.010 + 0.015 * burden
        simulated_delayed = delayed_rate + 0.012 + 0.012 * burden
        simulated_useful = useful_rate + 0.015
        simulated_critical = critical_rate - (0.015 if critical_rate < 0.995 else 0.002)
    else:
        simulated_false = false_rate
        simulated_ignored = ignored_rate
        simulated_delayed = delayed_rate
        simulated_useful = useful_rate
        simulated_critical = critical_rate

    severe_review_context = bool(context.get("high_drift_or_review_context", False))
    safety_violation = bool(
        (severe_review_context and action != "keep_threshold")
        or (action == "lower_threshold" and burden >= 0.18)
        or (action == "lower_threshold" and critical_rate < 0.995)
        or (action == "raise_threshold" and simulated_critical < 0.985)
    )

    reliability_delta = (
        0.30 * (false_rate - simulated_false)
        + 0.22 * (ignored_rate - simulated_ignored)
        + 0.18 * (delayed_rate - simulated_delayed)
        + 0.25 * (simulated_critical - critical_rate)
        + 0.12 * (simulated_useful - useful_rate)
    )
    if safety_violation:
        reliability_delta -= 0.04
    if severe_review_context and action == "keep_threshold":
        reliability_delta += 0.01

    return {
        "current_threshold": _round_rate(current_threshold),
        "proposed_threshold": _round_rate(proposed_threshold),
        "threshold_delta": _round_rate(actual_delta),
        "simulated_false_alert_rate": _clamp_rate(simulated_false),
        "simulated_ignored_rate": _clamp_rate(simulated_ignored),
        "simulated_delayed_rate": _clamp_rate(simulated_delayed),
        "simulated_useful_alert_rate": _clamp_rate(simulated_useful),
        "simulated_critical_preservation_rate": _clamp_rate(simulated_critical),
        "simulated_reliability_score": _clamp_rate(reliability + reliability_delta),
        "safety_violation_flag": safety_violation,
    }


def calculate_reward(
    context: dict[str, Any],
    simulated_effect: dict[str, Any],
    action: str,
) -> float:
    """Calculate conservative reward for one simulated action."""
    false_reduction = float(context["false_alert_rate"]) - simulated_effect["simulated_false_alert_rate"]
    ignored_reduction = float(context["ignored_alert_rate"]) - simulated_effect["simulated_ignored_rate"]
    delayed_reduction = float(context["delayed_alert_rate"]) - simulated_effect["simulated_delayed_rate"]
    useful_change = simulated_effect["simulated_useful_alert_rate"] - float(context["useful_alert_rate"])
    critical_change = (
        simulated_effect["simulated_critical_preservation_rate"]
        - float(context["critical_preservation_rate"])
    )
    reliability_change = (
        simulated_effect["simulated_reliability_score"] - float(context["reliability_score"])
    )

    reward = (
        1.70 * false_reduction
        + 1.20 * ignored_reduction
        + 1.10 * delayed_reduction
        + 0.90 * useful_change
        + 4.00 * critical_change
        + 1.60 * reliability_change
    )

    if action == "keep_threshold" and bool(context.get("high_drift_or_review_context", False)):
        reward += 0.20
    if action == "raise_threshold" and float(context.get("useful_alert_rate", 0.0)) > 0.80:
        reward -= 0.05
    if action == "lower_threshold" and _workflow_burden(context) > 0.16:
        reward -= 0.20
    if simulated_effect["safety_violation_flag"]:
        reward -= 1.00

    return _round_reward(reward)


def run_threshold_agent_simulation(
    context: dict[str, Any],
    episodes: int = 75,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a static-context bandit simulation over threshold actions."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for episode in range(1, episodes + 1):
        epsilon = max(0.05, 0.28 * (1.0 - (episode - 1) / max(episodes, 1)))
        action = choose_action(context, epsilon=epsilon, random_state=rng)
        effect = simulate_threshold_effect(context, action)
        reward = calculate_reward(context, effect, action)
        records.append(
            {
                "episode": episode,
                "current_threshold": effect["current_threshold"],
                "action": action,
                "proposed_threshold": effect["proposed_threshold"],
                "simulated_false_alert_rate": effect["simulated_false_alert_rate"],
                "simulated_ignored_rate": effect["simulated_ignored_rate"],
                "simulated_delayed_rate": effect["simulated_delayed_rate"],
                "simulated_useful_alert_rate": effect["simulated_useful_alert_rate"],
                "simulated_critical_preservation_rate": effect[
                    "simulated_critical_preservation_rate"
                ],
                "simulated_reliability_score": effect["simulated_reliability_score"],
                "reward": reward,
                "safety_violation_flag": bool(effect["safety_violation_flag"]),
                "simulation_note": (
                    "Simulation-only threshold experiment; no real threshold or model file changed."
                ),
            }
        )

    return pd.DataFrame(records, columns=REQUIRED_SIMULATION_COLUMNS)


def summarize_policy(simulation_df: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    """Summarize the safest recommended simulated threshold policy."""
    _validate_columns(simulation_df, REQUIRED_SIMULATION_COLUMNS, "RL simulation results")
    safe_df = simulation_df[~simulation_df["safety_violation_flag"].astype(bool)]
    if safe_df.empty:
        best_row = simulation_df.sort_values("reward", ascending=False).iloc[0]
    else:
        by_action = (
            safe_df.groupby("action", as_index=False)
            .agg(
                average_reward=("reward", "mean"),
                best_reward=("reward", "max"),
                proposed_threshold=("proposed_threshold", "median"),
            )
            .sort_values(["average_reward", "best_reward"], ascending=False)
        )
        best_action = _safety_first_recommended_action(by_action, context)
        best_row = by_action[by_action["action"] == best_action].iloc[0]

    recommended_action = str(best_row["action"])
    recommended_threshold = _clamp_threshold(float(best_row["proposed_threshold"]))
    average_reward = _round_reward(simulation_df["reward"].mean())
    best_reward = _round_reward(simulation_df["reward"].max())
    safety_violation_count = int(simulation_df["safety_violation_flag"].astype(bool).sum())
    human_review_required = bool(
        context.get("human_review_required", False)
        or context.get("severe_drift_count", 0) > 0
        or safety_violation_count > 0
    )

    return {
        "recommended_action": recommended_action,
        "current_threshold": _round_rate(context["current_threshold"]),
        "recommended_threshold": recommended_threshold,
        "average_reward": average_reward,
        "best_reward": best_reward,
        "safety_violation_count": safety_violation_count,
        "human_review_required": human_review_required,
        "reason": _policy_reason(recommended_action, context, safety_violation_count),
        "action_reward_summary": _action_reward_summary(simulation_df),
        "simulation_only_note": (
            "Simulation-only recommendation for review; it must not update deployed thresholds "
            "or be used for medical decision-making."
        ),
    }


def save_simulation_results(df: pd.DataFrame, path: str | Path) -> Path:
    """Save RL threshold simulation episodes to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_policy_summary(summary: dict[str, Any], path: str | Path) -> Path:
    """Save RL policy summary to JSON."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def run_rl_threshold_pipeline(
    model_update_path: str | Path = DEFAULT_MODEL_UPDATE_PATH,
    threshold_summary_path: str | Path = DEFAULT_THRESHOLD_SUMMARY_PATH,
    audited_path: str | Path = DEFAULT_AUDITED_PATH,
    fatigue_path: str | Path = DEFAULT_FATIGUE_PATH,
    response_path: str | Path = DEFAULT_RESPONSE_PATH,
    drift_path: str | Path = DEFAULT_DRIFT_PATH,
    reliability_path: str | Path = DEFAULT_RELIABILITY_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> pd.DataFrame:
    """Run the full Step 15 simulation-only threshold-agent pipeline."""
    model_update_df = load_model_update_results(model_update_path)
    threshold_summary = load_threshold_summary(threshold_summary_path)
    audited_df = load_audited_alerts(audited_path)
    fatigue_df = load_fatigue_reduced_alerts(fatigue_path)
    response_df = load_response_logs(response_path)
    drift_df = load_drift_results(drift_path)
    reliability_df = load_reliability_results(reliability_path)

    context = build_context_state(
        model_update_df,
        threshold_summary,
        audited_df,
        fatigue_df,
        response_df,
        drift_df,
        reliability_df,
    )
    simulation_df = run_threshold_agent_simulation(
        context,
        episodes=int(context.get("total_episodes_recommended", 75)),
        seed=42,
    )
    policy_summary = summarize_policy(simulation_df, context)

    results_saved_path = save_simulation_results(simulation_df, results_path)
    summary_saved_path = save_policy_summary(policy_summary, summary_path)
    simulation_df.attrs["context"] = context
    simulation_df.attrs["policy_summary"] = policy_summary
    simulation_df.attrs["results_path"] = str(results_saved_path)
    simulation_df.attrs["summary_path"] = str(summary_saved_path)
    return simulation_df


class ThresholdAgent:
    """Small wrapper for the simulation-only threshold policy functions."""

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        self.context = context or {}

    def choose_action(self, epsilon: float = 0.0, random_state: Any = None) -> str:
        """Choose a simulated action; this does not change any real threshold."""
        return choose_action(self.context, epsilon=epsilon, random_state=random_state)


def _action_scores(context: dict[str, Any]) -> dict[str, float]:
    false_rate = float(context.get("false_alert_rate", 0.0))
    useful_rate = float(context.get("useful_alert_rate", 0.0))
    delayed_rate = float(context.get("delayed_alert_rate", 0.0))
    critical_rate = float(context.get("critical_preservation_rate", 1.0))
    burden = _workflow_burden(context)
    high_drift_or_review = bool(context.get("high_drift_or_review_context", False))

    scores = {
        "keep_threshold": 0.55,
        "raise_threshold": 0.25 + 0.80 * false_rate + 0.35 * burden,
        "lower_threshold": 0.18 + 0.25 * delayed_rate + 0.30 * useful_rate,
    }

    if useful_rate > 0.80:
        scores["raise_threshold"] -= 0.22
    if critical_rate < 1.0:
        scores["raise_threshold"] -= 0.25
        scores["lower_threshold"] -= 0.35
    if burden > 0.16:
        scores["lower_threshold"] -= 0.25
        scores["keep_threshold"] += 0.08
    if high_drift_or_review:
        scores["keep_threshold"] += 0.85
        scores["raise_threshold"] -= 0.70
        scores["lower_threshold"] -= 0.90

    return scores


def _exploration_probabilities(context: dict[str, Any]) -> dict[str, float]:
    if bool(context.get("high_drift_or_review_context", False)):
        return {
            "keep_threshold": 0.78,
            "raise_threshold": 0.12,
            "lower_threshold": 0.10,
        }
    return {
        "keep_threshold": 0.45,
        "raise_threshold": 0.30,
        "lower_threshold": 0.25,
    }


def _safety_first_recommended_action(by_action: pd.DataFrame, context: dict[str, Any]) -> str:
    if bool(context.get("high_drift_or_review_context", False)):
        return "keep_threshold"
    return str(by_action.iloc[0]["action"])


def _policy_reason(action: str, context: dict[str, Any], safety_violation_count: int) -> str:
    if bool(context.get("high_drift_or_review_context", False)):
        return (
            "Severe drift or Step 14 human-review requirement is present, so the safest "
            "simulation policy is to keep the threshold and require human review."
        )
    if action == "raise_threshold":
        return "False-alert or workflow burden suggests a small simulated threshold increase for review."
    if action == "lower_threshold":
        return "Useful-alert preservation suggests a cautious simulated threshold decrease for review."
    if safety_violation_count:
        return "Unsafe exploratory actions occurred, so the summary keeps the current threshold."
    return "Rewards did not justify changing the simulated threshold."


def _action_reward_summary(simulation_df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for action, group in simulation_df.groupby("action"):
        summary[str(action)] = {
            "episodes": int(len(group)),
            "average_reward": _round_reward(group["reward"].mean()),
            "best_reward": _round_reward(group["reward"].max()),
            "safety_violations": int(group["safety_violation_flag"].astype(bool).sum()),
        }
    return summary


def _workflow_burden(context: dict[str, Any]) -> float:
    return _clamp_rate(
        0.35 * float(context.get("false_alert_rate", 0.0))
        + 0.25 * float(context.get("ignored_alert_rate", 0.0))
        + 0.25 * float(context.get("delayed_alert_rate", 0.0))
        + 0.15 * float(context.get("alert_reduction_rate", 0.0))
    )


def _load_csv(path: str | Path, label: str) -> pd.DataFrame:
    input_path = _resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{label} file not found: {input_path}")
    return pd.read_csv(input_path)


def _validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _as_rng(random_state: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(int(random_state))


def _prefer_metric(primary: Any, fallback: Any) -> float:
    try:
        value = float(primary)
    except (TypeError, ValueError):
        value = float(fallback)
    if pd.isna(value):
        value = float(fallback)
    return value


def _clamp_threshold(value: Any) -> float:
    return _round_rate(min(max(float(value), MIN_THRESHOLD), MAX_THRESHOLD))


def _clamp_rate(value: Any) -> float:
    return _round_rate(min(max(float(value), 0.0), 1.0))


def _round_rate(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value), 4)


def _round_reward(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value), 5)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _project_root() / candidate


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    results = run_rl_threshold_pipeline()
    summary = results.attrs["policy_summary"]

    print(f"Current threshold: {summary['current_threshold']:.4f}")
    print(f"Recommended action: {summary['recommended_action']}")
    print(f"Recommended threshold: {summary['recommended_threshold']:.4f}")
    print(f"Average reward: {summary['average_reward']:.5f}")
    print(f"Safety violation count: {summary['safety_violation_count']}")
    print(f"Human review required: {summary['human_review_required']}")
    print(f"\nSaved RL threshold simulation to {results.attrs['results_path']}")
    print(f"Saved RL policy summary to {results.attrs['summary_path']}")
