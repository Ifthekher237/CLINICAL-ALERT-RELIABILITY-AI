"""Generate the final Step 24 simulated evaluation report.

The report summarizes existing project artifacts only. It does not retrain
models, run alert logic, call LLM/Ollama, or claim clinical validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_ONLY_DISCLAIMER = (
    "This report describes a simulated healthcare AI engineering prototype. "
    "It uses synthetic/demo artifacts only, is not clinically validated, is not "
    "a medical device, and must not be used for real patient care."
)

DATA_PATHS = {
    "metrics_summary": "data/processed/project_metrics_summary.json",
    "metrics_table": "data/processed/project_metrics_table.csv",
    "generated_alerts": "data/processed/generated_alerts.csv",
    "fatigue_reduced_alerts": "data/processed/fatigue_reduced_alerts.csv",
    "clinician_response_summary": "data/processed/clinician_response_summary.json",
    "reliability_summary": "data/processed/reliability_summary.json",
    "drift_summary": "data/processed/drift_summary.json",
    "model_update_results": "data/processed/model_update_simulation_results.csv",
    "rl_policy_summary": "data/processed/rl_threshold_policy_summary.json",
    "alert_explanations": "data/processed/alert_explanations.csv",
    "action_recommendations": "data/processed/action_recommendations.csv",
    "outcome_effectiveness_summary": "data/processed/outcome_effectiveness_summary.json",
    "outcome_effectiveness_results": "data/processed/outcome_effectiveness_results.csv",
    "failure_mode_summary": "data/processed/failure_mode_summary.json",
    "failure_mode_results": "data/processed/failure_mode_results.csv",
    "scenario_test_summary": "data/processed/scenario_test_summary.json",
    "scenario_test_results": "data/processed/scenario_test_results.csv",
}


def safe_load_json(path: str) -> dict[str, Any]:
    """Load a JSON file safely; return an empty dict when unavailable."""
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def safe_load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file safely; return an empty DataFrame when unavailable."""
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def format_number(value: Any) -> str:
    """Format numbers for markdown report tables."""
    numeric = _safe_float(value)
    if abs(numeric - int(numeric)) < 1e-9:
        return f"{int(numeric):,}"
    return f"{numeric:,.4f}".rstrip("0").rstrip(".")


def format_percent(value: Any) -> str:
    """Format a rate as a percentage."""
    return f"{_safe_float(value) * 100:.2f}%"


def get_nested_metric(
    metrics: dict[str, Any],
    category: str,
    metric: str,
    default: Any = None,
) -> Any:
    """Read a nested metric with a safe default."""
    category_values = metrics.get(category, {})
    if not isinstance(category_values, dict):
        return default
    return category_values.get(metric, default)


def generate_project_overview_section(metrics: dict[str, Any]) -> str:
    return f"""## 1. Project Overview

`clinical-alert-reliability-ai` is a simulated healthcare AI engineering project that models an end-to-end clinical alert reliability workflow. It starts with synthetic patient-monitoring data, creates future-risk and anomaly signals, generates structured alerts, applies safety guardrails, audits alerts, reduces repeated low-value alert burden, simulates workflow responses, monitors reliability and drift, and produces safe explanation/action-support artifacts.

Clinical alert reliability matters because noisy, repeated, poorly calibrated, or unexplained alerts can increase workflow burden and make it harder to identify alerts that deserve timely human review. This project focuses on engineering reliability and transparency rather than clinical decision-making.

**Simulation boundary:** {SIMULATION_ONLY_DISCLAIMER}
"""


def generate_dataset_section(metrics: dict[str, Any]) -> str:
    rows = [
        ("Total patients", format_number(get_nested_metric(metrics, "dataset", "total_patients", 0))),
        ("Total vital rows", format_number(get_nested_metric(metrics, "dataset", "total_vital_rows", 0))),
        ("Missing data rate", format_percent(get_nested_metric(metrics, "dataset", "missing_data_rate", 0))),
        ("Sensor noise rate", format_percent(get_nested_metric(metrics, "dataset", "sensor_noise_rate", 0))),
        (
            "Deterioration event rate",
            format_percent(get_nested_metric(metrics, "dataset", "deterioration_event_rate", 0)),
        ),
    ]
    return f"""## 2. Dataset Summary

The dataset is synthetic patient-monitoring data designed to exercise reliability and alert-management logic. It includes simulated missingness, sensor noise, and deterioration events so the downstream modules can be evaluated under imperfect-data conditions.

{_markdown_table(["Metric", "Value"], rows)}
"""


def generate_alert_section(metrics: dict[str, Any], alerts_df: pd.DataFrame) -> str:
    severity_rows = _value_count_rows(alerts_df, "severity")
    severity_table = _markdown_table(["Severity", "Count"], severity_rows) if severity_rows else "No severity distribution was available."
    rows = [
        ("Total raw alerts", format_number(get_nested_metric(metrics, "alerts", "total_raw_alerts", 0))),
        ("Critical alert count", format_number(get_nested_metric(metrics, "alerts", "critical_alert_count", 0))),
    ]
    return f"""## 3. Alert Generation Summary

The alert generator converts simulated risk, anomaly, time-series, and current-instability signals into structured alert records. These alerts are engineering artifacts for later guardrail, audit, workflow, and monitoring steps.

{_markdown_table(["Metric", "Value"], rows)}

Severity distribution:

{severity_table}
"""


def generate_fatigue_section(metrics: dict[str, Any]) -> str:
    rows = [
        (
            "Active alerts after reduction",
            format_number(get_nested_metric(metrics, "alerts", "active_alerts_after_reduction", 0)),
        ),
        ("Alert reduction rate", format_percent(get_nested_metric(metrics, "alerts", "alert_reduction_rate", 0))),
        (
            "Critical preservation rate",
            format_percent(get_nested_metric(metrics, "alerts", "critical_preservation_rate", 0)),
        ),
        ("Grouped alerts", format_number(get_nested_metric(metrics, "audit_fatigue", "grouped_alert_count", 0))),
        ("Delayed alerts", format_number(get_nested_metric(metrics, "audit_fatigue", "delayed_alert_count", 0))),
        ("Downgraded alerts", format_number(get_nested_metric(metrics, "audit_fatigue", "downgraded_alert_count", 0))),
    ]
    return f"""## 4. Alert Auditing and Fatigue Reduction

The audit and fatigue-reduction layers label alerts for actionability, repetition, likely noise, and workflow burden. Repeated low-value alerts can be grouped, delayed, or downgraded in priority, but rows are retained for auditability.

{_markdown_table(["Metric", "Value"], rows)}

Critical alerts were preserved because the fatigue-reduction logic is safety-first: critical alerts, `critical_flag=True` alerts, immediate-priority alerts, and immediate-escalation alerts are retained as active rather than grouped away or delayed.
"""


def generate_workflow_section(metrics: dict[str, Any]) -> str:
    rows = [
        ("Total responses", format_number(get_nested_metric(metrics, "workflow", "total_responses", 0))),
        ("Ignored alert rate", format_percent(get_nested_metric(metrics, "workflow", "ignored_alert_rate", 0))),
        ("Delayed alert rate", format_percent(get_nested_metric(metrics, "workflow", "delayed_alert_rate", 0))),
        ("Escalation rate", format_percent(get_nested_metric(metrics, "workflow", "escalation_rate", 0))),
        (
            "Average response time minutes",
            format_number(get_nested_metric(metrics, "workflow", "average_response_time_minutes", 0)),
        ),
        (
            "Average clinician burden score",
            format_number(get_nested_metric(metrics, "workflow", "average_clinician_burden_score", 0)),
        ),
        (
            "Average perceived alert usefulness",
            format_number(get_nested_metric(metrics, "workflow", "average_perceived_alert_usefulness", 0)),
        ),
    ]
    return f"""## 5. Workflow Simulation Summary

The workflow simulation estimates how a care team might respond to fatigue-reduced alerts in a hospital-like monitoring queue. Response behavior is simulated from alert severity, safety priority, actionability, fatigue risk, false-positive likelihood, and time-of-day assumptions.

{_markdown_table(["Metric", "Value"], rows)}

These workflow values are useful for engineering evaluation of alert burden and escalation patterns, but they do not represent real clinical operations.
"""


def generate_reliability_section(metrics: dict[str, Any]) -> str:
    rows = [
        (
            "Average reliability score",
            format_number(get_nested_metric(metrics, "reliability", "average_reliability_score", 0)),
        ),
        ("Stable windows", format_number(get_nested_metric(metrics, "reliability", "stable_window_count", 0))),
        ("Watch windows", format_number(get_nested_metric(metrics, "reliability", "watch_window_count", 0))),
        ("Degraded windows", format_number(get_nested_metric(metrics, "reliability", "degraded_window_count", 0))),
        ("Unsafe review windows", format_number(get_nested_metric(metrics, "reliability", "unsafe_window_count", 0))),
        (
            "Windows requiring review",
            format_number(get_nested_metric(metrics, "reliability", "windows_requiring_review", 0)),
        ),
    ]
    return f"""## 6. Reliability Monitoring Summary

The reliability monitor evaluates the alert system over time windows using simulated false alerts, ignored alerts, delayed responses, alert volume, response time, and critical alert preservation.

{_markdown_table(["Metric", "Value"], rows)}

Review recommendations indicate where thresholds, workflow burden, or monitoring assumptions should be inspected by a human reviewer before any configuration change is considered.
"""


def generate_drift_section(metrics: dict[str, Any], drift_summary: dict[str, Any]) -> str:
    drift_type_rows = _dict_count_rows(drift_summary.get("drift_type_distribution", {}))
    drift_type_table = _markdown_table(["Drift type", "Count"], drift_type_rows) if drift_type_rows else "No drift type distribution was available."
    rows = [
        ("Average drift score", format_number(get_nested_metric(metrics, "drift", "average_drift_score", 0))),
        ("Severe drift count", format_number(get_nested_metric(metrics, "drift", "severe_drift_count", 0))),
        ("Moderate drift count", format_number(get_nested_metric(metrics, "drift", "moderate_drift_count", 0))),
        (
            "Drift checks requiring review",
            format_number(get_nested_metric(metrics, "drift", "drift_checks_requiring_review", 0)),
        ),
        ("Most common drift type", str(get_nested_metric(metrics, "drift", "most_common_drift_type", ""))),
    ]
    return f"""## 7. Drift Detection Summary

The drift detector compares earlier and later windows across patient vitals, alert behavior, clinician response behavior, alert volume, and reliability score. Severe drift is treated as a review signal because it can mean the simulated data distribution or workflow behavior has shifted enough that thresholds and model assumptions should be inspected before further updates.

{_markdown_table(["Metric", "Value"], rows)}

Drift type distribution:

{drift_type_table}

Severe drift does not automatically trigger retraining in this prototype. It triggers human review or retraining review because automatic model changes would be unsafe for a healthcare-style system, even in simulation.
"""


def generate_model_update_section(
    metrics: dict[str, Any],
    model_update_df: pd.DataFrame,
    rl_summary: dict[str, Any],
) -> str:
    last_update = model_update_df.iloc[-1].to_dict() if not model_update_df.empty else {}
    update_reason = str(last_update.get("update_reason", "No model update record was available."))
    expected_effect = str(last_update.get("expected_effect", "No expected-effect summary was available."))
    rows = [
        ("Current threshold", format_number(get_nested_metric(metrics, "model_update_rl", "current_threshold", 0))),
        ("Proposed threshold", format_number(get_nested_metric(metrics, "model_update_rl", "proposed_threshold", 0))),
        ("Threshold change", format_number(get_nested_metric(metrics, "model_update_rl", "threshold_change", 0))),
        (
            "Deployment recommendation",
            str(get_nested_metric(metrics, "model_update_rl", "deployment_recommendation", "")),
        ),
        ("RL recommended action", str(get_nested_metric(metrics, "model_update_rl", "rl_recommended_action", ""))),
        (
            "RL recommended threshold",
            format_number(get_nested_metric(metrics, "model_update_rl", "rl_recommended_threshold", 0)),
        ),
        (
            "RL safety violation count",
            format_number(get_nested_metric(metrics, "model_update_rl", "rl_safety_violation_count", 0)),
        ),
        (
            "Human review required",
            str(get_nested_metric(metrics, "model_update_rl", "human_review_required", False)),
        ),
    ]
    return f"""## 8. Model Update and RL Threshold Simulation

The model-update and RL-threshold modules simulate how feedback, drift, reliability, and workflow burden could inform threshold-review recommendations. They do not replace trained model files, deploy thresholds, or perform automatic retraining.

{_markdown_table(["Metric", "Value"], rows)}

Update reason: {update_reason}

Expected effect: {expected_effect}

Thresholds were not automatically changed because this prototype treats severe drift, safety violations, and human-review requirements as blockers for direct deployment. The RL policy summary also remains simulation-only: `{rl_summary.get("simulation_only_note", "No policy note available.")}`.
"""


def generate_llm_action_section(metrics: dict[str, Any]) -> str:
    total_actions = _safe_float(get_nested_metric(metrics, "llm_action", "total_action_recommendations", 0))
    immediate = _safe_float(get_nested_metric(metrics, "llm_action", "immediate_action_count", 0))
    urgent = _safe_float(get_nested_metric(metrics, "llm_action", "urgent_action_count", 0))
    routine_or_lower = max(total_actions - immediate - urgent, 0)
    rows = [
        ("Explanations generated", format_number(get_nested_metric(metrics, "llm_action", "total_explanations", 0))),
        (
            "Fallback explanation count",
            format_number(get_nested_metric(metrics, "llm_action", "fallback_explanation_count", 0)),
        ),
        ("Action recommendations", format_number(total_actions)),
        ("Immediate recommendations", format_number(immediate)),
        ("Urgent recommendations", format_number(urgent)),
        ("Routine/lower-priority recommendations", format_number(routine_or_lower)),
        ("RAG source coverage", format_percent(get_nested_metric(metrics, "llm_action", "rag_coverage_rate", 0))),
    ]
    return f"""## 9. LLM/RAG/Action Recommendation Summary

The explanation and recommendation layers produce support text and workflow-oriented next-step suggestions from existing system artifacts and local project rules. The system can use rule-based fallback explanations and local RAG source references, but it must remain limited to engineering support text and cannot provide medical advice or replace clinician judgment.

{_markdown_table(["Metric", "Value"], rows)}

The safety boundary is explicit: explanations are support text only, action recommendations are workflow recommendations only, and safety-sensitive alerts require human review.
"""


def generate_outcome_effectiveness_section(
    outcome_summary: dict[str, Any],
    outcome_df: pd.DataFrame,
) -> str:
    rows = [
        ("Total evaluated alerts", format_number(outcome_summary.get("total_evaluated_alerts", len(outcome_df)))),
        ("Useful alert rate", format_percent(outcome_summary.get("useful_alert_rate", 0))),
        ("Useless alert rate", format_percent(outcome_summary.get("useless_alert_rate", 0))),
        ("Action-to-outcome success rate", format_percent(outcome_summary.get("action_to_outcome_success_rate", 0))),
        (
            "Average outcome effectiveness score",
            format_number(outcome_summary.get("average_outcome_effectiveness_score", 0)),
        ),
        (
            "Average delayed response impact score",
            format_number(outcome_summary.get("average_delayed_response_impact_score", 0)),
        ),
        ("Improved count", format_number(outcome_summary.get("improved_count", 0))),
        ("Unchanged count", format_number(outcome_summary.get("unchanged_count", 0))),
        ("Worsened count", format_number(outcome_summary.get("worsened_count", 0))),
        ("Unknown count", format_number(outcome_summary.get("unknown_count", 0))),
    ]
    label_rows = _value_count_rows(outcome_df, "outcome_label")
    label_table = _markdown_table(["Outcome label", "Count"], label_rows) if label_rows else "No outcome-label distribution was available."
    return f"""## 10. Outcome Effectiveness Evaluation

The outcome-effectiveness evaluator links simulated alerts, workflow responses, action recommendations, and synthetic outcome fields. It checks whether an alert was associated with useful simulated workflow behavior, rather than only asking whether a model predicted risk.

{_markdown_table(["Metric", "Value"], rows)}

Outcome label distribution:

{label_table}

These results are simulated associations only. They are not proof of clinical effect, real patient benefit, or real-world safety.
"""


def generate_failure_mode_section(
    failure_summary: dict[str, Any],
    failure_df: pd.DataFrame,
) -> str:
    del failure_df
    failure_mode_rows = _dict_count_rows(failure_summary.get("failure_mode_distribution", {}))
    severity_rows = _dict_count_rows(failure_summary.get("severity_distribution", {}))
    safety_rows = _dict_count_rows(failure_summary.get("safety_status_distribution", {}))
    rows = [
        ("Total failure events", format_number(failure_summary.get("total_failure_events", 0))),
        ("Unsafe review required count", format_number(failure_summary.get("unsafe_review_required_count", 0))),
        ("Human review required rate", format_percent(failure_summary.get("human_review_required_rate", 0))),
        ("Average alert volume impact", format_number(failure_summary.get("average_alert_volume_impact", 0))),
        (
            "Average clinician burden impact",
            format_number(failure_summary.get("average_clinician_burden_impact", 0)),
        ),
        (
            "Average reliability impact",
            format_number(failure_summary.get("average_reliability_score_impact", 0)),
        ),
        ("Average drift risk impact", format_number(failure_summary.get("average_drift_risk_impact", 0))),
        ("Average outcome risk impact", format_number(failure_summary.get("average_outcome_risk_impact", 0))),
    ]
    return f"""## 11. Failure Mode Simulation

The failure-mode simulator stress-tests the alert reliability workflow under deployment-style engineering risks such as noisy sensor spikes, missing patient data, alert overload, repeated low-value alerts, delayed responses, confidence degradation, and distribution shift.

{_markdown_table(["Metric", "Value"], rows)}

Failure mode distribution:

{_markdown_table(["Failure mode", "Count"], failure_mode_rows) if failure_mode_rows else "No failure-mode distribution was available."}

Severity distribution:

{_markdown_table(["Severity", "Count"], severity_rows) if severity_rows else "No severity distribution was available."}

Safety status distribution:

{_markdown_table(["Safety status", "Count"], safety_rows) if safety_rows else "No safety-status distribution was available."}

The failure simulation shows where engineering review is needed under noisy data, workload pressure, delayed responses, drift, and degraded reliability. Mitigations remain workflow-focused: inspect sensor reliability, review thresholds, investigate drift patterns, and review workload assumptions.
"""


def generate_scenario_testing_section(
    scenario_summary: dict[str, Any],
    scenario_df: pd.DataFrame,
) -> str:
    del scenario_df
    scenario_rows = _dict_count_rows(scenario_summary.get("scenario_distribution", {}))
    status_rows = _dict_count_rows(scenario_summary.get("overall_status_distribution", {}))
    safety_rows = _dict_count_rows(scenario_summary.get("safety_check_distribution", {}))
    rows = [
        ("Total scenarios", format_number(scenario_summary.get("total_scenarios", 0))),
        ("Average reliability score", format_number(scenario_summary.get("average_reliability_score", 0))),
        ("Average ignored alert rate", format_percent(scenario_summary.get("average_ignored_alert_rate", 0))),
        ("Average delayed alert rate", format_percent(scenario_summary.get("average_delayed_alert_rate", 0))),
        (
            "Average outcome effectiveness score",
            format_number(scenario_summary.get("average_outcome_effectiveness_score", 0)),
        ),
        ("Failure mode trigger rate", format_percent(scenario_summary.get("failure_mode_trigger_rate", 0))),
        ("Human review required rate", format_percent(scenario_summary.get("human_review_required_rate", 0))),
        ("Passed safety checks", format_number(scenario_summary.get("passed_safety_checks", 0))),
        ("Warning safety checks", format_number(scenario_summary.get("warning_safety_checks", 0))),
        ("Failed safety checks", format_number(scenario_summary.get("failed_safety_checks", 0))),
    ]
    return f"""## 12. Scenario Testing Summary

Scenario testing evaluates the system across deployment-style patient-monitoring situations: stable monitoring, gradual deterioration, sudden critical events, noisy false alarms, repeated low-risk alerts, missing-data episodes, and high-volume workload stress. These tests summarize existing artifacts rather than rerunning the full pipeline.

{_markdown_table(["Metric", "Value"], rows)}

Scenario distribution:

{_markdown_table(["Scenario category", "Count"], scenario_rows) if scenario_rows else "No scenario distribution was available."}

Overall status distribution:

{_markdown_table(["Overall status", "Count"], status_rows) if status_rows else "No scenario status distribution was available."}

Safety check distribution:

{_markdown_table(["Safety check", "Count"], safety_rows) if safety_rows else "No safety check distribution was available."}

The scenario tests check whether safety rules continue to hold across stable, deteriorating, critical, noisy, missing-data, repeated-alert, and overload conditions.
"""


def generate_deployment_readiness_discussion_section(
    outcome_summary: dict[str, Any],
    failure_summary: dict[str, Any],
    scenario_summary: dict[str, Any],
) -> str:
    return f"""## 13. Real-World Deployment Readiness Discussion

This project now evaluates beyond prediction accuracy. It checks simulated alert-to-outcome association across {format_number(outcome_summary.get("total_evaluated_alerts", 0))} evaluated alerts, tests {format_number(failure_summary.get("total_failure_events", 0))} simulated failure events, and summarizes {format_number(scenario_summary.get("total_scenarios", 0))} deployment-style workflow scenarios.

That makes the portfolio stronger from an engineering perspective because it shows monitoring, safety review, alert fatigue, drift, workflow burden, and failure behavior as connected system concerns. The project still remains a simulated prototype: the data, workflow responses, outcomes, failure modes, and scenario tests are synthetic approximations and are not evidence of clinical validation or real-world deployment readiness.

Human review remains central whenever critical alerts, severe drift, high uncertainty, workload overload, or unsafe-review-required failure states appear.
"""


def generate_key_findings_section(
    metrics: dict[str, Any],
    outcome_summary: dict[str, Any] | None = None,
    failure_summary: dict[str, Any] | None = None,
    scenario_summary: dict[str, Any] | None = None,
) -> str:
    outcome_summary = outcome_summary or {}
    failure_summary = failure_summary or {}
    scenario_summary = scenario_summary or {}
    critical_preservation = format_percent(get_nested_metric(metrics, "alerts", "critical_preservation_rate", 0))
    reliability_score = format_number(get_nested_metric(metrics, "reliability", "average_reliability_score", 0))
    severe_drift = format_number(get_nested_metric(metrics, "drift", "severe_drift_count", 0))
    total_recommendations = format_number(get_nested_metric(metrics, "llm_action", "total_action_recommendations", 0))
    deployment_recommendation = str(get_nested_metric(metrics, "model_update_rl", "deployment_recommendation", ""))
    useful_alert_rate = format_percent(outcome_summary.get("useful_alert_rate", 0))
    unsafe_failure_count = format_number(failure_summary.get("unsafe_review_required_count", 0))
    warning_scenarios = format_number(scenario_summary.get("warning_safety_checks", 0))
    return f"""## 14. Key Findings

- Alert fatigue reduction preserved critical alerts with a critical preservation rate of {critical_preservation}.
- Reliability monitoring remained mostly stable in the simulated windows, with an average reliability score of {reliability_score}, while still flagging review needs.
- Drift detection found {severe_drift} severe drift checks, so threshold updates were handled conservatively.
- The model update simulation recommended `{deployment_recommendation}` rather than automatic deployment.
- The workflow-aware action recommender generated {total_recommendations} simulated workflow recommendations with safety notes and RAG source coverage.
- Outcome evaluation estimated a useful-alert rate of {useful_alert_rate} in the simulated workflow, while keeping outcome interpretation as association only.
- Failure-mode simulation identified {unsafe_failure_count} unsafe-review-required simulated failure events under noisy data, alert overload, drift, and delayed-response conditions.
- Scenario testing produced {warning_scenarios} warning safety checks across stable, deteriorating, critical, noisy, missing-data, repeated-alert, and overload situations.
- Human review remains central when uncertainty, severe drift, critical alerts, or unsafe failure states appear.
- The project demonstrates end-to-end reliability engineering patterns, but all findings remain simulation-only.
"""


def generate_limitations_section() -> str:
    return """## 15. Limitations

- The dataset is simulated and contains no real patient data.
- The project is not clinically validated.
- The system is not deployed and is not suitable for real patient monitoring.
- Model, rule, alert, workflow, and response assumptions are simplified for portfolio-scale engineering.
- Reliability and drift scores are engineering signals from synthetic artifacts, not real-world safety evidence.
- Outcome labels are synthetic and do not prove patient benefit.
- Failure modes are simulated approximations of engineering risks.
- Scenario tests are workflow stress tests, not clinical trials.
- Workflow response behavior is synthetic and does not represent real staff actions.
- Dashboard and report outputs are for engineering demonstration only.
- LLM/fallback explanations are support text only and may be incomplete.
- RAG uses local project guidance only and does not establish medical correctness.
- Human-in-the-loop review is required for safety-sensitive simulated alerts.
"""


def generate_conclusion_section() -> str:
    return f"""## 16. Conclusion

This project is valuable as a healthcare AI engineering portfolio prototype because it goes beyond basic model training and shows deployment-aware thinking: simulated data quality, future-risk modeling, anomaly detection, alert generation, safety guardrails, auditability, fatigue reduction, workflow simulation, reliability monitoring, drift detection, model-update caution, RL-threshold simulation, LLM/RAG support text, action recommendations, metrics, and dashboard presentation.

The strongest engineering value is the system-level framing: alerts are not treated as isolated predictions, but as outputs that must be monitored, explained, audited, and reviewed safely over time.

Final safety boundary: {SIMULATION_ONLY_DISCLAIMER}
"""


def generate_evaluation_report(output_path: str = "reports/evaluation_results.md") -> str:
    """Generate and save the final Step 24 markdown evaluation report."""
    metrics = safe_load_json(DATA_PATHS["metrics_summary"])
    _ = safe_load_csv(DATA_PATHS["metrics_table"])
    alerts_df = safe_load_csv(DATA_PATHS["generated_alerts"])
    _ = safe_load_csv(DATA_PATHS["fatigue_reduced_alerts"])
    _ = safe_load_json(DATA_PATHS["clinician_response_summary"])
    _ = safe_load_json(DATA_PATHS["reliability_summary"])
    drift_summary = safe_load_json(DATA_PATHS["drift_summary"])
    model_update_df = safe_load_csv(DATA_PATHS["model_update_results"])
    rl_summary = safe_load_json(DATA_PATHS["rl_policy_summary"])
    _ = safe_load_csv(DATA_PATHS["alert_explanations"])
    _ = safe_load_csv(DATA_PATHS["action_recommendations"])
    outcome_summary = safe_load_json(DATA_PATHS["outcome_effectiveness_summary"])
    outcome_df = safe_load_csv(DATA_PATHS["outcome_effectiveness_results"])
    failure_summary = safe_load_json(DATA_PATHS["failure_mode_summary"])
    failure_df = safe_load_csv(DATA_PATHS["failure_mode_results"])
    scenario_summary = safe_load_json(DATA_PATHS["scenario_test_summary"])
    scenario_df = safe_load_csv(DATA_PATHS["scenario_test_results"])

    sections = [
        "# Clinical Alert Reliability AI - Final Evaluation Report",
        f"**Report type:** Simulated engineering evaluation\n\n**Safety boundary:** {SIMULATION_ONLY_DISCLAIMER}",
        generate_project_overview_section(metrics),
        generate_dataset_section(metrics),
        generate_alert_section(metrics, alerts_df),
        generate_fatigue_section(metrics),
        generate_workflow_section(metrics),
        generate_reliability_section(metrics),
        generate_drift_section(metrics, drift_summary),
        generate_model_update_section(metrics, model_update_df, rl_summary),
        generate_llm_action_section(metrics),
        generate_outcome_effectiveness_section(outcome_summary, outcome_df),
        generate_failure_mode_section(failure_summary, failure_df),
        generate_scenario_testing_section(scenario_summary, scenario_df),
        generate_deployment_readiness_discussion_section(
            outcome_summary,
            failure_summary,
            scenario_summary,
        ),
        generate_key_findings_section(
            metrics,
            outcome_summary=outcome_summary,
            failure_summary=failure_summary,
            scenario_summary=scenario_summary,
        ),
        generate_limitations_section(),
        generate_conclusion_section(),
    ]
    report = "\n\n".join(section.strip() for section in sections) + "\n"

    file_path = _resolve_project_path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(report, encoding="utf-8")
    return report


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _markdown_table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def _value_count_rows(df: pd.DataFrame, column: str) -> list[tuple[str, str]]:
    if df.empty or column not in df.columns:
        return []
    counts = df[column].fillna("missing").astype(str).value_counts()
    return [(str(key), format_number(value)) for key, value in counts.items()]


def _dict_count_rows(values: Any) -> list[tuple[str, str]]:
    if not isinstance(values, dict):
        return []
    return [(str(key), format_number(value)) for key, value in values.items()]


if __name__ == "__main__":
    generated_report = generate_evaluation_report()
    output = _resolve_project_path("reports/evaluation_results.md")
    metrics = safe_load_json(DATA_PATHS["metrics_summary"])
    print(f"Generated evaluation report: {output}")
    print(f"Total patients: {format_number(get_nested_metric(metrics, 'dataset', 'total_patients', 0))}")
    print(f"Total raw alerts: {format_number(get_nested_metric(metrics, 'alerts', 'total_raw_alerts', 0))}")
    print(
        "Average reliability score: "
        f"{format_number(get_nested_metric(metrics, 'reliability', 'average_reliability_score', 0))}"
    )
    print(f"Report length: {format_number(len(generated_report.splitlines()))} lines")
