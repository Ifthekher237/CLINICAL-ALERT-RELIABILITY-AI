"""Scenario testing for the simulated clinical alert reliability system.

Step 24C summarizes existing project artifacts into deployment-style scenarios.
It does not rerun heavy pipelines or modify alert/model logic. All outputs are
simulation-only engineering test artifacts, not clinical validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_RESULTS_PATH = Path("data/processed/scenario_test_results.csv")
OUTPUT_SUMMARY_PATH = Path("data/processed/scenario_test_summary.json")
SIMULATION_ONLY_NOTE = (
    "Simulation-only scenario test; not clinical validation, not deployment "
    "readiness evidence, and not medical-device safety evidence."
)

SCENARIO_NAMES = [
    "stable_patient_monitoring",
    "gradual_deterioration",
    "sudden_critical_event",
    "noisy_sensor_false_alarm",
    "repeated_low_risk_alerts",
    "missing_data_episode",
    "high_patient_volume_overload",
]
VALID_SCENARIO_CATEGORIES = {
    "baseline_monitoring",
    "deterioration_monitoring",
    "critical_event",
    "sensor_quality",
    "alert_fatigue",
    "data_quality",
    "workload_stress",
}
VALID_SAFETY_CHECK_STATUSES = {"passed", "warning", "failed"}
VALID_OVERALL_SCENARIO_STATUSES = {"stable", "monitored", "degraded", "unsafe_review_required"}
VALID_DRIFT_RISK_LEVELS = {"low", "medium", "high", "severe"}

REQUIRED_OUTPUT_COLUMNS = [
    "scenario_test_id",
    "scenario_name",
    "scenario_category",
    "simulated_patient_count",
    "simulated_alert_count",
    "critical_alert_count",
    "reduced_alert_count",
    "ignored_alert_rate",
    "delayed_alert_rate",
    "reliability_score",
    "drift_risk_level",
    "outcome_effectiveness_score",
    "failure_mode_triggered",
    "safety_check_status",
    "human_review_required",
    "overall_scenario_status",
    "scenario_summary",
    "simulation_note",
    "timestamp",
]


def safe_load_csv(path: str) -> pd.DataFrame:
    """Load a CSV safely, returning an empty dataframe when unavailable."""
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def safe_load_json(path: str) -> dict[str, Any]:
    """Load JSON safely, returning an empty dictionary when unavailable."""
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def simulate_stable_patient_monitoring(context: dict[str, Any]) -> dict[str, Any]:
    """Baseline scenario with low alert pressure and stable monitoring signals."""
    patients = int(context.get("total_patients", 0))
    low_alert_count = int(context.get("low_alert_count", 0))
    simulated_alert_count = min(max(patients * 2, 1), max(low_alert_count, 1)) if patients else min(low_alert_count, 5)
    return {
        "scenario_name": "stable_patient_monitoring",
        "scenario_category": "baseline_monitoring",
        "simulated_patient_count": patients,
        "simulated_alert_count": simulated_alert_count,
        "critical_alert_count": 0,
        "reduced_alert_count": 0,
        "ignored_alert_rate": _clip_rate(context.get("ignored_alert_rate", 0.0) * 0.25),
        "delayed_alert_rate": _clip_rate(context.get("delayed_alert_rate", 0.0) * 0.25),
        "reliability_score": _clip_rate(max(context.get("average_reliability_score", 0.9), 0.90)),
        "drift_risk_level": "low",
        "outcome_effectiveness_score": _clip_rate(max(context.get("average_outcome_effectiveness_score", 0.0), 0.65)),
        "failure_mode_triggered": False,
        "human_review_required": False,
    }


def simulate_gradual_deterioration(context: dict[str, Any]) -> dict[str, Any]:
    """Scenario where medium/high alerts and drift indicate gradual worsening."""
    alert_count = int(context.get("medium_alert_count", 0) + context.get("high_alert_count", 0))
    drift_level = "high" if context.get("severe_drift_count", 0) else "medium"
    return {
        "scenario_name": "gradual_deterioration",
        "scenario_category": "deterioration_monitoring",
        "simulated_patient_count": int(context.get("total_patients", 0)),
        "simulated_alert_count": alert_count,
        "critical_alert_count": 0,
        "reduced_alert_count": int(context.get("downgraded_alert_count", 0)),
        "ignored_alert_rate": _clip_rate(context.get("ignored_alert_rate", 0.0)),
        "delayed_alert_rate": _clip_rate(context.get("delayed_alert_rate", 0.0) * 1.15),
        "reliability_score": _clip_rate(context.get("average_reliability_score", 0.0) - 0.03),
        "drift_risk_level": drift_level,
        "outcome_effectiveness_score": _clip_rate(context.get("average_outcome_effectiveness_score", 0.0) - 0.03),
        "failure_mode_triggered": bool(context.get("data_distribution_shift_events", 0)),
        "human_review_required": alert_count > 0 or drift_level in {"high", "severe"},
    }


def simulate_sudden_critical_event(context: dict[str, Any]) -> dict[str, Any]:
    """Scenario for critical alerts requiring safety-first review."""
    critical_count = int(context.get("critical_alert_count", 0))
    preserved = _clip_rate(context.get("critical_preservation_rate", 0.0))
    return {
        "scenario_name": "sudden_critical_event",
        "scenario_category": "critical_event",
        "simulated_patient_count": int(context.get("total_patients", 0)),
        "simulated_alert_count": max(critical_count, int(context.get("high_alert_count", 0))),
        "critical_alert_count": critical_count,
        "reduced_alert_count": 0,
        "ignored_alert_rate": 0.0 if preserved >= 1.0 else _clip_rate(context.get("ignored_alert_rate", 0.0)),
        "delayed_alert_rate": 0.0 if preserved >= 1.0 else _clip_rate(context.get("delayed_alert_rate", 0.0)),
        "reliability_score": _clip_rate(context.get("average_reliability_score", 0.0)),
        "drift_risk_level": "medium" if preserved >= 1.0 else "high",
        "outcome_effectiveness_score": _clip_rate(context.get("average_outcome_effectiveness_score", 0.0)),
        "failure_mode_triggered": bool(context.get("unsafe_failure_count", 0)),
        "human_review_required": True,
        "critical_preservation_rate": preserved,
    }


def simulate_noisy_sensor_false_alarm(context: dict[str, Any]) -> dict[str, Any]:
    """Scenario where noisy sensor artifacts create false-alert pressure."""
    triggered = bool(context.get("noisy_sensor_spike_events", 0))
    return {
        "scenario_name": "noisy_sensor_false_alarm",
        "scenario_category": "sensor_quality",
        "simulated_patient_count": int(context.get("total_patients", 0)),
        "simulated_alert_count": int(context.get("noisy_sensor_spike_events", 0)),
        "critical_alert_count": 0,
        "reduced_alert_count": 0,
        "ignored_alert_rate": _clip_rate(context.get("ignored_alert_rate", 0.0) * 1.3),
        "delayed_alert_rate": _clip_rate(context.get("delayed_alert_rate", 0.0)),
        "reliability_score": _clip_rate(context.get("average_reliability_score", 0.0) - 0.08),
        "drift_risk_level": "high" if triggered else "medium",
        "outcome_effectiveness_score": _clip_rate(context.get("average_outcome_effectiveness_score", 0.0) - 0.08),
        "failure_mode_triggered": triggered,
        "human_review_required": triggered,
    }


def simulate_repeated_low_risk_alerts(context: dict[str, Any]) -> dict[str, Any]:
    """Scenario for alert fatigue from repeated low/medium alerts."""
    grouped = int(context.get("grouped_alert_count", 0))
    delayed = int(context.get("delayed_alert_count", 0))
    downgraded = int(context.get("downgraded_alert_count", 0))
    return {
        "scenario_name": "repeated_low_risk_alerts",
        "scenario_category": "alert_fatigue",
        "simulated_patient_count": int(context.get("total_patients", 0)),
        "simulated_alert_count": int(context.get("low_alert_count", 0) + context.get("medium_alert_count", 0)),
        "critical_alert_count": 0,
        "reduced_alert_count": grouped + delayed + downgraded,
        "ignored_alert_rate": _clip_rate(context.get("ignored_alert_rate", 0.0) * 1.2),
        "delayed_alert_rate": _clip_rate(context.get("delayed_alert_rate", 0.0) * 1.2),
        "reliability_score": _clip_rate(context.get("average_reliability_score", 0.0) - 0.04),
        "drift_risk_level": "medium",
        "outcome_effectiveness_score": _clip_rate(context.get("average_outcome_effectiveness_score", 0.0) - 0.04),
        "failure_mode_triggered": bool(context.get("repeated_low_value_alerts_events", 0)),
        "human_review_required": False,
    }


def simulate_missing_data_episode(context: dict[str, Any]) -> dict[str, Any]:
    """Scenario for data-quality uncertainty from incomplete monitoring data."""
    missing_rate = _clip_rate(context.get("missing_data_rate", 0.0))
    missing_events = int(context.get("missing_patient_data_events", 0))
    return {
        "scenario_name": "missing_data_episode",
        "scenario_category": "data_quality",
        "simulated_patient_count": int(context.get("total_patients", 0)),
        "simulated_alert_count": missing_events,
        "critical_alert_count": 0,
        "reduced_alert_count": 0,
        "ignored_alert_rate": _clip_rate(context.get("ignored_alert_rate", 0.0)),
        "delayed_alert_rate": _clip_rate(context.get("delayed_alert_rate", 0.0)),
        "reliability_score": _clip_rate(context.get("average_reliability_score", 0.0) - missing_rate),
        "drift_risk_level": "medium" if missing_events else "low",
        "outcome_effectiveness_score": _clip_rate(context.get("average_outcome_effectiveness_score", 0.0) - 0.05),
        "failure_mode_triggered": bool(missing_events),
        "human_review_required": bool(missing_events),
    }


def simulate_high_patient_volume_overload(context: dict[str, Any]) -> dict[str, Any]:
    """Scenario for workload stress from high alert volume and response delays."""
    total_alerts = int(context.get("total_raw_alerts", 0))
    overload_events = int(context.get("alert_overload_events", 0))
    return {
        "scenario_name": "high_patient_volume_overload",
        "scenario_category": "workload_stress",
        "simulated_patient_count": int(context.get("total_patients", 0)),
        "simulated_alert_count": total_alerts,
        "critical_alert_count": int(context.get("critical_alert_count", 0)),
        "reduced_alert_count": int(context.get("total_reduced_alerts", 0)),
        "ignored_alert_rate": _clip_rate(context.get("ignored_alert_rate", 0.0) * 1.6),
        "delayed_alert_rate": _clip_rate(context.get("delayed_alert_rate", 0.0) * 1.6),
        "reliability_score": _clip_rate(context.get("average_reliability_score", 0.0) - 0.10),
        "drift_risk_level": "high" if overload_events else "medium",
        "outcome_effectiveness_score": _clip_rate(context.get("average_outcome_effectiveness_score", 0.0) - 0.10),
        "failure_mode_triggered": bool(overload_events),
        "human_review_required": True,
    }


def calculate_scenario_status(row: dict[str, Any]) -> str:
    """Assign overall scenario status from reliability, drift, rates, and safety."""
    safety = str(row.get("safety_check_status", "")).lower()
    drift = str(row.get("drift_risk_level", "")).lower()
    reliability = _safe_float(row.get("reliability_score"), 0.0)
    ignored = _safe_float(row.get("ignored_alert_rate"), 0.0)
    delayed = _safe_float(row.get("delayed_alert_rate"), 0.0)

    if safety == "failed" or drift == "severe" or reliability < 0.65:
        return "unsafe_review_required"
    if safety == "warning" or drift == "high" or ignored > 0.10 or delayed > 0.12 or reliability < 0.80:
        return "degraded"
    if row.get("human_review_required") or drift == "medium" or ignored > 0.03 or delayed > 0.03:
        return "monitored"
    return "stable"


def calculate_safety_check_status(row: dict[str, Any]) -> str:
    """Assign scenario safety-check status."""
    critical_count = int(_safe_float(row.get("critical_alert_count"), 0.0))
    human_review = _coerce_bool(row.get("human_review_required"))
    critical_preservation = _safe_float(row.get("critical_preservation_rate"), 1.0)
    reliability = _safe_float(row.get("reliability_score"), 0.0)
    drift = str(row.get("drift_risk_level", "")).lower()
    failure_triggered = _coerce_bool(row.get("failure_mode_triggered"))

    if critical_count > 0 and not human_review:
        return "failed"
    if critical_count > 0 and critical_preservation < 1.0:
        return "failed"
    if reliability < 0.65 or drift == "severe":
        return "failed"
    if critical_count > 0 or human_review or failure_triggered or drift in {"medium", "high"} or reliability < 0.85:
        return "warning"
    return "passed"


def generate_scenario_summary(row: dict[str, Any]) -> str:
    """Generate concise engineering-focused scenario summary text."""
    name = str(row.get("scenario_name", "scenario"))
    category = str(row.get("scenario_category", "unknown"))
    status = str(row.get("overall_scenario_status", "monitored"))
    safety = str(row.get("safety_check_status", "warning"))
    review_text = "requires human review" if _coerce_bool(row.get("human_review_required")) else "does not require immediate human review"
    failure_text = "with failure-mode interaction" if _coerce_bool(row.get("failure_mode_triggered")) else "without a triggered failure mode"
    return (
        f"Simulation-only {name} scenario in the {category} category: "
        f"{int(_safe_float(row.get('simulated_alert_count'), 0))} alerts, "
        f"{int(_safe_float(row.get('critical_alert_count'), 0))} critical alerts, "
        f"drift risk {row.get('drift_risk_level', 'low')}, safety check {safety}, "
        f"overall status {status}, {review_text}, {failure_text}. "
        "This is an engineering workflow test, not clinical validation."
    )


def build_scenario_context(
    raw_alerts_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
    audited_df: pd.DataFrame,
    response_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    outcome_results_df: pd.DataFrame,
    outcome_summary: dict[str, Any],
    failure_results_df: pd.DataFrame,
    failure_summary: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Summarize artifacts into reusable context for scenario simulation."""
    del audited_df
    dataset_metrics = _nested_dict(metrics, "dataset")
    alert_metrics = _nested_dict(metrics, "alerts")
    audit_fatigue_metrics = _nested_dict(metrics, "audit_fatigue")
    workflow_metrics = _nested_dict(metrics, "workflow")
    reliability_metrics = _nested_dict(metrics, "reliability")
    drift_metrics = _nested_dict(metrics, "drift")

    severity_counts = _value_counts(raw_alerts_df, "severity")
    failure_counts = failure_summary.get("failure_mode_distribution", {})
    if not isinstance(failure_counts, dict):
        failure_counts = {}

    context = {
        "total_patients": int(dataset_metrics.get("total_patients", raw_alerts_df.get("patient_id", pd.Series(dtype=str)).nunique() if not raw_alerts_df.empty and "patient_id" in raw_alerts_df.columns else 0)),
        "missing_data_rate": _safe_float(dataset_metrics.get("missing_data_rate"), 0.0),
        "total_raw_alerts": int(alert_metrics.get("total_raw_alerts", len(raw_alerts_df))),
        "critical_alert_count": int(alert_metrics.get("critical_alert_count", severity_counts.get("critical", 0))),
        "critical_preservation_rate": _safe_float(alert_metrics.get("critical_preservation_rate"), 0.0),
        "active_alerts_after_reduction": int(alert_metrics.get("active_alerts_after_reduction", _status_count(fatigue_df, "final_alert_status", "active"))),
        "total_reduced_alerts": max(int(len(raw_alerts_df)) - int(alert_metrics.get("active_alerts_after_reduction", 0)), 0),
        "grouped_alert_count": int(audit_fatigue_metrics.get("grouped_alert_count", _status_count(fatigue_df, "final_alert_status", "grouped"))),
        "delayed_alert_count": int(audit_fatigue_metrics.get("delayed_alert_count", _status_count(fatigue_df, "final_alert_status", "delayed"))),
        "downgraded_alert_count": int(audit_fatigue_metrics.get("downgraded_alert_count", _status_count(fatigue_df, "final_alert_status", "priority_downgraded"))),
        "ignored_alert_rate": _safe_float(workflow_metrics.get("ignored_alert_rate"), _rate(response_df, "simulated_response", "ignored")),
        "delayed_alert_rate": _safe_float(workflow_metrics.get("delayed_alert_rate"), _rate(response_df, "simulated_response", "delayed")),
        "average_reliability_score": _safe_float(reliability_metrics.get("average_reliability_score"), _mean(reliability_df, "reliability_score")),
        "severe_drift_count": int(drift_metrics.get("severe_drift_count", _status_count(drift_df, "drift_status", "severe_shift"))),
        "average_drift_score": _safe_float(drift_metrics.get("average_drift_score"), _mean(drift_df, "drift_score")),
        "average_outcome_effectiveness_score": _safe_float(
            outcome_summary.get("average_outcome_effectiveness_score"),
            _mean(outcome_results_df, "outcome_effectiveness_score"),
        ),
        "unsafe_failure_count": int(failure_summary.get("unsafe_review_required_count", _status_count(failure_results_df, "safety_status", "unsafe_review_required"))),
    }
    for severity in ["low", "medium", "high", "critical"]:
        context[f"{severity}_alert_count"] = int(severity_counts.get(severity, 0))
    for failure_mode in [
        "noisy_sensor_spike",
        "missing_patient_data",
        "alert_overload",
        "repeated_low_value_alerts",
        "delayed_response_failure",
        "model_confidence_drop",
        "data_distribution_shift",
    ]:
        context[f"{failure_mode}_events"] = int(failure_counts.get(failure_mode, 0))
    return context


def build_scenario_results_table(context: dict[str, Any]) -> pd.DataFrame:
    """Build all scenario rows from summarized context."""
    scenario_functions = [
        simulate_stable_patient_monitoring,
        simulate_gradual_deterioration,
        simulate_sudden_critical_event,
        simulate_noisy_sensor_false_alarm,
        simulate_repeated_low_risk_alerts,
        simulate_missing_data_episode,
        simulate_high_patient_volume_overload,
    ]
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, Any]] = []
    for index, scenario_function in enumerate(scenario_functions, start=1):
        row = scenario_function(context)
        row["scenario_test_id"] = f"SCENARIO-{index:03d}"
        row["safety_check_status"] = calculate_safety_check_status(row)
        row["overall_scenario_status"] = calculate_scenario_status(row)
        row["scenario_summary"] = generate_scenario_summary(row)
        row["simulation_note"] = SIMULATION_ONLY_NOTE
        row["timestamp"] = timestamp
        rows.append(row)

    results = pd.DataFrame(rows)
    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in results.columns:
            results[column] = "" if column not in {"human_review_required", "failure_mode_triggered"} else False

    for rate_column in [
        "ignored_alert_rate",
        "delayed_alert_rate",
        "reliability_score",
        "outcome_effectiveness_score",
    ]:
        results[rate_column] = pd.to_numeric(results[rate_column], errors="coerce").fillna(0.0).clip(0, 1)
    results["human_review_required"] = results["human_review_required"].map(lambda value: bool(_coerce_bool(value))).astype(object)
    results["failure_mode_triggered"] = results["failure_mode_triggered"].map(lambda value: bool(_coerce_bool(value))).astype(object)
    return results[REQUIRED_OUTPUT_COLUMNS].copy()


def calculate_scenario_summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize scenario test results for reporting/dashboard use."""
    if df.empty:
        return {
            "total_scenarios": 0,
            "scenario_distribution": {},
            "overall_status_distribution": {},
            "safety_check_distribution": {},
            "average_reliability_score": 0.0,
            "average_ignored_alert_rate": 0.0,
            "average_delayed_alert_rate": 0.0,
            "average_outcome_effectiveness_score": 0.0,
            "failure_mode_trigger_rate": 0.0,
            "human_review_required_rate": 0.0,
            "passed_safety_checks": 0,
            "warning_safety_checks": 0,
            "failed_safety_checks": 0,
            "simulation_only_note": SIMULATION_ONLY_NOTE,
        }

    total = len(df)
    return {
        "total_scenarios": int(total),
        "scenario_distribution": _value_counts(df, "scenario_category"),
        "overall_status_distribution": _value_counts(df, "overall_scenario_status"),
        "safety_check_distribution": _value_counts(df, "safety_check_status"),
        "average_reliability_score": _mean(df, "reliability_score"),
        "average_ignored_alert_rate": _mean(df, "ignored_alert_rate"),
        "average_delayed_alert_rate": _mean(df, "delayed_alert_rate"),
        "average_outcome_effectiveness_score": _mean(df, "outcome_effectiveness_score"),
        "failure_mode_trigger_rate": _safe_rate(_bool_column(df, "failure_mode_triggered").sum(), total),
        "human_review_required_rate": _safe_rate(_bool_column(df, "human_review_required").sum(), total),
        "passed_safety_checks": _status_count(df, "safety_check_status", "passed"),
        "warning_safety_checks": _status_count(df, "safety_check_status", "warning"),
        "failed_safety_checks": _status_count(df, "safety_check_status", "failed"),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


def save_scenario_results(df: pd.DataFrame, path: str) -> Path:
    """Save scenario-level result rows."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_scenario_summary(summary: dict[str, Any], path: str) -> Path:
    """Save scenario summary JSON."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def run_scenario_testing_pipeline(
    raw_alerts_path: str = "data/processed/generated_alerts.csv",
    fatigue_path: str = "data/processed/fatigue_reduced_alerts.csv",
    audited_path: str = "data/processed/audited_alerts.csv",
    response_path: str = "data/processed/clinician_response_logs.csv",
    reliability_path: str = "data/processed/reliability_monitoring_results.csv",
    drift_path: str = "data/processed/drift_detection_results.csv",
    outcome_results_path: str = "data/processed/outcome_effectiveness_results.csv",
    outcome_summary_path: str = "data/processed/outcome_effectiveness_summary.json",
    failure_results_path: str = "data/processed/failure_mode_results.csv",
    failure_summary_path: str = "data/processed/failure_mode_summary.json",
    metrics_path: str = "data/processed/project_metrics_summary.json",
    output_path: str = str(OUTPUT_RESULTS_PATH),
    summary_path: str = str(OUTPUT_SUMMARY_PATH),
) -> pd.DataFrame:
    """Run the complete scenario testing pipeline."""
    raw_alerts_df = safe_load_csv(raw_alerts_path)
    fatigue_df = safe_load_csv(fatigue_path)
    audited_df = safe_load_csv(audited_path)
    response_df = safe_load_csv(response_path)
    reliability_df = safe_load_csv(reliability_path)
    drift_df = safe_load_csv(drift_path)
    outcome_results_df = safe_load_csv(outcome_results_path)
    outcome_summary = safe_load_json(outcome_summary_path)
    failure_results_df = safe_load_csv(failure_results_path)
    failure_summary = safe_load_json(failure_summary_path)
    metrics = safe_load_json(metrics_path)

    context = build_scenario_context(
        raw_alerts_df=raw_alerts_df,
        fatigue_df=fatigue_df,
        audited_df=audited_df,
        response_df=response_df,
        reliability_df=reliability_df,
        drift_df=drift_df,
        outcome_results_df=outcome_results_df,
        outcome_summary=outcome_summary,
        failure_results_df=failure_results_df,
        failure_summary=failure_summary,
        metrics=metrics,
    )
    results = build_scenario_results_table(context)
    summary = calculate_scenario_summary_metrics(results)
    save_scenario_results(results, output_path)
    save_scenario_summary(summary, summary_path)
    return results


def run_scenario_tests() -> pd.DataFrame:
    """Compatibility wrapper for the earlier placeholder function."""
    return run_scenario_testing_pipeline()


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _nested_dict(values: dict[str, Any], key: str) -> dict[str, Any]:
    nested = values.get(key, {})
    return nested if isinstance(nested, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip_rate(value: Any) -> float:
    return round(max(0.0, min(1.0, _safe_float(value, 0.0))), 4)


def _safe_rate(numerator: Any, denominator: Any) -> float:
    denominator_float = _safe_float(denominator, 0.0)
    if denominator_float <= 0:
        return 0.0
    return round(_safe_float(numerator, 0.0) / denominator_float, 4)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(False, index=df.index)
    text_true = df[column].astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
    numeric_true = pd.to_numeric(df[column], errors="coerce").fillna(0).ne(0)
    return text_true | numeric_true


def _mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return round(float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean()), 4)


def _rate(df: pd.DataFrame, column: str, value: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return _safe_rate(df[column].astype(str).str.lower().eq(value.lower()).sum(), len(df))


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    counts = df[column].fillna("missing").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _status_count(df: pd.DataFrame, column: str, status: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.lower().eq(status.lower()).sum())


if __name__ == "__main__":
    scenario_results = run_scenario_testing_pipeline()
    scenario_summary = calculate_scenario_summary_metrics(scenario_results)
    print(f"Total scenarios: {scenario_summary['total_scenarios']}")
    print("Overall status distribution:")
    print(pd.Series(scenario_summary["overall_status_distribution"]).to_string() if scenario_summary["overall_status_distribution"] else "none")
    print("Safety check distribution:")
    print(pd.Series(scenario_summary["safety_check_distribution"]).to_string() if scenario_summary["safety_check_distribution"] else "none")
    print(f"Average reliability score: {scenario_summary['average_reliability_score']:.4f}")
    print(f"Average outcome effectiveness score: {scenario_summary['average_outcome_effectiveness_score']:.4f}")
    print(f"Saved results to {_resolve_project_path(OUTPUT_RESULTS_PATH)}")
    print(f"Saved summary to {_resolve_project_path(OUTPUT_SUMMARY_PATH)}")
