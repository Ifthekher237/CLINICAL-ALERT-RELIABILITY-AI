"""Simulated deployment-style failure mode testing for alert reliability.

Step 24B creates controlled failure events from existing simulated artifacts.
The goal is to test engineering behavior under noisy sensors, missing data,
alert overload, delayed responses, drift, and reliability degradation.

This is simulation-only and does not make clinical claims, recommend treatment,
or validate a medical device.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_RESULTS_PATH = Path("data/processed/failure_mode_results.csv")
OUTPUT_SUMMARY_PATH = Path("data/processed/failure_mode_summary.json")
SIMULATION_ONLY_NOTE = (
    "Simulation-only failure-mode test; not clinical validation, not treatment "
    "guidance, and not medical-device evidence."
)

VALID_FAILURE_MODES = {
    "noisy_sensor_spike",
    "missing_patient_data",
    "alert_overload",
    "repeated_low_value_alerts",
    "delayed_response_failure",
    "model_confidence_drop",
    "data_distribution_shift",
}
VALID_SEVERITY_LEVELS = {"low", "medium", "high", "critical"}
VALID_SAFETY_STATUSES = {"monitored", "warning", "degraded", "unsafe_review_required"}

REQUIRED_OUTPUT_COLUMNS = [
    "failure_event_id",
    "timestamp",
    "patient_id",
    "related_alert_id",
    "failure_mode",
    "severity_level",
    "failure_trigger_reason",
    "simulated_system_impact",
    "affected_component",
    "alert_volume_impact",
    "clinician_burden_impact",
    "reliability_score_impact",
    "drift_risk_impact",
    "outcome_risk_impact",
    "requires_human_review",
    "mitigation_recommendation",
    "safety_status",
    "failure_simulation_note",
]


def safe_load_csv(path: str) -> pd.DataFrame:
    """Load a CSV artifact safely; return an empty dataframe if missing."""
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def safe_load_json(path: str) -> dict[str, Any]:
    """Load a JSON artifact safely; return an empty dict if missing."""
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def simulate_noisy_sensor_spike(
    patient_df: pd.DataFrame,
    alerts_df: pd.DataFrame | None = None,
    limit: int = 25,
) -> pd.DataFrame:
    """Create events for noisy or physiologically implausible simulated readings."""
    if patient_df.empty:
        return _empty_base_events()
    df = patient_df.copy()
    noisy_mask = _bool_column(df, "sensor_noise_flag")
    spike_mask = pd.Series(False, index=df.index)
    if "heart_rate" in df.columns:
        spike_mask |= pd.to_numeric(df["heart_rate"], errors="coerce").gt(140)
    if "oxygen_saturation" in df.columns:
        spike_mask |= pd.to_numeric(df["oxygen_saturation"], errors="coerce").lt(82)
    if "respiratory_rate" in df.columns:
        spike_mask |= pd.to_numeric(df["respiratory_rate"], errors="coerce").gt(35)

    candidates = df[noisy_mask | spike_mask].head(limit)
    records = []
    for _, row in candidates.iterrows():
        related_alert_id = _find_related_alert_id(row, alerts_df)
        records.append(
            _base_event(
                row=row,
                related_alert_id=related_alert_id,
                failure_mode="noisy_sensor_spike",
                trigger_reason="Noisy sensor flag or unrealistic vital-sign spike detected in simulated monitoring data.",
                system_impact="May inflate alert volume and increase false-alert review burden.",
                affected_component="vital_sign_ingestion",
                context={
                    "sensor_noise_flag": row.get("sensor_noise_flag"),
                    "vital_spike_flag": bool(spike_mask.loc[row.name]),
                },
            )
        )
    return pd.DataFrame(records)


def simulate_missing_patient_data(
    patient_df: pd.DataFrame,
    alerts_df: pd.DataFrame | None = None,
    limit: int = 25,
) -> pd.DataFrame:
    """Create events for incomplete simulated vital-sign rows."""
    if patient_df.empty:
        return _empty_base_events()
    df = patient_df.copy()
    missing_flag = _bool_column(df, "missing_data_flag")
    vital_columns = [
        column
        for column in [
            "heart_rate",
            "oxygen_saturation",
            "systolic_bp",
            "diastolic_bp",
            "respiratory_rate",
            "temperature",
        ]
        if column in df.columns
    ]
    missing_vitals = df[vital_columns].isna().any(axis=1) if vital_columns else pd.Series(False, index=df.index)
    candidates = df[missing_flag | missing_vitals].head(limit)

    records = []
    for _, row in candidates.iterrows():
        records.append(
            _base_event(
                row=row,
                related_alert_id=_find_related_alert_id(row, alerts_df),
                failure_mode="missing_patient_data",
                trigger_reason="Incomplete simulated vitals can reduce confidence in alert interpretation.",
                system_impact="Increases uncertainty and may require manual verification before downstream interpretation.",
                affected_component="data_quality",
                context={"missing_data_flag": row.get("missing_data_flag")},
            )
        )
    return pd.DataFrame(records)


def simulate_alert_overload(
    alerts_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    limit: int = 20,
) -> pd.DataFrame:
    """Create events for windows or alert bursts with unusually high alert volume."""
    records = []
    if not reliability_df.empty and "total_alerts" in reliability_df.columns:
        threshold = max(10.0, float(pd.to_numeric(reliability_df["total_alerts"], errors="coerce").quantile(0.75)))
        candidates = reliability_df[pd.to_numeric(reliability_df["total_alerts"], errors="coerce").fillna(0) >= threshold]
        for _, row in candidates.head(limit).iterrows():
            records.append(
                {
                    "timestamp": _string_value(row.get("window_start")),
                    "patient_id": "system",
                    "related_alert_id": "",
                    "failure_mode": "alert_overload",
                    "failure_trigger_reason": f"Monitoring window contained {row.get('total_alerts', 0)} simulated alerts.",
                    "simulated_system_impact": "High alert volume can increase queue pressure and missed-review risk.",
                    "affected_component": "alert_queue",
                    "total_alerts": _safe_float(row.get("total_alerts"), 0.0),
                    "reliability_score": _safe_float(row.get("reliability_score"), 1.0),
                    "average_response_time_minutes": _safe_float(row.get("average_response_time_minutes"), 0.0),
                }
            )
    elif not alerts_df.empty:
        df = _normalize_timestamp(alerts_df.copy(), "timestamp")
        grouped = df.groupby("timestamp").size().reset_index(name="total_alerts").sort_values("total_alerts", ascending=False)
        for _, row in grouped.head(limit).iterrows():
            records.append(
                {
                    "timestamp": row.get("timestamp"),
                    "patient_id": "system",
                    "related_alert_id": "",
                    "failure_mode": "alert_overload",
                    "failure_trigger_reason": f"Timestamp had {row.get('total_alerts', 0)} simulated alerts.",
                    "simulated_system_impact": "Potential alert burst may increase workload pressure.",
                    "affected_component": "alert_queue",
                    "total_alerts": _safe_float(row.get("total_alerts"), 0.0),
                }
            )
    return pd.DataFrame(records)


def simulate_repeated_low_value_alerts(
    fatigue_df: pd.DataFrame,
    limit: int = 30,
) -> pd.DataFrame:
    """Create events for grouped/downgraded repeated non-critical alerts."""
    if fatigue_df.empty:
        return _empty_base_events()
    df = fatigue_df.copy()
    severity = df.get("severity", pd.Series("", index=df.index)).astype(str).str.lower()
    repeated_mask = (
        df.get("fatigue_action", pd.Series("", index=df.index)).astype(str).str.lower().isin(
            {"group_repeated", "downgrade_priority", "delay_non_critical"}
        )
        | df.get("final_alert_status", pd.Series("", index=df.index)).astype(str).str.lower().isin(
            {"grouped", "delayed", "priority_downgraded"}
        )
        | pd.to_numeric(df.get("grouped_alert_count", pd.Series(0, index=df.index)), errors="coerce").fillna(0).gt(1)
    )
    candidates = df[repeated_mask & severity.isin(["low", "medium"])].head(limit)

    records = []
    for _, row in candidates.iterrows():
        records.append(
            _base_event(
                row=row,
                related_alert_id=row.get("alert_id", ""),
                failure_mode="repeated_low_value_alerts",
                trigger_reason="Repeated non-critical alert pattern may increase fatigue risk.",
                system_impact="Can increase clinician burden and reduce perceived alert usefulness.",
                affected_component="fatigue_reduction",
                context={
                    "grouped_alert_count": row.get("grouped_alert_count", 1),
                    "fatigue_risk_score": row.get("fatigue_risk_score", 0),
                },
            )
        )
    return pd.DataFrame(records)


def simulate_delayed_response_failure(
    responses_df: pd.DataFrame,
    limit: int = 30,
) -> pd.DataFrame:
    """Create events for ignored or delayed simulated workflow responses."""
    if responses_df.empty:
        return _empty_base_events()
    df = responses_df.copy()
    response = df.get("simulated_response", pd.Series("", index=df.index)).astype(str).str.lower()
    delayed_mask = response.isin(["delayed", "ignored"])
    if "response_time_minutes" in df.columns:
        thresholds = df.get("severity", pd.Series("", index=df.index)).astype(str).str.lower().map(_severity_time_threshold_minutes)
        delayed_mask |= pd.to_numeric(df["response_time_minutes"], errors="coerce").fillna(0).gt(thresholds)
    candidates = df[delayed_mask].head(limit)

    records = []
    for _, row in candidates.iterrows():
        records.append(
            _base_event(
                row=row,
                related_alert_id=row.get("alert_id", ""),
                failure_mode="delayed_response_failure",
                trigger_reason="Simulated response was delayed, ignored, or exceeded severity-based timing expectations.",
                system_impact="May reduce escalation quality and increase simulated outcome risk.",
                affected_component="workflow_response",
                context={
                    "simulated_response": row.get("simulated_response"),
                    "response_time_minutes": row.get("response_time_minutes"),
                },
            )
        )
    return pd.DataFrame(records)


def simulate_model_confidence_drop(
    reliability_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    metrics: dict[str, Any] | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    """Create events when reliability score or drift evidence suggests lower confidence."""
    del metrics
    records = []
    if not reliability_df.empty:
        score = pd.to_numeric(reliability_df.get("reliability_score", pd.Series(dtype=float)), errors="coerce").fillna(1.0)
        review_needed = reliability_df.get("review_recommendation", pd.Series("", index=reliability_df.index)).astype(str).ne("no_action_needed")
        candidates = reliability_df[(score < 0.90) | review_needed].head(limit)
        for _, row in candidates.iterrows():
            records.append(
                {
                    "timestamp": _string_value(row.get("window_start")),
                    "patient_id": "system",
                    "related_alert_id": "",
                    "failure_mode": "model_confidence_drop",
                    "failure_trigger_reason": "Reliability score or monitoring recommendation suggests reduced confidence.",
                    "simulated_system_impact": "Requires review before relying on threshold or confidence assumptions.",
                    "affected_component": "reliability_monitor",
                    "reliability_score": _safe_float(row.get("reliability_score"), 1.0),
                    "review_recommendation": row.get("review_recommendation", ""),
                }
            )
    elif not drift_df.empty:
        severe = drift_df[drift_df.get("drift_status", pd.Series("", index=drift_df.index)).astype(str).str.lower().eq("severe_shift")]
        for _, row in severe.head(limit).iterrows():
            records.append(
                {
                    "timestamp": _string_value(row.get("window_start")),
                    "patient_id": "system",
                    "related_alert_id": "",
                    "failure_mode": "model_confidence_drop",
                    "failure_trigger_reason": "Severe drift suggests confidence should be reviewed.",
                    "simulated_system_impact": "Confidence assumptions should be inspected before deployment-style use.",
                    "affected_component": "model_monitoring",
                    "drift_score": _safe_float(row.get("drift_score"), 0.0),
                    "drift_status": row.get("drift_status", ""),
                }
            )
    return pd.DataFrame(records)


def simulate_data_distribution_shift(
    drift_df: pd.DataFrame,
    limit: int = 30,
) -> pd.DataFrame:
    """Create events for severe or review-required drift checks."""
    if drift_df.empty:
        return _empty_base_events()
    df = drift_df.copy()
    review_mask = _bool_column(df, "requires_review")
    severe_mask = df.get("drift_status", pd.Series("", index=df.index)).astype(str).str.lower().eq("severe_shift")
    candidates = df[review_mask | severe_mask].sort_values(
        by="drift_score",
        ascending=False,
        key=lambda series: pd.to_numeric(series, errors="coerce").fillna(0),
    ).head(limit)

    records = []
    for _, row in candidates.iterrows():
        records.append(
            {
                "timestamp": _string_value(row.get("window_start")),
                "patient_id": "system",
                "related_alert_id": "",
                "failure_mode": "data_distribution_shift",
                "failure_trigger_reason": (
                    f"{row.get('monitored_feature', 'feature')} showed "
                    f"{row.get('drift_status', 'drift')} in simulated drift monitoring."
                ),
                "simulated_system_impact": "Distribution shift can reduce confidence in downstream alert behavior.",
                "affected_component": "drift_monitor",
                "drift_score": _safe_float(row.get("drift_score"), 0.0),
                "drift_status": row.get("drift_status", ""),
                "monitored_feature": row.get("monitored_feature", ""),
            }
        )
    return pd.DataFrame(records)


def calculate_failure_impacts(row: pd.Series) -> dict[str, float]:
    """Calculate normalized impact scores where higher means worse impact."""
    mode = str(row.get("failure_mode", "")).lower()
    severity = str(row.get("severity", "")).lower()
    response_time = _safe_float(row.get("response_time_minutes"), 0.0)
    reliability_score = _safe_float(row.get("reliability_score"), 1.0)
    drift_score = _safe_float(row.get("drift_score"), 0.0)
    total_alerts = _safe_float(row.get("total_alerts"), 0.0)
    grouped_count = _safe_float(row.get("grouped_alert_count"), 1.0)

    defaults = {
        "alert_volume_impact": 0.20,
        "clinician_burden_impact": 0.20,
        "reliability_score_impact": 0.20,
        "drift_risk_impact": 0.10,
        "outcome_risk_impact": 0.20,
    }
    if mode == "noisy_sensor_spike":
        defaults.update(
            alert_volume_impact=0.75,
            clinician_burden_impact=0.40,
            reliability_score_impact=0.30,
            drift_risk_impact=0.25,
            outcome_risk_impact=0.30,
        )
    elif mode == "missing_patient_data":
        defaults.update(
            alert_volume_impact=0.25,
            clinician_burden_impact=0.35,
            reliability_score_impact=0.45,
            drift_risk_impact=0.25,
            outcome_risk_impact=0.35,
        )
    elif mode == "alert_overload":
        defaults.update(
            alert_volume_impact=min(total_alerts / 50.0, 1.0),
            clinician_burden_impact=min(0.45 + total_alerts / 80.0, 1.0),
            reliability_score_impact=1.0 - reliability_score,
            drift_risk_impact=0.20,
            outcome_risk_impact=0.35,
        )
    elif mode == "repeated_low_value_alerts":
        defaults.update(
            alert_volume_impact=0.35,
            clinician_burden_impact=min(0.45 + grouped_count / 10.0, 1.0),
            reliability_score_impact=0.25,
            drift_risk_impact=0.15,
            outcome_risk_impact=0.25,
        )
    elif mode == "delayed_response_failure":
        threshold = _severity_time_threshold_minutes(severity)
        delay_excess = max(response_time - threshold, 0.0)
        defaults.update(
            alert_volume_impact=0.15,
            clinician_burden_impact=0.55,
            reliability_score_impact=min(0.35 + delay_excess / 120.0, 1.0),
            drift_risk_impact=0.10,
            outcome_risk_impact=min(0.35 + delay_excess / 90.0, 1.0),
        )
    elif mode == "model_confidence_drop":
        defaults.update(
            alert_volume_impact=0.25,
            clinician_burden_impact=0.35,
            reliability_score_impact=max(1.0 - reliability_score, 0.35),
            drift_risk_impact=min(drift_score / 0.50, 1.0) if drift_score else 0.40,
            outcome_risk_impact=0.40,
        )
    elif mode == "data_distribution_shift":
        defaults.update(
            alert_volume_impact=0.35,
            clinician_burden_impact=0.35,
            reliability_score_impact=0.50,
            drift_risk_impact=min(drift_score / 0.35, 1.0),
            outcome_risk_impact=0.45,
        )

    return {key: _clip_score(value) for key, value in defaults.items()}


def determine_safety_status(row: pd.Series) -> str:
    """Map severity and impact context to a safety status."""
    mode = str(row.get("failure_mode", "")).lower()
    severity = str(row.get("severity_level", "")).lower()
    drift_status = str(row.get("drift_status", "")).lower()
    max_impact = max(
        _safe_float(row.get("alert_volume_impact"), 0.0),
        _safe_float(row.get("clinician_burden_impact"), 0.0),
        _safe_float(row.get("reliability_score_impact"), 0.0),
        _safe_float(row.get("drift_risk_impact"), 0.0),
        _safe_float(row.get("outcome_risk_impact"), 0.0),
    )

    if severity == "critical" or (mode == "data_distribution_shift" and drift_status == "severe_shift"):
        return "unsafe_review_required"
    if severity == "high" or max_impact >= 0.65:
        return "degraded"
    if severity == "medium" or max_impact >= 0.35:
        return "warning"
    return "monitored"


def generate_mitigation_recommendation(row: pd.Series) -> str:
    """Return workflow-safe mitigation guidance without clinical advice."""
    mode = str(row.get("failure_mode", "")).lower()
    recommendations = {
        "noisy_sensor_spike": "Inspect sensor reliability, verify the simulated vital-sign stream, and review alert thresholds before trusting the spike.",
        "missing_patient_data": "Perform manual verification of missing vitals and review data-quality handling before downstream interpretation.",
        "alert_overload": "Review alert thresholds, queue volume, and clinician workload assumptions for this simulated window.",
        "repeated_low_value_alerts": "Review repeated-alert grouping rules and monitor clinician workload before changing fatigue settings.",
        "delayed_response_failure": "Escalate workflow review and inspect response-time bottlenecks for delayed simulated alerts.",
        "model_confidence_drop": "Review reliability trends, calibration assumptions, and threshold settings before any simulated deployment change.",
        "data_distribution_shift": "Investigate drift pattern and require human review before recalibration or retraining review.",
    }
    return recommendations.get(mode, "Review the simulated failure context and require human verification before changing system behavior.")


def build_failure_mode_table(
    patient_df: pd.DataFrame,
    generated_alerts_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
    responses_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    metrics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the complete failure-mode event table."""
    del outcome_df
    metrics = metrics or {}
    alert_source = fatigue_df if not fatigue_df.empty else generated_alerts_df

    base_events = pd.concat(
        [
            simulate_noisy_sensor_spike(patient_df, alert_source),
            simulate_missing_patient_data(patient_df, alert_source),
            simulate_alert_overload(alert_source, reliability_df),
            simulate_repeated_low_value_alerts(fatigue_df),
            simulate_delayed_response_failure(responses_df),
            simulate_model_confidence_drop(reliability_df, drift_df, metrics),
            simulate_data_distribution_shift(drift_df),
        ],
        ignore_index=True,
        sort=False,
    )
    if base_events.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    rows = []
    for index, raw_row in base_events.iterrows():
        row = raw_row.copy()
        impacts = calculate_failure_impacts(row)
        for key, value in impacts.items():
            row[key] = value
        row["severity_level"] = _assign_severity_level(row)
        row["safety_status"] = determine_safety_status(row)
        row["requires_human_review"] = _requires_human_review(row)
        row["mitigation_recommendation"] = generate_mitigation_recommendation(row)
        row["failure_event_id"] = f"FAILURE-{index + 1:06d}"
        row["failure_simulation_note"] = SIMULATION_ONLY_NOTE
        rows.append(row)

    results = pd.DataFrame(rows)
    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in results.columns:
            results[column] = ""
    return results[REQUIRED_OUTPUT_COLUMNS].copy()


def calculate_failure_summary(results_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize failure-mode simulation results."""
    if results_df.empty:
        return {
            "total_failure_events": 0,
            "failure_mode_distribution": {},
            "severity_distribution": {},
            "safety_status_distribution": {},
            "average_alert_volume_impact": 0.0,
            "average_clinician_burden_impact": 0.0,
            "average_reliability_score_impact": 0.0,
            "average_drift_risk_impact": 0.0,
            "average_outcome_risk_impact": 0.0,
            "unsafe_review_required_count": 0,
            "human_review_required_rate": 0.0,
            "simulation_only_note": SIMULATION_ONLY_NOTE,
        }

    total = len(results_df)
    return {
        "total_failure_events": int(total),
        "failure_mode_distribution": _value_counts(results_df, "failure_mode"),
        "severity_distribution": _value_counts(results_df, "severity_level"),
        "safety_status_distribution": _value_counts(results_df, "safety_status"),
        "average_alert_volume_impact": _mean(results_df, "alert_volume_impact"),
        "average_clinician_burden_impact": _mean(results_df, "clinician_burden_impact"),
        "average_reliability_score_impact": _mean(results_df, "reliability_score_impact"),
        "average_drift_risk_impact": _mean(results_df, "drift_risk_impact"),
        "average_outcome_risk_impact": _mean(results_df, "outcome_risk_impact"),
        "unsafe_review_required_count": int(results_df["safety_status"].eq("unsafe_review_required").sum()),
        "human_review_required_rate": _safe_rate(_bool_column(results_df, "requires_human_review").sum(), total),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


def save_failure_results(df: pd.DataFrame, path: str) -> Path:
    """Save detailed failure-mode events."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_failure_summary(summary: dict[str, Any], path: str) -> Path:
    """Save failure-mode summary JSON."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_path


def run_failure_mode_pipeline(
    patient_path: str = "data/simulated/patient_monitoring.csv",
    generated_alerts_path: str = "data/processed/generated_alerts.csv",
    fatigue_alerts_path: str = "data/processed/fatigue_reduced_alerts.csv",
    response_path: str = "data/processed/clinician_response_logs.csv",
    reliability_path: str = "data/processed/reliability_monitoring_results.csv",
    drift_path: str = "data/processed/drift_detection_results.csv",
    outcome_path: str = "data/processed/outcome_effectiveness_results.csv",
    metrics_path: str = "data/processed/project_metrics_summary.json",
    results_path: str = str(OUTPUT_RESULTS_PATH),
    summary_path: str = str(OUTPUT_SUMMARY_PATH),
) -> pd.DataFrame:
    """Run the complete Step 24B failure-mode simulation pipeline."""
    patient_df = safe_load_csv(patient_path)
    generated_alerts_df = safe_load_csv(generated_alerts_path)
    fatigue_df = safe_load_csv(fatigue_alerts_path)
    responses_df = safe_load_csv(response_path)
    reliability_df = safe_load_csv(reliability_path)
    drift_df = safe_load_csv(drift_path)
    outcome_df = safe_load_csv(outcome_path)
    metrics = safe_load_json(metrics_path)

    results_df = build_failure_mode_table(
        patient_df=patient_df,
        generated_alerts_df=generated_alerts_df,
        fatigue_df=fatigue_df,
        responses_df=responses_df,
        reliability_df=reliability_df,
        drift_df=drift_df,
        outcome_df=outcome_df,
        metrics=metrics,
    )
    summary = calculate_failure_summary(results_df)
    save_failure_results(results_df, results_path)
    save_failure_summary(summary, summary_path)
    return results_df


def simulate_failure_modes() -> pd.DataFrame:
    """Compatibility wrapper for earlier placeholder imports."""
    return run_failure_mode_pipeline()


def _base_event(
    row: pd.Series,
    related_alert_id: Any,
    failure_mode: str,
    trigger_reason: str,
    system_impact: str,
    affected_component: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "timestamp": _string_value(row.get("timestamp")),
        "patient_id": _string_value(row.get("patient_id")) or "unknown",
        "related_alert_id": _string_value(related_alert_id),
        "failure_mode": failure_mode,
        "failure_trigger_reason": trigger_reason,
        "simulated_system_impact": system_impact,
        "affected_component": affected_component,
        "severity": row.get("severity", ""),
        "risk_score": row.get("risk_score", 0),
    }
    if context:
        event.update(context)
    return event


def _find_related_alert_id(row: pd.Series, alerts_df: pd.DataFrame | None) -> str:
    if alerts_df is None or alerts_df.empty or not {"patient_id", "timestamp", "alert_id"}.issubset(alerts_df.columns):
        return ""
    timestamp = _string_value(row.get("timestamp"))
    patient_id = _string_value(row.get("patient_id"))
    alerts = _normalize_timestamp(alerts_df.copy(), "timestamp")
    matches = alerts[(alerts["patient_id"].astype(str) == patient_id) & (alerts["timestamp"].astype(str) == timestamp)]
    if matches.empty:
        return ""
    return _string_value(matches.iloc[0].get("alert_id"))


def _assign_severity_level(row: pd.Series) -> str:
    if str(row.get("drift_status", "")).lower() == "severe_shift":
        return "critical"
    max_impact = max(
        _safe_float(row.get("alert_volume_impact"), 0.0),
        _safe_float(row.get("clinician_burden_impact"), 0.0),
        _safe_float(row.get("reliability_score_impact"), 0.0),
        _safe_float(row.get("drift_risk_impact"), 0.0),
        _safe_float(row.get("outcome_risk_impact"), 0.0),
    )
    if max_impact >= 0.85:
        return "critical"
    if max_impact >= 0.65:
        return "high"
    if max_impact >= 0.35:
        return "medium"
    return "low"


def _requires_human_review(row: pd.Series) -> bool:
    severity = str(row.get("severity_level", "")).lower()
    safety_status = str(row.get("safety_status", "")).lower()
    return severity in {"high", "critical"} or safety_status == "unsafe_review_required"


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _empty_base_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "patient_id",
            "related_alert_id",
            "failure_mode",
            "failure_trigger_reason",
            "simulated_system_impact",
            "affected_component",
        ]
    )


def _normalize_timestamp(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(False, index=df.index)
    text_true = df[column].astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
    numeric_true = pd.to_numeric(df[column], errors="coerce").fillna(0).ne(0)
    return text_true | numeric_true


def _severity_time_threshold_minutes(severity: str) -> float:
    return {
        "critical": 10.0,
        "high": 15.0,
        "medium": 30.0,
        "low": 60.0,
    }.get(str(severity).lower(), 60.0)


def _string_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip_score(value: Any) -> float:
    return round(max(0.0, min(1.0, _safe_float(value, 0.0))), 4)


def _safe_rate(numerator: Any, denominator: Any) -> float:
    denominator_float = _safe_float(denominator, 0.0)
    if denominator_float <= 0:
        return 0.0
    return round(_safe_float(numerator, 0.0) / denominator_float, 4)


def _mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return round(float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean()), 4)


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    counts = df[column].fillna("missing").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


if __name__ == "__main__":
    results = run_failure_mode_pipeline()
    summary = calculate_failure_summary(results)
    print(f"Total failure events: {summary['total_failure_events']}")
    print("Failure mode distribution:")
    print(pd.Series(summary["failure_mode_distribution"]).to_string() if summary["failure_mode_distribution"] else "none")
    print("Severity distribution:")
    print(pd.Series(summary["severity_distribution"]).to_string() if summary["severity_distribution"] else "none")
    print(f"Unsafe review count: {summary['unsafe_review_required_count']}")
    print(f"Average reliability impact: {summary['average_reliability_score_impact']:.4f}")
    print(f"Saved results to {_resolve_project_path(OUTPUT_RESULTS_PATH)}")
    print(f"Saved summary to {_resolve_project_path(OUTPUT_SUMMARY_PATH)}")
