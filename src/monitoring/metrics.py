"""Centralized project metrics for simulated alert-reliability evaluation.

Step 22 summarizes existing artifacts for dashboards and future reports. These
metrics describe a simulated engineering prototype only; they do not establish
clinical validation, medical safety, or real-world performance.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_JSON_PATH = Path("data/processed/project_metrics_summary.json")
DEFAULT_OUTPUT_CSV_PATH = Path("data/processed/project_metrics_table.csv")
SIMULATION_ONLY_NOTE = (
    "Metrics are calculated from simulated project artifacts only; they are not "
    "clinical validation and must not be interpreted as real-world medical safety."
)


def safe_load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV if present; return an empty dataframe if missing/unreadable."""
    csv_path = _resolve_project_path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def safe_load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object if present; return an empty dict if missing/unreadable."""
    json_path = _resolve_project_path(path)
    if not json_path.exists():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with a safe default."""
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_rate(numerator: Any, denominator: Any) -> float:
    """Calculate a bounded rate while avoiding divide-by-zero."""
    denominator_float = safe_float(denominator)
    if denominator_float <= 0:
        return 0.0
    return _round_metric(safe_float(numerator) / denominator_float)


def calculate_dataset_metrics(processed_df: pd.DataFrame) -> dict[str, Any]:
    """Calculate high-level dataset metrics."""
    total_rows = int(len(processed_df))
    return {
        "total_patients": int(processed_df["patient_id"].nunique()) if "patient_id" in processed_df.columns else 0,
        "total_vital_rows": total_rows,
        "deterioration_event_rate": _boolean_rate(processed_df, "deterioration_event"),
        "missing_data_rate": _missing_data_rate(processed_df),
        "sensor_noise_rate": _boolean_rate(processed_df, "sensor_noise_flag"),
    }


def calculate_alert_metrics(
    raw_alerts_df: pd.DataFrame,
    audited_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate alert volume and preservation metrics."""
    total_fatigue = int(len(fatigue_df))
    active_count = _status_count(fatigue_df, "final_alert_status", "active")
    critical_mask = _bool_series(fatigue_df, "critical_flag")
    critical_count = int(critical_mask.sum()) if not fatigue_df.empty else 0
    if critical_count and "final_alert_status" in fatigue_df.columns:
        critical_active_count = int(
            fatigue_df.loc[critical_mask, "final_alert_status"]
            .astype(str)
            .str.lower()
            .eq("active")
            .sum()
        )
        critical_preservation_rate = safe_rate(critical_active_count, critical_count)
    elif critical_count:
        critical_preservation_rate = 0.0
    else:
        critical_preservation_rate = 1.0

    return {
        "total_raw_alerts": int(len(raw_alerts_df)),
        "total_audited_alerts": int(len(audited_df)),
        "total_fatigue_reduced_alerts": total_fatigue,
        "active_alerts_after_reduction": active_count,
        "critical_alert_count": critical_count,
        "alert_reduction_rate": safe_rate(total_fatigue - active_count, total_fatigue),
        "critical_preservation_rate": critical_preservation_rate,
    }


def calculate_audit_fatigue_metrics(
    audited_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate audit quality and fatigue-reduction metrics."""
    return {
        "average_actionability_score": _mean(audited_df, "actionability_score"),
        "average_fatigue_risk_score": _mean(audited_df, "fatigue_risk_score"),
        "average_false_positive_likelihood": _mean(audited_df, "false_positive_likelihood"),
        "grouped_alert_count": _combined_status_count(
            fatigue_df,
            ("final_alert_status", "grouped"),
            ("fatigue_action", "group_repeated"),
        ),
        "delayed_alert_count": _combined_status_count(
            fatigue_df,
            ("final_alert_status", "delayed"),
            ("fatigue_action", "delay_non_critical"),
        ),
        "downgraded_alert_count": _combined_status_count(
            fatigue_df,
            ("final_alert_status", "priority_downgraded"),
            ("fatigue_action", "downgrade_priority"),
        ),
    }


def calculate_workflow_metrics(
    response_df: pd.DataFrame,
    response_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate simulated workflow-response metrics."""
    response_summary = response_summary or {}
    return {
        "total_responses": int(len(response_df)),
        "ignored_alert_rate": _summary_or_rate(response_summary, "ignored_alert_rate", response_df, "simulated_response", "ignored"),
        "delayed_alert_rate": _summary_or_rate(response_summary, "delayed_alert_rate", response_df, "simulated_response", "delayed"),
        "escalation_rate": _summary_or_rate(response_summary, "escalation_rate", response_df, "simulated_response", "escalated"),
        "average_response_time_minutes": _summary_or_mean(response_summary, "average_response_time_minutes", response_df, "response_time_minutes"),
        "average_clinician_burden_score": _summary_or_mean(response_summary, "average_clinician_burden_score", response_df, "clinician_burden_score"),
        "average_perceived_alert_usefulness": _summary_or_mean(response_summary, "average_perceived_alert_usefulness", response_df, "perceived_alert_usefulness"),
    }


def calculate_reliability_metrics(
    reliability_df: pd.DataFrame,
    reliability_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate reliability-window metrics."""
    reliability_summary = reliability_summary or {}
    status_counts = _value_counts(reliability_df, "reliability_status")
    return {
        "average_reliability_score": _summary_or_mean(reliability_summary, "average_reliability_score", reliability_df, "reliability_score"),
        "stable_window_count": int(status_counts.get("stable", 0)),
        "watch_window_count": int(status_counts.get("watch", 0)),
        "degraded_window_count": int(status_counts.get("degraded", 0)),
        "unsafe_window_count": int(
            reliability_summary.get(
                "unsafe_windows",
                status_counts.get("unsafe_review_required", 0),
            )
        ),
        "windows_requiring_review": int(
            reliability_summary.get(
                "windows_requiring_review",
                (reliability_df.get("review_recommendation", pd.Series(dtype=str)).astype(str) != "no_action_needed").sum()
                if not reliability_df.empty
                else 0,
            )
        ),
    }


def calculate_drift_metrics(
    drift_df: pd.DataFrame,
    drift_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate drift monitoring metrics."""
    drift_summary = drift_summary or {}
    status_counts = _value_counts(drift_df, "drift_status")
    drift_type_counts = _value_counts(drift_df, "drift_type")
    most_common_drift_type = max(drift_type_counts, key=drift_type_counts.get) if drift_type_counts else ""
    return {
        "average_drift_score": _summary_or_mean(drift_summary, "average_drift_score", drift_df, "drift_score"),
        "severe_drift_count": int(drift_summary.get("severe_drift_count", status_counts.get("severe_shift", 0))),
        "moderate_drift_count": int(status_counts.get("moderate_shift", 0)),
        "drift_checks_requiring_review": int(
            drift_summary.get(
                "checks_requiring_review",
                _bool_series(drift_df, "requires_review").sum() if not drift_df.empty else 0,
            )
        ),
        "most_common_drift_type": most_common_drift_type,
    }


def calculate_model_update_metrics(
    model_update_df: pd.DataFrame,
    rl_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate model-update and RL-threshold simulation metrics."""
    rl_summary = rl_summary or {}
    row = model_update_df.iloc[-1].to_dict() if not model_update_df.empty else {}
    human_review_required = _coerce_bool(row.get("human_review_required")) or _coerce_bool(
        rl_summary.get("human_review_required")
    )
    return {
        "current_threshold": _round_metric(row.get("current_risk_threshold", rl_summary.get("current_threshold", 0.0))),
        "proposed_threshold": _round_metric(row.get("proposed_risk_threshold", rl_summary.get("recommended_threshold", 0.0))),
        "threshold_change": _round_metric(row.get("threshold_change", 0.0)),
        "deployment_recommendation": str(row.get("deployment_recommendation", "")),
        "rl_recommended_action": str(rl_summary.get("recommended_action", "")),
        "rl_recommended_threshold": _round_metric(rl_summary.get("recommended_threshold", 0.0)),
        "rl_safety_violation_count": int(rl_summary.get("safety_violation_count", 0)),
        "human_review_required": bool(human_review_required),
    }


def calculate_llm_action_metrics(
    explanations_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate explanation, RAG, and action recommendation metrics."""
    total_recommendations = int(len(recommendations_df))
    rag_nonempty = int(
        recommendations_df.get("rag_sources", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    return {
        "total_explanations": int(len(explanations_df)),
        "fallback_explanation_count": int(_bool_series(explanations_df, "fallback_used").sum()) if not explanations_df.empty else 0,
        "total_action_recommendations": total_recommendations,
        "immediate_action_count": _status_count(recommendations_df, "action_priority", "immediate"),
        "urgent_action_count": _status_count(recommendations_df, "action_priority", "urgent"),
        "rag_coverage_rate": safe_rate(rag_nonempty, total_recommendations),
    }


def calculate_database_readiness_metrics(
    db_path: str | Path = "data/processed/clinical_alert_reliability.db",
) -> dict[str, Any]:
    """Calculate SQLite demo-readiness metrics."""
    database_path = _resolve_project_path(db_path)
    if not database_path.exists():
        return {
            "database_available": False,
            "database_table_count": 0,
            "database_total_rows_loaded": 0,
        }

    table_count = 0
    total_rows = 0
    try:
        with sqlite3.connect(database_path) as connection:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                connection,
            )["name"].tolist()
            table_count = len(tables)
            for table in tables:
                quoted_table = '"' + str(table).replace('"', '""') + '"'
                row = connection.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()
                total_rows += int(row[0]) if row else 0
    except Exception:
        return {
            "database_available": False,
            "database_table_count": 0,
            "database_total_rows_loaded": 0,
        }

    return {
        "database_available": True,
        "database_table_count": int(table_count),
        "database_total_rows_loaded": int(total_rows),
    }


def compile_project_metrics(
    processed_df: pd.DataFrame,
    raw_alerts_df: pd.DataFrame,
    audited_df: pd.DataFrame,
    fatigue_df: pd.DataFrame,
    response_df: pd.DataFrame,
    response_summary: dict[str, Any] | None,
    reliability_df: pd.DataFrame,
    reliability_summary: dict[str, Any] | None,
    drift_df: pd.DataFrame,
    drift_summary: dict[str, Any] | None,
    model_update_df: pd.DataFrame,
    rl_summary: dict[str, Any] | None,
    explanations_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    db_path: str | Path = "data/processed/clinical_alert_reliability.db",
) -> dict[str, Any]:
    """Compile all project metric categories into one nested dictionary."""
    return {
        "dataset": calculate_dataset_metrics(processed_df),
        "alerts": calculate_alert_metrics(raw_alerts_df, audited_df, fatigue_df),
        "audit_fatigue": calculate_audit_fatigue_metrics(audited_df, fatigue_df),
        "workflow": calculate_workflow_metrics(response_df, response_summary),
        "reliability": calculate_reliability_metrics(reliability_df, reliability_summary),
        "drift": calculate_drift_metrics(drift_df, drift_summary),
        "model_update_rl": calculate_model_update_metrics(model_update_df, rl_summary),
        "llm_action": calculate_llm_action_metrics(explanations_df, recommendations_df),
        "database_demo_readiness": calculate_database_readiness_metrics(db_path),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


def flatten_metrics_for_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Flatten nested metric categories into a table for dashboards/reports."""
    rows: list[dict[str, Any]] = []
    for category, values in metrics.items():
        if category == "simulation_only_note":
            rows.append(
                {
                    "category": "project",
                    "metric": "simulation_only_note",
                    "value": values,
                }
            )
            continue
        if not isinstance(values, dict):
            continue
        for metric, value in values.items():
            rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows, columns=["category", "metric", "value"])


def save_metrics_summary(metrics: dict[str, Any], path: str | Path) -> Path:
    """Save nested metrics JSON."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return output_path


def save_metrics_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Save flattened metrics table CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_metrics_pipeline(
    processed_path: str | Path = "data/processed/processed_data.csv",
    raw_alerts_path: str | Path = "data/processed/generated_alerts.csv",
    audited_path: str | Path = "data/processed/audited_alerts.csv",
    fatigue_path: str | Path = "data/processed/fatigue_reduced_alerts.csv",
    response_path: str | Path = "data/processed/clinician_response_logs.csv",
    response_summary_path: str | Path = "data/processed/clinician_response_summary.json",
    reliability_path: str | Path = "data/processed/reliability_monitoring_results.csv",
    reliability_summary_path: str | Path = "data/processed/reliability_summary.json",
    drift_path: str | Path = "data/processed/drift_detection_results.csv",
    drift_summary_path: str | Path = "data/processed/drift_summary.json",
    model_update_path: str | Path = "data/processed/model_update_simulation_results.csv",
    rl_summary_path: str | Path = "data/processed/rl_threshold_policy_summary.json",
    explanations_path: str | Path = "data/processed/alert_explanations.csv",
    recommendations_path: str | Path = "data/processed/action_recommendations.csv",
    db_path: str | Path = "data/processed/clinical_alert_reliability.db",
    output_json_path: str | Path = DEFAULT_OUTPUT_JSON_PATH,
    output_csv_path: str | Path = DEFAULT_OUTPUT_CSV_PATH,
) -> dict[str, Any]:
    """Run the full Step 22 metrics pipeline."""
    metrics = compile_project_metrics(
        processed_df=safe_load_csv(processed_path),
        raw_alerts_df=safe_load_csv(raw_alerts_path),
        audited_df=safe_load_csv(audited_path),
        fatigue_df=safe_load_csv(fatigue_path),
        response_df=safe_load_csv(response_path),
        response_summary=safe_load_json(response_summary_path),
        reliability_df=safe_load_csv(reliability_path),
        reliability_summary=safe_load_json(reliability_summary_path),
        drift_df=safe_load_csv(drift_path),
        drift_summary=safe_load_json(drift_summary_path),
        model_update_df=safe_load_csv(model_update_path),
        rl_summary=safe_load_json(rl_summary_path),
        explanations_df=safe_load_csv(explanations_path),
        recommendations_df=safe_load_csv(recommendations_path),
        db_path=db_path,
    )
    table = flatten_metrics_for_table(metrics)
    save_metrics_summary(metrics, output_json_path)
    save_metrics_table(table, output_csv_path)
    return metrics


def summarize_metrics() -> dict[str, Any]:
    """Compatibility wrapper for older placeholder imports."""
    return run_metrics_pipeline()


def _missing_data_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    if "missing_data_flag" in df.columns:
        return _boolean_rate(df, "missing_data_flag")
    if "had_missing_vitals_before_imputation" in df.columns:
        return _boolean_rate(df, "had_missing_vitals_before_imputation")
    if "missing_value_count" in df.columns:
        return safe_rate((pd.to_numeric(df["missing_value_count"], errors="coerce").fillna(0) > 0).sum(), len(df))
    return 0.0


def _boolean_rate(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return safe_rate(_bool_series(df, column).sum(), len(df))


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(False)
    text_true = values.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
    numeric_true = pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    return text_true | numeric_true


def _status_count(df: pd.DataFrame, column: str, status: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.lower().eq(status.lower()).sum())


def _action_count(df: pd.DataFrame, column: str, action: str) -> int:
    return _status_count(df, column, action)


def _combined_status_count(df: pd.DataFrame, *column_value_pairs: tuple[str, str]) -> int:
    if df.empty:
        return 0
    mask = pd.Series(False, index=df.index)
    for column, expected_value in column_value_pairs:
        if column not in df.columns:
            continue
        mask = mask | df[column].fillna("").astype(str).str.lower().eq(expected_value.lower())
    return int(mask.sum())


def _mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return _round_metric(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean())


def _summary_or_mean(summary: dict[str, Any], key: str, df: pd.DataFrame, column: str) -> float:
    if key in summary:
        return _round_metric(summary.get(key))
    return _mean(df, column)


def _summary_or_rate(
    summary: dict[str, Any],
    key: str,
    df: pd.DataFrame,
    column: str,
    value: str,
) -> float:
    if key in summary:
        return _round_metric(summary.get(key))
    if df.empty or column not in df.columns:
        return 0.0
    return safe_rate(df[column].astype(str).str.lower().eq(value.lower()).sum(), len(df))


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    normalized = df[column].fillna("").astype(str).str.strip().str.lower()
    return {str(key): int(value) for key, value in normalized.value_counts().items()}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _round_metric(value: Any) -> float:
    return round(safe_float(value), 4)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


if __name__ == "__main__":
    project_metrics = run_metrics_pipeline()
    print("Metric categories:")
    for category in project_metrics:
        print(f"  {category}")
    print("\nHeadline metrics:")
    print(f"  total patients: {project_metrics['dataset']['total_patients']}")
    print(f"  total raw alerts: {project_metrics['alerts']['total_raw_alerts']}")
    print(
        "  active alerts after reduction: "
        f"{project_metrics['alerts']['active_alerts_after_reduction']}"
    )
    print(
        "  average reliability score: "
        f"{project_metrics['reliability']['average_reliability_score']}"
    )
    print(f"  severe drift count: {project_metrics['drift']['severe_drift_count']}")
    print(
        "  total action recommendations: "
        f"{project_metrics['llm_action']['total_action_recommendations']}"
    )
    print(f"\nSaved metrics JSON to {_resolve_project_path(DEFAULT_OUTPUT_JSON_PATH)}")
    print(f"Saved metrics table to {_resolve_project_path(DEFAULT_OUTPUT_CSV_PATH)}")
