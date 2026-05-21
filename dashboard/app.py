"""Streamlit dashboard for the simulated clinical alert reliability prototype.

This dashboard is intentionally read-only. It displays artifacts created by the
roadmap pipeline and does not train models, call LLMs, or make clinical claims.
All data shown here is simulated demo data for engineering/portfolio review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_NOTE = (
    "Simulation-only research dashboard. This project is not clinically "
    "validated, is not a medical device, and must not be used for real patient care."
)

DATA_PATHS = {
    "metrics_summary": "data/processed/project_metrics_summary.json",
    "metrics_table": "data/processed/project_metrics_table.csv",
    "processed": "data/processed/processed_data.csv",
    "raw_alerts": "data/processed/generated_alerts.csv",
    "audited": "data/processed/audited_alerts.csv",
    "fatigue": "data/processed/fatigue_reduced_alerts.csv",
    "responses": "data/processed/clinician_response_logs.csv",
    "reliability": "data/processed/reliability_monitoring_results.csv",
    "drift": "data/processed/drift_detection_results.csv",
    "explanations": "data/processed/alert_explanations.csv",
    "recommendations": "data/processed/action_recommendations.csv",
    "model_updates": "data/processed/model_update_simulation_results.csv",
    "rl_policy": "data/processed/rl_threshold_policy_summary.json",
}

PAGES = [
    "Overview",
    "Patient vitals simulation",
    "Active alerts",
    "Alert auditing",
    "Alert fatigue reduction",
    "Clinician workflow simulation",
    "Reliability monitoring",
    "Drift detection",
    "LLM explanations",
    "Action recommendations",
    "System limitations",
]


def safe_load_csv(path: str) -> pd.DataFrame:
    """Load a CSV artifact without crashing the dashboard if it is missing."""
    file_path = resolve_project_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def safe_load_json(path: str) -> dict[str, Any]:
    """Load a JSON artifact without crashing the dashboard if it is missing."""
    file_path = resolve_project_path(path)
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def format_percentage(value: Any) -> str:
    """Format a rate-like value as a percentage string."""
    try:
        if value is None or pd.isna(value):
            return "0.00%"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def render_metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    """Render a compact Streamlit metric card."""
    st.metric(label=label, value=value, help=help_text)


def render_simulation_disclaimer() -> None:
    """Display the dashboard safety boundary near the top of each page."""
    st.warning(SIMULATION_NOTE)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_dashboard_data() -> dict[str, Any]:
    """Load all dashboard inputs once per render cycle."""
    return {
        "metrics_summary": safe_load_json(DATA_PATHS["metrics_summary"]),
        "metrics_table": safe_load_csv(DATA_PATHS["metrics_table"]),
        "processed": safe_load_csv(DATA_PATHS["processed"]),
        "raw_alerts": safe_load_csv(DATA_PATHS["raw_alerts"]),
        "audited": safe_load_csv(DATA_PATHS["audited"]),
        "fatigue": safe_load_csv(DATA_PATHS["fatigue"]),
        "responses": safe_load_csv(DATA_PATHS["responses"]),
        "reliability": safe_load_csv(DATA_PATHS["reliability"]),
        "drift": safe_load_csv(DATA_PATHS["drift"]),
        "explanations": safe_load_csv(DATA_PATHS["explanations"]),
        "recommendations": safe_load_csv(DATA_PATHS["recommendations"]),
        "model_updates": safe_load_csv(DATA_PATHS["model_updates"]),
        "rl_policy": safe_load_json(DATA_PATHS["rl_policy"]),
    }


def render_overview(data: dict[str, Any]) -> None:
    st.header("Overview")
    metrics = data.get("metrics_summary", {})
    _warn_missing(DATA_PATHS["metrics_summary"], metrics)

    dataset = metrics.get("dataset", {})
    alerts = metrics.get("alerts", {})
    reliability = metrics.get("reliability", {})
    drift = metrics.get("drift", {})
    llm_action = metrics.get("llm_action", {})

    first_row = st.columns(5)
    with first_row[0]:
        render_metric_card("Patients", dataset.get("total_patients", 0))
    with first_row[1]:
        render_metric_card("Vital rows", dataset.get("total_vital_rows", 0))
    with first_row[2]:
        render_metric_card("Raw alerts", alerts.get("total_raw_alerts", 0))
    with first_row[3]:
        render_metric_card("Active alerts", alerts.get("active_alerts_after_reduction", 0))
    with first_row[4]:
        render_metric_card("Action recs", llm_action.get("total_action_recommendations", 0))

    second_row = st.columns(4)
    with second_row[0]:
        render_metric_card("Alert reduction", format_percentage(alerts.get("alert_reduction_rate", 0)))
    with second_row[1]:
        render_metric_card(
            "Critical preservation",
            format_percentage(alerts.get("critical_preservation_rate", 0)),
            "Critical simulated alerts should remain active.",
        )
    with second_row[2]:
        render_metric_card("Reliability score", reliability.get("average_reliability_score", 0))
    with second_row[3]:
        render_metric_card("Severe drift checks", drift.get("severe_drift_count", 0))

    metrics_table = data.get("metrics_table", pd.DataFrame())
    if _has_data(metrics_table):
        st.subheader("Project Metrics Table")
        st.dataframe(metrics_table.head(80), use_container_width=True, hide_index=True)


def render_patient_vitals_simulation(data: dict[str, Any]) -> None:
    st.header("Patient Vitals Simulation")
    vitals = data.get("processed", pd.DataFrame())
    if not _require_dataframe(vitals, DATA_PATHS["processed"]):
        return

    filtered = _apply_select_filter(vitals, "patient_id", "Patient", "vitals_patient")
    st.dataframe(_limited_table(filtered), use_container_width=True, hide_index=True)

    count_cols = st.columns(3)
    with count_cols[0]:
        render_metric_card("Deterioration events", _truthy_count(filtered, "deterioration_event"))
    with count_cols[1]:
        render_metric_card("Sensor noise rows", _truthy_count(filtered, "sensor_noise_flag"))
    with count_cols[2]:
        render_metric_card("Missing-data rows", _truthy_count(filtered, "missing_data_flag"))

    chart_columns = [
        column
        for column in ["heart_rate", "oxygen_saturation", "respiratory_rate"]
        if column in filtered.columns
    ]
    if chart_columns and "timestamp" in filtered.columns:
        chart_df = filtered.copy()
        chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], errors="coerce")
        chart_df = chart_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if _has_data(chart_df):
            st.subheader("Vital Sign Trends")
            st.line_chart(chart_df.set_index("timestamp")[chart_columns])


def render_active_alerts(data: dict[str, Any]) -> None:
    st.header("Active Alerts")
    alerts = data.get("fatigue", pd.DataFrame())
    if not _require_dataframe(alerts, DATA_PATHS["fatigue"]):
        return

    filtered = _apply_alert_filters(alerts, prefix="active_alerts")
    cols = st.columns(2)
    with cols[0]:
        _render_distribution(filtered, "severity", "Severity Distribution")
    with cols[1]:
        _render_distribution(filtered, "final_alert_status", "Final Alert Status")
    st.dataframe(_limited_table(filtered), use_container_width=True, hide_index=True)


def render_alert_auditing(data: dict[str, Any]) -> None:
    st.header("Alert Auditing")
    audited = data.get("audited", pd.DataFrame())
    if not _require_dataframe(audited, DATA_PATHS["audited"]):
        return

    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Avg actionability", _mean_value(audited, "actionability_score"))
    with cols[1]:
        render_metric_card("Avg fatigue risk", _mean_value(audited, "fatigue_risk_score"))
    with cols[2]:
        render_metric_card("Avg false-positive likelihood", _mean_value(audited, "false_positive_likelihood"))

    _render_distribution(audited, "audit_status", "Audit Status Distribution")
    st.dataframe(_limited_table(audited), use_container_width=True, hide_index=True)


def render_alert_fatigue_reduction(data: dict[str, Any]) -> None:
    st.header("Alert Fatigue Reduction")
    raw_alerts = data.get("raw_alerts", pd.DataFrame())
    fatigue = data.get("fatigue", pd.DataFrame())
    metrics = data.get("metrics_summary", {}).get("alerts", {})
    if not _require_dataframe(fatigue, DATA_PATHS["fatigue"]):
        return

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Raw alerts", len(raw_alerts))
    with cols[1]:
        render_metric_card("Active after reduction", metrics.get("active_alerts_after_reduction", 0))
    with cols[2]:
        render_metric_card("Reduction rate", format_percentage(metrics.get("alert_reduction_rate", 0)))
    with cols[3]:
        render_metric_card("Critical preservation", format_percentage(metrics.get("critical_preservation_rate", 0)))

    st.info("Critical and safety-sensitive simulated alerts are preserved; reduced alerts remain in the table for auditability.")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        _render_distribution(fatigue, "fatigue_action", "Fatigue Action Distribution")
    with chart_cols[1]:
        _render_distribution(fatigue, "final_alert_status", "Final Alert Status Distribution")
    st.dataframe(_limited_table(fatigue), use_container_width=True, hide_index=True)


def render_clinician_workflow_simulation(data: dict[str, Any]) -> None:
    st.header("Clinician Workflow Simulation")
    responses = data.get("responses", pd.DataFrame())
    workflow = data.get("metrics_summary", {}).get("workflow", {})
    if not _require_dataframe(responses, DATA_PATHS["responses"]):
        return

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Avg response time", workflow.get("average_response_time_minutes", 0))
    with cols[1]:
        render_metric_card("Ignored rate", format_percentage(workflow.get("ignored_alert_rate", 0)))
    with cols[2]:
        render_metric_card("Delayed rate", format_percentage(workflow.get("delayed_alert_rate", 0)))
    with cols[3]:
        render_metric_card("Escalation rate", format_percentage(workflow.get("escalation_rate", 0)))

    _render_distribution(responses, "simulated_response", "Simulated Response Distribution")
    if {"severity", "simulated_response"}.issubset(responses.columns):
        st.subheader("Response by Severity")
        summary = (
            responses.groupby(["severity", "simulated_response"])
            .size()
            .reset_index(name="count")
            .sort_values(["severity", "simulated_response"])
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
    st.dataframe(_limited_table(responses), use_container_width=True, hide_index=True)


def render_reliability_monitoring(data: dict[str, Any]) -> None:
    st.header("Reliability Monitoring")
    reliability = data.get("reliability", pd.DataFrame())
    if not _require_dataframe(reliability, DATA_PATHS["reliability"]):
        return

    if "reliability_score" in reliability.columns:
        chart_df = reliability.copy()
        index_column = "window_start" if "window_start" in chart_df.columns else None
        if index_column:
            chart_df[index_column] = pd.to_datetime(chart_df[index_column], errors="coerce")
            chart_df = chart_df.dropna(subset=[index_column]).sort_values(index_column)
            st.line_chart(chart_df.set_index(index_column)["reliability_score"])
        else:
            st.line_chart(chart_df["reliability_score"])

    cols = st.columns(2)
    with cols[0]:
        _render_distribution(reliability, "reliability_status", "Reliability Status")
    with cols[1]:
        _render_distribution(reliability, "review_recommendation", "Review Recommendations")
    st.dataframe(_limited_table(reliability), use_container_width=True, hide_index=True)


def render_drift_detection(data: dict[str, Any]) -> None:
    st.header("Drift Detection")
    drift = data.get("drift", pd.DataFrame())
    metrics = data.get("metrics_summary", {}).get("drift", {})
    if not _require_dataframe(drift, DATA_PATHS["drift"]):
        return

    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Severe drift checks", metrics.get("severe_drift_count", 0))
    with cols[1]:
        render_metric_card("Moderate drift checks", metrics.get("moderate_drift_count", 0))
    with cols[2]:
        render_metric_card("Checks requiring review", metrics.get("drift_checks_requiring_review", 0))

    _render_distribution(drift, "drift_status", "Drift Status Distribution")
    if {"monitored_feature", "drift_score"}.issubset(drift.columns):
        top_features = (
            drift.assign(drift_score=pd.to_numeric(drift["drift_score"], errors="coerce"))
            .dropna(subset=["drift_score"])
            .sort_values("drift_score", ascending=False)
            .head(10)[["monitored_feature", "drift_type", "drift_score", "drift_status"]]
        )
        st.subheader("Top Drift Features")
        st.dataframe(top_features, use_container_width=True, hide_index=True)
    st.dataframe(_limited_table(drift), use_container_width=True, hide_index=True)


def render_llm_explanations(data: dict[str, Any]) -> None:
    st.header("LLM Explanations")
    explanations = data.get("explanations", pd.DataFrame())
    if not _require_dataframe(explanations, DATA_PATHS["explanations"]):
        return

    st.info("Explanations are simulation-only support text. They are not medical advice and may be rule-based fallback text.")
    filtered = _apply_select_filter(explanations, "severity", "Severity", "explanation_severity")
    filtered = _apply_select_filter(filtered, "fallback_used", "Fallback used", "explanation_fallback")
    st.dataframe(_limited_table(filtered), use_container_width=True, hide_index=True)

    if "explanation_text" in filtered.columns:
        st.subheader("Sample Explanation Text")
        for _, row in filtered.head(5).iterrows():
            alert_id = row.get("alert_id", "unknown")
            severity = row.get("severity", "unknown")
            with st.expander(f"{alert_id} | {severity}"):
                st.write(str(row.get("explanation_text", "")))
                st.caption(str(row.get("safety_note", SIMULATION_NOTE)))


def render_action_recommendations(data: dict[str, Any]) -> None:
    st.header("Action Recommendations")
    recommendations = data.get("recommendations", pd.DataFrame())
    if not _require_dataframe(recommendations, DATA_PATHS["recommendations"]):
        return

    st.info("Recommendations are workflow suggestions only. They do not diagnose, recommend treatment, or replace human review.")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        _render_distribution(recommendations, "recommended_action", "Recommended Action")
    with chart_cols[1]:
        _render_distribution(recommendations, "action_priority", "Action Priority")

    rag_rate = _rag_coverage_rate(recommendations)
    render_metric_card("RAG source coverage", format_percentage(rag_rate))
    st.dataframe(_limited_table(recommendations), use_container_width=True, hide_index=True)


def render_system_limitations(data: dict[str, Any]) -> None:
    st.header("System Limitations")
    st.markdown(
        """
        - This dashboard uses simulated data only.
        - The project is not clinically validated and is not for real patient use.
        - Model drift, alert fatigue, clinician responses, and outcomes are simulated engineering artifacts.
        - LLM explanations may use local fallback or rule-based text and can be incomplete.
        - Action recommendations are workflow-oriented only and must not be interpreted as treatment guidance.
        - Human review is required for safety-sensitive simulated alerts.
        - Metrics summarize a portfolio prototype, not real-world medical safety or clinical performance.
        """
    )

    rl_policy = data.get("rl_policy", {})
    model_updates = data.get("model_updates", pd.DataFrame())
    if rl_policy:
        st.subheader("RL Threshold Policy Summary")
        st.json(rl_policy)
    if _has_data(model_updates):
        st.subheader("Model Update Simulation Records")
        st.dataframe(_limited_table(model_updates), use_container_width=True, hide_index=True)


def _apply_alert_filters(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    filtered = df.copy()
    for column, label in [
        ("patient_id", "Patient"),
        ("severity", "Severity"),
        ("alert_type", "Alert type"),
        ("final_alert_status", "Final status"),
    ]:
        filtered = _apply_select_filter(filtered, column, label, f"{prefix}_{column}")
    return filtered


def _apply_select_filter(df: pd.DataFrame, column: str, label: str, key: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    values = sorted(str(value) for value in df[column].dropna().unique())
    if not values:
        return df
    selected = st.selectbox(label, ["All", *values], key=key)
    if selected == "All":
        return df
    return df[df[column].astype(str) == selected]


def _render_distribution(df: pd.DataFrame, column: str, title: str) -> None:
    st.subheader(title)
    if df.empty or column not in df.columns:
        st.info(f"No `{column}` data available.")
        return
    counts = (
        df[column]
        .fillna("missing")
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )
    st.bar_chart(counts.set_index(column)["count"])


def _limited_table(df: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    return df.head(limit).copy() if _has_data(df) else pd.DataFrame()


def _require_dataframe(df: pd.DataFrame, path: str) -> bool:
    if _has_data(df):
        return True
    st.warning(f"No data available for `{path}`. Run the upstream pipeline step that creates this file.")
    return False


def _warn_missing(path: str, loaded_value: Any) -> None:
    if loaded_value:
        return
    if not resolve_project_path(path).exists():
        st.warning(f"Missing dashboard input: `{path}`")


def _has_data(df: Any) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def _truthy_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    text_true = df[column].astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
    numeric_true = pd.to_numeric(df[column], errors="coerce").fillna(0).ne(0)
    return int((text_true | numeric_true).sum())


def _mean_value(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return round(float(pd.to_numeric(df[column], errors="coerce").fillna(0).mean()), 4)


def _rag_coverage_rate(recommendations: pd.DataFrame) -> float:
    if recommendations.empty or "rag_sources" not in recommendations.columns:
        return 0.0
    covered = recommendations["rag_sources"].fillna("").astype(str).str.strip().ne("").sum()
    return float(covered) / float(len(recommendations)) if len(recommendations) else 0.0


def main() -> None:
    st.set_page_config(
        page_title="Clinical Alert Reliability AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Clinical Alert Reliability AI")
    st.caption("End-to-end simulated alert reliability dashboard for engineering review.")
    render_simulation_disclaimer()

    page = st.sidebar.radio("Dashboard section", PAGES)
    st.sidebar.caption("Read-only demo dashboard. No models or LLMs run here.")
    data = load_dashboard_data()

    page_renderers = {
        "Overview": render_overview,
        "Patient vitals simulation": render_patient_vitals_simulation,
        "Active alerts": render_active_alerts,
        "Alert auditing": render_alert_auditing,
        "Alert fatigue reduction": render_alert_fatigue_reduction,
        "Clinician workflow simulation": render_clinician_workflow_simulation,
        "Reliability monitoring": render_reliability_monitoring,
        "Drift detection": render_drift_detection,
        "LLM explanations": render_llm_explanations,
        "Action recommendations": render_action_recommendations,
        "System limitations": render_system_limitations,
    }
    page_renderers[page](data)


if __name__ == "__main__":
    main()
