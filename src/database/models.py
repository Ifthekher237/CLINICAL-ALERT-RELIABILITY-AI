"""Centralized SQLite schema for simulated demo storage.

The database stores local simulation artifacts only. It is not designed for
real patient data, clinical operations, or regulated medical-device storage.
"""

from __future__ import annotations


SCHEMA_STATEMENTS: dict[str, str] = {
    "patients": """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            created_at TEXT,
            simulation_note TEXT
        )
    """,
    "vitals": """
        CREATE TABLE IF NOT EXISTS vitals (
            vitals_id TEXT PRIMARY KEY,
            patient_id TEXT,
            timestamp TEXT,
            heart_rate REAL,
            oxygen_saturation REAL,
            systolic_bp REAL,
            diastolic_bp REAL,
            respiratory_rate REAL,
            temperature REAL,
            patient_condition_label TEXT,
            deterioration_event INTEGER,
            sensor_noise_flag INTEGER,
            missing_data_flag INTEGER
        )
    """,
    "alerts": """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            patient_id TEXT,
            timestamp TEXT,
            severity TEXT,
            alert_type TEXT,
            risk_score REAL,
            trigger_reason TEXT,
            source_model TEXT,
            recommended_review_time TEXT,
            critical_flag INTEGER,
            final_alert_status TEXT,
            fatigue_action TEXT
        )
    """,
    "alert_audits": """
        CREATE TABLE IF NOT EXISTS alert_audits (
            audit_id TEXT PRIMARY KEY,
            alert_id TEXT,
            audit_status TEXT,
            actionability_score REAL,
            fatigue_risk_score REAL,
            urgency_score REAL,
            false_positive_likelihood REAL,
            confidence_score REAL,
            escalation_recommendation TEXT,
            audit_reason TEXT
        )
    """,
    "clinician_responses": """
        CREATE TABLE IF NOT EXISTS clinician_responses (
            response_id TEXT PRIMARY KEY,
            alert_id TEXT,
            patient_id TEXT,
            timestamp TEXT,
            simulated_response TEXT,
            response_time_minutes REAL,
            response_reason TEXT,
            clinician_burden_score REAL,
            perceived_alert_usefulness REAL,
            workflow_stage TEXT,
            escalation_completed INTEGER
        )
    """,
    "patient_outcomes": """
        CREATE TABLE IF NOT EXISTS patient_outcomes (
            outcome_id TEXT PRIMARY KEY,
            patient_id TEXT,
            alert_id TEXT,
            outcome_timestamp TEXT,
            outcome_label TEXT,
            outcome_severity_change REAL,
            simulation_note TEXT
        )
    """,
    "failure_mode_logs": """
        CREATE TABLE IF NOT EXISTS failure_mode_logs (
            failure_mode_id TEXT PRIMARY KEY,
            scenario_name TEXT,
            failure_mode_type TEXT,
            detected_at TEXT,
            severity TEXT,
            impact_summary TEXT,
            safety_warning TEXT
        )
    """,
    "scenario_test_results": """
        CREATE TABLE IF NOT EXISTS scenario_test_results (
            scenario_id TEXT PRIMARY KEY,
            scenario_name TEXT,
            run_timestamp TEXT,
            pass_fail_status TEXT,
            safety_check_passed INTEGER,
            summary TEXT
        )
    """,
    "drift_logs": """
        CREATE TABLE IF NOT EXISTS drift_logs (
            drift_id TEXT PRIMARY KEY,
            drift_window_id TEXT,
            drift_type TEXT,
            monitored_feature TEXT,
            drift_score REAL,
            drift_status TEXT,
            recalibration_recommendation TEXT,
            requires_review INTEGER
        )
    """,
    "reliability_logs": """
        CREATE TABLE IF NOT EXISTS reliability_logs (
            reliability_id TEXT PRIMARY KEY,
            monitoring_window_id TEXT,
            window_start TEXT,
            window_end TEXT,
            reliability_score REAL,
            reliability_status TEXT,
            reliability_warning TEXT,
            review_recommendation TEXT
        )
    """,
    "model_versions": """
        CREATE TABLE IF NOT EXISTS model_versions (
            model_version_id TEXT PRIMARY KEY,
            previous_model_version TEXT,
            proposed_model_version TEXT,
            update_timestamp TEXT,
            current_risk_threshold REAL,
            proposed_risk_threshold REAL,
            deployment_recommendation TEXT,
            human_review_required INTEGER,
            simulation_note TEXT
        )
    """,
}


def get_schema_statements() -> dict[str, str]:
    """Return table-name to CREATE TABLE SQL mappings."""
    return dict(SCHEMA_STATEMENTS)
