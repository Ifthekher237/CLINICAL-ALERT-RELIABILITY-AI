"""CRUD helpers for loading simulated demo artifacts into SQLite.

All inserts are local, idempotent, and simulation-only. The database layer does
not store real patient data and is not a clinical data store.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from database.db import get_connection, get_table_counts, initialize_database
    from database.models import get_schema_statements
else:
    from .db import get_connection, get_table_counts, initialize_database
    from .models import get_schema_statements


SIMULATION_NOTE = "Simulated demo data only; not real patient data."
DEFAULT_SIMULATED_CSV = Path("data/simulated/patient_monitoring.csv")
DEFAULT_FATIGUE_CSV = Path("data/processed/fatigue_reduced_alerts.csv")
DEFAULT_AUDITED_CSV = Path("data/processed/audited_alerts.csv")
DEFAULT_RESPONSES_CSV = Path("data/processed/clinician_response_logs.csv")
DEFAULT_DRIFT_CSV = Path("data/processed/drift_detection_results.csv")
DEFAULT_RELIABILITY_CSV = Path("data/processed/reliability_monitoring_results.csv")
DEFAULT_MODEL_UPDATE_CSV = Path("data/processed/model_update_simulation_results.csv")


def insert_patients_from_simulated_data(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert one simulated patient row per patient_id."""
    df = _read_optional_csv(csv_path, "simulated patient data")
    if df.empty or "patient_id" not in df.columns:
        return 0
    initialize_database(db_path)

    patient_rows = (
        df.groupby("patient_id", as_index=False)["timestamp"]
        .min()
        .rename(columns={"timestamp": "created_at"})
    )
    rows = [
        (
            _text(row["patient_id"]),
            _text(row["created_at"]),
            SIMULATION_NOTE,
        )
        for _, row in patient_rows.iterrows()
    ]
    return _insert_rows(
        """
        INSERT OR REPLACE INTO patients (
            patient_id, created_at, simulation_note
        ) VALUES (?, ?, ?)
        """,
        rows,
        db_path,
    )


def insert_vitals_from_simulated_data(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert simulated vital-sign rows."""
    df = _read_optional_csv(csv_path, "simulated vital-sign data")
    required = {"patient_id", "timestamp"}
    if df.empty or not required.issubset(df.columns):
        return 0
    initialize_database(db_path)

    rows = []
    for index, row in df.reset_index(drop=True).iterrows():
        rows.append(
            (
                _record_id("VITAL", row.get("patient_id"), row.get("timestamp"), index),
                _text(row.get("patient_id")),
                _text(row.get("timestamp")),
                _float(row.get("heart_rate")),
                _float(row.get("oxygen_saturation")),
                _float(row.get("systolic_bp")),
                _float(row.get("diastolic_bp")),
                _float(row.get("respiratory_rate")),
                _float(row.get("temperature")),
                _text(row.get("patient_condition_label")),
                _bool_int(row.get("deterioration_event")),
                _bool_int(row.get("sensor_noise_flag")),
                _bool_int(row.get("missing_data_flag")),
            )
        )
    return _insert_rows(
        """
        INSERT OR REPLACE INTO vitals (
            vitals_id, patient_id, timestamp, heart_rate, oxygen_saturation,
            systolic_bp, diastolic_bp, respiratory_rate, temperature,
            patient_condition_label, deterioration_event, sensor_noise_flag,
            missing_data_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        db_path,
    )


def insert_alerts_from_fatigue_data(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert alerts enriched with fatigue-reduction status."""
    df = _read_optional_csv(csv_path, "fatigue-reduced alerts")
    if df.empty or "alert_id" not in df.columns:
        return 0
    initialize_database(db_path)

    rows = [
        (
            _text(row.get("alert_id")),
            _text(row.get("patient_id")),
            _text(row.get("timestamp")),
            _text(row.get("severity")),
            _text(row.get("alert_type")),
            _float(row.get("risk_score")),
            _text(row.get("trigger_reason")),
            _text(row.get("source_model")),
            _text(row.get("recommended_review_time")),
            _bool_int(row.get("critical_flag")),
            _text(row.get("final_alert_status")),
            _text(row.get("fatigue_action")),
        )
        for _, row in df.iterrows()
    ]
    return _insert_rows(
        """
        INSERT OR REPLACE INTO alerts (
            alert_id, patient_id, timestamp, severity, alert_type, risk_score,
            trigger_reason, source_model, recommended_review_time, critical_flag,
            final_alert_status, fatigue_action
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        db_path,
    )


def insert_alert_audits_from_audited_data(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert alert audit records from Step 9."""
    df = _read_optional_csv(csv_path, "audited alerts")
    if df.empty or "alert_id" not in df.columns:
        return 0
    initialize_database(db_path)

    rows = [
        (
            _record_id("AUDIT", row.get("alert_id"), None, index),
            _text(row.get("alert_id")),
            _text(row.get("audit_status")),
            _float(row.get("actionability_score")),
            _float(row.get("fatigue_risk_score")),
            _float(row.get("urgency_score")),
            _float(row.get("false_positive_likelihood")),
            _float(row.get("confidence_score")),
            _text(row.get("escalation_recommendation")),
            _text(row.get("audit_reason")),
        )
        for index, row in df.reset_index(drop=True).iterrows()
    ]
    return _insert_rows(
        """
        INSERT OR REPLACE INTO alert_audits (
            audit_id, alert_id, audit_status, actionability_score,
            fatigue_risk_score, urgency_score, false_positive_likelihood,
            confidence_score, escalation_recommendation, audit_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        db_path,
    )


def insert_clinician_responses(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert simulated clinician workflow responses."""
    df = _read_optional_csv(csv_path, "clinician response logs")
    if df.empty or "response_id" not in df.columns:
        return 0
    initialize_database(db_path)

    rows = [
        (
            _text(row.get("response_id")),
            _text(row.get("alert_id")),
            _text(row.get("patient_id")),
            _text(row.get("timestamp")),
            _text(row.get("simulated_response")),
            _float(row.get("response_time_minutes")),
            _text(row.get("response_reason")),
            _float(row.get("clinician_burden_score")),
            _float(row.get("perceived_alert_usefulness")),
            _text(row.get("workflow_stage")),
            _bool_int(row.get("escalation_completed")),
        )
        for _, row in df.iterrows()
    ]
    return _insert_rows(
        """
        INSERT OR REPLACE INTO clinician_responses (
            response_id, alert_id, patient_id, timestamp, simulated_response,
            response_time_minutes, response_reason, clinician_burden_score,
            perceived_alert_usefulness, workflow_stage, escalation_completed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        db_path,
    )


def insert_patient_outcomes_from_simulated_data(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert available simulated patient outcome labels, if present."""
    df = _read_optional_csv(csv_path, "simulated patient outcomes")
    if df.empty or "patient_id" not in df.columns:
        return 0
    if "patient_outcome_after_alert" not in df.columns:
        return 0
    initialize_database(db_path)

    rows = []
    for index, row in df.reset_index(drop=True).iterrows():
        outcome_label = _text(row.get("patient_outcome_after_alert"))
        outcome_timestamp = _text(row.get("outcome_timestamp"))
        severity_change = _float(row.get("outcome_severity_change"))
        if not outcome_label and not outcome_timestamp and severity_change is None:
            continue
        if outcome_label == "unknown" and not outcome_timestamp and severity_change is None:
            continue
        rows.append(
            (
                _record_id("OUTCOME", row.get("patient_id"), outcome_timestamp or row.get("timestamp"), index),
                _text(row.get("patient_id")),
                None,
                outcome_timestamp,
                outcome_label,
                severity_change,
                SIMULATION_NOTE,
            )
        )
    if not rows:
        return 0
    return _insert_rows(
        """
        INSERT OR REPLACE INTO patient_outcomes (
            outcome_id, patient_id, alert_id, outcome_timestamp, outcome_label,
            outcome_severity_change, simulation_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        db_path,
    )


def insert_drift_logs(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert drift detection rows from Step 13."""
    df = _read_optional_csv(csv_path, "drift detection results")
    required = {"drift_window_id", "drift_type", "monitored_feature"}
    if df.empty or not required.issubset(df.columns):
        return 0
    initialize_database(db_path)

    rows = [
        (
            _record_id(
                "DRIFT",
                f"{row.get('drift_window_id')}-{row.get('drift_type')}",
                row.get("monitored_feature"),
                index,
            ),
            _text(row.get("drift_window_id")),
            _text(row.get("drift_type")),
            _text(row.get("monitored_feature")),
            _float(row.get("drift_score")),
            _text(row.get("drift_status")),
            _text(row.get("recalibration_recommendation")),
            _bool_int(row.get("requires_review")),
        )
        for index, row in df.reset_index(drop=True).iterrows()
    ]
    return _insert_rows(
        """
        INSERT OR REPLACE INTO drift_logs (
            drift_id, drift_window_id, drift_type, monitored_feature,
            drift_score, drift_status, recalibration_recommendation,
            requires_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        db_path,
    )


def insert_reliability_logs(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert reliability monitoring rows from Step 12."""
    df = _read_optional_csv(csv_path, "reliability monitoring results")
    if df.empty or "monitoring_window_id" not in df.columns:
        return 0
    initialize_database(db_path)

    rows = [
        (
            _record_id("RELIABILITY", row.get("monitoring_window_id"), None, index),
            _text(row.get("monitoring_window_id")),
            _text(row.get("window_start")),
            _text(row.get("window_end")),
            _float(row.get("reliability_score")),
            _text(row.get("reliability_status")),
            _text(row.get("reliability_warning")),
            _text(row.get("review_recommendation")),
        )
        for index, row in df.reset_index(drop=True).iterrows()
    ]
    return _insert_rows(
        """
        INSERT OR REPLACE INTO reliability_logs (
            reliability_id, monitoring_window_id, window_start, window_end,
            reliability_score, reliability_status, reliability_warning,
            review_recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        db_path,
    )


def insert_model_versions(
    csv_path: str | Path,
    db_path: str | None = None,
) -> int:
    """Insert simulated model-version update rows from Step 14."""
    df = _read_optional_csv(csv_path, "model update simulation results")
    if df.empty or "proposed_model_version" not in df.columns:
        return 0
    initialize_database(db_path)

    rows = [
        (
            _text(row.get("update_id")) or _record_id(
                "MODEL",
                row.get("proposed_model_version"),
                row.get("update_timestamp"),
                index,
            ),
            _text(row.get("previous_model_version")),
            _text(row.get("proposed_model_version")),
            _text(row.get("update_timestamp")),
            _float(row.get("current_risk_threshold")),
            _float(row.get("proposed_risk_threshold")),
            _text(row.get("deployment_recommendation")),
            _bool_int(row.get("human_review_required")),
            "Simulation-only model version record; no model artifact was deployed.",
        )
        for index, row in df.reset_index(drop=True).iterrows()
    ]
    return _insert_rows(
        """
        INSERT OR REPLACE INTO model_versions (
            model_version_id, previous_model_version, proposed_model_version,
            update_timestamp, current_risk_threshold, proposed_risk_threshold,
            deployment_recommendation, human_review_required, simulation_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        db_path,
    )


def load_all_demo_data(db_path: str | None = None) -> dict[str, int]:
    """Load all available local demo artifacts into SQLite."""
    initialize_database(db_path)
    counts = {
        "patients": insert_patients_from_simulated_data(DEFAULT_SIMULATED_CSV, db_path),
        "vitals": insert_vitals_from_simulated_data(DEFAULT_SIMULATED_CSV, db_path),
        "alerts": insert_alerts_from_fatigue_data(DEFAULT_FATIGUE_CSV, db_path),
        "alert_audits": insert_alert_audits_from_audited_data(DEFAULT_AUDITED_CSV, db_path),
        "clinician_responses": insert_clinician_responses(DEFAULT_RESPONSES_CSV, db_path),
        "patient_outcomes": insert_patient_outcomes_from_simulated_data(
            DEFAULT_SIMULATED_CSV,
            db_path,
        ),
        "drift_logs": insert_drift_logs(DEFAULT_DRIFT_CSV, db_path),
        "reliability_logs": insert_reliability_logs(DEFAULT_RELIABILITY_CSV, db_path),
        "model_versions": insert_model_versions(DEFAULT_MODEL_UPDATE_CSV, db_path),
        "failure_mode_logs": 0,
        "scenario_test_results": 0,
    }
    return counts


def fetch_table(
    table_name: str,
    limit: int = 10,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Fetch a table preview into a pandas DataFrame."""
    if table_name not in get_schema_statements():
        raise ValueError(f"Unknown table name: {table_name}")
    initialize_database(db_path)
    limit = max(int(limit), 0)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(
            f"SELECT * FROM {table_name} LIMIT ?",
            connection,
            params=(limit,),
        )


def _insert_rows(
    sql: str,
    rows: list[tuple[Any, ...]],
    db_path: str | None = None,
) -> int:
    if not rows:
        return 0
    with get_connection(db_path) as connection:
        connection.executemany(sql, rows)
        connection.commit()
    return len(rows)


def _read_optional_csv(csv_path: str | Path, label: str) -> pd.DataFrame:
    path = _resolve_project_path(csv_path)
    if not path.exists():
        print(f"Skipping {label}: file not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _record_id(prefix: str, value: Any, timestamp: Any, index: int) -> str:
    pieces = [
        prefix,
        _safe_token(value) or "UNKNOWN",
        _safe_token(timestamp) or "NO_TIME",
        str(index).zfill(6),
    ]
    return "-".join(pieces)


def _safe_token(value: Any) -> str:
    text = _text(value)
    if text is None:
        return ""
    return "".join(character if character.isalnum() else "" for character in text)[:80]


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip().lower()
    return int(text in {"true", "1", "yes", "y"})


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _project_root() / candidate


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    initialize_database()
    inserted_counts = load_all_demo_data()
    print("Inserted or replaced row counts:")
    for table, count in inserted_counts.items():
        print(f"  {table}: {count}")

    print("\nFinal table counts:")
    for table, count in get_table_counts().items():
        print(f"  {table}: {count}")
