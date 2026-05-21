"""Read-only alert endpoints for the local simulation API.

These routes expose generated demo artifacts only. They must not be used with
real patient data and are not a clinically validated alert API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query


router = APIRouter(prefix="/alerts", tags=["alerts"])

SIMULATION_ONLY_NOTE = (
    "Local simulated demo data only; not clinically validated and not for patient care."
)
RAW_ALERTS_PATH = Path("data/processed/generated_alerts.csv")
AUDITED_ALERTS_PATH = Path("data/processed/audited_alerts.csv")
FATIGUE_ALERTS_PATH = Path("data/processed/fatigue_reduced_alerts.csv")
RESPONSES_PATH = Path("data/processed/clinician_response_logs.csv")


@router.get("/raw")
def get_raw_alerts(
    limit: int = Query(20, ge=1, le=500),
    patient_id: str | None = None,
    severity: str | None = None,
) -> dict[str, Any]:
    """Return generated alerts from Step 7."""
    df = safe_load_csv(RAW_ALERTS_PATH, "raw alerts")
    filtered = apply_filters(df, patient_id=patient_id, severity=severity)
    return _records_response(filtered, limit, "raw_alerts")


@router.get("/audited")
def get_audited_alerts(
    limit: int = Query(20, ge=1, le=500),
    patient_id: str | None = None,
    severity: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Return audited alerts from Step 9."""
    df = safe_load_csv(AUDITED_ALERTS_PATH, "audited alerts")
    filtered = apply_filters(df, patient_id=patient_id, severity=severity, status=status)
    return _records_response(filtered, limit, "audited_alerts")


@router.get("/fatigue-reduced")
def get_fatigue_reduced_alerts(
    limit: int = Query(20, ge=1, le=500),
    patient_id: str | None = None,
    severity: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Return fatigue-reduced alerts from Step 10."""
    df = safe_load_csv(FATIGUE_ALERTS_PATH, "fatigue-reduced alerts")
    filtered = apply_filters(df, patient_id=patient_id, severity=severity, status=status)
    return _records_response(filtered, limit, "fatigue_reduced_alerts")


@router.get("/summary")
def get_alert_summary() -> dict[str, Any]:
    """Return high-level alert counts for dashboard-style views."""
    raw = safe_load_csv(RAW_ALERTS_PATH, "raw alerts")
    audited = safe_load_csv(AUDITED_ALERTS_PATH, "audited alerts")
    fatigue = safe_load_csv(FATIGUE_ALERTS_PATH, "fatigue-reduced alerts")

    total_fatigue = len(fatigue)
    active_alerts = int(
        fatigue.get("final_alert_status", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .eq("active")
        .sum()
    )
    critical_alerts = int(
        fatigue.get("critical_flag", pd.Series(dtype=bool)).apply(_coerce_bool).sum()
    )
    reduction_rate = (
        round((total_fatigue - active_alerts) / total_fatigue, 4)
        if total_fatigue
        else 0.0
    )

    return {
        "total_raw_alerts": int(len(raw)),
        "total_audited_alerts": int(len(audited)),
        "total_fatigue_reduced_alerts": int(total_fatigue),
        "active_alerts_after_reduction": active_alerts,
        "critical_alerts": critical_alerts,
        "alert_reduction_rate": reduction_rate,
        "critical_preservation_note": (
            "Critical and safety-sensitive alerts are preserved in the simulation; "
            "this is not clinical validation."
        ),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


@router.get("/{alert_id}")
def get_alert_by_id(alert_id: str) -> dict[str, Any]:
    """Return one alert enriched with audit, fatigue, and response context."""
    record = build_alert_detail(alert_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
    return {
        "alert": record,
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


def build_alert_detail(alert_id: str) -> dict[str, Any] | None:
    """Build one alert detail record from available Step 7-11 artifacts."""
    fatigue = safe_load_csv(FATIGUE_ALERTS_PATH, "fatigue-reduced alerts")
    alert_rows = fatigue[fatigue["alert_id"].astype(str) == str(alert_id)]
    if alert_rows.empty:
        raw = safe_load_csv(RAW_ALERTS_PATH, "raw alerts")
        alert_rows = raw[raw["alert_id"].astype(str) == str(alert_id)]
    if alert_rows.empty:
        return None

    record = json_safe_record(alert_rows.iloc[0].to_dict())

    audited = safe_load_csv(AUDITED_ALERTS_PATH, "audited alerts")
    audit_rows = audited[audited["alert_id"].astype(str) == str(alert_id)]
    if not audit_rows.empty:
        audit_record = json_safe_record(audit_rows.iloc[0].to_dict())
        for key in [
            "audit_status",
            "actionability_score",
            "fatigue_risk_score",
            "urgency_score",
            "false_positive_likelihood",
            "confidence_score",
            "escalation_recommendation",
            "audit_reason",
        ]:
            if key in audit_record:
                record[key] = audit_record[key]

    responses = safe_load_csv(RESPONSES_PATH, "clinician response logs")
    response_rows = responses[responses["alert_id"].astype(str) == str(alert_id)]
    if not response_rows.empty:
        response_record = json_safe_record(response_rows.iloc[0].to_dict())
        for key in [
            "simulated_response",
            "response_time_minutes",
            "response_reason",
            "workflow_stage",
            "escalation_completed",
        ]:
            if key in response_record:
                record[key] = response_record[key]

    return record


def safe_load_csv(path: str | Path, label: str) -> pd.DataFrame:
    """Load a CSV artifact or raise a helpful API error."""
    csv_path = resolve_project_path(path)
    if not csv_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{label} file is missing: {csv_path}",
        )
    return pd.read_csv(csv_path)


def apply_filters(
    df: pd.DataFrame,
    patient_id: str | None = None,
    severity: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """Apply common API filters without mutating the source dataframe."""
    filtered = df.copy()
    if patient_id and "patient_id" in filtered.columns:
        filtered = filtered[filtered["patient_id"].astype(str) == patient_id]
    if severity and "severity" in filtered.columns:
        filtered = filtered[filtered["severity"].astype(str).str.lower() == severity.lower()]
    if status:
        status_columns = [
            "final_alert_status",
            "audit_status",
            "reliability_status",
            "drift_status",
            "simulated_response",
        ]
        matching_columns = [column for column in status_columns if column in filtered.columns]
        if matching_columns:
            mask = False
            for column in matching_columns:
                column_mask = filtered[column].astype(str).str.lower() == status.lower()
                mask = column_mask if isinstance(mask, bool) else (mask | column_mask)
            filtered = filtered[mask]
    return filtered


def dataframe_to_records(df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    """Convert a DataFrame preview into JSON-safe records."""
    limited = df.head(max(int(limit), 0))
    return [json_safe_record(record) for record in limited.to_dict(orient="records")]


def json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert pandas/numpy values into JSON-safe Python values."""
    safe: dict[str, Any] = {}
    for key, value in record.items():
        safe[key] = json_safe_value(value)
    return safe


def json_safe_value(value: Any) -> Any:
    """Convert common pandas values to JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_value(item) for item in value]
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


def _records_response(df: pd.DataFrame, limit: int, response_key: str) -> dict[str, Any]:
    return {
        "count": int(len(df)),
        "limit": int(limit),
        response_key: dataframe_to_records(df, limit),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
