"""Read-only monitoring endpoints for the local simulation API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.routes_alerts import (
    FATIGUE_ALERTS_PATH,
    RESPONSES_PATH,
    SIMULATION_ONLY_NOTE,
    dataframe_to_records,
    json_safe_record,
    resolve_project_path,
    safe_load_csv,
)
from src.database.db import get_database_path, get_table_counts


router = APIRouter(tags=["monitoring"])

RELIABILITY_PATH = Path("data/processed/reliability_monitoring_results.csv")
DRIFT_PATH = Path("data/processed/drift_detection_results.csv")
MODEL_UPDATES_PATH = Path("data/processed/model_update_simulation_results.csv")
RL_RESULTS_PATH = Path("data/processed/rl_threshold_simulation_results.csv")
RL_POLICY_PATH = Path("data/processed/rl_threshold_policy_summary.json")


@router.get("/monitoring/reliability")
def get_reliability_monitoring(
    limit: int = Query(20, ge=1, le=500),
    status: str | None = None,
) -> dict[str, Any]:
    """Return Step 12 reliability monitoring windows."""
    df = safe_load_csv(RELIABILITY_PATH, "reliability monitoring results")
    if status and "reliability_status" in df.columns:
        df = df[df["reliability_status"].astype(str).str.lower() == status.lower()]
    return {
        "count": int(len(df)),
        "limit": int(limit),
        "average_reliability_score": _mean(df, "reliability_score"),
        "reliability": dataframe_to_records(df, limit),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


@router.get("/monitoring/drift")
def get_drift_monitoring(
    limit: int = Query(20, ge=1, le=500),
    status: str | None = None,
    drift_type: str | None = None,
) -> dict[str, Any]:
    """Return Step 13 drift detection records."""
    df = safe_load_csv(DRIFT_PATH, "drift detection results")
    if status and "drift_status" in df.columns:
        df = df[df["drift_status"].astype(str).str.lower() == status.lower()]
    if drift_type and "drift_type" in df.columns:
        df = df[df["drift_type"].astype(str).str.lower() == drift_type.lower()]
    return {
        "count": int(len(df)),
        "limit": int(limit),
        "severe_drift_count": int(
            df.get("drift_status", pd.Series(dtype=str)).astype(str).eq("severe_shift").sum()
        ),
        "average_drift_score": _mean(df, "drift_score"),
        "drift": dataframe_to_records(df, limit),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


@router.get("/monitoring/model-updates")
def get_model_update_simulations(
    limit: int = Query(20, ge=1, le=500),
) -> dict[str, Any]:
    """Return Step 14 model update simulation records."""
    df = safe_load_csv(MODEL_UPDATES_PATH, "model update simulation results")
    return {
        "count": int(len(df)),
        "limit": int(limit),
        "model_updates": dataframe_to_records(df, limit),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


@router.get("/monitoring/rl-threshold-policy")
def get_rl_threshold_policy(
    limit: int = Query(20, ge=1, le=500),
) -> dict[str, Any]:
    """Return Step 15 RL threshold policy summary and recent episodes."""
    policy = safe_load_json(RL_POLICY_PATH, "RL threshold policy summary")
    episodes = safe_load_csv(RL_RESULTS_PATH, "RL threshold simulation results")
    return {
        "policy_summary": policy,
        "episode_count": int(len(episodes)),
        "episodes": dataframe_to_records(episodes, limit),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


@router.get("/monitoring/workflow-responses")
def get_workflow_responses(
    limit: int = Query(20, ge=1, le=500),
    patient_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Return Step 11 simulated clinician response logs."""
    df = safe_load_csv(RESPONSES_PATH, "clinician response logs")
    if patient_id and "patient_id" in df.columns:
        df = df[df["patient_id"].astype(str) == patient_id]
    if status and "simulated_response" in df.columns:
        df = df[df["simulated_response"].astype(str).str.lower() == status.lower()]
    return {
        "count": int(len(df)),
        "limit": int(limit),
        "ignored_alert_rate": _rate(df, "simulated_response", "ignored"),
        "delayed_alert_rate": _rate(df, "simulated_response", "delayed"),
        "workflow_responses": dataframe_to_records(df, limit),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


@router.get("/dashboard-summary")
def get_dashboard_summary() -> dict[str, Any]:
    """Return a compact summary for local demo dashboards."""
    counts = _safe_table_counts()
    fatigue = safe_load_csv(FATIGUE_ALERTS_PATH, "fatigue-reduced alerts")
    reliability = safe_load_csv(RELIABILITY_PATH, "reliability monitoring results")
    drift = safe_load_csv(DRIFT_PATH, "drift detection results")
    responses = safe_load_csv(RESPONSES_PATH, "clinician response logs")
    policy = safe_load_json(RL_POLICY_PATH, "RL threshold policy summary")

    active_alerts = int(
        fatigue.get("final_alert_status", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .eq("active")
        .sum()
    )

    return {
        "total_patients": int(counts.get("patients", 0)),
        "total_vitals": int(counts.get("vitals", 0)),
        "total_alerts": int(counts.get("alerts", len(fatigue))),
        "active_alerts": active_alerts,
        "average_reliability_score": _mean(reliability, "reliability_score"),
        "severe_drift_count": int(
            drift.get("drift_status", pd.Series(dtype=str)).astype(str).eq("severe_shift").sum()
        ),
        "ignored_alert_rate": _rate(responses, "simulated_response", "ignored"),
        "delayed_alert_rate": _rate(responses, "simulated_response", "delayed"),
        "rl_recommended_action": policy.get("recommended_action"),
        "database_path": get_database_path(),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


def safe_load_json(path: str | Path, label: str) -> dict[str, Any]:
    """Load a JSON artifact or raise a helpful API error."""
    json_path = resolve_project_path(path)
    if not json_path.exists():
        raise HTTPException(status_code=503, detail=f"{label} file is missing: {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail=f"{label} must be a JSON object.")
    return json_safe_record(data)


def _safe_table_counts() -> dict[str, int]:
    try:
        return get_table_counts()
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc


def _mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return round(float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean()), 4)


def _rate(df: pd.DataFrame, column: str, value: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return round(float(df[column].astype(str).str.lower().eq(value.lower()).mean()), 4)
