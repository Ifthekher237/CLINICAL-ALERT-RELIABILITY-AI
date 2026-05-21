"""Focused tests for Step 17 FastAPI backend."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def _known_alert_id() -> str:
    alerts = pd.read_csv("data/processed/fatigue_reduced_alerts.csv")
    return str(alerts.iloc[0]["alert_id"])


def test_health_returns_200() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["project"] == "clinical-alert-reliability-ai"
    assert "simulation_only_note" in payload


def test_alerts_summary_returns_expected_keys() -> None:
    response = client.get("/alerts/summary")

    assert response.status_code == 200
    payload = response.json()
    expected = {
        "total_raw_alerts",
        "total_audited_alerts",
        "total_fatigue_reduced_alerts",
        "active_alerts_after_reduction",
        "critical_alerts",
        "alert_reduction_rate",
        "critical_preservation_note",
    }
    assert expected.issubset(payload)
    assert "simulation_only_note" in payload


def test_alerts_raw_returns_list_response() -> None:
    response = client.get("/alerts/raw?limit=3")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["raw_alerts"], list)
    assert len(payload["raw_alerts"]) <= 3


def test_alerts_audited_returns_list_response() -> None:
    response = client.get("/alerts/audited?limit=3")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["audited_alerts"], list)
    assert len(payload["audited_alerts"]) <= 3


def test_alerts_fatigue_reduced_returns_list_response() -> None:
    response = client.get("/alerts/fatigue-reduced?limit=3")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["fatigue_reduced_alerts"], list)
    assert len(payload["fatigue_reduced_alerts"]) <= 3


def test_monitoring_reliability_returns_list_and_summary() -> None:
    response = client.get("/monitoring/reliability?limit=3")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["reliability"], list)
    assert "average_reliability_score" in payload
    assert "simulation_only_note" in payload


def test_monitoring_drift_returns_list_and_summary() -> None:
    response = client.get("/monitoring/drift?limit=3")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["drift"], list)
    assert "severe_drift_count" in payload
    assert "simulation_only_note" in payload


def test_dashboard_summary_returns_expected_keys() -> None:
    response = client.get("/dashboard-summary")

    assert response.status_code == 200
    payload = response.json()
    expected = {
        "total_patients",
        "total_vitals",
        "total_alerts",
        "active_alerts",
        "average_reliability_score",
        "severe_drift_count",
        "ignored_alert_rate",
        "delayed_alert_rate",
        "rl_recommended_action",
        "simulation_only_note",
    }
    assert expected.issubset(payload)


def test_explain_alert_works_for_known_alert() -> None:
    alert_id = _known_alert_id()
    response = client.get(f"/explain-alert/{alert_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["alert_id"] == alert_id
    assert "explanation" in payload
    assert "simulation_only_note" in payload


def test_missing_alert_id_returns_404() -> None:
    response = client.get("/explain-alert/DOES-NOT-EXIST")

    assert response.status_code == 404


def test_alert_detail_includes_simulation_note() -> None:
    alert_id = _known_alert_id()
    response = client.get(f"/alerts/{alert_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["alert"]["alert_id"] == alert_id
    assert "simulation_only_note" in payload
