"""FastAPI resilience and graceful failure checks for Step 25."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import api.routes_alerts as routes_alerts
import api.routes_monitoring as routes_monitoring
from api.main import app


client = TestClient(app)


def test_core_api_endpoints_return_200() -> None:
    for path in [
        "/health",
        "/",
        "/alerts/summary",
        "/alerts/raw?limit=2",
        "/alerts/audited?limit=2",
        "/alerts/fatigue-reduced?limit=2",
        "/monitoring/reliability?limit=2",
        "/monitoring/drift?limit=2",
        "/monitoring/model-updates?limit=2",
        "/monitoring/rl-threshold-policy?limit=2",
        "/monitoring/workflow-responses?limit=2",
        "/dashboard-summary",
    ]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert isinstance(response.json(), dict)


def test_missing_alert_ids_return_valid_errors() -> None:
    for path in ["/alerts/NOT-A-REAL-ALERT", "/explain-alert/NOT-A-REAL-ALERT"]:
        response = client.get(path)
        assert response.status_code == 404
        payload = response.json()
        assert "detail" in payload
        assert "not found" in str(payload["detail"]).lower()


def test_invalid_limits_are_rejected_safely() -> None:
    for path in ["/alerts/raw?limit=0", "/monitoring/drift?limit=9999"]:
        response = client.get(path)
        assert response.status_code == 422
        assert isinstance(response.json(), dict)


def test_dashboard_summary_always_returns_dict_with_simulation_note() -> None:
    response = client.get("/dashboard-summary")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert "simulation_only_note" in payload


def test_explanation_endpoint_never_crashes_for_known_alert() -> None:
    alerts_response = client.get("/alerts/fatigue-reduced?limit=1")
    alert_id = alerts_response.json()["fatigue_reduced_alerts"][0]["alert_id"]

    response = client.get(f"/explain-alert/{alert_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["alert_id"] == alert_id
    assert "explanation" in payload
    assert "simulation_only_note" in payload


def test_api_responds_with_helpful_error_when_optional_file_missing(monkeypatch, tmp_path) -> None:
    missing_csv = tmp_path / "missing_reliability.csv"
    monkeypatch.setattr(routes_monitoring, "RELIABILITY_PATH", Path(missing_csv))

    response = client.get("/monitoring/reliability")

    assert response.status_code == 503
    payload = response.json()
    assert "file is missing" in str(payload["detail"]).lower()


def test_alert_endpoint_missing_file_returns_503_not_crash(monkeypatch, tmp_path) -> None:
    missing_csv = tmp_path / "missing_alerts.csv"
    monkeypatch.setattr(routes_alerts, "RAW_ALERTS_PATH", Path(missing_csv))

    response = client.get("/alerts/raw")

    assert response.status_code == 503
    assert "file is missing" in str(response.json()["detail"]).lower()
