"""Focused tests for Step 23 Streamlit dashboard helpers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd


def test_dashboard_app_imports_without_running_streamlit_server() -> None:
    dashboard_app = importlib.import_module("dashboard.app")

    assert hasattr(dashboard_app, "main")
    assert callable(dashboard_app.main)


def test_safe_load_csv_handles_missing_file(tmp_path: Path) -> None:
    from dashboard import app

    loaded = app.safe_load_csv(str(tmp_path / "missing.csv"))

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.empty


def test_safe_load_json_handles_missing_file(tmp_path: Path) -> None:
    from dashboard import app

    loaded = app.safe_load_json(str(tmp_path / "missing.json"))

    assert loaded == {}


def test_safe_load_csv_reads_existing_file(tmp_path: Path) -> None:
    from dashboard import app

    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([{"patient_id": "P001", "value": 1}]).to_csv(csv_path, index=False)

    loaded = app.safe_load_csv(str(csv_path))

    assert not loaded.empty
    assert loaded.loc[0, "patient_id"] == "P001"


def test_safe_load_json_reads_existing_file(tmp_path: Path) -> None:
    from dashboard import app

    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps({"total_patients": 5}), encoding="utf-8")

    loaded = app.safe_load_json(str(json_path))

    assert loaded["total_patients"] == 5


def test_format_percentage_works() -> None:
    from dashboard import app

    assert app.format_percentage(0.1719) == "17.19%"
    assert app.format_percentage(None) == "0.00%"
    assert app.format_percentage("not-a-number") == "0.00%"


def test_required_helper_functions_exist() -> None:
    from dashboard import app

    for function_name in [
        "safe_load_csv",
        "safe_load_json",
        "format_percentage",
        "render_metric_card",
        "render_simulation_disclaimer",
    ]:
        assert hasattr(app, function_name)
        assert callable(getattr(app, function_name))


def test_page_rendering_functions_exist() -> None:
    from dashboard import app

    expected_renderers = [
        "render_overview",
        "render_patient_vitals_simulation",
        "render_active_alerts",
        "render_alert_auditing",
        "render_alert_fatigue_reduction",
        "render_clinician_workflow_simulation",
        "render_reliability_monitoring",
        "render_drift_detection",
        "render_llm_explanations",
        "render_action_recommendations",
        "render_system_limitations",
    ]

    for function_name in expected_renderers:
        assert hasattr(app, function_name)
        assert callable(getattr(app, function_name))


def test_dashboard_navigation_contains_required_pages() -> None:
    from dashboard import app

    expected_pages = {
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
    }

    assert expected_pages.issubset(set(app.PAGES))


def test_dashboard_does_not_require_external_services() -> None:
    from dashboard import app

    source_paths = [str(path) for path in app.DATA_PATHS.values()]

    assert all("http" not in path for path in source_paths)
    assert all("ollama" not in path.lower() for path in source_paths)
