"""Dashboard data-loading and empty-state resilience checks for Step 25."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dashboard import app as dashboard_app


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_streamlit_noops(monkeypatch) -> None:
    for name in [
        "header",
        "subheader",
        "info",
        "warning",
        "metric",
        "dataframe",
        "bar_chart",
        "line_chart",
        "markdown",
        "json",
        "caption",
        "write",
    ]:
        monkeypatch.setattr(dashboard_app.st, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard_app.st, "columns", lambda count: [_DummyContext() for _ in range(count)])
    monkeypatch.setattr(dashboard_app.st, "expander", lambda *args, **kwargs: _DummyContext())
    monkeypatch.setattr(dashboard_app.st, "selectbox", lambda label, options, key=None: options[0])


def test_dashboard_safe_load_csv_and_json_handle_missing_files(tmp_path: Path) -> None:
    assert dashboard_app.safe_load_csv(str(tmp_path / "missing.csv")).empty
    assert dashboard_app.safe_load_json(str(tmp_path / "missing.json")) == {}


def test_dashboard_safe_loaders_read_existing_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    json_path = tmp_path / "sample.json"
    pd.DataFrame([{"metric": "x", "value": 1}]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"metric": "x"}), encoding="utf-8")

    assert dashboard_app.safe_load_csv(str(csv_path)).iloc[0]["value"] == 1
    assert dashboard_app.safe_load_json(str(json_path)) == {"metric": "x"}


def test_dashboard_fallback_metric_helpers_are_safe() -> None:
    empty = pd.DataFrame()

    assert dashboard_app.format_percentage(None) == "0.00%"
    assert dashboard_app.format_percentage("not-a-number") == "0.00%"
    assert dashboard_app._limited_table(empty).empty
    assert dashboard_app._truthy_count(empty, "missing") == 0
    assert dashboard_app._mean_value(empty, "missing") == 0.0
    assert dashboard_app._rag_coverage_rate(empty) == 0.0


def test_dashboard_distribution_render_does_not_crash_on_empty_data(monkeypatch) -> None:
    _patch_streamlit_noops(monkeypatch)

    dashboard_app._render_distribution(pd.DataFrame(), "missing_column", "Empty Distribution")


def test_dashboard_sections_render_minimal_empty_state(monkeypatch) -> None:
    _patch_streamlit_noops(monkeypatch)
    empty_data = {
        "metrics_summary": {},
        "metrics_table": pd.DataFrame(),
        "processed": pd.DataFrame(),
        "raw_alerts": pd.DataFrame(),
        "audited": pd.DataFrame(),
        "fatigue": pd.DataFrame(),
        "responses": pd.DataFrame(),
        "reliability": pd.DataFrame(),
        "drift": pd.DataFrame(),
        "explanations": pd.DataFrame(),
        "recommendations": pd.DataFrame(),
        "model_updates": pd.DataFrame(),
        "rl_policy": {},
    }

    for render_function in [
        dashboard_app.render_overview,
        dashboard_app.render_patient_vitals_simulation,
        dashboard_app.render_active_alerts,
        dashboard_app.render_alert_auditing,
        dashboard_app.render_alert_fatigue_reduction,
        dashboard_app.render_clinician_workflow_simulation,
        dashboard_app.render_reliability_monitoring,
        dashboard_app.render_drift_detection,
        dashboard_app.render_llm_explanations,
        dashboard_app.render_action_recommendations,
        dashboard_app.render_system_limitations,
    ]:
        render_function(empty_data)


def test_load_dashboard_data_returns_safe_shapes_when_paths_missing(monkeypatch, tmp_path: Path) -> None:
    missing_paths = {key: str(tmp_path / f"missing_{key}.csv") for key in dashboard_app.DATA_PATHS}
    missing_paths["metrics_summary"] = str(tmp_path / "missing_metrics.json")
    missing_paths["rl_policy"] = str(tmp_path / "missing_policy.json")
    monkeypatch.setattr(dashboard_app, "DATA_PATHS", missing_paths)

    data = dashboard_app.load_dashboard_data()

    assert data["metrics_summary"] == {}
    assert data["rl_policy"] == {}
    assert all(
        isinstance(value, pd.DataFrame) and value.empty
        for key, value in data.items()
        if key not in {"metrics_summary", "rl_policy"}
    )
