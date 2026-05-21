"""Simulated alert generation for clinical alert reliability experiments.

Step 7 converts model outputs and time-series signals into structured alert
records for later auditing, safety guardrails, and fatigue-reduction work. This
is a research/engineering prototype only. It is not a validated clinical alert
system and must not be used for real patient care.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import anomaly_model
from src.models.risk_model import _prepare_input_for_prediction


DEFAULT_SCORED_DATA_PATH = Path("data/processed/timeseries_risk_scored.csv")
DEFAULT_ALERT_OUTPUT_PATH = Path("data/processed/generated_alerts.csv")
DEFAULT_RANDOM_FOREST_PATH = Path("models/random_forest.pkl")
DEFAULT_ANOMALY_MODEL_PATH = Path("models/isolation_forest_anomaly.pkl")

REQUIRED_ALERT_COLUMNS = [
    "alert_id",
    "patient_id",
    "timestamp",
    "severity",
    "alert_type",
    "risk_score",
    "trigger_reason",
    "source_model",
    "recommended_review_time",
    "critical_flag",
]

SEVERITY_LEVELS = {"low", "medium", "high", "critical"}

LEAKAGE_COLUMNS = {
    "patient_condition_label",
    "deterioration_event",
    "patient_outcome_after_alert",
    "outcome_timestamp",
    "outcome_severity_change",
    "target_label",
    "future_deterioration_label",
}


def load_scored_data(path: str | Path = DEFAULT_SCORED_DATA_PATH) -> pd.DataFrame:
    """Load time-series scored data for simulated alert generation."""
    data_path = _resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Scored data file not found: {data_path}")
    return pd.read_csv(data_path)


def load_model(path: str | Path) -> Any:
    """Load a saved local model artifact."""
    model_path = _resolve_project_path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def calculate_combined_alert_risk(row: pd.Series | dict[str, Any]) -> float:
    """Combine transparent simulated risk signals into one alert risk score."""
    time_series_risk = _clip01(_numeric(row, "time_series_risk_score", 0.0))
    baseline_risk = _clip01(_numeric(row, "baseline_future_deterioration_risk", 0.0))
    anomaly_component = 1.0 if int(_numeric(row, "anomaly_label", 0)) == 1 else 0.0
    abnormal_component = min(_numeric(row, "abnormal_value_count", 0.0) / 4.0, 1.0)
    instability_component = min(_numeric(row, "instability_score", 0.0) / 1.4, 1.0)
    oxygen_component = _oxygen_risk_component(row)
    respiratory_component = _respiratory_risk_component(row)

    score = (
        0.34 * time_series_risk
        + 0.22 * baseline_risk
        + 0.14 * anomaly_component
        + 0.10 * abnormal_component
        + 0.08 * oxygen_component
        + 0.07 * instability_component
        + 0.05 * respiratory_component
    )

    # Current vital patterns can raise priority, but this remains simulation-only
    # alert logic and is not a clinical rule.
    if _has_critical_vital_pattern(row):
        score = max(score, 0.90)
    elif _has_high_risk_vital_pattern(row):
        score = max(score, 0.70)

    return round(float(np.clip(score, 0.0, 1.0)), 4)


def assign_alert_severity(risk_score: float, row: pd.Series | dict[str, Any] | None = None) -> str:
    """Assign alert severity from combined risk and current simulated vitals."""
    if row is not None and _has_critical_vital_pattern(row):
        return "critical"

    score = float(risk_score)
    if score >= 0.85:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def assign_alert_type(row: pd.Series | dict[str, Any]) -> str:
    """Pick the most specific alert type for the row's active risk signals."""
    risk_signal_count = _active_risk_signal_count(row)
    if risk_signal_count >= 3:
        return "multiple_risk_signals"
    if bool(_row_value(row, "sustained_oxygen_saturation_drop", False)) or _numeric(
        row,
        "oxygen_saturation_change_3",
        0.0,
    ) <= -2.0:
        return "oxygen_saturation_decline"
    if bool(_row_value(row, "sustained_high_respiratory_rate", False)) or _numeric(
        row,
        "respiratory_rate_change_3",
        0.0,
    ) >= 3.0:
        return "respiratory_rate_increase"
    if _numeric(row, "time_series_risk_score", 0.0) >= 0.35:
        return "sustained_vital_instability"
    if int(_numeric(row, "anomaly_label", 0)) == 1:
        return "anomaly_detected"
    return "future_deterioration_risk"


def generate_trigger_reason(row: pd.Series | dict[str, Any]) -> str:
    """Generate an explainable reason for the simulated alert."""
    reasons = []
    time_series_reason = str(_row_value(row, "time_series_risk_reason", "")).strip()
    anomaly_reason = str(_row_value(row, "anomaly_reason", "")).strip()

    if _numeric(row, "time_series_risk_score", 0.0) >= 0.35 and time_series_reason:
        reasons.append(time_series_reason)
    if int(_numeric(row, "anomaly_label", 0)) == 1 and anomaly_reason:
        reasons.append(anomaly_reason)
    if _numeric(row, "baseline_future_deterioration_risk", 0.0) >= 0.45:
        reasons.append("Elevated baseline future deterioration risk")
    if _has_critical_vital_pattern(row):
        reasons.append("Critical simulated vital-sign pattern")
    elif _has_high_risk_vital_pattern(row):
        reasons.append("High-risk simulated vital-sign pattern")
    if _numeric(row, "abnormal_value_count", 0.0) >= 2:
        reasons.append("Multiple abnormal current vital signs")
    if _numeric(row, "instability_score", 0.0) >= 0.75:
        reasons.append("High current instability score")

    cleaned_reasons = []
    for reason in reasons:
        if reason and reason not in cleaned_reasons and reason != "No sustained deterioration pattern":
            cleaned_reasons.append(reason)

    if cleaned_reasons:
        return "; ".join(cleaned_reasons[:3])
    return "Elevated combined simulated alert risk"


def recommended_review_time(severity: str) -> str:
    """Return the simulated review-time target for an alert severity."""
    mapping = {
        "low": "review within 60 minutes",
        "medium": "review within 30 minutes",
        "high": "review within 10 minutes",
        "critical": "immediate review",
    }
    return mapping.get(severity, "review within 60 minutes")


def generate_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Generate structured simulated alert records from scored monitoring rows."""
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_ALERT_COLUMNS)
    _validate_required_input_columns(df)

    working = df.copy().reset_index(drop=True)
    if "baseline_future_deterioration_risk" not in working.columns:
        working["baseline_future_deterioration_risk"] = 0.0
    if "anomaly_label" not in working.columns:
        working["anomaly_label"] = 0
    if "anomaly_score" not in working.columns:
        working["anomaly_score"] = 0.0
    if "anomaly_reason" not in working.columns:
        working["anomaly_reason"] = ""

    alert_records: list[dict[str, Any]] = []
    for row_number, row in working.iterrows():
        risk_score = calculate_combined_alert_risk(row)
        if not _should_generate_alert(row, risk_score):
            continue

        severity = assign_alert_severity(risk_score, row=row)
        critical_flag = bool(severity == "critical" or _has_critical_vital_pattern(row))
        alert_records.append(
            {
                "alert_id": _make_alert_id(row, len(alert_records) + 1),
                "patient_id": row["patient_id"],
                "timestamp": row["timestamp"],
                "severity": severity,
                "alert_type": assign_alert_type(row),
                "risk_score": risk_score,
                "trigger_reason": generate_trigger_reason(row),
                "source_model": _source_model_summary(row),
                "recommended_review_time": recommended_review_time(severity),
                "critical_flag": critical_flag,
            }
        )

    alerts_df = pd.DataFrame(alert_records, columns=REQUIRED_ALERT_COLUMNS)
    if not alerts_df.empty:
        alerts_df["critical_flag"] = alerts_df["critical_flag"].astype(bool)
    return alerts_df


def save_alerts(alerts_df: pd.DataFrame, path: str | Path) -> Path:
    """Save generated simulated alerts to CSV."""
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alerts_df.to_csv(output_path, index=False)
    return output_path


def run_alert_generation_pipeline(
    scored_data_path: str | Path = DEFAULT_SCORED_DATA_PATH,
    output_path: str | Path = DEFAULT_ALERT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run Step 7 alert generation and save structured alert records."""
    df = load_scored_data(scored_data_path)
    df = _add_baseline_model_risk(df)
    df = _add_anomaly_signals(df)
    alerts_df = generate_alerts(df)
    saved_path = save_alerts(alerts_df, output_path)
    alerts_df.attrs["output_path"] = str(saved_path)
    return alerts_df


def _add_baseline_model_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Attach baseline ML future-risk probabilities when a model is available."""
    model_path = _resolve_project_path(DEFAULT_RANDOM_FOREST_PATH)
    enriched = df.copy()
    enriched["baseline_future_deterioration_risk"] = 0.0
    if not model_path.exists():
        return enriched

    model = load_model(model_path)
    if not hasattr(model, "predict_proba"):
        return enriched

    X_prepared = _prepare_input_for_prediction(enriched, model=model)
    probabilities = model.predict_proba(X_prepared)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        positive_index = classes.index(1)
        enriched["baseline_future_deterioration_risk"] = probabilities[:, positive_index]
    return enriched


def _add_anomaly_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach anomaly model output if Step 5 model artifact is available."""
    if {"anomaly_label", "anomaly_score", "anomaly_reason"}.issubset(df.columns):
        return df.copy()

    model_path = _resolve_project_path(DEFAULT_ANOMALY_MODEL_PATH)
    enriched = df.copy()
    enriched["anomaly_label"] = 0
    enriched["anomaly_score"] = 0.0
    enriched["anomaly_severity"] = "normal"
    enriched["anomaly_reason"] = ""
    if not model_path.exists():
        return enriched

    model = load_model(model_path)
    X, _ = anomaly_model.select_anomaly_features(enriched)
    anomaly_scored = anomaly_model.score_anomalies(model, X, original_df=enriched)
    for column in ["anomaly_score", "anomaly_label", "anomaly_severity", "anomaly_reason"]:
        enriched[column] = anomaly_scored[column].values
    return enriched


def _should_generate_alert(row: pd.Series, risk_score: float) -> bool:
    """Decide whether a row should become a simulated alert record."""
    return bool(
        risk_score >= 0.25
        or _numeric(row, "time_series_risk_score", 0.0) >= 0.35
        or int(_numeric(row, "anomaly_label", 0)) == 1
        or _has_high_risk_vital_pattern(row)
    )


def _active_risk_signal_count(row: pd.Series | dict[str, Any]) -> int:
    """Count active risk sources for alert typing."""
    signals = [
        _numeric(row, "baseline_future_deterioration_risk", 0.0) >= 0.45,
        int(_numeric(row, "anomaly_label", 0)) == 1,
        _numeric(row, "time_series_risk_score", 0.0) >= 0.35,
        _numeric(row, "abnormal_value_count", 0.0) >= 2,
        _numeric(row, "instability_score", 0.0) >= 0.75,
        bool(_row_value(row, "sustained_oxygen_saturation_drop", False)),
        bool(_row_value(row, "sustained_high_respiratory_rate", False)),
        _has_high_risk_vital_pattern(row),
    ]
    return int(sum(bool(signal) for signal in signals))


def _source_model_summary(row: pd.Series | dict[str, Any]) -> str:
    """Summarize the contributing Step 4-6 sources for downstream audit steps."""
    sources = []
    if _numeric(row, "baseline_future_deterioration_risk", 0.0) > 0:
        sources.append("random_forest")
    if int(_numeric(row, "anomaly_label", 0)) == 1:
        sources.append("isolation_forest")
    if _numeric(row, "time_series_risk_score", 0.0) > 0:
        sources.append("time_series_rules")
    if not sources:
        sources.append("vital_sign_rules")
    return "+".join(sources)


def _make_alert_id(row: pd.Series, sequence_number: int) -> str:
    """Create a deterministic readable alert id."""
    timestamp = pd.to_datetime(row["timestamp"], errors="coerce")
    if pd.isna(timestamp):
        timestamp_part = "unknown-time"
    else:
        timestamp_part = timestamp.strftime("%Y%m%d%H%M%S")
    patient_part = str(row["patient_id"]).replace(" ", "_")
    return f"ALERT-{patient_part}-{timestamp_part}-{sequence_number:06d}"


def _validate_required_input_columns(df: pd.DataFrame) -> None:
    """Ensure the alert generator has the non-label signals it needs."""
    required_columns = {"patient_id", "timestamp", "heart_rate", "oxygen_saturation"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required alert input columns: {missing_columns}")


def _oxygen_risk_component(row: pd.Series | dict[str, Any]) -> float:
    """Return a normalized oxygen-related risk signal."""
    oxygen = _numeric(row, "oxygen_saturation", 97.0)
    oxygen_change = _numeric(row, "oxygen_saturation_change_3", 0.0)
    sustained_drop = bool(_row_value(row, "sustained_oxygen_saturation_drop", False))
    if oxygen <= 88.0:
        return 1.0
    if sustained_drop or oxygen <= 92.0 or oxygen_change <= -3.0:
        return 0.75
    if oxygen <= 94.0 or oxygen_change <= -2.0:
        return 0.45
    return 0.0


def _respiratory_risk_component(row: pd.Series | dict[str, Any]) -> float:
    """Return a normalized respiratory-related risk signal."""
    respiratory_rate = _numeric(row, "respiratory_rate", 16.0)
    respiratory_change = _numeric(row, "respiratory_rate_change_3", 0.0)
    sustained_high = bool(_row_value(row, "sustained_high_respiratory_rate", False))
    if respiratory_rate >= 32.0:
        return 1.0
    if sustained_high or respiratory_rate >= 26.0 or respiratory_change >= 4.0:
        return 0.75
    if respiratory_rate >= 22.0 or respiratory_change >= 3.0:
        return 0.45
    return 0.0


def _has_high_risk_vital_pattern(row: pd.Series | dict[str, Any]) -> bool:
    """Check current simulated vitals for high-priority patterns."""
    return bool(
        _numeric(row, "oxygen_saturation", 97.0) <= 90.0
        or _numeric(row, "heart_rate", 75.0) >= 130.0
        or _numeric(row, "systolic_bp", 120.0) <= 85.0
        or _numeric(row, "respiratory_rate", 16.0) >= 30.0
    )


def _has_critical_vital_pattern(row: pd.Series | dict[str, Any]) -> bool:
    """Check current simulated vitals for critical-priority patterns."""
    return bool(
        _numeric(row, "oxygen_saturation", 97.0) <= 86.0
        or _numeric(row, "heart_rate", 75.0) >= 145.0
        or _numeric(row, "systolic_bp", 120.0) <= 78.0
        or _numeric(row, "respiratory_rate", 16.0) >= 35.0
    )


def _numeric(row: pd.Series | dict[str, Any], key: str, default: float) -> float:
    """Read a numeric row value safely."""
    value = _row_value(row, key, default)
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    """Read a value from a row-like object with a fallback."""
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _clip01(value: float) -> float:
    """Clip a numeric value to the unit interval."""
    return float(np.clip(value, 0.0, 1.0))


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return _project_root() / path_obj


def _project_root() -> Path:
    """Return the repository root for this project."""
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    generated_alerts = run_alert_generation_pipeline()
    output_path = generated_alerts.attrs.get(
        "output_path",
        str(_resolve_project_path(DEFAULT_ALERT_OUTPUT_PATH)),
    )

    print("Step 7 alert generation complete")
    print(f"Total alerts generated: {len(generated_alerts)}")
    print("Severity distribution:")
    print(generated_alerts["severity"].value_counts().to_dict())
    print("Alert type distribution:")
    print(generated_alerts["alert_type"].value_counts().to_dict())
    print(f"Saved alerts to: {output_path}")
    print("First few alerts:")
    print(generated_alerts.head())
