"""Focused tests for the Step 5 anomaly detection model."""

from __future__ import annotations

from pathlib import Path

from src.models import anomaly_model


def test_processed_data_can_be_loaded() -> None:
    df = anomaly_model.load_processed_data("data/processed/processed_data.csv")

    assert not df.empty
    assert {"patient_id", "timestamp"}.issubset(df.columns)


def test_feature_selection_excludes_leakage_columns() -> None:
    df = anomaly_model.load_processed_data("data/processed/processed_data.csv")
    X, feature_columns = anomaly_model.select_anomaly_features(df)

    assert not X.empty
    assert anomaly_model.LEAKAGE_COLUMNS.isdisjoint(feature_columns)
    assert all(column in X.columns for column in feature_columns)


def test_isolation_forest_trains_successfully() -> None:
    df = anomaly_model.load_processed_data("data/processed/processed_data.csv")
    X, feature_columns = anomaly_model.select_anomaly_features(df)
    model = anomaly_model.train_isolation_forest(X, contamination=0.05)

    assert getattr(model, "anomaly_feature_columns_") == feature_columns
    assert hasattr(model, "predict")


def test_anomaly_scoring_returns_required_columns() -> None:
    df = anomaly_model.load_processed_data("data/processed/processed_data.csv")
    X, _ = anomaly_model.select_anomaly_features(df)
    model = anomaly_model.train_isolation_forest(X, contamination=0.05)
    scored = anomaly_model.score_anomalies(model, X, original_df=df)

    assert set(anomaly_model.REQUIRED_SCORE_COLUMNS).issubset(scored.columns)
    assert set(scored["anomaly_label"].unique()).issubset({0, 1})
    assert scored["anomaly_severity"].notna().all()
    assert scored["anomaly_reason"].notna().all()


def test_model_file_is_saved(tmp_path: Path) -> None:
    df = anomaly_model.load_processed_data("data/processed/processed_data.csv")
    X, _ = anomaly_model.select_anomaly_features(df)
    model = anomaly_model.train_isolation_forest(X, contamination=0.05)
    model_path = anomaly_model.save_anomaly_model(
        model,
        path=tmp_path / "isolation_forest_anomaly.pkl",
    )

    assert model_path.exists()
    assert model_path.name == "isolation_forest_anomaly.pkl"
