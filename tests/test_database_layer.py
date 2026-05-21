"""Focused tests for Step 16 SQLite database layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.database import crud, db, models


def _temp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "clinical_alert_reliability_test.db")


def test_database_initializes_successfully(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)

    assert Path(db_path).exists()


def test_all_required_tables_exist(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)

    for table_name in models.get_schema_statements():
        assert db.table_exists(table_name, db_path)


def test_table_counts_return_dictionary(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)
    counts = db.get_table_counts(db_path)

    assert isinstance(counts, dict)
    assert set(models.get_schema_statements()).issubset(counts)


def test_inserting_demo_data_returns_integer_counts(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)
    inserted_counts = crud.load_all_demo_data(db_path)

    assert isinstance(inserted_counts, dict)
    assert all(isinstance(value, int) for value in inserted_counts.values())
    assert inserted_counts["patients"] > 0
    assert inserted_counts["vitals"] > 0
    assert inserted_counts["alerts"] > 0


def test_repeated_inserts_do_not_duplicate_primary_key_records(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)

    first_insert = crud.insert_patients_from_simulated_data(
        "data/simulated/patient_monitoring.csv",
        db_path,
    )
    first_count = db.get_table_counts(db_path)["patients"]
    second_insert = crud.insert_patients_from_simulated_data(
        "data/simulated/patient_monitoring.csv",
        db_path,
    )
    second_count = db.get_table_counts(db_path)["patients"]

    assert first_insert == second_insert
    assert first_count == second_count


def test_fetch_table_returns_dataframe(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)
    crud.insert_alerts_from_fatigue_data("data/processed/fatigue_reduced_alerts.csv", db_path)
    alerts = crud.fetch_table("alerts", limit=5, db_path=db_path)

    assert isinstance(alerts, pd.DataFrame)
    assert len(alerts) <= 5
    assert "alert_id" in alerts.columns


def test_reset_database_removes_and_recreates_tables(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)
    crud.insert_alerts_from_fatigue_data("data/processed/fatigue_reduced_alerts.csv", db_path)
    assert db.get_table_counts(db_path)["alerts"] > 0

    db.reset_database(db_path)
    counts = db.get_table_counts(db_path)

    assert db.table_exists("alerts", db_path)
    assert counts["alerts"] == 0


def test_optional_missing_files_are_handled_safely(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)
    missing_count = crud.insert_drift_logs(tmp_path / "missing_drift_file.csv", db_path)

    assert missing_count == 0


def test_no_real_patient_data_assumption_is_made(tmp_path: Path) -> None:
    db_path = _temp_db_path(tmp_path)
    db.initialize_database(db_path)
    crud.insert_patients_from_simulated_data("data/simulated/patient_monitoring.csv", db_path)
    patients = crud.fetch_table("patients", limit=3, db_path=db_path)

    assert not patients.empty
    assert patients["simulation_note"].str.contains("Simulated demo data only").all()
