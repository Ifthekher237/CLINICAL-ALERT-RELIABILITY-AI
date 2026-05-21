"""SQLite database utilities for local simulated demo storage.

This layer is intentionally lightweight and uses only SQLite. It stores
generated prototype artifacts for demos/tests and must not be used with real
patient data or as a clinical database.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from database.models import get_schema_statements
else:
    from .models import get_schema_statements


DEFAULT_DATABASE_PATH = Path("data/processed/clinical_alert_reliability.db")


def get_database_path() -> str:
    """Return the default SQLite database path for local simulation data."""
    return str(_project_root() / DEFAULT_DATABASE_PATH)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""
    database_path = Path(db_path) if db_path is not None else Path(get_database_path())
    if not database_path.is_absolute():
        database_path = _project_root() / database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: str | None = None) -> None:
    """Create all Step 16 tables if they do not already exist."""
    schema = get_schema_statements()
    with get_connection(db_path) as connection:
        for statement in schema.values():
            connection.execute(statement)
        connection.commit()


def reset_database(db_path: str | None = None) -> None:
    """Drop and recreate all simulation tables."""
    schema = get_schema_statements()
    with get_connection(db_path) as connection:
        for table_name in reversed(list(schema.keys())):
            connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        for statement in schema.values():
            connection.execute(statement)
        connection.commit()


def table_exists(table_name: str, db_path: str | None = None) -> bool:
    """Return whether a table exists in the SQLite database."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        ).fetchone()
    return row is not None


def get_table_counts(db_path: str | None = None) -> dict[str, int]:
    """Return row counts for all known schema tables."""
    initialize_database(db_path)
    counts: dict[str, int] = {}
    with get_connection(db_path) as connection:
        for table_name in get_schema_statements():
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
            counts[table_name] = int(row["count"])
    return counts


def get_database_url() -> str:
    """Return a SQLite URL for compatibility with older placeholder imports."""
    return f"sqlite:///{get_database_path()}"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized SQLite database: {get_database_path()}")
    print("Table counts:")
    for table, count in get_table_counts().items():
        print(f"  {table}: {count}")
