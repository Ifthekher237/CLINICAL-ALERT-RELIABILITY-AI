"""Shared helpers for project-wide reliability validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SIMULATED_DIR = PROJECT_ROOT / "data" / "simulated"


def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path)


def load_json(relative_path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / relative_path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def assert_unique(df: pd.DataFrame, column: str) -> None:
    assert column in df.columns
    assert df[column].notna().all()
    assert df[column].is_unique


def assert_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"


def assert_between_zero_and_one(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        assert column in df.columns
        values = pd.to_numeric(df[column], errors="coerce")
        assert values.notna().all(), f"{column} contains non-numeric values"
        assert values.between(0, 1).all(), f"{column} values outside [0, 1]"


def bool_series(series: pd.Series) -> pd.Series:
    text_true = series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
    numeric_true = pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return text_true | numeric_true
