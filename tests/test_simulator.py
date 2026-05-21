"""Placeholder tests for the future data simulator."""

from __future__ import annotations

from src.data import simulator


def test_simulator_module_imports() -> None:
    """The Step 1 simulator module should be importable."""
    assert simulator.PatientDataSimulator is not None

