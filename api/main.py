"""FastAPI backend for the local clinical-alert reliability simulation.

The API is read-only for Step 17. It exposes existing generated artifacts and
SQLite summaries for local demos/tests only. It must not be used with real
patient data or interpreted as a clinically validated system.
"""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.routes_alerts import SIMULATION_ONLY_NOTE, router as alerts_router
from api.routes_explanations import router as explanations_router
from api.routes_monitoring import router as monitoring_router
from src.database.db import get_database_path


app = FastAPI(
    title="Clinical Alert Reliability AI",
    description=(
        "Local research and engineering prototype for simulated clinical-alert "
        "reliability workflows. Not clinically validated and not for patient care."
    ),
    version="0.17.0",
)

app.include_router(alerts_router)
app.include_router(monitoring_router)
app.include_router(explanations_router)


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "project": "clinical-alert-reliability-ai",
        "status": "running",
        "docs": "/docs",
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


@app.get("/health")
def health_check() -> dict[str, object]:
    """Return local backend health for demo checks."""
    database_path = Path(get_database_path())
    return {
        "status": "ok",
        "project": "clinical-alert-reliability-ai",
        "database_available": database_path.exists(),
        "simulation_only_note": SIMULATION_ONLY_NOTE,
    }


if __name__ == "__main__":
    print("Run the local demo API with:")
    print("  .venv/bin/uvicorn api.main:app --reload")
    print("Then open:")
    print("  http://127.0.0.1:8000/health")
    print("  http://127.0.0.1:8000/docs")
    print("  http://127.0.0.1:8000/dashboard-summary")
