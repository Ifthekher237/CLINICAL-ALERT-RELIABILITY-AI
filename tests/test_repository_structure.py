from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_project_directories_exist() -> None:
    required_directories = [
        "src",
        "api",
        "dashboard",
        "data",
        "data/processed",
        "reports",
        "knowledge_base",
        "tests",
        "docs",
    ]

    for directory in required_directories:
        assert (ROOT / directory).is_dir(), f"Missing required directory: {directory}"


def test_portfolio_packaging_files_exist() -> None:
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "setup_project.sh",
        "PROJECT_STRUCTURE.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "docs/project_overview.md",
    ]

    for file_path in required_files:
        path = ROOT / file_path
        assert path.is_file(), f"Missing packaging file: {file_path}"
        assert path.stat().st_size > 0, f"Packaging file is empty: {file_path}"


def test_demo_entry_points_exist() -> None:
    required_files = [
        "api/main.py",
        "dashboard/app.py",
        "reports/evaluation_report_generator.py",
        "reports/evaluation_results.md",
    ]

    for file_path in required_files:
        assert (ROOT / file_path).is_file(), f"Missing demo entry point: {file_path}"


def test_important_processed_outputs_exist() -> None:
    required_outputs = [
        "data/processed/project_metrics_summary.json",
        "data/processed/project_metrics_table.csv",
        "data/processed/outcome_effectiveness_summary.json",
        "data/processed/failure_mode_summary.json",
        "data/processed/scenario_test_summary.json",
        "data/processed/clinical_alert_reliability.db",
    ]

    for file_path in required_outputs:
        assert (ROOT / file_path).is_file(), f"Missing curated demo output: {file_path}"


def test_gitignore_keeps_curated_outputs_visible() -> None:
    gitignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    active_patterns = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    forbidden_patterns = [
        "data/processed/*",
        "data/processed/",
        "models/",
        "*.csv",
        "*.json",
        "*.db",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in active_patterns, f".gitignore hides curated portfolio artifact pattern: {pattern}"

    required_patterns = [
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".ipynb_checkpoints/",
        ".DS_Store",
        "logs/",
        ".vscode/",
        ".idea/",
    ]

    for pattern in required_patterns:
        assert pattern in gitignore_text, f".gitignore should ignore local artifact: {pattern}"


def test_setup_script_is_lightweight_and_clear() -> None:
    script = (ROOT / "setup_project.sh").read_text(encoding="utf-8")

    assert "python3 -m venv" in script
    assert "pip install -r requirements.txt" in script
    assert "pytest -q" in script
    assert "uvicorn api.main:app" in script
    assert "streamlit run dashboard/app.py" in script

    forbidden_heavy_commands = [
        "risk_model.py",
        "anomaly_model.py",
        "drift_detector.py",
        "scenario_tester.py",
        "run_metrics_pipeline",
    ]
    for command in forbidden_heavy_commands:
        assert command not in script, f"Setup script should not run heavy pipeline command: {command}"


def test_documentation_contains_simulation_boundary() -> None:
    docs_to_check = [
        "README.md",
        "PROJECT_STRUCTURE.md",
        "CONTRIBUTING.md",
        "docs/project_overview.md",
    ]

    for file_path in docs_to_check:
        text = (ROOT / file_path).read_text(encoding="utf-8").lower()
        assert "simulated" in text or "simulation" in text
        assert "not clinically validated" in text or "clinical validation" in text
