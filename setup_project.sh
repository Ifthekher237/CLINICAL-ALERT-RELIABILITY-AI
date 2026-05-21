#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="clinical-alert-reliability-ai"
VENV_DIR=".venv"

echo "Setting up ${PROJECT_NAME}..."

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment in ${VENV_DIR}/"
  python3 -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists at ${VENV_DIR}/"
fi

echo "Installing Python dependencies..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

echo
echo "Setup complete."
echo
echo "Next useful commands:"
echo "  ${VENV_DIR}/bin/python -m pytest -q"
echo "  ${VENV_DIR}/bin/python -m uvicorn api.main:app --reload"
echo "  ${VENV_DIR}/bin/python -m streamlit run dashboard/app.py"
echo
echo "Note: this project uses simulated data only and is not a clinically validated medical device."
