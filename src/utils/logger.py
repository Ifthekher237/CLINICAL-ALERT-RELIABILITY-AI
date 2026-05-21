"""Logging helpers for the prototype.

Later steps will use this module for consistent structured logging across data,
modeling, alerting, monitoring, API, and dashboard code.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a standard library logger for the requested module name."""
    return logging.getLogger(name)

