"""General helper utilities for the prototype.

Later steps can add shared validation, path resolution, and configuration
helpers here when reuse becomes worthwhile.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root path based on this utility module location."""
    return Path(__file__).resolve().parents[2]

