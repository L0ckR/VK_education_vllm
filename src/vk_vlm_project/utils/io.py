"""Input/output helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str) -> Path:
    """Create directory (including parents) if it does not exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
