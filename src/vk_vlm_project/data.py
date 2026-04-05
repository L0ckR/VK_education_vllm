"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path


def read_jsonl(path: str) -> list[str]:
    """Read a JSONL-like text file and return non-empty lines."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as source:
        return [line.strip() for line in source if line.strip()]
