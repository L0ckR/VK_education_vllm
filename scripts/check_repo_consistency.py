#!/usr/bin/env python3
"""Repository consistency checks for VK VLM project scaffold."""

from __future__ import annotations

from pathlib import Path


REQUIRED_DIRS = [
    "src",
    "configs",
    "scripts",
    "docs",
    "data",
    "notebooks",
    "notebooks/eda",
]

REQUIRED_FILES = [
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "docs/project_description.md",
    "docs/solution_detailed.md",
    "docs/final_report_template.md",
    "scripts/train.py",
    "scripts/eval.py",
]

EXPERIMENT_REFERENCE_KEYS = [
    "data_config",
    "model_config",
    "train_config",
    "eval_config",
    "inference_config",
]


def parse_simple_yaml_map(file_path: Path) -> dict[str, str]:
    """Parse top-level `key: value` entries from a simple YAML file."""
    parsed: dict[str, str] = {}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value and not key.startswith("-"):
            parsed[key] = value
    return parsed


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    errors: list[str] = []

    for directory in REQUIRED_DIRS:
        if not (repo_root / directory).is_dir():
            errors.append(f"Missing directory: {directory}")

    for file_path in REQUIRED_FILES:
        if not (repo_root / file_path).is_file():
            errors.append(f"Missing file: {file_path}")

    experiments_dir = repo_root / "configs" / "experiments"
    experiment_files = sorted(experiments_dir.glob("*.yaml")) if experiments_dir.exists() else []
    if not experiment_files:
        errors.append("No experiment configs found in configs/experiments")

    for exp_file in experiment_files:
        parsed = parse_simple_yaml_map(exp_file)
        for ref_key in EXPERIMENT_REFERENCE_KEYS:
            ref_value = parsed.get(ref_key)
            if not ref_value:
                errors.append(f"{exp_file.relative_to(repo_root)}: missing key `{ref_key}`")
                continue

            ref_path = repo_root / ref_value
            if not ref_path.exists():
                errors.append(
                    f"{exp_file.relative_to(repo_root)}: `{ref_key}` points to missing file `{ref_value}`"
                )

    if errors:
        print("[consistency] FAILED")
        for error in errors:
            print(f" - {error}")
        return 1

    print("[consistency] OK")
    print(f"[consistency] checked experiments: {len(experiment_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
