"""Evaluation entry point."""

from __future__ import annotations

from .config import ProjectConfig
from .utils.logging import get_logger


def run_evaluation(config: ProjectConfig) -> dict[str, float]:
    """Run a dummy evaluation routine and return metrics."""
    logger = get_logger(__name__)
    logger.info("Evaluating model %s", config.model_name)
    return {"accuracy": 0.95}


def main() -> None:
    """CLI entry point for model evaluation."""
    metrics = run_evaluation(ProjectConfig())
    print(metrics)


if __name__ == "__main__":
    main()
