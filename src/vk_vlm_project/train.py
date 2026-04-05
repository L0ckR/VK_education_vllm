"""Training entry point."""

from __future__ import annotations

from .config import ProjectConfig
from .utils.logging import get_logger
from .utils.seed import set_seed


def run_training(config: ProjectConfig) -> dict[str, float]:
    """Run a dummy training routine and return metrics."""
    logger = get_logger(__name__)
    set_seed(config.seed)
    logger.info("Training model %s with batch_size=%d", config.model_name, config.batch_size)
    return {"train_loss": 0.1234}


def main() -> None:
    """CLI entry point for model training."""
    metrics = run_training(ProjectConfig())
    print(metrics)


if __name__ == "__main__":
    main()
