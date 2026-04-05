"""Random seed utilities."""

import random


def set_seed(seed: int) -> None:
    """Seed random number generators used in the project."""
    random.seed(seed)
