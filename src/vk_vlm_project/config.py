"""Configuration models for training and evaluation."""

from dataclasses import dataclass


@dataclass(slots=True)
class ProjectConfig:
    """Simple project configuration container."""

    model_name: str = "baseline-vlm"
    train_data_path: str = "data/train.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    batch_size: int = 8
    learning_rate: float = 1e-4
    seed: int = 42
