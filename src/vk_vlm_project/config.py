"""Configuration models and YAML loaders for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    """Dataset-related paths and field names."""

    name: str
    train_path: str
    val_path: str
    test_path: str
    image_root: str
    question_field: str
    answer_field: str
    metric_primary: str


@dataclass(slots=True)
class ModelConfig:
    """Model and processor identifiers."""

    name: str
    model_name_or_path: str
    processor_name_or_path: str
    trust_remote_code: bool = True
    max_seq_length: int = 2048
    image_resolution: int = 448


@dataclass(slots=True)
class LoRAConfig:
    """PEFT LoRA hyperparameters."""

    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclass(slots=True)
class CheckpointConfig:
    """Checkpoint saving strategy."""

    output_dir: str
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass(slots=True)
class LoggingConfig:
    """Trainer logging configuration."""

    logging_steps: int = 25
    report_to: list[str] | None = None


@dataclass(slots=True)
class TrainConfig:
    """Training hyperparameters."""

    name: str
    seed: int
    mixed_precision: str
    batch_size: int
    grad_accumulation_steps: int
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_grad_norm: float
    lora: LoRAConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig


@dataclass(slots=True)
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 4
    max_new_tokens: int = 32


@dataclass(slots=True)
class InferenceConfig:
    """Inference configuration."""

    batch_size: int = 1
    max_new_tokens: int = 64


@dataclass(slots=True)
class RunConfig:
    """Run-level paths and metric names."""

    output_dir: str
    checkpoint_dir: str
    eval_metric: str = "accuracy"


@dataclass(slots=True)
class ExperimentConfig:
    """Fully resolved experiment configuration."""

    experiment_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    inference: InferenceConfig
    run: RunConfig
    raw_paths: dict[str, str]


def _read_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {file_path}")

    with file_path.open("r", encoding="utf-8") as source:
        content = yaml.safe_load(source) or {}

    if not isinstance(content, dict):
        raise ValueError(f"Expected mapping in config file: {file_path}")
    return content


def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment config and all referenced child configs."""

    experiment_path = Path(config_path).resolve()
    experiment_raw = _read_yaml(experiment_path)

    root_dir = experiment_path.parents[2]

    child_paths = {
        "data_config": _resolve_path(root_dir, experiment_raw["data_config"]),
        "model_config": _resolve_path(root_dir, experiment_raw["model_config"]),
        "train_config": _resolve_path(root_dir, experiment_raw["train_config"]),
        "eval_config": _resolve_path(root_dir, experiment_raw["eval_config"]),
        "inference_config": _resolve_path(root_dir, experiment_raw["inference_config"]),
    }

    data_raw = _read_yaml(child_paths["data_config"])
    model_raw = _read_yaml(child_paths["model_config"])
    train_raw = _read_yaml(child_paths["train_config"])
    eval_raw = _read_yaml(child_paths["eval_config"])
    inference_raw = _read_yaml(child_paths["inference_config"])
    run_raw = experiment_raw.get("run", {})

    data = DataConfig(
        name=data_raw["name"],
        train_path=_resolve_path(root_dir, data_raw["train_path"]),
        val_path=_resolve_path(root_dir, data_raw["val_path"]),
        test_path=_resolve_path(root_dir, data_raw["test_path"]),
        image_root=_resolve_path(root_dir, data_raw["image_root"]),
        question_field=data_raw.get("question_field", "question"),
        answer_field=data_raw.get("answer_field", "answer"),
        metric_primary=data_raw.get("metric_primary", "accuracy"),
    )

    model = ModelConfig(
        name=model_raw["name"],
        model_name_or_path=model_raw["model_name_or_path"],
        processor_name_or_path=model_raw["processor_name_or_path"],
        trust_remote_code=bool(model_raw.get("trust_remote_code", True)),
        max_seq_length=int(model_raw.get("max_seq_length", 2048)),
        image_resolution=int(model_raw.get("image_resolution", 448)),
    )

    lora = train_raw.get("lora", {})
    checkpoint = train_raw.get("checkpoint", {})
    logging = train_raw.get("logging", {})
    train = TrainConfig(
        name=train_raw["name"],
        seed=int(train_raw.get("seed", experiment_raw.get("seed", 42))),
        mixed_precision=str(train_raw.get("mixed_precision", "bf16")),
        batch_size=int(train_raw["batch_size"]),
        grad_accumulation_steps=int(train_raw.get("grad_accumulation_steps", 1)),
        num_train_epochs=float(train_raw.get("num_train_epochs", 1)),
        learning_rate=float(train_raw["learning_rate"]),
        weight_decay=float(train_raw.get("weight_decay", 0.0)),
        warmup_ratio=float(train_raw.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(train_raw.get("lr_scheduler_type", "cosine")),
        max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
        lora=LoRAConfig(
            r=int(lora.get("r", 16)),
            alpha=int(lora.get("alpha", 32)),
            dropout=float(lora.get("dropout", 0.0)),
            target_modules=list(lora.get("target_modules", [])),
        ),
        checkpoint=CheckpointConfig(
            output_dir=_resolve_path(root_dir, checkpoint.get("output_dir", "checkpoints/default")),
            save_strategy=str(checkpoint.get("save_strategy", "steps")),
            save_steps=int(checkpoint.get("save_steps", 500)),
            eval_steps=int(checkpoint.get("eval_steps", checkpoint.get("save_steps", 500))),
            save_total_limit=int(checkpoint.get("save_total_limit", 3)),
            load_best_model_at_end=bool(checkpoint.get("load_best_model_at_end", False)),
            metric_for_best_model=str(checkpoint.get("metric_for_best_model", "eval_loss")),
            greater_is_better=bool(checkpoint.get("greater_is_better", False)),
        ),
        logging=LoggingConfig(
            logging_steps=int(logging.get("logging_steps", 25)),
            report_to=list(logging.get("report_to", [])) or None,
        ),
    )

    eval_prediction = eval_raw.get("prediction", {})
    inference_prediction = inference_raw.get("prediction", {})
    eval_config = EvalConfig(
        batch_size=int(eval_raw.get("batch_size", 4)),
        max_new_tokens=int(eval_prediction.get("max_new_tokens", eval_raw.get("max_new_tokens", 32))),
    )
    inference = InferenceConfig(
        batch_size=int(inference_raw.get("batch_size", 1)),
        max_new_tokens=int(
            inference_prediction.get("max_new_tokens", inference_raw.get("max_new_tokens", 64))
        ),
    )
    run = RunConfig(
        output_dir=_resolve_path(root_dir, run_raw.get("output_dir", f"runs/{experiment_raw['experiment_name']}")),
        checkpoint_dir=_resolve_path(
            root_dir,
            run_raw.get("checkpoint_dir", train.checkpoint.output_dir),
        ),
        eval_metric=str(run_raw.get("eval_metric", data.metric_primary)),
    )

    return ExperimentConfig(
        experiment_name=experiment_raw["experiment_name"],
        seed=int(experiment_raw.get("seed", train.seed)),
        data=data,
        model=model,
        train=train,
        eval=eval_config,
        inference=inference,
        run=run,
        raw_paths=child_paths,
    )
