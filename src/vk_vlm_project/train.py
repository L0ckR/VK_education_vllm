"""Training pipeline for multimodal LoRA fine-tuning."""

from __future__ import annotations

import json
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import (
    VisionLanguageDataCollator,
    VisionLanguageTrainDataset,
    build_vqa_samples,
)
from .utils.io import ensure_dir
from .utils.logging import get_logger
from .utils.seed import set_seed

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Trainer, TrainingArguments

    try:
        from transformers import AutoModelForImageTextToText as AutoVisionLanguageModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVisionLanguageModel
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "Training dependencies are missing. Install project deps first, for example: "
            "`pip install -e .` or `uv sync`."
        ) from IMPORT_ERROR


def _resolve_dtype(mixed_precision: str) -> Any:
    precision = mixed_precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _prepare_model_and_processor(config: ExperimentConfig) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(
        config.model.processor_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )

    model = AutoVisionLanguageModel.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=_resolve_dtype(config.train.mixed_precision),
    )

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if config.train.gradient_checkpointing:
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=config.train.lora.r,
        lora_alpha=config.train.lora.alpha,
        lora_dropout=config.train.lora.dropout,
        target_modules=config.train.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    return model, processor


def _build_training_arguments(
    config: ExperimentConfig,
    output_dir: str | None,
    run_dir: Path,
) -> Any:
    checkpoint_dir = output_dir or config.run.checkpoint_dir
    logging_dir = run_dir / "tensorboard"
    use_bf16 = config.train.mixed_precision.lower() == "bf16"
    use_fp16 = config.train.mixed_precision.lower() == "fp16"
    signature = inspect.signature(TrainingArguments.__init__)
    supported_kwargs = set(signature.parameters)

    kwargs: dict[str, Any] = {
        "output_dir": checkpoint_dir,
        "run_name": config.experiment_name,
        "num_train_epochs": config.train.num_train_epochs,
        "per_device_train_batch_size": config.train.batch_size,
        "per_device_eval_batch_size": config.eval.batch_size,
        "gradient_accumulation_steps": config.train.grad_accumulation_steps,
        "learning_rate": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
        "warmup_ratio": config.train.warmup_ratio,
        "lr_scheduler_type": config.train.lr_scheduler_type,
        "max_grad_norm": config.train.max_grad_norm,
        "logging_steps": config.train.logging.logging_steps,
        "save_strategy": config.train.checkpoint.save_strategy,
        "save_steps": config.train.checkpoint.save_steps,
        "save_total_limit": config.train.checkpoint.save_total_limit,
        "eval_steps": config.train.checkpoint.eval_steps,
        "load_best_model_at_end": config.train.checkpoint.load_best_model_at_end,
        "metric_for_best_model": config.train.checkpoint.metric_for_best_model,
        "greater_is_better": config.train.checkpoint.greater_is_better,
        "remove_unused_columns": False,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": config.train.logging.report_to,
        "logging_dir": str(logging_dir),
        "dataloader_num_workers": 0,
        "save_safetensors": True,
        "seed": config.seed,
        "data_seed": config.seed,
        "label_names": ["labels"],
        "gradient_checkpointing": config.train.gradient_checkpointing,
        "ddp_find_unused_parameters": False if torch.cuda.device_count() > 1 else None,
        "do_train": True,
        "do_eval": True,
    }

    if "overwrite_output_dir" in supported_kwargs:
        kwargs["overwrite_output_dir"] = True
    if "evaluation_strategy" in supported_kwargs:
        kwargs["evaluation_strategy"] = "steps"
    if "eval_strategy" in supported_kwargs:
        kwargs["eval_strategy"] = "steps"

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_kwargs and value is not None
    }
    return TrainingArguments(**filtered_kwargs)


def _find_last_checkpoint(checkpoint_dir: str | Path) -> str | None:
    target_dir = Path(checkpoint_dir)
    if not target_dir.exists():
        return None

    checkpoint_paths: list[tuple[int, Path]] = []
    for path in target_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        if not (path / "trainer_state.json").exists():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoint_paths.append((step, path))

    if not checkpoint_paths:
        return None
    return str(max(checkpoint_paths, key=lambda item: item[0])[1])


def run_training(
    config: ExperimentConfig,
    device: str | None = None,
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> dict[str, float]:
    """Run LoRA fine-tuning for the configured VLM experiment."""

    _require_runtime_dependencies()

    logger = get_logger(__name__)
    set_seed(config.seed)

    run_dir = ensure_dir(config.run.output_dir)
    ensure_dir(str(run_dir / "tensorboard"))
    ensure_dir(output_dir or config.run.checkpoint_dir)

    logger.info("Loading train/val samples for experiment `%s`", config.experiment_name)
    train_samples = build_vqa_samples(config.data, split="train")
    val_samples = build_vqa_samples(config.data, split="val")
    if config.train.max_train_samples is not None:
        train_samples = train_samples[: config.train.max_train_samples]
    if config.train.max_eval_samples is not None:
        val_samples = val_samples[: config.train.max_eval_samples]
    logger.info(
        "Loaded %d train and %d validation samples",
        len(train_samples),
        len(val_samples),
    )

    model, processor = _prepare_model_and_processor(config)
    collator = VisionLanguageDataCollator(
        processor=processor, model_config=config.model
    )
    train_dataset = VisionLanguageTrainDataset(train_samples)
    eval_dataset = VisionLanguageTrainDataset(val_samples)

    training_args = _build_training_arguments(
        config, output_dir=output_dir, run_dir=run_dir
    )
    logger.info("Training checkpoints will be saved to %s", training_args.output_dir)

    trainer_signature = inspect.signature(Trainer.__init__)
    trainer_supported_kwargs = set(trainer_signature.parameters)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    if "processing_class" in trainer_supported_kwargs:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_supported_kwargs:
        trainer_kwargs["tokenizer"] = processor.tokenizer

    trainer = Trainer(**trainer_kwargs)

    if device:
        logger.info("Device override requested: %s", device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is false"
            )

    resolved_resume_checkpoint = resume_from_checkpoint or _find_last_checkpoint(
        training_args.output_dir
    )
    if resolved_resume_checkpoint:
        logger.info("Resuming training from checkpoint %s", resolved_resume_checkpoint)
    else:
        logger.info("Starting training from scratch")

    train_result = trainer.train(resume_from_checkpoint=resolved_resume_checkpoint)
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)

    metrics = dict(train_result.metrics)
    metrics.update(trainer.evaluate())

    metrics_path = run_dir / "train_metrics.json"
    config_snapshot_path = run_dir / "resolved_experiment.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_snapshot_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved resolved config to %s", config_snapshot_path)
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
