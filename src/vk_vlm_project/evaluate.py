"""Evaluation helpers for GQA-ru style question answering."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import ExperimentConfig, load_experiment_config
from .data import VQASample, build_vqa_samples
from .utils.io import ensure_dir
from .utils.logging import get_logger
from .utils.seed import set_seed

try:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _require_runtime_dependencies() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "Evaluation dependencies are missing. Install project deps first, for example: "
            "`pip install -e .` or `uv sync`."
        ) from IMPORT_ERROR


def _resolve_dtype(mixed_precision: str) -> Any:
    precision = mixed_precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def normalize_answer(text: str) -> str:
    """Normalize a Russian short answer for exact-match style metrics."""

    normalized = text.strip().lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s-]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def token_f1(prediction: str, reference: str) -> float:
    """Compute simple token-level F1 after answer normalization."""

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def clean_generated_answer(text: str) -> str:
    """Extract the first short-answer span from a generated continuation."""

    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", " ").replace("</think>", " ")
    cleaned = cleaned.replace("assistant", " ")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = cleaned.strip()

    for prefix in ("Ответ:", "ответ:"):
        while cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()

    stop_patterns = (
        "\nВопрос:",
        "\nвопрос:",
        "\nСледующий вопрос:",
        "\nследующий вопрос:",
        "\nОтвет:",
        "\nответ:",
        "\n\n",
    )
    stop_positions = [cleaned.find(pattern) for pattern in stop_patterns if cleaned.find(pattern) >= 0]
    if stop_positions:
        cleaned = cleaned[: min(stop_positions)].strip()

    if "\n" in cleaned:
        cleaned = next((line.strip() for line in cleaned.splitlines() if line.strip()), cleaned)

    cleaned = cleaned.strip(" \t\r\n.:;,-")
    for prefix in ("Ответ", "ответ"):
        if cleaned == prefix:
            return ""
    return cleaned


def _build_prompt(question: str) -> str:
    return (
        "Ответь кратко на русском языке на вопрос по изображению. "
        "Если вопрос требует выбора, верни только ответ.\n"
        f"Вопрос: {question}\n"
        "Ответ:"
    )


def _load_tokenizer(model_name_or_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _prepare_adapter_path(adapter_path: str | Path) -> str:
    """Return adapter path, remapping legacy language_model LoRA keys when needed."""

    source_dir = Path(adapter_path)
    weights_path = source_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        return str(source_dir)

    state = load_safetensors(str(weights_path))
    needs_remap = any(".language_model.layers." in key for key in state)
    if not needs_remap:
        return str(source_dir)

    digest = hashlib.sha1(str(source_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    target_dir = Path(tempfile.gettempdir()) / f"vk_vlm_adapter_remap_{digest}"
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_name in (
        "adapter_config.json",
        "README.md",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ):
        source_file = source_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, target_dir / file_name)

    remapped = {
        key.replace(".language_model.layers.", ".layers."): value
        for key, value in state.items()
    }
    save_safetensors(remapped, str(target_dir / "adapter_model.safetensors"))
    return str(target_dir)


def _load_model(
    config: ExperimentConfig,
    device: str | None,
    adapter_path: str | None,
) -> tuple[Any, Any]:
    tokenizer = _load_tokenizer(config.model.processor_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=_resolve_dtype(config.train.mixed_precision),
    )
    if adapter_path:
        resolved_adapter_path = _prepare_adapter_path(adapter_path)
        model = PeftModel.from_pretrained(model, resolved_adapter_path)

    target_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _answer_loss(
    model: Any,
    tokenizer: Any,
    sample: VQASample,
    device: torch.device,
    max_seq_length: int,
) -> tuple[float, int]:
    prompt = _build_prompt(sample.question)
    full_text = f"{prompt} {sample.answer}{tokenizer.eos_token or ''}"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    prompt_length = min(prompt_ids.shape[1], labels.shape[1])
    labels[:, :prompt_length] = -100
    answer_tokens = int((labels != -100).sum().item())
    if answer_tokens == 0:
        return 0.0, 0

    with torch.no_grad():
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    return float(loss.item()) * answer_tokens, answer_tokens


def _generate_answer(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    max_seq_length: int,
    max_new_tokens: int,
) -> str:
    prompt = _build_prompt(question)
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer_ids = generated[0, prompt_length:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def _evaluate_loaded_model(
    model: Any,
    tokenizer: Any,
    samples: list[VQASample],
    config: ExperimentConfig,
    output_path: Path | None,
    generate_predictions: bool,
    model_label: str,
) -> dict[str, float]:
    logger = get_logger(__name__)
    device = next(model.parameters()).device
    start_time = perf_counter()
    total_loss = 0.0
    total_answer_tokens = 0
    exact_matches = 0
    f1_sum = 0.0

    output_file = None
    if output_path is not None:
        ensure_dir(output_path.parent)
        output_file = output_path.open("w", encoding="utf-8")

    try:
        for index, sample in enumerate(samples):
            loss_sum, answer_tokens = _answer_loss(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                device=device,
                max_seq_length=config.model.max_seq_length,
            )
            total_loss += loss_sum
            total_answer_tokens += answer_tokens

            raw_prediction = ""
            prediction = ""
            exact_match = 0
            f1 = 0.0
            if generate_predictions:
                raw_prediction = _generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    question=sample.question,
                    device=device,
                    max_seq_length=config.model.max_seq_length,
                    max_new_tokens=config.eval.max_new_tokens,
                )
                prediction = clean_generated_answer(raw_prediction)
                exact_match = int(normalize_answer(prediction) == normalize_answer(sample.answer))
                f1 = token_f1(prediction, sample.answer)
                exact_matches += exact_match
                f1_sum += f1

            if output_file is not None:
                output_file.write(
                    json.dumps(
                        {
                            "model": model_label,
                            "index": index,
                            "image_path": sample.image_path,
                            "question": sample.question,
                            "reference": sample.answer,
                            "raw_prediction": raw_prediction,
                            "prediction": prediction,
                            "exact_match": exact_match,
                            "token_f1": f1,
                            "answer_tokens": answer_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if (index + 1) % 25 == 0:
                logger.info("Evaluated %d/%d samples for %s", index + 1, len(samples), model_label)
    finally:
        if output_file is not None:
            output_file.close()

    eval_runtime = perf_counter() - start_time
    mean_loss = total_loss / max(total_answer_tokens, 1)
    metrics = {
        "samples": float(len(samples)),
        "answer_tokens": float(total_answer_tokens),
        "answer_loss": mean_loss,
        "answer_perplexity": math.exp(min(mean_loss, 100.0)),
        "eval_runtime": eval_runtime,
        "samples_per_second": len(samples) / eval_runtime if eval_runtime > 0 else 0.0,
    }
    if generate_predictions:
        metrics["exact_match"] = exact_matches / max(len(samples), 1)
        metrics["token_f1"] = f1_sum / max(len(samples), 1)
    return metrics


def _free_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_evaluation(
    config: ExperimentConfig,
    device: str | None = None,
    output_dir: str | None = None,
    adapter_path: str | None = None,
    split: str = "val",
    max_samples: int | None = None,
    compare_base: bool = False,
    generate_predictions: bool = True,
) -> dict[str, Any]:
    """Run loss and optional generation evaluation for base and LoRA models."""

    _require_runtime_dependencies()
    logger = get_logger(__name__)
    set_seed(config.seed)

    samples = build_vqa_samples(config.data, split=split)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No samples selected for split={split}")

    run_dir = Path(output_dir or config.run.output_dir)
    ensure_dir(run_dir)
    metrics: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "split": split,
        "max_samples": max_samples,
        "generate_predictions": generate_predictions,
        "config": asdict(config),
        "models": {},
    }

    adapter_path = adapter_path or config.run.checkpoint_dir
    jobs: list[tuple[str, str | None]] = []
    if compare_base:
        jobs.append(("base", None))
    jobs.append(("adapter", adapter_path))

    for label, current_adapter_path in jobs:
        logger.info("Loading %s model", label)
        model, tokenizer = _load_model(config, device=device, adapter_path=current_adapter_path)
        prediction_path = run_dir / f"{split}_{label}_predictions.jsonl"
        metrics["models"][label] = _evaluate_loaded_model(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            config=config,
            output_path=prediction_path if generate_predictions else None,
            generate_predictions=generate_predictions,
            model_label=label,
        )
        metrics["models"][label]["predictions_path"] = str(prediction_path) if generate_predictions else ""
        _free_model(model)

    if compare_base:
        base_loss = metrics["models"]["base"]["answer_loss"]
        adapter_loss = metrics["models"]["adapter"]["answer_loss"]
        metrics["comparison"] = {
            "answer_loss_delta_adapter_minus_base": adapter_loss - base_loss,
            "answer_loss_relative_improvement": (base_loss - adapter_loss) / base_loss
            if base_loss
            else 0.0,
        }
        if generate_predictions:
            base_em = metrics["models"]["base"].get("exact_match", 0.0)
            adapter_em = metrics["models"]["adapter"].get("exact_match", 0.0)
            metrics["comparison"]["exact_match_delta_adapter_minus_base"] = adapter_em - base_em

    metrics_path = run_dir / f"{split}_evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved evaluation metrics to %s", metrics_path)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real QA evaluation")
    parser.add_argument("--config", required=True, help="Path to the experiment config file")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0")
    parser.add_argument("--output-dir", default=None, help="Directory for metrics and predictions")
    parser.add_argument("--resume", default=None, help="Path to LoRA adapter/checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compare-base", action="store_true", help="Evaluate base model first")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Only compute teacher-forced answer loss/perplexity",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    metrics = run_evaluation(
        config=config,
        device=args.device,
        output_dir=args.output_dir,
        adapter_path=args.resume,
        split=args.split,
        max_samples=args.max_samples,
        compare_base=args.compare_base,
        generate_predictions=not args.no_generate,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
