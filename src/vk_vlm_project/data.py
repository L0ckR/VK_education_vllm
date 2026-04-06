"""Dataset and collation helpers for VLM training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on runtime environment
    Image = None

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - depends on runtime environment
    class Dataset:  # type: ignore[override]
        """Fallback Dataset base when torch is not installed."""

        pass

from .config import DataConfig, ModelConfig

IMAGE_KEYS = ("image", "image_path", "image_file", "image_filename")


@dataclass(slots=True)
class VQASample:
    """Single VQA-style record."""

    image_path: str
    question: str
    answer: str


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read JSONL records from disk."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            "Dataset file does not exist: "
            f"{file_path}\n"
            "Prepare local JSONL manifests before training. Expected one JSON object per line, "
            'for example: {"image": "000123.jpg", "question": "Что изображено?", '
            '"answer": "Красный автобус"}'
        )

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {file_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {file_path}:{line_number}")
            records.append(payload)
    return records


def _extract_image_path(record: dict[str, Any], image_root: Path) -> str:
    for key in IMAGE_KEYS:
        raw_path = record.get(key)
        if raw_path:
            path = Path(str(raw_path))
            return str(path if path.is_absolute() else (image_root / path).resolve())
    raise KeyError(
        "Image path is missing. Expected one of keys: "
        + ", ".join(IMAGE_KEYS)
    )


def build_vqa_samples(config: DataConfig, split: str) -> list[VQASample]:
    """Build normalized VQA samples for the requested split."""

    split_to_path = {
        "train": config.train_path,
        "val": config.val_path,
        "test": config.test_path,
    }
    if split not in split_to_path:
        raise ValueError(f"Unsupported split: {split}")

    image_root = Path(config.image_root)
    records = read_jsonl(split_to_path[split])
    samples: list[VQASample] = []
    for index, record in enumerate(records):
        try:
            image_path = _extract_image_path(record, image_root)
            question = str(record[config.question_field]).strip()
            answer = str(record[config.answer_field]).strip()
        except KeyError as exc:
            raise KeyError(f"Bad record in split={split} at index={index}: {exc}") from exc

        if not question:
            raise ValueError(f"Empty question in split={split} at index={index}")
        if not answer:
            raise ValueError(f"Empty answer in split={split} at index={index}")

        samples.append(VQASample(image_path=image_path, question=question, answer=answer))

    return samples


class VisionLanguageTrainDataset(Dataset):
    """PyTorch dataset for image-question-answer training samples."""

    def __init__(self, samples: list[VQASample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VQASample:
        return self.samples[index]


class VisionLanguageDataCollator:
    """Prepare multimodal batches for causal LM fine-tuning."""

    def __init__(self, processor: Any, model_config: ModelConfig) -> None:
        self.processor = processor
        self.model_config = model_config
        self.image_token_id = getattr(processor.tokenizer, "image_token_id", None)

    @staticmethod
    def _messages(question: str, answer: str | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
        return messages

    def _build_prompt(self, question: str, answer: str | None = None) -> str:
        return self.processor.apply_chat_template(
            self._messages(question=question, answer=answer),
            tokenize=False,
            add_generation_prompt=answer is None,
        )

    def __call__(self, features: list[VQASample]) -> dict[str, Any]:
        if Image is None:
            raise RuntimeError("Pillow is required for image loading. Install project dependencies first.")

        images: list[Image.Image] = []
        prompt_texts: list[str] = []
        full_texts: list[str] = []

        for feature in features:
            image = Image.open(feature.image_path).convert("RGB")
            if self.model_config.image_resolution > 0:
                image = image.resize(
                    (self.model_config.image_resolution, self.model_config.image_resolution)
                )
            images.append(image)
            prompt_texts.append(self._build_prompt(feature.question))
            full_texts.append(self._build_prompt(feature.question, feature.answer))

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.model_config.max_seq_length,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.model_config.max_seq_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels = labels.masked_fill(batch["attention_mask"] == 0, -100)

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row_index, :prompt_length] = -100

        if self.image_token_id is not None:
            labels = labels.masked_fill(batch["input_ids"] == self.image_token_id, -100)

        batch["labels"] = labels
        return batch
