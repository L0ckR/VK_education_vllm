#!/usr/bin/env python3
"""Prepare local JSONL manifests and images from a GQA-ru HF snapshot."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vk_vlm_project.utils.io import ensure_dir
from vk_vlm_project.utils.logging import get_logger


DATASET_ID = "deepvk/GQA-ru"
TRAIN_IMAGES_DIR = "train_balanced_images"
TRAIN_INSTRUCTIONS_DIR = "train_balanced_instructions"
TEST_IMAGES_DIR = "testdev_balanced_images"
TEST_INSTRUCTIONS_DIR = "testdev_balanced_instructions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DeepVK GQA-ru under data/gqa_ru/")
    parser.add_argument("--output-root", default="data/gqa_ru", help="Output dataset directory")
    parser.add_argument(
        "--source-dir",
        default="data/raw/hf_gqa_ru_snapshot",
        help="Local snapshot directory created from the HF dataset repo",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face datasets cache directory")
    parser.add_argument("--val-size", type=float, default=0.05, help="Validation fraction by imageId")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument(
        "--answer-field",
        choices=("answer", "fullAnswer"),
        default="answer",
        help="Which answer field from DeepVK to store in JSONL",
    )
    parser.add_argument("--train-max-samples", type=int, default=None, help="Optional limit for train records")
    parser.add_argument("--test-max-samples", type=int, default=None, help="Optional limit for test records")
    parser.add_argument("--train-max-images", type=int, default=None, help="Optional limit for train images")
    parser.add_argument("--test-max-images", type=int, default=None, help="Optional limit for test images")
    parser.add_argument(
        "--skip-existing-images",
        action="store_true",
        help="Do not overwrite already downloaded images",
    )
    return parser


def _read_snapshot_table(snapshot_dir: Path, subdir: str) -> pd.DataFrame:
    parquet_paths = sorted((snapshot_dir / subdir).glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {(snapshot_dir / subdir).resolve()}")
    frames = [pd.read_parquet(path) for path in parquet_paths]
    return pd.concat(frames, ignore_index=True)


def _load_hub_split(config_name: str, split_name: str, cache_dir: str | None):
    dataset = load_dataset(
        DATASET_ID,
        config_name,
        split=split_name,
        cache_dir=cache_dir,
        streaming=True,
    )
    return list(dataset)


def _load_source_tables(source_dir: Path, cache_dir: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger = get_logger(__name__)
    if source_dir.exists():
        logger.info("Loading local snapshot from %s", source_dir)
        train_images = _read_snapshot_table(source_dir, TRAIN_IMAGES_DIR)
        train_instructions = _read_snapshot_table(source_dir, TRAIN_INSTRUCTIONS_DIR)
        test_images = _read_snapshot_table(source_dir, TEST_IMAGES_DIR)
        test_instructions = _read_snapshot_table(source_dir, TEST_INSTRUCTIONS_DIR)
        return train_images, train_instructions, test_images, test_instructions

    logger.info("Local snapshot %s does not exist, loading from HF Hub", source_dir)
    train_images = pd.DataFrame(_load_hub_split("train_balanced_images", "train", cache_dir))
    train_instructions = pd.DataFrame(_load_hub_split("train_balanced_instructions", "train", cache_dir))
    test_images = pd.DataFrame(_load_hub_split("testdev_balanced_images", "testdev", cache_dir))
    test_instructions = pd.DataFrame(_load_hub_split("testdev_balanced_instructions", "testdev", cache_dir))
    return train_images, train_instructions, test_images, test_instructions


def _iter_image_rows(df: pd.DataFrame, max_images: int | None) -> Iterable[dict]:
    count = 0
    for row in df.to_dict(orient="records"):
        yield row
        count += 1
        if max_images is not None and count >= max_images:
            break


def _extract_image_bytes(image_payload: object) -> bytes:
    if isinstance(image_payload, dict):
        image_bytes = image_payload.get("bytes")
        if image_bytes is not None:
            return image_bytes
    raise ValueError("Unsupported image payload format in snapshot parquet")


def _save_images(df: pd.DataFrame, images_dir: Path, skip_existing_images: bool, max_images: int | None) -> set[str]:
    logger = get_logger(__name__)
    image_ids: set[str] = set()

    for row_index, row in enumerate(_iter_image_rows(df, max_images=max_images), start=1):
        image_id = str(row["id"])
        image_ids.add(image_id)

        image_path = images_dir / f"{image_id}.jpg"
        if skip_existing_images and image_path.exists():
            continue

        image_bytes = _extract_image_bytes(row["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(image_path, format="JPEG", quality=95)

        if row_index % 1000 == 0:
            logger.info("Saved %d images into %s", row_index, images_dir)

    logger.info("Finished saving %d images into %s", len(image_ids), images_dir)
    return image_ids


def _write_jsonl_records(
    df: pd.DataFrame,
    output_path: Path,
    allowed_image_ids: set[str],
    val_image_ids: set[str],
    answer_field: str,
    split_name: str,
    max_samples: int | None,
) -> int:
    written = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for row in df.to_dict(orient="records"):
            image_id = str(row["imageId"])
            if image_id not in allowed_image_ids:
                continue

            target_split = "val" if split_name == "train" and image_id in val_image_ids else split_name
            if output_path.stem != target_split:
                continue

            record = {
                "id": str(row["id"]),
                "image": f"{image_id}.jpg",
                "question": str(row["question"]).strip(),
                "answer": str(row[answer_field]).strip(),
                "image_id": image_id,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if max_samples is not None and written >= max_samples:
                break
    return written


def main() -> int:
    args = build_parser().parse_args()
    logger = get_logger(__name__)

    if not 0.0 < args.val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1")

    output_root = Path(args.output_root).resolve()
    source_dir = Path(args.source_dir).resolve()
    images_dir = ensure_dir(str(output_root / "images"))
    output_root.mkdir(parents=True, exist_ok=True)

    train_images_df, train_instructions_df, test_images_df, test_instructions_df = _load_source_tables(
        source_dir=source_dir,
        cache_dir=args.cache_dir,
    )

    logger.info("Saving train images")
    train_image_ids = _save_images(
        train_images_df,
        images_dir,
        args.skip_existing_images,
        args.train_max_images,
    )

    logger.info("Saving test images")
    test_image_ids = _save_images(
        test_images_df,
        images_dir,
        args.skip_existing_images,
        args.test_max_images,
    )

    all_train_image_ids = sorted(train_image_ids)
    rng = random.Random(args.seed)
    rng.shuffle(all_train_image_ids)
    val_image_count = max(1, int(len(all_train_image_ids) * args.val_size))
    val_image_ids = set(all_train_image_ids[:val_image_count])

    logger.info(
        "Split train images by imageId: train=%d val=%d",
        len(all_train_image_ids) - len(val_image_ids),
        len(val_image_ids),
    )

    train_path = output_root / "train.jsonl"
    val_path = output_root / "val.jsonl"
    test_path = output_root / "test.jsonl"

    train_count = _write_jsonl_records(
        train_instructions_df,
        train_path,
        train_image_ids,
        val_image_ids,
        args.answer_field,
        "train",
        args.train_max_samples,
    )
    val_count = _write_jsonl_records(
        train_instructions_df,
        val_path,
        train_image_ids,
        val_image_ids,
        args.answer_field,
        "train",
        args.train_max_samples,
    )
    test_count = _write_jsonl_records(
        test_instructions_df,
        test_path,
        test_image_ids,
        set(),
        args.answer_field,
        "test",
        args.test_max_samples,
    )

    summary = {
        "dataset_id": DATASET_ID,
        "source_dir": str(source_dir),
        "answer_field": args.answer_field,
        "output_root": str(output_root),
        "images_dir": str(images_dir),
        "train_records": train_count,
        "val_records": val_count,
        "test_records": test_count,
        "unique_train_images": len(train_image_ids),
        "unique_val_images": len(val_image_ids),
        "unique_test_images": len(test_image_ids),
    }
    summary_path = output_root / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Prepared dataset under %s", output_root)
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
