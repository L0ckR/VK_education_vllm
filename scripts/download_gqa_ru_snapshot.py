#!/usr/bin/env python3
"""Download the full DeepVK GQA-ru dataset repository snapshot from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vk_vlm_project.utils.logging import get_logger


DATASET_ID = "deepvk/GQA-ru"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download DeepVK GQA-ru dataset repo snapshot")
    parser.add_argument(
        "--output-dir",
        default="data/raw/hf_gqa_ru_snapshot",
        help="Directory where the full snapshot will be downloaded",
    )
    parser.add_argument("--revision", default=None, help="Optional dataset revision")
    parser.add_argument("--token", default=None, help="Optional HF token")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = get_logger(__name__)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=args.revision,
        token=args.token,
        local_dir=str(output_dir),
    )
    logger.info("Downloaded dataset snapshot to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
