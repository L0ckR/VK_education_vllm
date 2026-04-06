#!/usr/bin/env python3
"""Training entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vk_vlm_project.config import load_experiment_config
from vk_vlm_project.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model training")
    parser.add_argument("--config", required=True, help="Path to the experiment config file")
    parser.add_argument("--device", default=None, help="Device override (e.g. cuda:0)")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint for resume")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = load_experiment_config(config_path)
    metrics = run_training(
        config=config,
        device=args.device,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
