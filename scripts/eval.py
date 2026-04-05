#!/usr/bin/env python3
"""Evaluation / prediction entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation or prediction")
    parser.add_argument("--config", required=True, help="Path to the evaluation config file")
    parser.add_argument("--device", default=None, help="Device override (e.g. cuda:0)")
    parser.add_argument("--output-dir", default=None, help="Directory for artifacts/results")
    parser.add_argument("--resume", default=None, help="Path to checkpoint/weights")
    parser.add_argument(
        "--mode",
        choices=("eval", "predict"),
        default="eval",
        help="Execution mode",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    print(f"[{args.mode}] starting")
    print(f"[{args.mode}] config={config_path}")
    print(f"[{args.mode}] device={args.device or 'default'}")
    print(f"[{args.mode}] output_dir={args.output_dir or 'default'}")
    print(f"[{args.mode}] resume={args.resume or 'none'}")
    # TODO: replace prints with actual evaluation/prediction pipeline invocation.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
