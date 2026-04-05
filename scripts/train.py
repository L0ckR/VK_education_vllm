#!/usr/bin/env python3
"""Training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model training")
    parser.add_argument("--config", required=True, help="Path to the training config file")
    parser.add_argument("--device", default=None, help="Device override (e.g. cuda:0)")
    parser.add_argument("--output-dir", default=None, help="Directory for artifacts/checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint for resume")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    print("[train] starting")
    print(f"[train] config={config_path}")
    print(f"[train] device={args.device or 'default'}")
    print(f"[train] output_dir={args.output_dir or 'default'}")
    print(f"[train] resume={args.resume or 'none'}")
    # TODO: replace prints with actual training pipeline invocation.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
