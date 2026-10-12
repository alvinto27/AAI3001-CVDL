"""Command-line entry point for the v6 bulk runner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lsb_batch import VERSION, generate, read_config, verify_output


def main(argv: list[str] | None = None) -> int:
    """Run generation or verification from command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate deterministic LSB clean/stego pairs")
    parser.add_argument("--config", type=Path, help="YAML run configuration")
    parser.add_argument("--workers", type=int, help="process worker count")
    parser.add_argument("--verify", type=Path, help="verify an existing output folder")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args(argv)
    if args.verify is not None and args.config is not None:
        parser.error("--verify cannot be combined with --config")
    if args.verify is None and args.config is None:
        parser.error("--config is required unless --verify is used")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        if args.verify is not None:
            report = verify_output(args.verify, workers=args.workers)
            print(json.dumps(report, sort_keys=True))
            return 0 if report["failed"] == 0 else 2
        config = read_config(args.config)
        report = generate(config, workers=args.workers)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "completed" else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
