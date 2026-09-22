"""Headless command-line entry point."""
import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from lsb_batch import Config, generate, read_config
from lsb_core import VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description="LSB Dataset Generator — strict RGB PNG, no dataset splitting")
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--input", dest="input_dir", help="source image folder")
    parser.add_argument("--output", dest="output_dir", help="new or empty run output folder")
    parser.add_argument("--payload-rate", type=float, default=0.4, help="fixed payload rate, or the range default in YAML")
    parser.add_argument("--run-seed", type=int, default=42, help="one seed for the whole run")
    args = parser.parse_args()
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())

    def progress(event: dict) -> None:
        print(event["message"], flush=True)

    try:
        if not args.config and (not args.input_dir or not args.output_dir):
            parser.error("Provide --config or both --input and --output.")
        config = read_config(args.config) if args.config else Config(
            args.input_dir, args.output_dir, args.payload_rate, args.run_seed)
        result = generate(config, progress, cancel)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
