"""Headless command-line entry point."""
import argparse
import json
import sys
import threading
import signal
from pathlib import Path

from lsb_batch import Config, generate, read_config, verify_run
from lsb_core import VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description="LSB Dataset Generator — strict RGB PNG, no dataset splitting")
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--verify", type=Path, metavar="RUN_FOLDER")
    parser.add_argument("--input", dest="input_dir")
    parser.add_argument("--output", dest="output_dir")
    parser.add_argument("--payload-rate", type=float, default=0.4)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--source-dataset", default="")
    parser.add_argument("--error-policy", choices=("skip", "stop"), default="skip")
    args = parser.parse_args()
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    def progress(event: dict) -> None:
        print(event["message"], flush=True)
    try:
        if args.verify:
            result = verify_run(args.verify, progress, cancel)
        else:
            if not args.config and (not args.input_dir or not args.output_dir):
                parser.error("Provide --config or both --input and --output.")
            config = read_config(args.config) if args.config else Config(
                args.input_dir, args.output_dir, args.payload_rate, args.run_seed,
                args.source_dataset, error_policy=args.error_policy)
            result = generate(config, progress, cancel)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in ("completed", "completed_with_rejections", "verified") else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
