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
    parser.add_argument("--character-count", type=int, default=40,
                        help="positive whole-number count of mixed-case ASCII letters")
    parser.add_argument("--payload-mode", choices=("fixed", "range"), default="fixed")
    parser.add_argument("--min-character-count", type=int, default=10,
                        help="inclusive minimum character count in range mode")
    parser.add_argument("--max-character-count", type=int, default=90,
                        help="inclusive maximum character count in range mode")
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
            input_dir=args.input_dir, output_dir=args.output_dir,
            character_count=args.character_count, run_seed=args.run_seed,
            payload_mode=args.payload_mode, min_character_count=args.min_character_count,
            max_character_count=args.max_character_count)
        result = generate(config, progress, cancel)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
