"""Command-line entry point for cover preparation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lsb_prepare import VERSION, PrepareConfig, prepare


def main(argv: list[str] | None = None) -> int:
    """Prepare source photos into fixed-size RGB PNG tiles."""
    parser = argparse.ArgumentParser(description="Prepare RGB PNG tiles for LSB generation")
    parser.add_argument("--input", type=Path, required=True, help="source photo folder")
    parser.add_argument("--output", type=Path, required=True, help="tile output folder")
    parser.add_argument("--tile", type=int, default=512, help="square tile size")
    parser.add_argument("--max-tiles-per-image", type=int, help="random tile limit per source")
    parser.add_argument("--seed", type=int, default=0, help="tile selection seed")
    parser.add_argument("--workers", type=int, help="process worker count")
    parser.add_argument("--png-compress-level", type=int, default=1, choices=range(10))
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        config = PrepareConfig(
            input=str(args.input),
            output=str(args.output),
            tile=args.tile,
            max_tiles_per_image=args.max_tiles_per_image,
            seed=args.seed,
            png_compress_level=args.png_compress_level,
        )
        result = prepare(config, workers=args.workers)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "completed" else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
