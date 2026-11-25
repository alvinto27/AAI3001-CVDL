#!/usr/bin/env python3
"""CPU-only freeze/check of the approved RGB-channel TEST diagnostic."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cvdl.channel_analysis import build, estimate_precision, validate_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/verification/rgb-channel-analysis-2026-10-24')
    parser.add_argument('--reading-rule', type=Path, default=ROOT/'runs/.scratch/rgb-channel-plan-2026-10-24/PLAN.md')
    parser.add_argument('--precision-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.precision_only:
        print(json.dumps(estimate_precision(ROOT), indent=2, allow_nan=False))
    elif args.verify:
        recipe, _, _ = validate_panel(ROOT, args.output)
        print(json.dumps(dict(verified=str(args.output), photos=recipe['photos'], tiles=recipe['tiles'])))
    else:
        sha, recipe = build(ROOT, args.output, args.reading_rule)
        print(json.dumps(dict(recipe_sha256=sha, photos=recipe['photos'], tiles=recipe['tiles'],
            images_per_model=recipe['images_per_model'], precision=recipe['precision']), indent=2))


if __name__ == '__main__':
    main()
