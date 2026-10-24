#!/usr/bin/env python3
"""Score the frozen channel panel, or make its predeclared CPU report. No fitting."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cvdl.channel_analysis import MODEL_SPECS, report, score, validate_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scoring = sub.add_parser('score')
    scoring.add_argument('--model', choices=tuple(MODEL_SPECS), required=True)
    scoring.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    scoring.add_argument('--batch-size', type=int, default=32)
    scoring.add_argument('--workers', type=int, default=4)
    for command in (scoring, sub.add_parser('report'), sub.add_parser('check')):
        command.add_argument('--panel', type=Path, default=ROOT/'runs/verification/rgb-channel-analysis-2026-10-24')
    args = parser.parse_args()
    if args.command == 'score':
        from scripts.pc_safety import own_process_group
        own_process_group()
        score(ROOT, args.panel, args.model, device=args.device, batch_size=args.batch_size, workers=args.workers)
    elif args.command == 'report':
        report(ROOT, args.panel)
    else:
        recipe, _, _ = validate_panel(ROOT, args.panel)
        print('VERIFIED: '+str(args.panel)+'; '+str(recipe['images_per_model'])+' images/model')


if __name__ == '__main__':
    from scripts.pc_safety import guarded, own_process_group
    own_process_group()  # Isolate check/report failures as well as GPU workers.
    guarded(main)
