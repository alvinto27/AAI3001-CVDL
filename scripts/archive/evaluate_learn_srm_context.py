#!/usr/bin/env python3
"""Freeze, score or report learn-SRM on original #11 bytes in a new sibling."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from cvdl import learn_srm_context as context
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','score','report','check'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--panel',type=Path,default=ROOT/'runs/verification/rgb-channel-analysis-2026-10-24')
    parser.add_argument('--evaluation',type=Path)
    args=parser.parse_args()
    if args.command=='freeze':
        if args.evaluation is None:parser.error('freeze requires --evaluation')
        context.freeze(ROOT,args.panel,args.evaluation,args.output)
    elif args.command=='check':context.checked(ROOT,args.output)
    else:getattr(context,args.command)(ROOT,args.output)


if __name__=='__main__':
    from scripts.pc_safety import guarded,own_process_group
    own_process_group();guarded(main)
