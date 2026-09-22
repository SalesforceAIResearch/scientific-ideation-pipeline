#!/usr/bin/env python3
"""Subsample the in-loop validation parquet to a small, fixed set of prompts.

WHY: verl scores the ENTIRE val set x val_kwargs.n at every TEST_FREQ (and at
val_before_train). With the full 100-prompt val set that is a burst of thousands
of judge calls offered near-simultaneously across the reward workers -- ~16x the
SciJudge server's max_num_seqs ceiling -- which builds a deep queue. Under the
tightened per-call timeout that queue makes healthy-but-queued calls time out and
impute, so the val metric gets noisy (this is what run 26070's step-0 val already
showed). Scoring a SMALLER val set fully is more faithful than scoring the full
set unreliably; keep the full val.parquet for a final/offline eval.

Val metrics on a fixed ~64-prompt subsample are still perfectly meaningful for
tracking training progress. The subsample is DETERMINISTIC (seeded) so every run
evaluates on the same prompts.

Usage:
    python data_prep/make_val_small.py \
        --src /path/to/grpo_ideation_v0/val.parquet \
        --out /path/to/grpo_ideation_v0/val_small.parquet \
        --n 64 --seed 0
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source val.parquet")
    ap.add_argument("--out", required=True, help="destination val_small.parquet")
    ap.add_argument("--n", type=int, default=64, help="number of prompts to keep (default 64)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the sample (default 0)")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    import os
    if os.path.exists(args.out) and not args.force:
        print(f"[make_val_small] {args.out} already exists; pass --force to overwrite. Skipping.")
        return 0

    df = pd.read_parquet(args.src)
    total = len(df)
    if total == 0:
        print(f"[make_val_small] ERROR: {args.src} is empty.", file=sys.stderr)
        return 1

    n = min(args.n, total)
    # Deterministic sample; sort the chosen indices so row order is stable/readable.
    small = df.sample(n=n, random_state=args.seed).sort_index().reset_index(drop=True)
    small.to_parquet(args.out, index=False)
    print(f"[make_val_small] wrote {n}/{total} prompts -> {args.out} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
