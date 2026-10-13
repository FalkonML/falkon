"""Benchmark the BLOKS system (falkon ``balkon`` branch) on TabArena and compare to the leaderboard.

BLOKS is integrated as a *system* (``tabarena.systems.bloks``): it receives the raw training data and
the per-split budget and does its own preprocessing, 8-fold bagged inner CV, Optuna HPO, calibration
and greedy ensembling (see ``tabarena/systems/bloks/_pipeline.py``). Its three presets map onto
TabArena's regimes: ``BLOKS_c1`` (tuned + ensemble), ``BLOKS_c2`` (tuned) and ``BLOKS_c3`` (default).

Install (torch first: falkon compiles an extension against it)::

    pip install torch
    uv pip install --prerelease=allow -e "./packages/tabarena[benchmark,bloks]"

Examples::

    # quick check: small datasets of TabArena-Lite, 5 minutes per split, in-process
    python run_bloks_system.py --subset small lite --time-limit 300

    # TabArena-Lite (first split of all 51 datasets) with the default 1 h budget
    python run_bloks_system.py --subset lite
"""

from __future__ import annotations

import argparse
from ast import parse
import datetime
from pathlib import Path
import sys

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from falkon.benchmarks.tabarena.hpo import gen_bloks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", nargs="*", default=["small", "lite"], help="TabArena task subset expressions")
    parser.add_argument("--datasets", nargs="*", default=None, help="Restrict to these dataset names")
    parser.add_argument("--time-limit", type=int, default=None, help="Fit budget per split in seconds (default 1h)")
    parser.add_argument("--num-gpus", type=int, default=None, help="GPUs per fit (default: auto-detect)")
    parser.add_argument("--num-cpus", type=int, default=None, help="CPUs per fit (default: auto-detect)")
    parser.add_argument("--run-name", default=None, required=False)
    parser.add_argument("--no-compare", action="store_true", help="Only run the jobs, skip the leaderboard")
    parser.add_argument("--no-debug-mode", action="store_true", help="Run jobs in subprocesses instead of in-process")
    args = parser.parse_args()

    here = Path(__file__).parent
    run_name = args.run_name or f"bloks_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # 1: one no-validation system experiment per BLOKS preset (c1: tuned + ensemble, c2: tuned, c3: default).
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(gen_bloks, 0)],
        system_experiments=True,
    ).build_experiments(
        time_limit=args.time_limit, 
        num_gpus=args.num_gpus, 
        num_cpus=args.num_cpus
    )

    # 2: run every experiment on every split of the subset and register the results.
    build_kwargs = {"dataset_names": args.datasets} if args.datasets else None
    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=args.subset,
        build_kwargs=build_kwargs,
        debug_mode=not args.no_debug_mode,
    )

    if args.no_compare:
        sys.exit(0)
    
    # 3: leaderboard against the TabArena baselines, restricted to the tasks BLOKS ran.
    leaderboard = context.compare(output_dir=eval_dir, subset=args.subset)
    print("\n=== TabArena leaderboard ===")
    print(context.leaderboard_to_website_format(leaderboard=leaderboard).to_markdown(index=False))
    print(f"\nResults cached in {results_dir}; figures in {eval_dir}")