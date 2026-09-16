"""End-to-end smoke run of the framework on synthetic data (no raw archives).

Generates the synthetic inputs, points the configuration at them, and runs the
experiments that can complete on a CPU, in dependency order, reporting which
succeed.  With ``--fake-mamba`` a CPU stand-in replaces ``mamba_ssm`` so the
learned-policy rows and the point-process pipeline run as well (their numbers
are meaningless; this checks the code paths, not the science).

    python -m xplat.smoke --work /tmp/xplat_smoke [--fake-mamba] [--only a,b] [--skip a,b]
"""
from __future__ import annotations

import argparse
import importlib
import os
import time
import traceback

from . import config
from .cli import EXPERIMENTS

CPU_ORDER = [
    ("component-ablation", []),
    ("error-analysis", []),
    ("audience-bootstrap", []),
    ("china-within-account", []),
    ("tg-org-bridge", []),
    ("tw-inplatform-profile", []),          # uses the stand-in archived split
]
SEQ_ORDER = [
    ("rep-task-matrix", []),                # rewrites the archived split and policies
    ("tw-inplatform-profile", []),
    ("policy-adapted-transfer", []),
    ("rd-policy-inplatform", []),
    ("xplat-seq-baselines", ["--models", "BLOC|tfidf_lr,ActionLSTM"]),
    ("tpp", ["--variant", "v5"]),
]
NOT_RUNNABLE = {
    "text-baseline": "needs organic timeline .json.bz2 files, Reddit recrawl folders, and a transformer download",
    "text-baseline-v2": "needs the text-baseline caches and a transformer download",
    "alizadeh-reimpl": "needs organic timeline .json.bz2 files and Reddit recrawl folders",
    "rd-inplatform": "needs Reddit recrawl folders and the MPNet embedding cache",
    "public-tpp-figure": "needs the private archived fits (XPLAT_ARCHIVE_DIR)",
    "public-case-summaries": "needs the private archived fits (XPLAT_ARCHIVE_DIR)",
}


def run_one(name, extra):
    module = importlib.import_module(EXPERIMENTS[name])
    t0 = time.time()
    try:
        module.main(list(extra))
        return True, time.time() - t0, ""
    except SystemExit as e:
        return (e.code in (0, None)), time.time() - t0, f"SystemExit({e.code})"
    except Exception:
        return False, time.time() - t0, traceback.format_exc(limit=3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="scratch directory for synthetic data and outputs")
    ap.add_argument("--fake-mamba", action="store_true", help="also run policy / point-process experiments with a CPU stand-in")
    ap.add_argument("--only", default="", help="comma list of experiment names to run")
    ap.add_argument("--skip", default="", help="comma list of experiment names to skip")
    ap.add_argument("--scale", type=float, default=1.0, help="synthetic account-count multiplier")
    ap.add_argument("--no-generate", action="store_true", help="reuse synthetic data already in --work")
    args = ap.parse_args(argv)

    work = os.path.abspath(args.work)
    io_dir, td_dir = os.path.join(work, "work"), os.path.join(work, "takedown")
    if not args.no_generate:
        from .synthetic import main as make
        make(["--out-dir", io_dir, "--takedown-dir", td_dir, "--scale", str(args.scale)])
    config.configure(io_dir=io_dir, takedown_dir=td_dir)
    os.environ["IO_RESULTS_DIR"] = io_dir
    os.environ["TWITTER_TAKEDOWN_DIR"] = td_dir

    plan = list(CPU_ORDER)
    if args.fake_mamba:
        from .fake_mamba import install
        print("mamba_ssm stand-in installed" if install() else "real mamba_ssm found; using it", flush=True)
        plan += SEQ_ORDER
    only = {s for s in args.only.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    plan = [(n, a) for n, a in plan if (not only or n in only) and n not in skip]

    report = []
    for name, extra in plan:
        print(f"\n===== {name} {' '.join(extra)} =====", flush=True)
        ok, secs, err = run_one(name, extra)
        report.append((name, ok, secs, err))
        if not ok:
            print(err, flush=True)
    print("\n===== smoke summary =====")
    width = max(len(n) for n, *_ in report) if report else 10
    for name, ok, secs, err in report:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {secs:7.1f}s")
    for name, why in NOT_RUNNABLE.items():
        if not only or name in only:
            print(f"SKIP  {name.ljust(width)}  {why}")
    failed = [n for n, ok, *_ in report if not ok]
    print(f"\n{len(report) - len(failed)} passed, {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
