"""Command-line entry point: ``xplat list`` and ``xplat run <experiment> [args]``.

Every experiment is an importable module with ``main(argv=None)``; the same
module can be run directly with ``python -m xplat.<group>.<name>``.
"""
from __future__ import annotations

import importlib
import sys

EXPERIMENTS = {
    # adaptation (Table 2)
    "rep-task-matrix": "xplat.adaptation.rep_task_matrix",
    "tw-inplatform-profile": "xplat.adaptation.tw_inplatform_profile",
    "rd-inplatform": "xplat.adaptation.rd_inplatform",
    "rd-policy-inplatform": "xplat.adaptation.rd_policy_inplatform",
    "policy-adapted-transfer": "xplat.adaptation.policy_adapted_transfer",
    "xplat-seq-baselines": "xplat.adaptation.xplat_seq_baselines",
    "alizadeh-reimpl": "xplat.adaptation.alizadeh_reimpl",
    # analysis
    "component-ablation": "xplat.analysis.component_ablation",
    "error-analysis": "xplat.analysis.error_analysis",
    "text-baseline": "xplat.analysis.text_baseline",
    "text-baseline-v2": "xplat.analysis.text_baseline_v2",
    "audience-bootstrap": "xplat.analysis.audience_bootstrap",
    "china-within-account": "xplat.analysis.china_within_account",
    "tg-org-bridge": "xplat.analysis.tg_org_bridge",
    # temporal point process (Figure 4)
    "tpp": "xplat.tpp.pipeline",
    # public exports (no fitting)
    "public-tpp-figure": "xplat.public.tpp_figure",
    "public-case-summaries": "xplat.public.case_summaries",
    "public-tex": "xplat.public.tex",
    # synthetic data for smoke runs
    "make-synthetic-data": "xplat.synthetic",
}

USAGE = """usage: xplat list
       xplat run <experiment> [experiment args...]
       xplat <experiment> [experiment args...]

Experiments (also runnable as python -m <module>):
"""


def _usage() -> str:
    width = max(len(k) for k in EXPERIMENTS)
    return USAGE + "\n".join(f"  {k.ljust(width)}  {v}" for k, v in EXPERIMENTS.items()) + "\n"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage(), end="")
        return 0
    if argv[0] == "list":
        for k, v in EXPERIMENTS.items():
            print(f"{k}\t{v}")
        return 0
    if argv[0] == "run":
        argv = argv[1:]
    if not argv or argv[0] not in EXPERIMENTS:
        print(_usage(), end="", file=sys.stderr)
        print(f"error: unknown experiment {argv[0] if argv else ''!r}", file=sys.stderr)
        return 2
    module = importlib.import_module(EXPERIMENTS[argv[0]])
    rc = module.main(argv[1:])
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
