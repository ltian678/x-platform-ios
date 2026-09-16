"""Platform-adapted policy transfer (TW->RD): archived Twitter policies, Reddit gap deciles.

The rep-task matrix scores Reddit accounts under the Twitter-fitted policies
with the Twitter gap cutpoints frozen ("source-frozen preprocessing").  This
experiment keeps the archived policies (rep_task_policy_TW_*.pt from
rep_task_matrix) and changes only the quantizer: Reddit gaps are mapped to
the empirical deciles of the organic-Reddit reference, so token k means "k-th
gap decile of the platform" on both sides.  This is the adaptation step of
the method section applied to the policy score.  A tw_edges pass reproduces
the archived cells as a check.

Cohort: RD:IRA (96) vs the 168 volume-matched organic controls (rng replay of
rep_task_matrix) and vs all 643.  Scoring truncates accounts to 5,000 events
as in the archive.  500-draw account bootstrap intervals.

Requires: rep_task_scores.csv and rep_task_policy_TW_*.pt from
``xplat.adaptation.rep_task_matrix``, china_policy_edges.json.

Run:    python -m xplat.adaptation.policy_adapted_transfer
Output: policy_adapted_transfer.json (inside IO_RESULTS_DIR)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..cohorts import load_archived_splits, load_reddit_cohort, matched_reddit_controls
from ..config import ensure_io_dir, io_path
from ..evaluation import auc_ci
from ..log import log
from ..policy import SCORE_CAP, account_loglik, decile_edges, device, load_policy, load_tw_edges, tokenize
from ..sequences import gaps_min

POLICIES = ("TW:IRA", "TW:Iran", "TW:Venezuela", "TW:China", "TW:organic")
OPS = ("TW:IRA", "TW:Iran", "TW:Venezuela", "TW:China")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    ensure_io_dir()
    dev = device()

    # ---------------------------------------------------------------- cohort and matched controls (rng replay)
    log("loading Reddit cohort")
    rd_ira, rd_org = load_reddit_cohort()
    arch = load_archived_splits()
    rd_pos = sorted(rd_ira)
    rd_neg = sorted(rd_org)
    nev = {**{u: len(v[0]) for u, v in rd_ira.items()}, **{u: len(v[0]) for u, v in rd_org.items()}}
    rd_neg_matched = matched_reddit_controls(arch, rd_pos, rd_neg, nev)
    log(f"RD cohort: {len(rd_pos)} IRA, {len(rd_neg)} organic, {len(rd_neg_matched)} matched (archive: 168)")

    # ---------------------------------------------------------------- quantizers
    tw_edges = load_tw_edges()
    rd_gaps = np.concatenate([gaps_min(rd_org[u][0])[1:] for u in rd_neg])
    rd_edges = decile_edges(rd_gaps)
    log(f"tw_edges {np.round(tw_edges, 2).tolist()}  rd_edges {np.round(rd_edges, 2).tolist()}")

    # ---------------------------------------------------------------- archived policies
    pol = {A: load_policy(A, dev=dev) for A in POLICIES}
    log("archived policies loaded")

    results = {"rd_edges_min": rd_edges.tolist(), "tw_edges_min": tw_edges.tolist(),
               "n": [len(rd_pos), len(rd_neg), len(rd_neg_matched)]}
    users = rd_pos + rd_neg
    pops = [rd_ira] * len(rd_pos) + [rd_org] * len(rd_neg)
    for qname, edges in (("tw_edges_frozen", tw_edges), ("rd_edges_adapted", rd_edges)):
        LL = {A: np.zeros(len(users)) for A in pol}
        for i, (u, P) in enumerate(zip(users, pops)):
            ts, ac = P[u]
            if len(ts) > SCORE_CAP:
                ts, ac = ts[:SCORE_CAP], ac[:SCORE_CAP]
            tok = tokenize(ts, ac, edges)
            for A in pol:
                LL[A][i] = account_loglik(pol[A], tok, dev=dev)
        res = {}
        for A in OPS:
            llr = LL[A] - LL["TW:organic"]
            s = dict(zip(users, llr))
            res[A] = {"matched": auc_ci([s[u] for u in rd_pos], [s[u] for u in rd_neg_matched]),
                      "unmatched": auc_ci([s[u] for u in rd_pos], [s[u] for u in rd_neg])}
            log(f"{qname} policy {A}: matched {res[A]['matched']} unmatched {res[A]['unmatched']}")
        results[qname] = res
    json.dump(results, open(io_path("policy_adapted_transfer.json"), "w"), indent=1)
    log("DONE -> policy_adapted_transfer.json")


if __name__ == "__main__":
    main()
