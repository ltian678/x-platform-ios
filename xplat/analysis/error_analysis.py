"""Detector error analysis on the common-cohort transfer scores.

Uses the per-account scores saved by component_ablation (ablation_scores.csv)
joined with re-extracted full-schema features (same window and code path).
For the primary scorers, reports:
  - recall at evaluation-domain 5% and 1% FPR (threshold = organic score
    quantile, the paper's stated threshold convention)
  - false-positive profile: flagged vs unflagged organics (medians of volume,
    schedule concentration, tempo, action mix)
  - false-negative profile: missed vs detected IRA accounts

Run:    python -m xplat.analysis.error_analysis
Output: IO_RESULTS_DIR/error_analysis.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..config import RD_WIN, SEQ_FILES, io_path
from ..features import account_features
from ..sequences import read_sequence_frame

SCORERS = ["G1G2_phase_sched|lr", "G1G2_phase_sched|gbm", "G1_phase|gbm", "full_profile|lr"]
PROFILE = ["n_events", "per_day", "day_density", "median_gap_min", "burstiness",
           "top6h_mass", "hour_entropy", "weekday_ratio", "frac_B", "frac_R"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)

    rows = []
    for key in ("rd_t", "rd_o"):
        # the archived loader keeps non-positive timestamps (only unparseable rows dropped)
        seq = read_sequence_frame(io_path(SEQ_FILES[key]), positive_only=False)
        for u, g in seq.groupby("user"):
            g = g.sort_values("ts")
            ts = g["ts"].values.astype(np.int64)
            ac = g["action"].values.astype(np.int64)
            m = (ts >= RD_WIN[0]) & (ts <= RD_WIN[1])
            ts, ac = ts[m], ac[m]
            if len(ts) < 10:
                continue
            rows.append({"user": u, **account_features(ts, ac)})
    F = pd.DataFrame(rows)

    S = pd.read_csv(io_path("ablation_scores.csv"))
    # the scores table already carries n_events and n_distinct_ts
    D = S.merge(F.drop(columns=["n_events", "n_distinct_ts"]), on="user", how="inner")
    print(f"joined {len(D)} accounts ({int(D.label.sum())} IRA)", flush=True)

    res = {"n": {"ira": int(D.label.sum()), "organic": int((1 - D.label).sum())}, "scorers": {}}
    org = D[D.label == 0]
    tro = D[D.label == 1]
    for sc in SCORERS:
        entry = {}
        for fpr in (0.05, 0.01):
            thr = float(np.quantile(org[sc], 1 - fpr))
            flagged_o = org[org[sc] >= thr]
            missed_t = tro[tro[sc] < thr]
            entry[f"recall_at_{int(fpr * 100)}pct_fpr"] = round(float((tro[sc] >= thr).mean()), 3)
            if fpr == 0.05:
                entry["threshold"] = round(thr, 4)
                entry["n_flagged_organic"] = int(len(flagged_o))
                entry["fp_profile"] = {c: [round(float(flagged_o[c].median()), 3),
                                           round(float(org[org[sc] < thr][c].median()), 3)]
                                       for c in PROFILE}
                entry["fn_profile"] = {c: [round(float(missed_t[c].median()), 3),
                                           round(float(tro[tro[sc] >= thr][c].median()), 3)]
                                       for c in PROFILE}
                entry["n_missed_ira"] = int(len(missed_t))
        res["scorers"][sc] = entry
        print(sc, {k: v for k, v in entry.items() if "profile" not in k}, flush=True)

    json.dump(res, open(io_path("error_analysis.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
