"""Common-cohort component ablation for the TW->RD transfer (stage 1).

One extraction pass, one cohort, one split; feature-group AUCs plus PAIRED
account-bootstrap CIs on AUC differences.

Cohort A = the paper's transfer cohort (membership taken from the existing
feat_* tables, so results are comparable to published numbers).
Cohort B = A restricted to TPP-style distinct-timestamp eligibility
(>=30 IRA / >=50 organic); the A->B funnel is reported.

CPU only, ~10-30 min.
Run:     python -m xplat.analysis.component_ablation
Outputs: IO_RESULTS_DIR/ablation_results.json, ablation_scores.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..config import EPS, FEAT_FILES, RD_WIN, SEQ_FILES, io_path
from ..evaluation import auc_point, volume_match
from ..features import FEATURE_GROUPS, featurize_population, design_matrix

N_BOOT = 1000

CONTRASTS = [
    ("G1G2_phase_sched", "G3_tempo"),
    ("G1_phase", "G3_tempo"),
    ("full_profile", "G1G2_phase_sched"),
    ("G1G2_phase_sched", "G0_volume"),
    ("G1G2G3_plus_tempo", "G1G2_phase_sched"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    rng = np.random.default_rng(0)

    print("loading membership from existing feat tables", flush=True)
    ft = {k: pd.read_csv(io_path(FEAT_FILES[k])) for k in ("tw_t", "tw_o", "rd_t", "rd_o")}
    members = {k: set(v["user"]) for k, v in ft.items()}
    print({k: len(v) for k, v in members.items()}, flush=True)

    print("re-extracting full-schema features (single pass)", flush=True)
    POPS = {
        "tw_t": featurize_population(io_path(SEQ_FILES["tw_t"]), members["tw_t"]),
        "tw_o": featurize_population(io_path(SEQ_FILES["tw_o"]), members["tw_o"]),
        "rd_t": featurize_population(io_path(SEQ_FILES["rd_t"]), members["rd_t"], window=RD_WIN),
        "rd_o": featurize_population(io_path(SEQ_FILES["rd_o"]), members["rd_o"], window=RD_WIN),
    }
    coverage = {k: {"feat_table": len(members[k]), "seq_extracted": len(v)} for k, v in POPS.items()}
    print("coverage:", coverage, flush=True)

    # validation 1: shared columns must agree with the archived pipeline
    validation = {}
    for k in ("tw_t", "tw_o", "rd_t", "rd_o"):
        m = POPS[k].merge(ft[k], on="user", suffixes=("_new", "_old"))
        cors = {}
        for c in ("median_gap_min", "burstiness", "hour_0", "hour_12", "weekday_ratio", "per_day"):
            a, b = f"{c}_new", f"{c}_old"
            if a in m and b in m and m[b].nunique() > 1:
                cors[c] = round(float(np.corrcoef(m[a], m[b])[0, 1]), 4)
            elif b in m:
                cors[c] = f"OLD_DEGENERATE(nunique={m[b].nunique()})"
        validation[k] = cors
    print("validation vs archived features:", validation, flush=True)

    # validation 2: no degenerate columns in the fresh extraction
    for k, df in POPS.items():
        for c in df.columns:
            if c == "user":
                continue
            assert df[c].nunique() > 1 or df[c].abs().max() < EPS or len(df) < 5, \
                f"degenerate column {c} in {k} (nunique={df[c].nunique()})"
        bad = [c for c in ("weekday_ratio", "median_gap_min", "burstiness") if df[c].nunique() <= 1]
        assert not bad, f"required feature degenerate in {k}: {bad}"

    # ---------- train on Twitter, score Reddit ----------
    Xsrc = pd.concat([POPS["tw_t"], POPS["tw_o"]], ignore_index=True)
    ysrc = np.r_[np.ones(len(POPS["tw_t"])), np.zeros(len(POPS["tw_o"]))]
    tgt = pd.concat([POPS["rd_t"], POPS["rd_o"]], ignore_index=True)
    ytgt = np.r_[np.ones(len(POPS["rd_t"])), np.zeros(len(POPS["rd_o"]))]

    scores = {}   # (group, model) -> np.array over tgt rows
    for gname, cols in FEATURE_GROUPS.items():
        sc = StandardScaler().fit(design_matrix(Xsrc, cols))
        Xtr, Xte = sc.transform(design_matrix(Xsrc, cols)), sc.transform(design_matrix(tgt, cols))
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ysrc)
        gbm = GradientBoostingClassifier(random_state=0).fit(Xtr, ysrc)
        scores[(gname, "lr")] = lr.predict_proba(Xte)[:, 1]
        scores[(gname, "gbm")] = gbm.predict_proba(Xte)[:, 1]
        print(f"{gname}: lr {roc_auc_score(ytgt, scores[(gname, 'lr')]):.3f} "
              f"gbm {roc_auc_score(ytgt, scores[(gname, 'gbm')]):.3f}", flush=True)

    # ---------- cohorts and matching ----------
    pos_idx = np.flatnonzero(ytgt == 1)
    neg_idx = np.flatnonzero(ytgt == 0)
    nev = tgt["n_events"].values

    # cohort B: TPP-style distinct-timestamp eligibility
    distinct = tgt["n_distinct_ts"].values
    B_pos = pos_idx[distinct[pos_idx] >= 30]
    B_neg = neg_idx[distinct[neg_idx] >= 50]
    funnel = {
        "cohortA": {"ira": int(len(pos_idx)), "organic": int(len(neg_idx))},
        "cohortB": {"ira": int(len(B_pos)), "organic": int(len(B_neg)),
                    "rule": "distinct ts >=30 IRA / >=50 organic"},
    }
    print("funnel:", funnel, flush=True)

    # rng consumption order: A matching, then B matching, then bootstrap draws
    COHORTS = {
        "A_unmatched": (pos_idx, neg_idx),
        "A_matched": (pos_idx, volume_match(nev, neg_idx, pos_idx, rng)),
        "B_unmatched": (B_pos, B_neg),
        "B_matched": (B_pos, volume_match(nev, B_neg, B_pos, rng)),
    }

    # ---------- paired bootstrap ----------
    results = {"coverage": coverage, "validation": validation, "funnel": funnel,
               "n_boot": N_BOOT, "cohorts": {}}
    for cname, (P, N) in COHORTS.items():
        point = {f"{g}|{m}": round(auc_point(scores[(g, m)], P, N), 4) for (g, m) in scores}
        draws = {k: np.empty(N_BOOT) for k in point}
        for b in range(N_BOOT):
            bp = rng.choice(P, len(P), replace=True)
            bn = rng.choice(N, len(N), replace=True)
            for (g, m), v in scores.items():
                draws[f"{g}|{m}"][b] = auc_point(v, bp, bn)
        cis = {k: [round(float(np.percentile(d, 2.5)), 4),
                   round(float(np.percentile(d, 97.5)), 4)] for k, d in draws.items()}
        deltas = {}
        for ga, gb in CONTRASTS:
            for m in ("lr", "gbm"):
                d = draws[f"{ga}|{m}"] - draws[f"{gb}|{m}"]
                deltas[f"{ga}-minus-{gb}|{m}"] = {
                    "point": round(point[f"{ga}|{m}"] - point[f"{gb}|{m}"], 4),
                    "ci": [round(float(np.percentile(d, 2.5)), 4),
                           round(float(np.percentile(d, 97.5)), 4)],
                    "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                }
        results["cohorts"][cname] = {"auc": point, "ci": cis, "delta": deltas,
                                     "n_pos": int(len(P)), "n_neg": int(len(N))}
        print(f"[{cname}] done (n={len(P)}/{len(N)})", flush=True)

    with open(io_path("ablation_results.json"), "w") as f:
        json.dump(results, f, indent=1)

    sc_df = tgt[["user", "n_events", "n_distinct_ts"]].copy()
    sc_df["label"] = ytgt
    for (g, m), v in scores.items():
        sc_df[f"{g}|{m}"] = v
    sc_df.to_csv(io_path("ablation_scores.csv"), index=False)
    print("wrote ablation_results.json + ablation_scores.csv", flush=True)


if __name__ == "__main__":
    main()
