"""Representation x task matrix: the same four representations on the same three tasks.

Representations
  phase        24 UTC-hour proportions + three fixed window masses
  phase_sched  phase + weekday ratio, top-six-hour mass, hour entropy, day density
  action       three action proportions
  policy       action-gap next-token policy (2-layer Mamba, 30 tokens, archived tw_edges)
Profile representations are scored by L2-logistic (lr) and gradient boosting (gbm).

Tasks (rows = training operation A in {IRA, Iran, Venezuela, China}; every
score is frozen after fitting on A's TRAIN split)
  T1 frozen operation-vs-reference ranking   AUC(B_test vs platform-reference_test)
       columns: TW:IRA TW:Iran TW:Venezuela TW:China TW:GRU RD:IRA(matched / unmatched)
  T2 frozen specificity                      AUC(A_test vs B_test) under A's own score
  T3 supervised pairwise classification      fit A_train vs B_train, AUC(A_test vs B_test)
       (policy analogue: log p_A - log p_B with policies fitted on the train splits)

Protocol: one extraction pass; per-population 70/30 account split (seed 0);
Twitter reference = organic 2016 timelines; Reddit accounts are test-only and
scored with source-frozen preprocessing; Reddit controls are volume-matched
with the ablation protocol (0.25-dex bins, 5 per IRA account) and also unmatched.
Account bootstrap (500 draws, test side) gives 95% intervals per cell.

This experiment must run first: it writes the archived account split
(rep_task_scores.csv) and the Twitter policies (rep_task_policy_TW_*.pt) that
the other Table 2 rows reuse.

Run:     python -m xplat.adaptation.rep_task_matrix
Outputs: rep_task_matrix.json, rep_task_scores.csv, rep_task_policy_TW_*.pt
         (all inside IO_RESULTS_DIR)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..config import (CHINA_FILES, FEAT_FILES, IRAN_FILES, N_BOOT, RD_WIN, SEED, SEQ_FILES,
                      TWITTER_TAKEDOWN_DIR, ensure_io_dir, io_path, takedown_paths)
from ..evaluation import bootstrap_auc, volume_match
from ..features import HOURS, account_features, design_matrix
from ..log import log
from ..policy import (SCORE_CAP, account_loglik, device, load_tw_edges, policy_artifact, tokenize,
                      train_policy, windows)
from ..sequences import load_accounts, parse_takedown_actions

MINEV_BIG, MINEV_SMALL = 50, 30
TEST_FRAC = 0.30
LOGCOLS = {"per_day"}          # this experiment only log-transforms volume

REPS = {
    "phase": HOURS + ["moscow_mass", "night_mass", "evening_mass"],
    "phase_sched": HOURS + ["moscow_mass", "night_mass", "evening_mass", "weekday_ratio", "top6h_mass",
                            "hour_entropy", "day_density"],
    "action": ["frac_B", "frac_R", "frac_A"],
}
TRAIN_OPS = ["TW:IRA", "TW:Iran", "TW:Venezuela", "TW:China"]
TW_TEST = ["TW:IRA", "TW:Iran", "TW:Venezuela", "TW:China", "TW:GRU"]
VENEZUELA_FILES = [f"venezuela_201901_1_tweets_csv_unhashed_{i}.csv" for i in (1, 2, 3)]
GRU_FILE = "2020_12/GRU_202012/GRU_202012_tweets_csv_unhashed.csv"   # relative to TWITTER_TAKEDOWN_DIR


def load_populations():
    log("loading populations")
    pops = {
        "TW:IRA": load_accounts(io_path(SEQ_FILES["tw_t"]), MINEV_BIG),
        "TW:organic": load_accounts(io_path(SEQ_FILES["tw_o"]), MINEV_BIG),
    }
    log(f"IRA {len(pops['TW:IRA'])} organic {len(pops['TW:organic'])}")
    pops["TW:Iran"] = parse_takedown_actions(takedown_paths(IRAN_FILES), MINEV_BIG, log=log)
    log(f"Iran {len(pops['TW:Iran'])}")
    pops["TW:Venezuela"] = parse_takedown_actions(takedown_paths(VENEZUELA_FILES), MINEV_BIG, log=log)
    log(f"Venezuela {len(pops['TW:Venezuela'])}")
    pops["TW:China"] = parse_takedown_actions(takedown_paths(CHINA_FILES), MINEV_BIG, log=log)
    log(f"China {len(pops['TW:China'])}")
    pops["TW:GRU"] = parse_takedown_actions([os.path.join(TWITTER_TAKEDOWN_DIR, GRU_FILE)], MINEV_SMALL, log=log)
    log(f"GRU {len(pops['TW:GRU'])}")
    rd_t_members = set(pd.read_csv(io_path(FEAT_FILES["rd_t"]))["user"].astype(str))
    rd_o_members = set(pd.read_csv(io_path(FEAT_FILES["rd_o"]))["user"].astype(str))
    pops["RD:IRA"] = load_accounts(io_path(SEQ_FILES["rd_t"]), 10, rd_t_members, RD_WIN)
    pops["RD:organic"] = load_accounts(io_path(SEQ_FILES["rd_o"]), 10, rd_o_members, RD_WIN)
    log(f"RD IRA {len(pops['RD:IRA'])} RD organic {len(pops['RD:organic'])}")
    return pops


def make_splits(pops, rng):
    """70/30 by account for every Twitter population; Reddit and GRU are test-only."""
    split = {}
    for name, pop in pops.items():
        keys = np.array(sorted(pop))
        if name.startswith("RD:"):
            split[name] = {"train": np.array([], dtype=object), "test": keys}
            continue
        perm = rng.permutation(len(keys))
        n_test = int(round(TEST_FRAC * len(keys)))
        split[name] = {"test": keys[perm[:n_test]], "train": keys[perm[n_test:]]}
    split["TW:GRU"] = {"train": np.array([], dtype=object), "test": np.array(sorted(pops["TW:GRU"]))}
    log("splits: " + ", ".join(f"{k} {len(v['train'])}/{len(v['test'])}" for k, v in split.items()))
    return split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    ensure_io_dir()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = device()
    tw_edges = load_tw_edges()

    pops = load_populations()
    split = make_splits(pops, rng)

    # one account table: features for every account, with pop/split labels
    log("featurizing")
    rows = []
    for name, pop in pops.items():
        tr = set(split[name]["train"])
        for u, (ts, ac) in pop.items():
            rows.append({"pop": name, "user": u, "split": "train" if u in tr else "test", **account_features(ts, ac)})
    acc = pd.DataFrame(rows).reset_index(drop=True)
    acc["row"] = np.arange(len(acc))
    idx = {(p, s): acc.index[(acc["pop"] == p) & (acc["split"] == s)].values for p in pops for s in ("train", "test")}

    # Reddit matched controls: ablation protocol (0.25-dex log-volume bins, 5 per IRA account)
    rd_pos = idx[("RD:IRA", "test")]
    rd_neg_all = idx[("RD:organic", "test")]
    rd_neg_matched = volume_match(acc["n_events"].values, rd_neg_all, rd_pos, rng)
    log(f"RD cohort: {len(rd_pos)} IRA, {len(rd_neg_all)} organic, {len(rd_neg_matched)} matched")

    # ------------------------------------------------------------------ scores
    scores = {}   # (rep, clf, A, B_or_None) -> np.array over acc rows (NaN where not scored)

    def fit_profile(rep, clf, pos_idx, neg_idx):
        cols = REPS[rep]
        Xtr = design_matrix(acc.iloc[np.r_[pos_idx, neg_idx]], cols, LOGCOLS)
        ytr = np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))]
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        if clf == "lr":
            model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        else:
            model = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
        return model.predict_proba(sc.transform(design_matrix(acc, cols, LOGCOLS)))[:, 1]

    log("profile scorers: frozen A-vs-reference")
    org_tr = idx[("TW:organic", "train")]
    for rep in REPS:
        for clf in ("lr", "gbm"):
            for A in TRAIN_OPS:
                scores[(rep, clf, A, None)] = fit_profile(rep, clf, idx[(A, "train")], org_tr)
            log(f"  {rep}/{clf} done")
    log("profile scorers: supervised pairwise A-vs-B")
    for rep in REPS:
        for clf in ("lr", "gbm"):
            for i, A in enumerate(TRAIN_OPS):
                for B in TRAIN_OPS[i + 1:]:
                    s = fit_profile(rep, clf, idx[(A, "train")], idx[(B, "train")])
                    scores[(rep, clf, A, B)] = s
                    scores[(rep, clf, B, A)] = 1 - s
            log(f"  {rep}/{clf} done")

    log("policies")
    policy = {}
    for A in TRAIN_OPS + ["TW:organic"]:
        policy[A], _ = train_policy(windows(pops[A], split[A]["train"], tw_edges), A, dev=dev)
        torch.save(policy[A].state_dict(), policy_artifact(A))
    log("scoring every account under every policy")
    pol_names = list(policy)
    LL = np.full((len(acc), len(policy)), np.nan)
    for r in range(len(acc)):
        ts, ac = pops[acc.at[r, "pop"]][acc.at[r, "user"]]
        if len(ts) > SCORE_CAP:
            ts, ac = ts[:SCORE_CAP], ac[:SCORE_CAP]
        t = tokenize(ts, ac, tw_edges)
        for j, name in enumerate(pol_names):
            LL[r, j] = account_loglik(policy[name], t, dev=dev)
        if r % 1000 == 0:
            log(f"  scored {r}/{len(acc)}")
    for j, name in enumerate(pol_names):
        acc[f"ll_{name}"] = LL[:, j]
    j_org = pol_names.index("TW:organic")
    for A in TRAIN_OPS:
        scores[("policy", "llr", A, None)] = LL[:, pol_names.index(A)] - LL[:, j_org]
        for B in TRAIN_OPS:
            if A != B:
                scores[("policy", "llr", A, B)] = LL[:, pol_names.index(A)] - LL[:, pol_names.index(B)]

    # ------------------------------------------------------------------ evaluation
    def auc_ci(score, pos, neg):
        # Shared-generator bootstrap (same rng as the splits and matching); non-finite scores dropped.
        pos = pos[np.isfinite(score[pos])]
        neg = neg[np.isfinite(score[neg])]
        point, lo, hi = bootstrap_auc(score[pos], score[neg], n_boot=N_BOOT, rng=rng)
        return {"auc": round(point, 4), "ci": [round(lo, 4), round(hi, 4)],
                "n_pos": int(len(pos)), "n_neg": int(len(neg))}

    log("evaluating")
    org_te = idx[("TW:organic", "test")]
    res = {"protocol": {"test_frac": TEST_FRAC, "seed": SEED, "n_boot": N_BOOT, "minev_big": MINEV_BIG,
                        "minev_small_gru": MINEV_SMALL, "score_cap_events": SCORE_CAP,
                        "tw_edges_min": tw_edges.tolist(), "rd_window": list(RD_WIN),
                        "reps": {k: v for k, v in REPS.items()},
                        "policy": "2-layer Mamba d=128, 30 tokens, WIN 512, <=20 windows/account, 3 epochs, AdamW 1e-3",
                        "splits": {k: {"train": int(len(v["train"])), "test": int(len(v["test"]))} for k, v in split.items()},
                        "rd_cohort": {"ira": int(len(rd_pos)), "organic": int(len(rd_neg_all)),
                                      "matched": int(len(rd_neg_matched))}},
           "T1_frozen_vs_reference": {}, "T2_frozen_specificity": {}, "T3_supervised_pairwise": {}}
    scorers = [(rep, clf) for rep in REPS for clf in ("lr", "gbm")] + [("policy", "llr")]
    for rep, clf in scorers:
        key = f"{rep}|{clf}"
        t1, t2, t3 = {}, {}, {}
        for A in TRAIN_OPS:
            s = scores[(rep, clf, A, None)]
            t1[A] = {B: auc_ci(s, idx[(B, "test")], org_te) for B in TW_TEST}
            t1[A]["RD:IRA|matched"] = auc_ci(s, rd_pos, rd_neg_matched)
            t1[A]["RD:IRA|unmatched"] = auc_ci(s, rd_pos, rd_neg_all)
            t2[A] = {B: auc_ci(s, idx[(A, "test")], idx[(B, "test")]) for B in TRAIN_OPS if B != A}
            t2[A]["TW:GRU"] = auc_ci(s, idx[(A, "test")], idx[("TW:GRU", "test")])
            t3[A] = {B: auc_ci(scores[(rep, clf, A, B)], idx[(A, "test")], idx[(B, "test")])
                     for B in TRAIN_OPS if B != A}
        res["T1_frozen_vs_reference"][key] = t1
        res["T2_frozen_specificity"][key] = t2
        res["T3_supervised_pairwise"][key] = t3
        log(f"  {key}: T1 diag " + " ".join(f"{A.split(':')[1]} {t1[A][A]['auc']:.3f}" for A in TRAIN_OPS)
            + " | RD " + " ".join(f"{A.split(':')[1]} {t1[A]['RD:IRA|matched']['auc']:.3f}" for A in TRAIN_OPS))

    json.dump(res, open(io_path("rep_task_matrix.json"), "w"), indent=1)
    keep = acc[["pop", "user", "split", "n_events"] + [c for c in acc.columns if c.startswith("ll_")]].copy()
    for (rep, clf, A, B), s in scores.items():
        keep[f"{rep}|{clf}|{A}|{B or 'ref'}"] = s
    keep.to_csv(io_path("rep_task_scores.csv"), index=False)
    log("DONE -> rep_task_matrix.json, rep_task_scores.csv")


if __name__ == "__main__":
    main()
