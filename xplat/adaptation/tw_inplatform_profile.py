"""Within-Twitter (TW->TW) production-signature profile rows for Table 2.

Features and classifiers as in the component ablation (full schema: phase,
schedule, tempo, action mix, volume).  Split: the archived rep-task 70/30
account split (rep_task_scores.csv), so the rows share the held-out accounts
of the policy / BLOC / phase rows: fit on TW:IRA train (2,042) vs TW:organic
train (1,220); AUC on TW:IRA test (875) vs TW:organic test (523), with
500-draw account-bootstrap intervals.  LR and GBM for every group.

Requires: rep_task_scores.csv from ``xplat.adaptation.rep_task_matrix``.

Run:    python -m xplat.adaptation.tw_inplatform_profile
Output: tw_inplatform_profile.json (inside IO_RESULTS_DIR)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..cohorts import archived_split_users, load_archived_splits
from ..config import SEQ_FILES, ensure_io_dir, io_path
from ..evaluation import auc_ci
from ..features import FEATURE_GROUPS, LOGCOLS, account_features, design_matrix
from ..log import log
from ..sequences import load_accounts

MINEV_BIG = 50
# Groups reported by this row, in the archived output order.
GROUPS = ["G0_volume", "G1_phase", "G2_schedule", "G3_tempo", "G4_actionmix",
          "G1G2_phase_sched", "G1G2G3_plus_tempo", "full_profile"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    ensure_io_dir()

    log("loading")
    pops = {"TW:IRA": load_accounts(io_path(SEQ_FILES["tw_t"]), MINEV_BIG),
            "TW:organic": load_accounts(io_path(SEQ_FILES["tw_o"]), MINEV_BIG)}
    arch = load_archived_splits()
    split = {p: {s: archived_split_users(arch, p, s, present_in=pops[p]) for s in ("train", "test")} for p in pops}
    log("splits: " + ", ".join(f"{p} {len(v['train'])}/{len(v['test'])}" for p, v in split.items()))
    F = {p: pd.DataFrame([account_features(*pops[p][u]) for u in split[p]["train"] + split[p]["test"]],
                         index=split[p]["train"] + split[p]["test"]) for p in pops}
    Xtr_df = pd.concat([F["TW:IRA"].loc[split["TW:IRA"]["train"]], F["TW:organic"].loc[split["TW:organic"]["train"]]])
    ytr = np.r_[np.ones(len(split["TW:IRA"]["train"])), np.zeros(len(split["TW:organic"]["train"]))]
    res = {"protocol": "rep-task 70/30 split; fit IRA train vs organic train; AUC IRA test vs organic test; "
                       "500-draw bootstrap", "auc": {}}
    for g in GROUPS:
        cols = FEATURE_GROUPS[g]
        sc = StandardScaler().fit(design_matrix(Xtr_df, cols, LOGCOLS))
        Xtr = sc.transform(design_matrix(Xtr_df, cols, LOGCOLS))
        Xp = sc.transform(design_matrix(F["TW:IRA"].loc[split["TW:IRA"]["test"]], cols, LOGCOLS))
        Xn = sc.transform(design_matrix(F["TW:organic"].loc[split["TW:organic"]["test"]], cols, LOGCOLS))
        for name, clf in (("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
                          ("gbm", GradientBoostingClassifier(random_state=0))):
            clf.fit(Xtr, ytr)
            res["auc"][f"{g}|{name}"] = auc_ci(clf.predict_proba(Xp)[:, 1], clf.predict_proba(Xn)[:, 1])
        log(f"{g}: lr {res['auc'][g + '|lr']} gbm {res['auc'][g + '|gbm']}")
    json.dump(res, open(io_path("tw_inplatform_profile.json"), "w"), indent=1)
    log("DONE -> tw_inplatform_profile.json")


if __name__ == "__main__":
    main()
