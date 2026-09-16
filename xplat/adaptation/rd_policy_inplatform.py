"""Within-Reddit (RD->RD) learned action-gap policy: the missing Table 2 cell.

Fits the same two-layer Mamba next-token policy used in rep_task_matrix
(30 tokens = 3 actions x 10 gap deciles, 512-token windows, <=20 windows per
account) on Reddit itself, and scores held-out accounts by the log-likelihood
ratio LLR(u) = mean log p_IRA(token | history) - mean log p_organic(...).

Cohort: the common transfer cohort of rd_inplatform (96 IRA / 643 organic,
2015-01-02..2018-04-11 window, >=10 events).  Splits: the first N_REP repeats
of the same RepeatedStratifiedKFold(5, 10, random_state=0) used there, so the
row shares folds with the in-platform profile / MPNet / empirical-policy rows.

Gap quantizer: (a) "rd_edges": empirical gap deciles of the organic-Reddit
TRAIN accounts of each fold (platform-adapted reference, as in the method
section); (b) "tw_edges": the archived Twitter edges, frozen (repeat 0 only),
as a sensitivity check.

The only deviation from the Twitter training recipe is a minimum number of
optimizer steps (MIN_STEPS): the Reddit IRA training fold has ~80 accounts and
~100 windows, so a fixed 3 epochs would be ~6 steps.

Requires: china_policy_edges.json (archived Twitter edges) and the Reddit
sequence and membership files in IO_RESULTS_DIR.

Run:     python -m xplat.adaptation.rd_policy_inplatform
Outputs: rd_policy_inplatform.json, rd_policy_inplatform_scores.csv (inside IO_RESULTS_DIR)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from ..cohorts import load_reddit_cohort, reddit_cv_splits
from ..config import N_BOOT, SEED, ensure_io_dir, io_path
from ..evaluation import bootstrap_auc
from ..log import log
from ..policy import account_loglik, decile_edges, device, load_tw_edges, tokenize, train_policy, windows
from ..sequences import gaps_min

MIN_STEPS, N_REP, N_FOLD = 150, 3, 5


def auc_ci(score, y, n_boot=N_BOOT):
    """(auc, lo, hi) with a fresh seeded generator; same draws as the archived script."""
    return bootstrap_auc(score[y == 1], score[y == 0], n_boot=n_boot, seed=SEED)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    ensure_io_dir()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = device()
    tw_edges = load_tw_edges()

    # ---------------------------------------------------------------- cohort
    log("loading Reddit cohort")
    rd_t, rd_o = load_reddit_cohort()
    pop = {}
    pop.update(rd_t)
    n_t = len(pop)
    pop.update(rd_o)
    users = list(pop.keys())
    y = np.r_[np.ones(n_t), np.zeros(len(users) - n_t)].astype(int)
    log(f"cohort: {int(y.sum())} IRA, {int((1 - y).sum())} organic")

    splits = reddit_cv_splits(y, n_splits=N_FOLD, n_repeats=10, random_state=0)  # identical to rd_inplatform

    def run_variant(name, n_rep, edges_fn):
        oof = np.full((n_rep, len(y)), np.nan)
        cen = np.full((n_rep, len(y)), np.nan)
        fold_info = []
        for r in range(n_rep):
            for k in range(N_FOLD):
                tr, te = splits[r * N_FOLD + k]
                tr_t = [users[i] for i in tr if y[i] == 1]
                tr_o = [users[i] for i in tr if y[i] == 0]
                edges = edges_fn(tr_o)
                m_t, x_t = train_policy(windows(pop, tr_t, edges), f"{name} r{r}k{k} IRA",
                                        min_steps=MIN_STEPS, log_every=10, dev=dev)
                m_o, x_o = train_policy(windows(pop, tr_o, edges), f"{name} r{r}k{k} organic",
                                        min_steps=MIN_STEPS, log_every=10, dev=dev)

                def llr(u):
                    tok = tokenize(*pop[u], edges)
                    return account_loglik(m_t, tok, dev=dev) - account_loglik(m_o, tok, dev=dev)

                for i in te:
                    oof[r, i] = llr(users[i])
                # fold-centering: standardize by the TRAIN organic accounts' LLR (no held-out information)
                tr_llr = np.array([llr(u) for u in tr_o])
                mu, sd = tr_llr.mean(), tr_llr.std() + 1e-9
                cen[r, te] = (oof[r, te] - mu) / sd
                a = roc_auc_score(y[te], oof[r, te])
                fold_info.append({"rep": r, "fold": k, "auc": round(float(a), 4), "xent_ira": round(x_t, 4),
                                  "xent_org": round(x_o, 4),
                                  "train_org_llr_mu_sd": [round(float(mu), 4), round(float(sd), 4)],
                                  "edges": [round(float(e), 2) for e in edges]})
                log(f"{name} rep {r} fold {k}: held-out AUC {a:.3f} (train-organic LLR mu {mu:.3f} sd {sd:.3f})")
                del m_t, m_o
                torch.cuda.empty_cache()

        def summarize(arr):
            per_rep = [float(roc_auc_score(y, arr[r])) for r in range(n_rep)]
            a0, lo0, hi0 = auc_ci(arr[0], y)
            am, lom, him = auc_ci(arr.mean(0), y)
            return {"per_repeat_auc": [round(a, 4) for a in per_rep], "auc_mean": round(float(np.mean(per_rep)), 4),
                    "auc_sd": round(float(np.std(per_rep, ddof=1)), 4) if n_rep > 1 else None,
                    "rep0_auc_ci": [round(a0, 4), round(lo0, 4), round(hi0, 4)],
                    "mean_score_auc_ci": [round(am, 4), round(lom, 4), round(him, 4)]}

        res = {"n_rep": n_rep, "pooled_raw_llr": summarize(oof), "pooled_fold_centered_llr": summarize(cen),
               "per_fold_auc_mean": round(float(np.mean([f["auc"] for f in fold_info])), 4), "folds": fold_info}
        log(f"{name}: raw pooled AUC {res['pooled_raw_llr']['auc_mean']:.3f}; fold-centered pooled AUC "
            f"{res['pooled_fold_centered_llr']['auc_mean']:.3f} rep0 {res['pooled_fold_centered_llr']['rep0_auc_ci']}; "
            f"per-fold mean {res['per_fold_auc_mean']:.3f}")
        return res, cen.mean(0)

    def rd_edges(tr_o):
        # deciles of the organic TRAIN accounts' gaps, duplicates merged
        return decile_edges(np.concatenate([gaps_min(pop[u][0])[1:] for u in tr_o]))

    results = {"cohort": {"n_ira": int(y.sum()), "n_organic": int((1 - y).sum())},
               "protocol": f"first {N_REP} repeats of RepeatedStratifiedKFold(5,10,rs=0) shared with rd_inplatform; "
                           f"LLR = mean log p_IRA - mean log p_organic; MIN_STEPS={MIN_STEPS}",
               "models": {}}
    scores = pd.DataFrame({"user": users, "label": y})
    res, s = run_variant("Policy|llr|rd_edges", N_REP, rd_edges)
    results["models"]["Policy|llr|rd_edges"] = res
    scores["Policy|llr|rd_edges"] = s
    res, s = run_variant("Policy|llr|tw_edges", 1, lambda tr_o: tw_edges)
    results["models"]["Policy|llr|tw_edges"] = res
    scores["Policy|llr|tw_edges"] = s
    json.dump(results, open(io_path("rd_policy_inplatform.json"), "w"), indent=1)
    scores.to_csv(io_path("rd_policy_inplatform_scores.csv"), index=False)
    log("DONE -> rd_policy_inplatform.json, rd_policy_inplatform_scores.csv")


if __name__ == "__main__":
    main()
