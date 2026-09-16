"""Reddit-to-Reddit (in-platform) baselines on the common transfer cohort.

All rows are account-level AUC under repeated stratified 5-fold CV (10 repeats,
shared splits), on the same 96 IRA / 643 organic Reddit cohort as the
TW->RD transfer (component_ablation cohort A), so they are comparable with
Table 2 of the ICWSM draft:

  (1) OS full profile and OS phase+schedule (LR / GBM), same features and
      classifier settings as component_ablation, refit within Reddit.
  (2) MPNet content baseline (LR) from the cached account embeddings.
  (3) Empirical-policy baseline of Schneider et al. (2026), reimplemented from
      the 12-state x 6-action encoding of Yuan et al. (WWW 2025) on the recrawl
      per-user files with agreement labels; random forest on the vectorized
      row-normalized visitation matrix.
  (4) Schneider-style protocol check for (3): 25 runs, 1:1 organics matched on
      trajectory length (organics truncated to troll length), stratified 3-fold,
      RF; macro-F1 and AUC (to compare with their reported 93.9% / 94.4%).

Run:     python -m xplat.adaptation.rd_inplatform [--dry N]
Outputs: rd_inplatform_results.json, rd_inplatform_scores.csv (in IO_RESULTS_DIR)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..cohorts import reddit_cv_splits
from .. import config
from ..config import EPS, FEAT_FILES, RD_WIN, SEQ_FILES, io_path
from ..features import FEATURE_GROUPS, design_matrix, featurize_population
from ..log import log

# Feature groups evaluated within Reddit (the ablation groups without tzfree).
GROUP_KEYS = ["G0_volume", "G1_phase", "G2_schedule", "G3_tempo", "G4_actionmix",
              "G1G2_phase_sched", "G1G2G3_plus_tempo", "full_profile"]
GROUPS = {k: FEATURE_GROUPS[k] for k in GROUP_KEYS}

# ---------------- Empirical policy (Yuan et al. 2025 encoding) ----------------
STATES = ["IT", "IRC", "IR+", "IR~", "IR-", "ERC", "ER+", "ER~", "ER-", "GR+", "GR~", "GR-"]
ACTIONS = ["WR", "CT", "RC", "PR+", "PR~", "PR-"]
SI = {s: i for i, s in enumerate(STATES)}
AI = {a: i for i, a in enumerate(ACTIONS)}
AGR = {0: "-", 1: "~", 2: "+"}   # agree_prediciton: 0=disagree, 1=neutral, 2=agree

N_REP, N_FOLD = 10, 5


def read_opt(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        log(f"  C parser failed on {path} ({type(e).__name__}); falling back to python engine, skipping bad lines")
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def user_events(u, rd_dir=None):
    """Return sorted list of (ts, kind, thread, agr) for user u.
    kind in {CT, RC, PR, GR}; agr in {+,~,-} or None."""
    d = os.path.join(rd_dir or config.REDDIT_RECRAWL_DIR, u)
    ev = []
    sub = read_opt(os.path.join(d, "all_submission.csv"), ["created_utc", "id"])
    for ts, i in zip(sub["created_utc"], sub["id"]):
        ev.append((ts, "CT", f"t3_{i}", None))
    rc = read_opt(os.path.join(d, "all_user_comment_submission.csv"), ["created_utc", "link_id"])
    for ts, l in zip(rc["created_utc"], rc["link_id"]):
        ev.append((ts, "RC", l, None))
    pr = read_opt(os.path.join(d, "all_user_comment_reply_w_agreement.csv"), ["created_utc", "link_id", "agree_prediciton"])
    for ts, l, a in zip(pr["created_utc"], pr["link_id"], pr["agree_prediciton"]):
        ev.append((ts, "PR", l, AGR.get(int(a) if pd.notna(pd.to_numeric(a, errors="coerce")) else 1, "~")))
    gr = read_opt(os.path.join(d, "all_reply_comments_w_agreement.csv"), ["created_utc", "link_id", "agree_prediciton"])
    for ts, l, a in zip(gr["created_utc"], gr["link_id"], gr["agree_prediciton"]):
        ev.append((ts, "GR", l, AGR.get(int(a) if pd.notna(pd.to_numeric(a, errors="coerce")) else 1, "~")))
    out = []
    for t, k, l, a in ev:
        t = pd.to_numeric(t, errors="coerce")
        if pd.isna(t) or t <= 0:
            continue
        out.append((int(t), k, str(l), a))
    ev = out
    ev.sort(key=lambda x: x[0])
    return ev


def state_action_pairs(ev, window=RD_WIN):
    ev = [e for e in ev if window[0] <= e[0] <= window[1]]
    seen = set()
    states = []
    for ts, k, th, a in ev:
        if k == "GR":
            states.append("GR" + a)
        else:
            init = th not in seen
            seen.add(th)
            if k == "CT":
                states.append("IT")
            elif k == "RC":
                states.append("IRC" if init else "ERC")
            else:
                states.append(("IR" if init else "ER") + a)
    pairs = []
    for i in range(len(ev) - 1):
        nk, na = ev[i + 1][1], ev[i + 1][3]
        act = "WR" if nk == "GR" else ("CT" if nk == "CT" else ("RC" if nk == "RC" else "PR" + na))
        pairs.append((states[i], act))
    return pairs


def policy_vector(pairs):
    C = np.zeros((12, 6))
    for s, a in pairs:
        C[SI[s], AI[a]] += 1
    rows = C.sum(1, keepdims=True)
    P = np.divide(C, rows, out=np.zeros_like(C), where=rows > 0)
    return P.ravel(), C.sum(), len(pairs)


# ---------------- classifiers ----------------
def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def gbm():
    return make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=0))


def rf():
    return RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=8)


def lr_txt():
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry", type=int, default=0, metavar="N",
                   help="use only the first N members per population and stop after policy encoding")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    DRY = args.dry

    # ---------------- cohort ----------------
    ft_rdt = pd.read_csv(io_path(FEAT_FILES["rd_t"]))
    ft_rdo = pd.read_csv(io_path(FEAT_FILES["rd_o"]))
    members = {"rd_t": set(ft_rdt["user"]), "rd_o": set(ft_rdo["user"])}
    if DRY:
        members = {k: set(sorted(v)[:DRY]) for k, v in members.items()}
    log("membership", {k: len(v) for k, v in members.items()})
    POPS = {"rd_t": featurize_population(io_path(SEQ_FILES["rd_t"]), members["rd_t"], window=RD_WIN),
            "rd_o": featurize_population(io_path(SEQ_FILES["rd_o"]), members["rd_o"], window=RD_WIN)}
    log("cohort A (window, >=10 events):", {k: len(v) for k, v in POPS.items()})
    tgt = pd.concat([POPS["rd_t"], POPS["rd_o"]], ignore_index=True)
    y = np.r_[np.ones(len(POPS["rd_t"])), np.zeros(len(POPS["rd_o"]))]
    users = tgt["user"].tolist()

    # content vectors
    VEC = {}
    for pop in ("rd_t", "rd_o"):
        z = np.load(io_path(f"textbase_emb_{pop}.npz"), allow_pickle=True)
        VEC.update(dict(zip(z["users"].tolist(), z["vecs"])))
    Xtxt = np.array([VEC.get(u.lower(), np.zeros(768)) for u in users])
    has_txt = np.array([u.lower() in VEC for u in users])
    Xtxt = Xtxt / (np.linalg.norm(Xtxt, axis=1, keepdims=True) + EPS)
    log("content vectors found for", int(has_txt.sum()), "of", len(users))

    # empirical policy vectors
    Xpol, npairs = [], []
    for i, u in enumerate(users):
        ev = user_events(u)
        pairs = state_action_pairs(ev)
        v, tot, n = policy_vector(pairs)
        Xpol.append(v)
        npairs.append(n)
        if DRY or i % 100 == 0:
            log(f"policy {i+1}/{len(users)} {u}: events={len(ev)} pairs={n} label={int(y[i])} "
                f"states={dict(Counter(s for s, _ in pairs).most_common(4))} "
                f"actions={dict(Counter(a for _, a in pairs).most_common(4))}")
    Xpol = np.array(Xpol)
    npairs = np.array(npairs)
    log("policy encoding done; pairs per account: IRA median", np.median(npairs[y == 1]),
        "organic median", np.median(npairs[y == 0]))
    if DRY:
        return 0

    # ---------------- shared repeated CV ----------------
    splits = reddit_cv_splits(y, n_splits=N_FOLD, n_repeats=N_REP, random_state=0)

    MODELS = {
        "OS_full|lr": (lambda: lr(), design_matrix(tgt, GROUPS["full_profile"]), np.ones(len(y), bool)),
        "OS_full|gbm": (lambda: gbm(), design_matrix(tgt, GROUPS["full_profile"]), np.ones(len(y), bool)),
        "OS_phase_sched|lr": (lambda: lr(), design_matrix(tgt, GROUPS["G1G2_phase_sched"]), np.ones(len(y), bool)),
        "OS_phase_sched|gbm": (lambda: gbm(), design_matrix(tgt, GROUPS["G1G2_phase_sched"]), np.ones(len(y), bool)),
        "MPNet|lr": (lambda: lr_txt(), Xtxt, has_txt),
        "EmpiricalPolicy|rf": (lambda: rf(), Xpol, npairs > 0),
        "EmpiricalPolicy|lr": (lambda: lr(), Xpol, npairs > 0),
    }
    results = {"cohort": {"n_ira": int(y.sum()), "n_organic": int((1 - y).sum())},
               "cv": f"{N_REP}x{N_FOLD}-fold stratified, shared splits", "models": {}}
    scores_df = pd.DataFrame({"user": users, "label": y, "n_pairs": npairs})
    for name, (mk, X, mask) in MODELS.items():
        aucs = []
        oof_all = np.zeros(len(y))
        for r in range(N_REP):
            oof = np.full(len(y), np.nan)
            for k in range(N_FOLD):
                tr, te = splits[r * N_FOLD + k]
                tr = tr[mask[tr]]
                te = te[mask[te]]
                m = mk().fit(X[tr], y[tr])
                oof[te] = m.predict_proba(X[te])[:, 1]
            ok = ~np.isnan(oof)
            aucs.append(roc_auc_score(y[ok], oof[ok]))
            oof_all += np.nan_to_num(oof) / N_REP
        aucs = np.array(aucs)
        results["models"][name] = {"auc_mean": round(float(aucs.mean()), 4), "auc_sd": round(float(aucs.std(ddof=1)), 4),
                                   "auc_min": round(float(aucs.min()), 4), "auc_max": round(float(aucs.max()), 4),
                                   "n_used": int(mask.sum()), "per_repeat": [round(float(a), 4) for a in aucs]}
        scores_df[name] = oof_all
        log(f"{name}: AUC {aucs.mean():.3f} +/- {aucs.std(ddof=1):.3f} (n={int(mask.sum())})")

    # ---------------- Schneider-style protocol check for the empirical policy ----------------
    rng = np.random.default_rng(0)
    pos = np.flatnonzero((y == 1) & (npairs > 0))
    neg = np.flatnonzero((y == 0) & (npairs > 0))
    f1s, aucs2 = [], []
    # need raw pairs to truncate organics to troll length: recompute pairs on demand (cache)
    PAIRS = {}

    def pairs_for(i):
        if i not in PAIRS:
            PAIRS[i] = state_action_pairs(user_events(users[i]))
        return PAIRS[i]

    for run in range(25):
        # 1:1 match each troll with an unused organic of >= its length, truncate organic to that length
        neg_pool = list(rng.permutation(neg))
        Xr, yr = [], []
        for i in pos:
            L = npairs[i]
            j = next((j for j in neg_pool if npairs[j] >= L), None)
            if j is None:
                j = neg_pool[0]
            neg_pool.remove(j)
            Xr.append(policy_vector(pairs_for(i))[0])
            yr.append(1)
            Xr.append(policy_vector(pairs_for(j)[:L])[0])
            yr.append(0)
        Xr, yr = np.array(Xr), np.array(yr)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=run)
        oof = np.zeros(len(yr))
        pred = np.zeros(len(yr), int)
        for tr, te in skf.split(Xr, yr):
            m = rf().fit(Xr[tr], yr[tr])
            oof[te] = m.predict_proba(Xr[te])[:, 1]
            pred[te] = m.predict(Xr[te])
        f1s.append(f1_score(yr, pred, average="macro"))
        aucs2.append(roc_auc_score(yr, oof))
        if run % 5 == 0:
            log(f"schneider-protocol run {run}: macroF1 {f1s[-1]:.3f} AUC {aucs2[-1]:.3f}")
    results["empirical_policy_schneider_protocol"] = {
        "protocol": "25 runs, 1:1 length-matched organics truncated to troll trajectory length, RF(500), stratified 3-fold",
        "macro_f1_median": round(float(np.median(f1s)), 4),
        "macro_f1_p5_p95": [round(float(np.percentile(f1s, 5)), 4), round(float(np.percentile(f1s, 95)), 4)],
        "auc_median": round(float(np.median(aucs2)), 4),
        "auc_p5_p95": [round(float(np.percentile(aucs2, 5)), 4), round(float(np.percentile(aucs2, 95)), 4)],
        "n_per_class": int(len(pos))}
    log("schneider-protocol:", results["empirical_policy_schneider_protocol"])

    with open(io_path("rd_inplatform_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    scores_df.to_csv(io_path("rd_inplatform_scores.csv"), index=False)
    log("wrote rd_inplatform_results.json + rd_inplatform_scores.csv")
    return 0


if __name__ == "__main__":
    main()
