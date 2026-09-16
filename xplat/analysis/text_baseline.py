"""Tier-1 content baseline: multilingual text-embedding classifier, head-to-head
with the behavioral features on the SAME transfer cohort.

Design: mean-pooled sentence embeddings (paraphrase-multilingual-mpnet-base-v2
via transformers) of up to N_TEXT posts per account -> account vector -> l2 ->
logistic regression trained on Twitter (IRA vs organic timelines), evaluated
zero-shot on the Reddit cohort A (same membership as component_ablation),
unmatched + volume-matched, with paired account-bootstrap deltas against the
archived behavioral scores (ablation_scores.csv).

Checkpointed: per-population text collection and embeddings are cached as
.npz/.jsonl under IO_RESULTS_DIR/textbase_* so the job resumes after drops.

Inputs: TWITTER_TAKEDOWN_DIR/tweet_csvs/ira_tweets_csv_unhashed.csv,
        TWITTER_TIMELINE_DIR (hourly .json.bz2), REDDIT_RECRAWL_DIR (one folder per account),
        IO_RESULTS_DIR/feat_*.csv, ablation_scores.csv
Run:    python -m xplat.analysis.text_baseline
Output: IO_RESULTS_DIR/text_baseline_results.json, text_baseline_scores.csv
"""
from __future__ import annotations

import argparse
import bz2
import glob
import json
import os

import numpy as np
import pandas as pd

from .. import config
from ..config import FEAT_FILES, IRA_TWEETS_FILE, io_path, takedown_path
from ..evaluation import volume_match

N_TEXT = 150
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TEXTCOLS = ["title", "selftext", "body", "text", "comment_body", "comment"]
COMP = ["text_lr", "G1G2_phase_sched|lr", "G1G2_phase_sched|gbm", "full_profile|lr"]
N_BOOT = 1000


# ---------- stage A: collect texts ----------

def cache_path(pop):
    return io_path(f"textbase_texts_{pop}.jsonl")


def save_texts(pop, d):
    with open(cache_path(pop), "w") as f:
        for u, texts in d.items():
            f.write(json.dumps({"user": u, "texts": texts[:N_TEXT]}) + "\n")
    print(f"[{pop}] cached {len(d)} accounts", flush=True)


def load_texts(pop):
    if not os.path.exists(cache_path(pop)):
        return None
    d = {}
    for line in open(cache_path(pop)):
        r = json.loads(line)
        d[r["user"]] = r["texts"]
    print(f"[{pop}] cache hit: {len(d)} accounts", flush=True)
    return d


def collect_tw_t(members):
    d = {}
    use = ["user_screen_name", "tweet_text"]
    for c in pd.read_csv(takedown_path(IRA_TWEETS_FILE), dtype=str, usecols=use,
                         chunksize=1_000_000, lineterminator="\n", on_bad_lines="skip"):
        sn = c["user_screen_name"].str.lower()
        m = sn.isin(members)
        for u, t in zip(sn[m], c["tweet_text"][m].fillna("")):
            if len(t) < 5:
                continue
            lst = d.setdefault(u, [])
            if len(lst) < N_TEXT:
                lst.append(t)
    return d


def collect_tw_o(members):
    d = {}
    files = sorted(glob.glob(f"{config.TWITTER_TIMELINE_DIR}/**/*.bz2", recursive=True))
    print(f"[tw_o] {len(files)} bz2 files", flush=True)
    for i, path in enumerate(files):
        try:
            with bz2.open(path, "rt", errors="replace") as f:
                for line in f:
                    try:
                        t = json.loads(line)
                    except Exception:
                        continue
                    u = t.get("user") or {}
                    uid = str(u.get("id_str", ""))  # feat table keys timelines by id_str
                    if uid not in members:
                        continue
                    txt = t.get("full_text") or t.get("text") or ""
                    if len(txt) < 5:
                        continue
                    lst = d.setdefault(uid, [])
                    if len(lst) < N_TEXT:
                        lst.append(txt)
        except Exception as e:
            print(f"[tw_o] skip {path}: {e}", flush=True)
        if i % 200 == 0:
            print(f"[tw_o] {i}/{len(files)} files, {len(d)} accounts", flush=True)
    return d


def read_user_folder(folder):
    texts = []
    for fn in ("all_submission.csv", "all_parent_comments.csv", "all_reply_comments.csv"):
        p = os.path.join(folder, fn)
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, dtype=str, on_bad_lines="skip", lineterminator="\n")
        except Exception:
            continue
        for c in TEXTCOLS:
            if c in df.columns:
                for t in df[c].dropna():
                    if len(t) >= 5 and t not in ("[deleted]", "[removed]"):
                        texts.append(t)
                        if len(texts) >= N_TEXT * 2:
                            return texts
    return texts


def collect_rd(pop, members):
    d = {}
    folders = {f.lower(): f for f in os.listdir(config.REDDIT_RECRAWL_DIR)}
    hit = 0
    for u in members:
        f = folders.get(u)
        if f is None:
            continue
        texts = read_user_folder(os.path.join(config.REDDIT_RECRAWL_DIR, f))
        if texts:
            d[u] = texts[:N_TEXT]
            hit += 1
    print(f"[{pop}] matched {hit} of {len(members)} accounts", flush=True)
    return d


def make_embedder(model_name, dev):
    """Mean-pooled frozen sentence embeddings (fp16), batched."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name).to(dev).eval().half()

    def embed_texts(texts, bs=256):
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128,
                        return_tensors="pt").to(dev)
                out = mod(**b).last_hidden_state
                mask = b["attention_mask"].unsqueeze(-1).float()
                v = (out.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                vecs.append(v.cpu().numpy())
        return np.vstack(vecs)
    return embed_texts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    rng = np.random.default_rng(0)

    members = {}
    for k in ("tw_t", "tw_o", "rd_t", "rd_o"):
        members[k] = set(pd.read_csv(io_path(FEAT_FILES[k]))["user"].astype(str).str.lower())
    print({k: len(v) for k, v in members.items()}, flush=True)

    POPS = {}
    for pop, fn in (("tw_t", lambda: collect_tw_t(members["tw_t"])),
                    ("tw_o", lambda: collect_tw_o(members["tw_o"])),
                    ("rd_t", lambda: collect_rd("rd_t", members["rd_t"])),
                    ("rd_o", lambda: collect_rd("rd_o", members["rd_o"]))):
        d = load_texts(pop)
        if d is None:
            d = fn()
            save_texts(pop, d)
        POPS[pop] = d

    # ---------- stage B: embed ----------
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    embed_texts = None
    VEC = {}
    for pop, d in POPS.items():
        npz = io_path(f"textbase_emb_{pop}.npz")
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            VEC[pop] = dict(zip(z["users"].tolist(), z["vecs"]))
            print(f"[{pop}] emb cache hit: {len(VEC[pop])}", flush=True)
            continue
        if embed_texts is None:
            embed_texts = make_embedder(MODEL, dev)
        users, vecs = [], []
        for j, (u, texts) in enumerate(sorted(d.items())):
            E = embed_texts(texts)
            v = E.mean(axis=0)
            v /= (np.linalg.norm(v) + 1e-9)
            users.append(u)
            vecs.append(v)
            if j % 200 == 0:
                print(f"[{pop}] embedded {j}/{len(d)}", flush=True)
        VEC[pop] = dict(zip(users, vecs))
        np.savez_compressed(npz, users=np.array(users), vecs=np.array(vecs))
        print(f"[{pop}] embedded {len(users)} accounts", flush=True)

    # ---------- stage C: classify + evaluate on the ablation cohort ----------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    Xtr = np.array(list(VEC["tw_t"].values()) + list(VEC["tw_o"].values()))
    ytr = np.r_[np.ones(len(VEC["tw_t"])), np.zeros(len(VEC["tw_o"]))]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    cv = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                         Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                         scoring="roc_auc")
    print(f"within-Twitter 5-fold CV AUC: {cv.mean():.3f} ± {cv.std():.3f}", flush=True)

    ab = pd.read_csv(io_path("ablation_scores.csv"))
    ab["user_l"] = ab["user"].astype(str).str.lower()
    have = ab["user_l"].map(lambda u: u in VEC["rd_t"] or u in VEC["rd_o"])
    D = ab[have].copy()
    Xte = np.array([VEC["rd_t"].get(u) if u in VEC["rd_t"] else VEC["rd_o"][u] for u in D["user_l"]])
    D["text_lr"] = lr.predict_proba(Xte)[:, 1]
    y = D["label"].values
    print(f"reddit eval coverage: {len(D)} accounts ({int(y.sum())} IRA) "
          f"of {len(ab)} cohort-A accounts", flush=True)

    res = {"model": MODEL, "n_text_cap": N_TEXT,
           "within_twitter_cv_auc": [round(float(cv.mean()), 3), round(float(cv.std()), 3)],
           "coverage": {"reddit_eval": int(len(D)), "reddit_ira": int(y.sum()),
                        "cohortA_total": int(len(ab))}}

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    nev = D["n_events"].values

    def auc_of(col, P, N):
        s = D[col].values
        return roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(N))], np.r_[s[P], s[N]])

    for tag, (P, N) in (("unmatched", (pos, neg)), ("matched", (pos, volume_match(nev, neg, pos, rng)))):
        point = {c: round(auc_of(c, P, N), 4) for c in COMP}
        draws = {c: np.empty(N_BOOT) for c in COMP}
        for b in range(N_BOOT):
            bp = rng.choice(P, len(P), replace=True)
            bn = rng.choice(N, len(N), replace=True)
            for c in COMP:
                draws[c][b] = auc_of(c, bp, bn)
        deltas = {}
        for c in COMP[1:]:
            dd = draws[c] - draws["text_lr"]
            deltas[f"{c}-minus-text"] = {
                "point": round(point[c] - point["text_lr"], 4),
                "ci": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]}
        res[tag] = {"auc": point,
                    "ci": {c: [round(float(np.percentile(d, 2.5)), 4),
                               round(float(np.percentile(d, 97.5)), 4)] for c, d in draws.items()},
                    "delta_vs_text": deltas, "n": [int(len(P)), int(len(N))]}
        print(tag, json.dumps(point), flush=True)

    json.dump(res, open(io_path("text_baseline_results.json"), "w"), indent=1)
    D[["user", "label", "n_events", "text_lr"]].to_csv(io_path("text_baseline_scores.csv"), index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
