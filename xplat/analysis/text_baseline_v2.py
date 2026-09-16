"""Stronger content baselines on the same transfer cohort (reviewer M3).

V2a  originals-only frozen embeddings: identical to v1 but texts starting
     with "RT @" are dropped before pooling (retweets are other people's
     words).  Reuses the v1 text caches; embeddings recomputed.
V2b  fine-tuned transformer: xlm-roberta-base fine-tuned at the POST level
     (IRA vs organic-timeline tweets, originals only, account-disjoint 80/20
     split), account score = mean post probability.  Held-out within-Twitter
     AUC + zero-shot Reddit transfer.

Both evaluated exactly like v1: cohort A join with ablation_scores.csv,
unmatched + volume-matched, paired 1,000-draw account bootstrap against the
v1 text baseline and the behavioral scorers.  Needs one CUDA device.

Inputs: IO_RESULTS_DIR/textbase_texts_*.jsonl (from text_baseline), ablation_scores.csv,
        text_baseline_scores.csv
Run:    python -m xplat.analysis.text_baseline_v2
Output: IO_RESULTS_DIR/text_baseline_v2_results.json, text_baseline_v2_scores.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from ..config import io_path
from ..evaluation import volume_match

N_TEXT = 150
RT = re.compile(r"^\s*RT @")
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
FT_MODEL = "xlm-roberta-base"
CAP_TRAIN = 60
COMP = ["text_lr", "text_orig_lr", "text_ft", "G1G2_phase_sched|lr", "G1G2_phase_sched|gbm", "full_profile|lr"]
N_BOOT = 1000


def load_cache(pop):
    d = {}
    for line in open(io_path(f"textbase_texts_{pop}.jsonl")):
        r = json.loads(line)
        d[r["user"]] = r["texts"]
    return d


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    rng = np.random.default_rng(0)

    POPS = {p: load_cache(p) for p in ("tw_t", "tw_o", "rd_t", "rd_o")}
    ORIG = {p: {u: [t for t in ts if not RT.match(t)] for u, ts in d.items()} for p, d in POPS.items()}
    ORIG = {p: {u: ts for u, ts in d.items() if len(ts) >= 3} for p, d in ORIG.items()}
    print({p: (len(POPS[p]), len(ORIG[p])) for p in POPS}, flush=True)

    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- V2a: originals-only frozen embeddings ----------
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    mod = AutoModel.from_pretrained(EMB_MODEL).to(dev).eval().half()

    def embed_texts(texts, bs=256):
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128,
                        return_tensors="pt").to(dev)
                o = mod(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                vecs.append(((o.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        return np.vstack(vecs)

    VEC = {}
    for pop, d in ORIG.items():
        npz = io_path(f"textbase_v2a_emb_{pop}.npz")
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            VEC[pop] = dict(zip(z["users"].tolist(), z["vecs"]))
            print(f"[v2a {pop}] cache hit {len(VEC[pop])}", flush=True)
            continue
        users, vecs = [], []
        for j, (u, texts) in enumerate(sorted(d.items())):
            E = embed_texts(texts[:N_TEXT])
            v = E.mean(0)
            v /= (np.linalg.norm(v) + 1e-9)
            users.append(u)
            vecs.append(v)
            if j % 400 == 0:
                print(f"[v2a {pop}] {j}/{len(d)}", flush=True)
        VEC[pop] = dict(zip(users, vecs))
        np.savez_compressed(npz, users=np.array(users), vecs=np.array(vecs))
        print(f"[v2a {pop}] embedded {len(users)}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    Xtr = np.array(list(VEC["tw_t"].values()) + list(VEC["tw_o"].values()))
    ytr = np.r_[np.ones(len(VEC["tw_t"])), np.zeros(len(VEC["tw_o"]))]
    lr2a = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    cv2a = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                           Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           scoring="roc_auc")
    print(f"[v2a] within-TW CV {cv2a.mean():.3f} ± {cv2a.std():.3f}", flush=True)

    # ---------- V2b: fine-tuned xlm-roberta at post level ----------
    CKPT = io_path("textbase_v2b_xlmr.pt")
    tw_t_users = sorted(ORIG["tw_t"])
    tw_o_users = sorted(ORIG["tw_o"])
    rng.shuffle(tw_t_users)
    rng.shuffle(tw_o_users)
    sp_t, sp_o = int(0.8 * len(tw_t_users)), int(0.8 * len(tw_o_users))
    hold_t, hold_o = tw_t_users[sp_t:], tw_o_users[sp_o:]

    train_texts, train_y = [], []
    for u in tw_t_users[:sp_t]:
        for t in ORIG["tw_t"][u][:CAP_TRAIN]:
            train_texts.append(t)
            train_y.append(1)
    for u in tw_o_users[:sp_o]:
        for t in ORIG["tw_o"][u][:CAP_TRAIN]:
            train_texts.append(t)
            train_y.append(0)
    idx = rng.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in idx]
    train_y = np.array(train_y)[idx]
    print(f"[v2b] train posts {len(train_texts)} (pos {int(train_y.sum())}), "
          f"held-out accounts {len(hold_t)}/{len(hold_o)}", flush=True)

    ftok = AutoTokenizer.from_pretrained(FT_MODEL)
    fmod = AutoModelForSequenceClassification.from_pretrained(FT_MODEL, num_labels=2).to(dev)
    if os.path.exists(CKPT):
        fmod.load_state_dict(torch.load(CKPT, map_location=dev))
        print("[v2b] ckpt hit", flush=True)
    else:
        fmod.train()
        opt = torch.optim.AdamW(fmod.parameters(), lr=2e-5)
        scaler = torch.amp.GradScaler("cuda")
        w1 = float((train_y == 0).mean() / max((train_y == 1).mean(), 1e-6))
        cw = torch.tensor([1.0, w1], device=dev)
        BS = 64
        for i in range(0, len(train_texts), BS):
            b = ftok(train_texts[i:i + BS], padding=True, truncation=True, max_length=128,
                     return_tensors="pt").to(dev)
            yy = torch.tensor(train_y[i:i + BS], device=dev)
            with torch.amp.autocast("cuda"):
                logits = fmod(**b).logits
                loss = torch.nn.functional.cross_entropy(logits, yy, weight=cw)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if (i // BS) % 200 == 0:
                print(f"[v2b] step {i // BS}/{len(train_texts) // BS} loss {loss.item():.3f}", flush=True)
        torch.save(fmod.state_dict(), CKPT)
    fmod.eval()

    def ft_account_score(texts, bs=128):
        ps = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b = ftok(texts[i:i + bs], padding=True, truncation=True, max_length=128,
                         return_tensors="pt").to(dev)
                with torch.amp.autocast("cuda"):
                    p = torch.softmax(fmod(**b).logits, -1)[:, 1]
                ps.append(p.float().cpu().numpy())
        return float(np.concatenate(ps).mean())

    SC = {}
    sc_npz = io_path("textbase_v2b_scores.npz")
    if os.path.exists(sc_npz):
        z = np.load(sc_npz, allow_pickle=True)
        SC = dict(zip(z["users"].tolist(), z["scores"].tolist()))
        print(f"[v2b] score cache {len(SC)}", flush=True)
    else:
        todo = [("tw_hold", u, ORIG["tw_t"][u]) for u in hold_t] + \
               [("tw_hold", u, ORIG["tw_o"][u]) for u in hold_o] + \
               [("rd", u, ts) for u, ts in ORIG["rd_t"].items()] + \
               [("rd", u, ts) for u, ts in ORIG["rd_o"].items()]
        for j, (_, u, ts) in enumerate(todo):
            SC[u] = ft_account_score(ts[:N_TEXT])
            if j % 100 == 0:
                print(f"[v2b] scored {j}/{len(todo)}", flush=True)
        np.savez_compressed(sc_npz, users=np.array(list(SC)), scores=np.array(list(SC.values())))

    yh = np.r_[np.ones(len(hold_t)), np.zeros(len(hold_o))]
    sh = np.array([SC[u] for u in hold_t] + [SC[u] for u in hold_o])
    ft_tw_auc = roc_auc_score(yh, sh)
    print(f"[v2b] held-out within-TW account AUC {ft_tw_auc:.3f}", flush=True)

    # ---------- joint evaluation on cohort A ----------
    ab = pd.read_csv(io_path("ablation_scores.csv"))
    ab["user_l"] = ab["user"].astype(str).str.lower()
    v1 = pd.read_csv(io_path("text_baseline_scores.csv"))
    v1["user_l"] = v1["user"].astype(str).str.lower()
    D = ab.merge(v1[["user_l", "text_lr"]], on="user_l", how="inner")
    D["text_orig_lr"] = [
        float(lr2a.predict_proba(VEC["rd_t"][u].reshape(1, -1))[0, 1]) if u in VEC["rd_t"]
        else float(lr2a.predict_proba(VEC["rd_o"][u].reshape(1, -1))[0, 1]) if u in VEC["rd_o"]
        else np.nan for u in D["user_l"]]
    D["text_ft"] = [SC.get(u, np.nan) for u in D["user_l"]]
    D = D.dropna(subset=["text_orig_lr", "text_ft"])
    y = D["label"].values
    print(f"joint coverage {len(D)} accounts ({int(y.sum())} IRA)", flush=True)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    nev = D["n_events"].values

    def auc_of(col, P, N):
        s = D[col].values
        return roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(N))], np.r_[s[P], s[N]])

    res = {"v2a_within_tw_cv": [round(float(cv2a.mean()), 3), round(float(cv2a.std()), 3)],
           "v2b_within_tw_heldout_auc": round(float(ft_tw_auc), 3),
           "v2b_train_posts": int(len(train_texts)),
           "coverage": {"joint": int(len(D)), "ira": int(y.sum())}}
    for tag, (P, N) in (("unmatched", (pos, neg)), ("matched", (pos, volume_match(nev, neg, pos, rng)))):
        point = {c: round(auc_of(c, P, N), 4) for c in COMP}
        draws = {c: np.empty(N_BOOT) for c in COMP}
        for b in range(N_BOOT):
            bp = rng.choice(P, len(P), replace=True)
            bn = rng.choice(N, len(N), replace=True)
            for c in COMP:
                draws[c][b] = auc_of(c, bp, bn)
        deltas = {}
        for base in ("text_orig_lr", "text_ft"):
            for c in ("G1G2_phase_sched|gbm", "full_profile|lr"):
                dd = draws[c] - draws[base]
                deltas[f"{c}-minus-{base}"] = {
                    "point": round(point[c] - point[base], 4),
                    "ci": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]}
        res[tag] = {"auc": point, "delta": deltas,
                    "ci": {c: [round(float(np.percentile(d, 2.5)), 4),
                               round(float(np.percentile(d, 97.5)), 4)] for c, d in draws.items()},
                    "n": [int(len(P)), int(len(N))]}
        print(tag, json.dumps(point), flush=True)

    json.dump(res, open(io_path("text_baseline_v2_results.json"), "w"), indent=1)
    D[["user", "label", "n_events", "text_lr", "text_orig_lr", "text_ft"]].to_csv(
        io_path("text_baseline_v2_scores.csv"), index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
