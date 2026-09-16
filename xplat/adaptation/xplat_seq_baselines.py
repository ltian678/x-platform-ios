"""Prior-work behavioral baselines on the Table 2 cohorts.

Two published behavioral representations, re-implemented on the paper's
content-free traces (timestamps + coarse actions) and evaluated on exactly the
cohorts of the learned-policy rows in Table 2:

  ActionLSTM  Ezzeddine et al. (2023): an LSTM classifier over the time-ordered
              sequence of account actions.  Their sequences also carry received
              feedback (likes, retweets), which our traces do not have, so this
              is the actions-only variant: 3 action symbols, no timing.
  BLOC        Nwala, Flammini & Menczer (2023): action symbols interleaved with
              pause symbols of fixed absolute duration (< 1 min, < 1 h, < 1 day,
              < 1 week, < 1 month, < 1 year, longer), TF-IDF over symbol bigrams,
              L2-logistic regression.  Fixed-duration pauses make BLOC a natural
              foil for the paper's platform-relative gap deciles.

Cohorts / splits (identical to rep_task_matrix):
  TW->TW   fit on TW:IRA train (2,042) vs TW:organic train (1,220);
           AUC on TW:IRA test (875) vs TW:organic test (523)
  TW->RD   same fitted model, frozen; AUC on RD:IRA (96) vs the 168
           volume-matched organic Reddit accounts (rng replay) and vs all 643
  RD->RD   5-fold stratified CV on the 96/643 cohort, first repeat of the
           rd_inplatform splits, out-of-fold scores pooled
500-draw account bootstrap intervals on every AUC.

Run:     python -m xplat.adaptation.xplat_seq_baselines [--models M1,M2] [--outname NAME] [--extra-ops]
         (these flags replace the MODELS / OUTNAME / EXTRA_OPS environment
         variables of the archived script; defaults are unchanged)
Outputs: <outname>.json, <outname>_scores.csv (in IO_RESULTS_DIR)
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from ..cohorts import archived_split_users, load_archived_splits, load_reddit_cohort, matched_reddit_controls, reddit_cv_splits
from ..config import CHINA_FILES, IRAN_FILES, SEED, SEQ_FILES, io_path, takedown_paths
from ..evaluation import auc_ci
from ..log import log
from ..policy import MAX_WIN_PER_ACC, WIN, decile_edges, device
from ..sequences import gaps_min, load_accounts, parse_takedown_actions

MINEV_BIG = 50
DEFAULT_MODELS = "BLOC|tfidf_lr,ActionLSTM"

# ---------------------------------------------------------------- BLOC-style representation
PAUSE_EDGES_MIN = np.array([1, 60, 1440, 10080, 43200, 525600])   # <1min <1h <1d <1w <1mo <1y, else longer
ACT_SYM = "Tpr"                                                   # post, reply, repost -> BLOC-like T / p / r
ACT_MAP = {0: "T", 1: "r", 2: "p"}
PAUSE_SYM = "abcdefg"

# ---------------------------------------------------------------- action-only LSTM (Ezzeddine-style)
N_ACT = 3
N_GAPB = 10   # platform-adapted gap deciles for the gap-token sequence variants


def bloc_string(ts, ac):
    g = gaps_min(ts)
    out = [ACT_MAP.get(int(ac[0]), "T")]
    for i in range(1, len(ts)):
        out.append(PAUSE_SYM[int(np.searchsorted(PAUSE_EDGES_MIN, g[i], side="right"))])
        out.append(ACT_MAP.get(int(ac[i]), "T"))
    return "".join(out)


class WinData(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.from_numpy(self.seqs[i].copy()), float(self.labels[i])


class SeqClf(nn.Module):
    def __init__(self, backbone, vocab, d=128, h=64, layers=2):
        super().__init__()
        self.backbone = backbone
        if backbone == "lstm":
            self.emb = nn.Embedding(vocab + 1, 32, padding_idx=vocab)
            self.rnn = nn.LSTM(32, h, num_layers=1, batch_first=True)
            out = h
        else:
            from mamba_ssm import Mamba
            self.emb = nn.Embedding(vocab + 1, d, padding_idx=vocab)
            self.blocks = nn.ModuleList([Mamba(d_model=d) for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            out = d
        self.head = nn.Linear(out, 1)

    def forward(self, x, m):
        h = self.emb(x)
        if self.backbone == "lstm":
            h, _ = self.rnn(h)
        else:
            for blk, nrm in zip(self.blocks, self.norms):
                h = h + blk(nrm(h))
        pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def build_models(POPS, EDGES):
    """Fit-and-score callables over ``(pop, user)`` keys of ``POPS``."""
    DEV = device()

    def bloc_fit_score(train_pos, train_neg, test_sets):
        docs = [bloc_string(*POPS[p][u]) for p, u in train_pos + train_neg]
        y = np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))]
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), lowercase=False, sublinear_tf=True)
        X = vec.fit_transform(docs)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(X, y)
        return {name: clf.predict_proba(vec.transform([bloc_string(*POPS[p][u]) for p, u in keys]))[:, 1]
                for name, keys in test_sets.items()}

    def tok_act(p, u):
        return POPS[p][u][1].astype(np.int64)

    def tok_gap(p, u):
        ts, ac = POPS[p][u]
        b = np.clip(np.searchsorted(EDGES[p[:2]], gaps_min(ts), side="right"), 0, N_GAPB - 1)
        return (ac * N_GAPB + b).astype(np.int64)

    def make_seq_fit_score(backbone, tokfn, vocab, label):
        def windows_of(keys):
            out = []
            r = np.random.default_rng(SEED)
            for p, u in keys:
                t = tokfn(p, u)
                if len(t) <= WIN:
                    out.append(t)
                else:
                    starts = np.arange(0, len(t) - WIN + 1, WIN)
                    if len(starts) > MAX_WIN_PER_ACC:
                        starts = r.choice(starts, MAX_WIN_PER_ACC, replace=False)
                    out += [t[s:s + WIN] for s in starts]
            return out

        def coll(batch):
            L = max(len(t) for t, _ in batch)
            x = torch.full((len(batch), L), vocab, dtype=torch.long)
            m = torch.zeros((len(batch), L))
            for i, (t, _) in enumerate(batch):
                x[i, :len(t)] = t
                m[i, :len(t)] = 1
            return x, m, torch.tensor([y for _, y in batch])

        def fit_score(train_pos, train_neg, test_sets, epochs=3, min_steps=150):
            torch.manual_seed(SEED)
            wp, wn = windows_of(train_pos), windows_of(train_neg)
            seqs = wp + wn
            labels = np.r_[np.ones(len(wp)), np.zeros(len(wn))]
            steps = math.ceil(len(seqs) / 48)
            epochs = max(epochs, math.ceil(min_steps / steps))
            model = SeqClf(backbone, vocab).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(wn) / max(1, len(wp))], device=DEV))
            dl = DataLoader(WinData(seqs, labels), batch_size=48, shuffle=True, collate_fn=coll)
            log(f"  {label}: {len(wp)}+{len(wn)} windows, {epochs} epochs")
            for ep in range(epochs):
                model.train()
                for x, m, yb in dl:
                    opt.zero_grad()
                    loss = lossf(model(x.to(DEV), m.to(DEV)), yb.to(DEV))
                    loss.backward()
                    opt.step()
            model.eval()

            def score(keys):
                out = []
                with torch.no_grad():
                    for p, u in keys:
                        t = tokfn(p, u)
                        segs = [t[s:s + WIN] for s in range(0, len(t), WIN)]
                        x, m, _ = coll([(torch.from_numpy(s.copy()), 0.0) for s in segs])
                        out.append(float(torch.sigmoid(model(x.to(DEV), m.to(DEV))).mean()))
                return np.array(out)
            return {name: score(keys) for name, keys in test_sets.items()}
        return fit_score

    return {"BLOC|tfidf_lr": bloc_fit_score,
            "ActionLSTM": make_seq_fit_score("lstm", tok_act, N_ACT, "ActionLSTM"),
            "MambaAct": make_seq_fit_score("mamba", tok_act, N_ACT, "MambaAct"),
            "MambaGap|adapted": make_seq_fit_score("mamba", tok_gap, 3 * N_GAPB, "MambaGap"),
            "LSTMGap|adapted": make_seq_fit_score("lstm", tok_gap, 3 * N_GAPB, "LSTMGap")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default=DEFAULT_MODELS,
                   help="comma-separated subset of BLOC|tfidf_lr, ActionLSTM, MambaAct, MambaGap|adapted, LSTMGap|adapted")
    p.add_argument("--outname", default="xplat_seq_baselines", help="output basename inside IO_RESULTS_DIR")
    p.add_argument("--extra-ops", action="store_true",
                   help="also fit each baseline on TW:Iran and TW:China vs organic Twitter and apply it to Reddit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log("loading populations")
    POPS = {
        "TW:IRA": load_accounts(io_path(SEQ_FILES["tw_t"]), MINEV_BIG),
        "TW:organic": load_accounts(io_path(SEQ_FILES["tw_o"]), MINEV_BIG),
    }
    POPS["RD:IRA"], POPS["RD:organic"] = load_reddit_cohort()
    log("sizes: " + ", ".join(f"{k} {len(v)}" for k, v in POPS.items()))

    # archived splits
    arch = load_archived_splits(usecols=("pop", "user", "split", "n_events"))
    SPLIT = {p: {s: archived_split_users(arch, p, s, present_in=POPS[p]) for s in ("train", "test")} for p in POPS}
    log("splits: " + ", ".join(f"{p} {len(v['train'])}/{len(v['test'])}" for p, v in SPLIT.items()))

    # matched Reddit controls: replay rep_task_matrix's rng consumption exactly
    rd_pos = sorted(POPS["RD:IRA"])
    rd_neg = sorted(POPS["RD:organic"])
    nev = {u: len(POPS[p][u][0]) for p in ("RD:IRA", "RD:organic") for u in POPS[p]}
    rd_neg_matched = matched_reddit_controls(arch, rd_pos, rd_neg, nev)
    log(f"RD cohort: {len(rd_pos)} IRA, {len(rd_neg)} organic, {len(rd_neg_matched)} matched (archive: 168)")

    # platform-adapted gap deciles (label-free platform reference: organic Twitter train accounts; all organic Reddit)
    EDGES = {"TW": decile_edges(np.concatenate([gaps_min(POPS["TW:organic"][u][0])[1:] for u in SPLIT["TW:organic"]["train"]]), N_GAPB),
             "RD": decile_edges(np.concatenate([gaps_min(POPS["RD:organic"][u][0])[1:] for u in POPS["RD:organic"]]), N_GAPB)}
    log(f"adapted deciles: TW {np.round(EDGES['TW'], 2).tolist()} RD {np.round(EDGES['RD'], 2).tolist()}")

    MODELS = build_models(POPS, EDGES)
    SEL = [m for m in args.models.split(",") if m in MODELS]
    OUTNAME = args.outname
    SUBSHARE = {u: float((POPS[p][u][1] == 0).mean()) for p in ("RD:IRA", "RD:organic") for u in POPS[p]}

    # ---------------------------------------------------------------- evaluation
    def K(p, keys):
        return [(p, u) for u in keys]

    tw_tr_pos, tw_tr_neg = K("TW:IRA", SPLIT["TW:IRA"]["train"]), K("TW:organic", SPLIT["TW:organic"]["train"])
    TEST = {"TW:IRA_test": K("TW:IRA", SPLIT["TW:IRA"]["test"]), "TW:organic_test": K("TW:organic", SPLIT["TW:organic"]["test"]),
            "RD:IRA": K("RD:IRA", rd_pos), "RD:organic_all": K("RD:organic", rd_neg), "RD:organic_matched": K("RD:organic", rd_neg_matched)}

    rd_users = rd_pos + rd_neg
    rd_y = np.r_[np.ones(len(rd_pos)), np.zeros(len(rd_neg))].astype(int)
    rd_pop = ["RD:IRA"] * len(rd_pos) + ["RD:organic"] * len(rd_neg)
    # rd_inplatform builds its cohort in the same (troll-then-organic, groupby-sorted) order
    splits = reddit_cv_splits(rd_y)[:5]

    results, scores = {}, {}
    for name in SEL:
        fit_score = MODELS[name]
        log(f"=== {name}: fit on Twitter ===")
        S = fit_score(tw_tr_pos, tw_tr_neg, TEST)
        rd_all = np.r_[S["RD:IRA"], S["RD:organic_all"]]
        rd_keys = [u for _, u in TEST["RD:IRA"]] + [u for _, u in TEST["RD:organic_all"]]
        res = {"TW->TW": auc_ci(S["TW:IRA_test"], S["TW:organic_test"]),
               "TW->RD|matched": auc_ci(S["RD:IRA"], S["RD:organic_matched"]),
               "TW->RD|unmatched": auc_ci(S["RD:IRA"], S["RD:organic_all"]),
               "RD_spearman_vs_submission_share": round(float(spearmanr(rd_all, [SUBSHARE[u] for u in rd_keys]).statistic), 3)}
        log(f"{name}: " + "; ".join(f"{k} {v}" for k, v in res.items()))
        for k, v in S.items():
            scores[f"{name}|TWfit|{k}"] = dict(zip([u for _, u in TEST[k]], v))
        log(f"=== {name}: 5-fold CV within Reddit ===")
        oof = np.full(len(rd_y), np.nan)
        for k, (tr, te) in enumerate(splits):
            trp = [(rd_pop[i], rd_users[i]) for i in tr if rd_y[i] == 1]
            trn = [(rd_pop[i], rd_users[i]) for i in tr if rd_y[i] == 0]
            S = fit_score(trp, trn, {"te": [(rd_pop[i], rd_users[i]) for i in te]})
            oof[te] = S["te"]
            log(f"  fold {k}: AUC {roc_auc_score(rd_y[te], oof[te]):.3f}")
        res["RD->RD|5fold"] = auc_ci(oof[rd_y == 1], oof[rd_y == 0])
        res["RD->RD|per_fold"] = [round(float(roc_auc_score(rd_y[te], oof[te])), 4) for _, te in splits]
        log(f"{name}: RD->RD {res['RD->RD|5fold']}")
        scores[f"{name}|RDcv|oof"] = dict(zip(rd_users, oof))
        results[name] = res

    results["_protocol"] = {"tw_train": [len(tw_tr_pos), len(tw_tr_neg)], "tw_test": [len(TEST["TW:IRA_test"]), len(TEST["TW:organic_test"])],
                            "rd": [len(rd_pos), len(rd_neg), len(rd_neg_matched)], "bloc_pause_edges_min": PAUSE_EDGES_MIN.tolist(),
                            "lstm": "emb32-LSTM64-meanpool, 512-action windows, <=20/account, AdamW 1e-3, 3 epochs (>=150 steps), account = mean window prob"}
    json.dump(results, open(io_path(f"{OUTNAME}.json"), "w"), indent=1)
    pd.DataFrame({k: pd.Series(v) for k, v in scores.items()}).to_csv(io_path(f"{OUTNAME}_scores.csv"))
    log(f"DONE -> {OUTNAME}.json, {OUTNAME}_scores.csv")

    # ---------------------------------------------------------------- other source operations (--extra-ops)
    # Same check the paper applies to the action-mix score: fit the baseline on a
    # different operation (China, Iran) vs organic Twitter and apply it to Reddit.
    if args.extra_ops:
        POPS["TW:Iran"] = parse_takedown_actions(takedown_paths(IRAN_FILES), MINEV_BIG, log=log)
        POPS["TW:China"] = parse_takedown_actions(takedown_paths(CHINA_FILES), MINEV_BIG, log=log)
        for op in ("TW:Iran", "TW:China"):
            tr = archived_split_users(arch, op, "train", present_in=POPS[op])
            te = archived_split_users(arch, op, "test", present_in=POPS[op])
            log(f"{op}: {len(POPS[op])} accounts, archived split {len(tr)}/{len(te)}")
            for name in SEL:
                fit_score = MODELS[name]
                S = fit_score(K(op, tr), tw_tr_neg, {"own_test": K(op, te), "TW:organic_test": TEST["TW:organic_test"],
                                                      "RD:IRA": TEST["RD:IRA"], "RD:organic_all": TEST["RD:organic_all"],
                                                      "RD:organic_matched": TEST["RD:organic_matched"]})
                r = {"TW->TW(own)": auc_ci(S["own_test"], S["TW:organic_test"]),
                     "TW->RD|matched": auc_ci(S["RD:IRA"], S["RD:organic_matched"]),
                     "TW->RD|unmatched": auc_ci(S["RD:IRA"], S["RD:organic_all"])}
                results[name][f"fit_on_{op}"] = r
                log(f"{name} fitted on {op}: " + "; ".join(f"{k} {v}" for k, v in r.items()))
        json.dump(results, open(io_path(f"{OUTNAME}.json"), "w"), indent=1)
        log(f"DONE (extra ops) -> {OUTNAME}.json")
    return 0


if __name__ == "__main__":
    main()
