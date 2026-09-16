"""Action-gap next-token policy: tokenizer, 2-layer Mamba language model, scoring.

Tokens: 3 actions x ``N_GAP`` platform-relative gap deciles (30 tokens), a
padding token, 512-token windows, at most ``MAX_WIN_PER_ACC`` windows per
account.  Account score under two policies = mean log-likelihood difference.
``mamba_ssm`` is imported lazily when a ``Policy`` is constructed, so this
module imports on CPU-only machines.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import SEED, io_path
from .log import log
from .sequences import gaps_min

N_GAP, VOCAB, PAD, WIN, MAX_WIN_PER_ACC = 10, 30, 30, 512, 20
SCORE_CAP = 5000            # events per account when scoring (archive protocol)
BATCH_SIZE, LR, WEIGHT_DECAY = 48, 1e-3, 0.01
TW_EDGES_FILE = "china_policy_edges.json"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tw_edges():
    """Archived Twitter gap-decile edges (minutes) shared by every policy row."""
    return np.array(json.load(open(io_path(TW_EDGES_FILE)))["tw_edges_min"])


def decile_edges(gaps, n_gap=N_GAP):
    """Empirical decile edges of a gap sample; duplicate edges merged."""
    return np.unique(np.quantile(gaps, np.arange(1, n_gap) / n_gap))


def tokenize(ts, ac, edges):
    b = np.clip(np.searchsorted(edges, gaps_min(ts), side="right"), 0, N_GAP - 1)
    return (ac * N_GAP + b).astype(np.int64)


def windows(pop, keys, edges, seed=SEED):
    """Non-overlapping ``WIN``-token windows, at most ``MAX_WIN_PER_ACC`` per account."""
    out = []
    r = np.random.default_rng(seed)
    for u in keys:
        ts, ac = pop[u]
        t = tokenize(ts, ac, edges)
        if len(t) <= WIN:
            out.append(t)
        else:
            starts = np.arange(0, len(t) - WIN + 1, WIN)
            if len(starts) > MAX_WIN_PER_ACC:
                starts = r.choice(starts, MAX_WIN_PER_ACC, replace=False)
            out += [t[s:s + WIN] for s in starts]
    return out


class LMData(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.from_numpy(self.seqs[i].copy())


def collate(batch):
    L = max(len(t) for t in batch)
    x = torch.full((len(batch), L), PAD, dtype=torch.long)
    for i, t in enumerate(batch):
        x[i, :len(t)] = t
    return x


class Policy(nn.Module):
    """Two residual Mamba blocks with pre-norm over a token embedding."""

    def __init__(self, d=128, layers=2):
        super().__init__()
        from mamba_ssm import Mamba
        self.emb = nn.Embedding(VOCAB + 1, d, padding_idx=PAD)
        self.blocks = nn.ModuleList([Mamba(d_model=d) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        h = self.emb(x)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))
        return self.head(h)


def train_policy(seqs, name, epochs=3, min_steps=None, log_every=1, seed=SEED, dev=None):
    """Fit a policy on token windows.  Returns ``(model, last_epoch_xent)``.

    ``min_steps`` raises the epoch count so that at least that many optimizer
    steps are taken (used for the small within-Reddit folds).
    """
    dev = dev or device()
    steps_per_epoch = math.ceil(len(seqs) / BATCH_SIZE)
    if min_steps is not None:
        epochs = max(epochs, math.ceil(min_steps / steps_per_epoch))
    log(f"training policy {name}: {len(seqs)} windows, {epochs} epochs ({steps_per_epoch} steps/epoch)")
    torch.manual_seed(seed)
    model = Policy().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)
    dl = DataLoader(LMData(seqs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    last = None
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for x in dl:
            x = x.to(dev)
            loss = lossf(model(x[:, :-1]).reshape(-1, VOCAB), x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        last = tot / nb
        if ep % log_every == 0 or ep == epochs - 1:
            log(f"  {name} epoch {ep}: xent {last:.4f} (ppl {np.exp(last):.2f})")
    model.eval()
    return model, last


def account_loglik(model, tokens, dev=None):
    """Mean next-token log-likelihood of an account's token sequence under ``model``."""
    dev = dev or device()
    lls = []
    with torch.no_grad():
        for s in range(0, max(1, len(tokens) - 1), WIN):
            seg = tokens[s:s + WIN + 1]
            if len(seg) < 2:
                continue
            x = torch.from_numpy(seg[None, :-1].copy()).to(dev)
            tgt = torch.from_numpy(seg[None, 1:].copy()).to(dev)
            logp = torch.log_softmax(model(x), dim=-1)
            lls.append(logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy().ravel())
    return float(np.concatenate(lls).mean()) if lls else 0.0


def policy_artifact(name: str) -> str:
    """Path of an archived policy state dict written by ``rep_task_matrix``."""
    return io_path(f"rep_task_policy_{name.replace(':', '_')}.pt")


def load_policy(name: str, dev=None):
    dev = dev or device()
    m = Policy().to(dev)
    m.load_state_dict(torch.load(policy_artifact(name), map_location=dev))
    m.eval()
    return m
