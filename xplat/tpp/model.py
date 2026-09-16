"""Interval-censored Mamba-TPP: windows, network, mixture likelihoods, training.

Stage B fits a per-platform baseline on minute-censored gaps; stage C maps
every entity's gaps through the baseline CDF with a randomized PIT to a
rescaled time ``tau`` with an Exp(1) null; stage D fits population policies
as log-normal-mixture densities on ``tau``.  ``mamba_ssm`` is imported lazily
when a ``TPP`` is constructed.
"""
from __future__ import annotations

import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .time_resolution import conservative_minute_bounds as bounds

WIN, MAXW, K, DM, BS, LR = 256, 20, 8, 64, 64, 1e-3
BASE_EPOCHS, POP_EPOCHS = 5, 3
MAX_STEPS, EVAL_EVERY, PATIENCE, VAL_CAP = 3000, 50, 8, 2000

NB = 32
BIN_EDGES = np.linspace(0, np.log1p(1e5), NB - 1)      # censored-gap input bins (minutes)
RB_EDGES = np.linspace(0, np.log1p(50.0), NB - 1)      # rescaled-time input bins
START = NB
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def in_bins(vals):
    return np.digitize(np.log1p(vals), BIN_EDGES).astype(np.int64)


def r_bins(v):
    return np.digitize(np.log1p(v), RB_EDGES).astype(np.int64)


def windows_of(d_arrays, maxw=MAXW):
    """Censored-gap windows ``(input bins, lower, upper, mask)`` from completed-minute gaps."""
    out = []
    wrng = np.random.default_rng(0)
    for d in d_arrays:
        if len(d) < 5:
            continue
        a, b = bounds(d)
        ib = np.concatenate([[START], in_bins(d[:-1])])
        starts = np.arange(0, max(len(d) - WIN, 0) + 1, WIN)
        if len(starts) > maxw:
            starts = wrng.choice(starts, maxw, replace=False)
        for s in starts:
            e = min(s + WIN, len(d))
            pad = WIN - (e - s)
            out.append((np.pad(ib[s:e], (0, pad), constant_values=START),
                        np.pad(a[s:e], (0, pad), constant_values=1.0),
                        np.pad(b[s:e], (0, pad), constant_values=2.0),
                        np.pad(np.ones(e - s), (0, pad))))
    return out


def r_windows(entries, maxw=MAXW):
    """Rescaled-time windows ``(input bins, tau, zeros, mask)`` from ``(tau, u)`` entries."""
    out = []
    wrng = np.random.default_rng(0)
    for tau, _ in entries:
        n = len(tau)
        if n < 5:
            continue
        t = np.maximum(tau, 1e-8)
        ib = np.concatenate([[START], r_bins(t[:-1])])
        starts = np.arange(0, max(n - WIN, 0) + 1, WIN)
        if len(starts) > maxw:
            starts = wrng.choice(starts, maxw, replace=False)
        for s in starts:
            e = min(s + WIN, n)
            pad = WIN - (e - s)
            out.append((np.pad(ib[s:e], (0, pad), constant_values=START),
                        np.pad(t[s:e], (0, pad), constant_values=1.0),
                        np.pad(np.zeros(e - s), (0, pad)),
                        np.pad(np.ones(e - s), (0, pad))))
    return out


def time_split(tau):
    n = len(tau)
    i, j = int(0.6 * n), int(0.8 * n)
    return tau[:i], tau[i:j], tau[j:]


class WData(Dataset):
    def __init__(self, w):
        self.w = w

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        ib, a, b, m = self.w[i]
        return (torch.from_numpy(ib), torch.from_numpy(a.astype(np.float32)),
                torch.from_numpy(b.astype(np.float32)), torch.from_numpy(m.astype(np.float32)))


class TPP(nn.Module):
    """Two residual Mamba blocks emitting a K-component log-normal mixture per step."""

    def __init__(self):
        super().__init__()
        from mamba_ssm import Mamba
        self.emb = nn.Embedding(NB + 1, DM)
        self.m1, self.m2 = Mamba(d_model=DM), Mamba(d_model=DM)
        self.n1, self.n2 = nn.LayerNorm(DM), nn.LayerNorm(DM)
        self.head = nn.Linear(DM, 3 * K)

    def forward(self, ib):
        h = self.emb(ib)
        h = h + self.m1(self.n1(h))
        h = h + self.m2(self.n2(h))
        return self.head(h)


def _wms(params):
    w = torch.log_softmax(params[..., :K], -1)
    mu = params[..., K:2 * K]
    s = torch.exp(torch.clamp(params[..., 2 * K:], -3.0, 3.0))
    return w, mu, s


def mix_cdf(params, x):
    w, mu, s = _wms(params)
    z = (torch.log(x.clamp_min(1e-6)).unsqueeze(-1) - mu) / s
    return (torch.exp(w) * torch.special.ndtr(z)).sum(-1)


def mix_logpdf(params, x):
    w, mu, s = _wms(params)
    lx = torch.log(x.clamp_min(1e-8)).unsqueeze(-1)
    lp = w - LOG_SQRT_2PI - torch.log(s) - lx - 0.5 * ((lx - mu) / s) ** 2
    return torch.logsumexp(lp, -1)


def loss_cens(params, a, b, m):
    F = (mix_cdf(params, b) - mix_cdf(params, a)).clamp_min(1e-9)
    return -(torch.log(F) * m).sum() / m.sum().clamp_min(1.0)


def loss_dens(params, x, _b, m):
    return -(mix_logpdf(params, x) * m).sum() / m.sum().clamp_min(1.0)


def artifact_path(out_dir, tag):
    return os.path.join(out_dir, f"tpp_{tag}.pt")


def train_tpp(wins, tag, epochs, out_dir, dens=False, dev=None):
    """Fixed-epoch fit (baselines); reloads ``tpp_{tag}.pt`` from ``out_dir`` if present."""
    dev = dev or device()
    path = artifact_path(out_dir, tag)
    model = TPP().to(dev)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=dev))
        return model
    dl = DataLoader(WData(wins), batch_size=BS, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = loss_dens if dens else loss_cens
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for ib, a, b, m in dl:
            ib, a, b, m = ib.to(dev), a.to(dev), b.to(dev), m.to(dev)
            loss = lossfn(model(ib), a, b, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"  {tag} ep{ep} nll/gap {tot / max(nb, 1):.4f}", flush=True)
    torch.save(model.state_dict(), path)
    return model


@torch.no_grad()
def eval_nll(model, wins, dens=False, dev=None):
    dev = dev or device()
    dl = DataLoader(WData(wins), batch_size=BS)
    tot, n = 0.0, 0.0
    for ib, a, b, m in dl:
        ib, a, b, m = ib.to(dev), a.to(dev), b.to(dev), m.to(dev)
        p = model(ib)
        if dens:
            nll = -mix_logpdf(p, a)
        else:
            nll = -torch.log((mix_cdf(p, b) - mix_cdf(p, a)).clamp_min(1e-9))
        tot += (nll * m).sum().item()
        n += m.sum().item()
    return tot / max(n, 1.0)


@torch.no_grad()
def rescale_entity(model, d, pit_rng, dev=None):
    """Randomized-PIT rescaling of one entity's completed-minute gaps -> ``(tau, u)``."""
    dev = dev or device()
    a, b = bounds(d)
    ib = np.concatenate([[START], in_bins(d[:-1])])
    ua_l, ub_l = [], []
    for s in range(0, len(d), 4096):
        e = min(s + 4096, len(d))
        p = model(torch.from_numpy(ib[s:e]).unsqueeze(0).to(dev))
        ua_l.append(mix_cdf(p, torch.tensor(a[s:e], dtype=torch.float32, device=dev).unsqueeze(0))[0].cpu().numpy())
        ub_l.append(mix_cdf(p, torch.tensor(b[s:e], dtype=torch.float32, device=dev).unsqueeze(0))[0].cpu().numpy())
    ua = np.clip(np.concatenate(ua_l), 0.0, 1 - 1e-7)
    ub = np.clip(np.concatenate(ub_l), 0.0, 1 - 1e-7)
    ub = np.maximum(ub, ua + 1e-9)
    u = np.clip(ua + pit_rng.random(len(ua)) * (ub - ua), 1e-9, 1 - 1e-7)
    return -np.log1p(-u), u


def train_population_policy(train_wins, val_wins, tag, out_dir, dev=None):
    """Early-stopped density fit on rescaled time (stage D).  Returns ``(model, info)``.

    Reloads ``tpp_{tag}.pt`` and its curve JSON from ``out_dir`` when both exist.
    """
    dev = dev or device()
    path = artifact_path(out_dir, tag)
    curve_path = os.path.join(out_dir, f"tpp_{tag}_curve.json")
    model = TPP().to(dev)
    if os.path.exists(path) and os.path.exists(curve_path):
        model.load_state_dict(torch.load(path, map_location=dev))
        return model, json.load(open(curve_path))
    val = val_wins
    if len(val) > VAL_CAP:
        vr = np.random.default_rng(0)
        val = [val[i] for i in vr.choice(len(val), VAL_CAP, replace=False)]
    dl = DataLoader(WData(train_wins), batch_size=BS, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, best_step, since, step, curve = float("inf"), None, 0, 0, 0, []
    done = False
    while not done:
        for ib, a, b, m in dl:
            ib, a, b, m = ib.to(dev), a.to(dev), b.to(dev), m.to(dev)
            loss = loss_dens(model(ib), a, b, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                v = eval_nll(model, val, dens=True, dev=dev)
                curve.append((step, round(loss.item(), 4), round(v, 4)))
                if v < best - 1e-4:
                    best, best_state, best_step, since = v, copy.deepcopy(model.state_dict()), step, 0
                else:
                    since += 1
                print(f"  {tag} step{step} train {loss.item():.4f} val {v:.4f} best {best:.4f}@{best_step}", flush=True)
            if step >= MAX_STEPS or since >= PATIENCE:
                done = True
                break
    model.load_state_dict(best_state)
    torch.save(best_state, path)
    info = {"best_val": round(best, 4), "best_step": best_step, "steps_run": step,
            "n_train_windows": len(train_wins), "curve": curve}
    json.dump(info, open(curve_path, "w"))
    return model, info
