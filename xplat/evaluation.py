"""Account-level evaluation: bootstrap AUC intervals and volume matching."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import N_BOOT, SEED

# 0.25-dex log10 event-volume bins used for matched organic controls.
VOLUME_BINS = np.arange(0.5, 6.5, 0.25)
_MATCH_SHIFTS = (0, 1, -1, 2, -2, 3, -3, 4, -4)


def bootstrap_auc(pos_scores, neg_scores, n_boot=N_BOOT, rng=None, seed=SEED):
    """AUC with a percentile account-bootstrap interval.

    Returns ``(auc, lo, hi)`` as floats.  Positives and negatives are resampled
    separately with replacement, positives drawn first, so a fresh
    ``default_rng(seed)`` reproduces every archived interval; pass ``rng`` to
    draw from a shared generator instead.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    s = np.r_[pos_scores, neg_scores]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = roc_auc_score(y, s)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        draws[b] = roc_auc_score(y[idx], s[idx])
    return float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def auc_ci(pos_scores, neg_scores, n_boot=N_BOOT, rng=None, seed=SEED, ndigits=4):
    """``[auc, lo, hi]`` rounded, the list format written by the Table 2 scripts."""
    a, lo, hi = bootstrap_auc(pos_scores, neg_scores, n_boot=n_boot, rng=rng, seed=seed)
    return [round(a, ndigits), round(lo, ndigits), round(hi, ndigits)]


def auc_point(scores, pos_idx, neg_idx):
    """Plain AUC of ``scores`` for the accounts at ``pos_idx`` vs ``neg_idx``."""
    return roc_auc_score(np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))],
                         np.r_[scores[pos_idx], scores[neg_idx]])


def volume_match_positions(pos_volumes, neg_volumes, rng, n_per=5):
    """Volume-matched control selection (ablation protocol).

    Each positive account draws ``n_per`` negatives (with replacement) from the
    nearest non-empty 0.25-dex log-volume bin; the union of draws is returned
    as sorted positions into ``neg_volumes``.
    """
    tb = np.digitize(np.log10(np.clip(pos_volumes, 1, None)), VOLUME_BINS)
    ob = np.digitize(np.log10(np.clip(neg_volumes, 1, None)), VOLUME_BINS)
    idx = []
    for b in tb:
        for shift in _MATCH_SHIFTS:
            cand = np.flatnonzero(ob == b + shift)
            if len(cand):
                idx.extend(rng.choice(cand, n_per, replace=True))
                break
    return sorted(set(idx))


def volume_match(n_events, neg_idx, pos_idx, rng, n_per=5):
    """Index-array form: ``n_events`` is aligned with the frame the indices refer to."""
    n_events = np.asarray(n_events)
    keep = volume_match_positions(n_events[pos_idx], n_events[neg_idx], rng, n_per=n_per)
    return np.asarray(neg_idx)[keep]
