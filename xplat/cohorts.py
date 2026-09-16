"""Cohorts shared across experiments.

* the archived 70/30 Twitter account split written by ``rep_task_matrix``
* the common Reddit transfer cohort (96 IRA / 643 organic, window, >=10 events)
* the volume-matched Reddit controls, obtained by replaying the random-number
  consumption of ``rep_task_matrix`` so every script selects the same 168
* the repeated stratified folds shared by every within-Reddit row
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import FEAT_FILES, RD_WIN, SEED, SEQ_FILES, io_path
from .evaluation import volume_match_positions
from .sequences import load_accounts

# Order in which rep_task_matrix permutes its Twitter populations before matching.
REP_TASK_SPLIT_ORDER = ("TW:IRA", "TW:organic", "TW:Iran", "TW:Venezuela", "TW:China", "TW:GRU")
ARCHIVED_SCORES = "rep_task_scores.csv"


def load_archived_splits(usecols=("pop", "user", "split")):
    """The account table written by ``rep_task_matrix`` (pop, user, split, ...)."""
    return pd.read_csv(io_path(ARCHIVED_SCORES), usecols=list(usecols), dtype={"user": str})


def archived_split_users(arch, pop, split, present_in=None):
    users = arch[(arch["pop"] == pop) & (arch["split"] == split)]["user"].tolist()
    if present_in is not None:
        users = [u for u in users if u in present_in]
    return users


def reddit_members(cast_str=True):
    """Membership sets of the Reddit cohort from the archived feature tables."""
    out = {}
    for key in ("rd_t", "rd_o"):
        col = pd.read_csv(io_path(FEAT_FILES[key]))["user"]
        out[key] = set(col.astype(str)) if cast_str else set(col)
    return out


def load_reddit_cohort(minev=10, window=RD_WIN):
    """``(RD_IRA, RD_organic)`` account dicts of the common transfer cohort."""
    m = reddit_members()
    rd_t = load_accounts(io_path(SEQ_FILES["rd_t"]), minev, m["rd_t"], window)
    rd_o = load_accounts(io_path(SEQ_FILES["rd_o"]), minev, m["rd_o"], window)
    return rd_t, rd_o


def matched_reddit_controls(arch, rd_pos, rd_neg, n_events, seed=SEED, n_per=5):
    """Replay ``rep_task_matrix``'s generator, then volume-match ``rd_neg`` to ``rd_pos``.

    ``arch`` is the archived split table; ``n_events`` maps user -> event count.
    Returns the matched negative keys in sorted-position order.
    """
    rng = np.random.default_rng(seed)
    for p in REP_TASK_SPLIT_ORDER:
        rng.permutation(int((arch["pop"] == p).sum()))
    keep = volume_match_positions([n_events[u] for u in rd_pos], [n_events[u] for u in rd_neg], rng, n_per=n_per)
    return [rd_neg[i] for i in keep]


def reddit_cv_splits(y, n_splits=5, n_repeats=10, random_state=0):
    """The shared RepeatedStratifiedKFold folds of the within-Reddit rows."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return list(rskf.split(np.zeros(len(y)), y))
