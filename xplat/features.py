"""Production-signature (profile) features and feature groups.

``account_features`` is the full schema used by the component ablation; every
other profile script in the paper uses a subset of its keys with identical
definitions, so all of them import this one function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EPS
from .sequences import gaps_min, read_sequence_frame

HOURS = [f"hour_{h}" for h in range(24)]

FEATURE_GROUPS = {
    "G0_volume": ["per_day"],
    "G1_phase": HOURS + ["moscow_mass", "night_mass", "evening_mass"],
    "G2_schedule": ["weekday_ratio", "top6h_mass", "hour_entropy", "day_density"],
    "G3_tempo": ["median_gap_min", "p90_gap_min", "burstiness", "frac_gap_under_1min"],
    "G4_actionmix": ["frac_B", "frac_R", "frac_A"],
}
FEATURE_GROUPS["G1G2_phase_sched"] = FEATURE_GROUPS["G1_phase"] + FEATURE_GROUPS["G2_schedule"]
FEATURE_GROUPS["G1G2G3_plus_tempo"] = FEATURE_GROUPS["G1G2_phase_sched"] + FEATURE_GROUPS["G3_tempo"]
FEATURE_GROUPS["full_profile"] = FEATURE_GROUPS["G1G2G3_plus_tempo"] + FEATURE_GROUPS["G4_actionmix"]
FEATURE_GROUPS["tzfree_G2G3G4"] = (FEATURE_GROUPS["G2_schedule"] + FEATURE_GROUPS["G3_tempo"]
                                   + FEATURE_GROUPS["G4_actionmix"])

# Columns log1p-transformed before standardization.
LOGCOLS = {"median_gap_min", "p90_gap_min", "per_day"}


def account_features(ts, ac):
    """Full-schema per-account features from sorted UTC seconds and actions."""
    n = len(ts)
    f = {}
    f["frac_B"] = float((ac == 0).sum() / n)
    f["frac_R"] = float((ac == 1).sum() / n)
    f["frac_A"] = float((ac == 2).sum() / n)
    di = pd.DatetimeIndex(pd.to_datetime(ts, unit="s"))
    hh = np.bincount(di.hour, minlength=24).astype(float)
    hh /= hh.sum()
    for h in range(24):
        f[f"hour_{h}"] = float(hh[h])
    f["moscow_mass"] = float(hh[6:16].sum())     # 09-18 MSK = 06-15 UTC
    f["night_mass"] = float(hh[0:6].sum())
    f["evening_mass"] = float(hh[15:21].sum())
    f["hour_entropy"] = float(-(hh * np.log(hh + EPS)).sum())
    f["top6h_mass"] = float(np.sort(hh)[-6:].sum())
    dow = np.bincount(di.dayofweek, minlength=7).astype(float)
    wk = dow[:5].sum() / 5.0
    we = dow[5:].sum() / 2.0
    f["weekday_ratio"] = min(float(wk / (we + EPS)) if we > 0 else 5.0, 5.0)
    g = gaps_min(ts)[1:]
    f["median_gap_min"] = float(np.median(g)) if len(g) else 0.0
    f["p90_gap_min"] = float(np.percentile(g, 90)) if len(g) else 0.0
    f["burstiness"] = float((np.std(g) - np.mean(g)) / (np.std(g) + np.mean(g) + EPS)) if len(g) else 0.0
    f["frac_gap_under_1min"] = float((g < 1).mean()) if len(g) else 0.0
    days = di.normalize()
    nd = days.nunique()
    span = max(1, (days.max() - days.min()).days + 1)
    f["day_density"] = float(nd / span)
    f["per_day"] = float(n / nd)
    f["n_events"] = int(n)
    f["n_distinct_ts"] = int(len(np.unique(ts)))
    return f


def design_matrix(df, cols, logcols=LOGCOLS):
    """Select ``cols`` from a feature frame, log1p the ``logcols``, fill NaN with 0."""
    X = df.reindex(columns=cols).copy()
    for c in set(logcols) & set(cols):
        X[c] = np.log1p(X[c])
    return X.fillna(0).values


def featurize_population(seq_path, members, window=None, min_events=10):
    """Feature frame (one row per account) for the members found in a sequence file."""
    seq = read_sequence_frame(seq_path, positive_only=True)
    rows = []
    for u, g in seq.groupby("user"):
        if u not in members:
            continue
        g = g.sort_values("ts")
        ts = g["ts"].values.astype(np.int64)
        ac = g["action"].values.astype(np.int64)
        if window is not None:
            m = (ts >= window[0]) & (ts <= window[1])
            ts, ac = ts[m], ac[m]
        if len(ts) < min_events:
            continue
        rows.append({"user": u, **account_features(ts, ac)})
    return pd.DataFrame(rows)
