"""Timestamp-only sequence stores for the point-process pipeline."""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

MINEV = 50


def pack_save(path, d):
    users = list(d)
    lens = np.array([len(d[u]) for u in users])
    np.savez_compressed(path, users=np.array(users, dtype=object),
                        lens=lens, ts=np.concatenate([d[u] for u in users]))


def pack_load(path):
    z = np.load(path, allow_pickle=True)
    out, off = {}, 0
    for u, L in zip(z["users"], z["lens"]):
        out[str(u)] = z["ts"][off:off + L]
        off += L
    return out


def load_seq_csv(path):
    """Sequence frame with integer-second ``ts`` (unparseable rows dropped)."""
    df = pd.read_csv(path)
    if not pd.api.types.is_numeric_dtype(df["ts"]):
        t = pd.to_datetime(df["ts"], errors="coerce")
        df["ts"] = t.astype("datetime64[s]").astype("int64")
        df = df[t.notna()]
    return df


def acc_ts(df, minev=MINEV, ucol="user"):
    """``{entity: sorted distinct timestamps}`` for entities with >= ``minev`` of them."""
    out = {}
    for u, g in df.groupby(ucol):
        ts = np.sort(np.unique(g["ts"].values.astype(np.int64)))
        if len(ts) >= minev:
            out[str(u)] = ts
    return out


def parse_takedown_ts(paths, cache, minev=MINEV):
    """Distinct timestamps per screen name from takedown CSVs, cached as .npz."""
    if os.path.exists(cache):
        return pack_load(cache)
    seq = defaultdict(list)
    for path in paths:
        print("parsing", os.path.basename(path), flush=True)
        for c in pd.read_csv(path, dtype=str, usecols=["user_screen_name", "tweet_time"],
                             chunksize=1_000_000, lineterminator="\n", on_bad_lines="skip"):
            ts = pd.to_datetime(c["tweet_time"], errors="coerce")
            ok = ts.notna().values
            df = pd.DataFrame({"sn": c["user_screen_name"].str.lower().values[ok],
                               "ts": ts.values.astype("datetime64[s]").astype("int64")[ok]})
            for name, g in df.groupby("sn", sort=False):
                seq[name].append(g["ts"].values)
    out = {}
    for name, parts in seq.items():
        ts = np.sort(np.unique(np.concatenate(parts)))
        if len(ts) >= minev:
            out[name] = ts
    pack_save(cache, out)
    return out
