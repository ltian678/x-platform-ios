"""Per-account event sequences: loading, timestamp conversion, gaps.

A sequence file (``seq_*.csv``) has columns ``user, ts, action`` where
``action`` is 0 = post / submission, 1 = reply / comment, 2 = repost / quote.
Timestamps are either UTC Unix seconds or ISO strings; strings are converted
to seconds here.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def to_epoch_s(series):
    """ISO strings -> (UTC seconds floored to the second, valid mask)."""
    ts = pd.to_datetime(series, errors="coerce")
    ok = ts.notna().values
    return ts.dt.floor("s").values.astype("datetime64[s]").astype("int64"), ok


def read_sequence_frame(path, positive_only=True):
    """Read a ``seq_*.csv`` as a frame with integer-second ``ts``.

    ``positive_only=True`` (the profile / policy scripts) drops rows whose
    timestamp is not strictly positive; ``False`` reproduces the loader of the
    audience and error-analysis scripts, which only drop unparseable rows.
    """
    df = pd.read_csv(path)
    if not pd.api.types.is_numeric_dtype(df["ts"]):
        ts = pd.to_datetime(df["ts"], errors="coerce")
        df["ts"] = ts.dt.floor("s").astype("datetime64[s]").astype("int64")
        df = df[ts.notna() & (df["ts"] > 0)] if positive_only else df[ts.notna()]
    return df


def load_accounts(path, minev, members=None, window=None):
    """Per-account ``{user: (ts, actions)}`` from a sequence file.

    Accounts are restricted to ``members`` (if given) and to ``window``
    (inclusive UTC-second bounds, if given) and kept when they have at least
    ``minev`` events.  Events are stably sorted by timestamp.
    """
    df = read_sequence_frame(path, positive_only=True)
    df["user"] = df["user"].astype(str)
    out = {}
    for u, g in df.groupby("user"):
        if members is not None and u not in members:
            continue
        ts = g["ts"].values.astype(np.int64)
        ac = g["action"].values.astype(np.int64)
        if window is not None:
            m = (ts >= window[0]) & (ts <= window[1])
            ts, ac = ts[m], ac[m]
        if len(ts) < minev:
            continue
        o = np.argsort(ts, kind="stable")
        out[u] = (ts[o], ac[o])
    return out


def gaps_min(ts):
    """Inter-event gaps in minutes; the first entry is 0 (prepended)."""
    return np.diff(ts, prepend=ts[0]) / 60.0


def parse_takedown_actions(paths, minev, log=None):
    """Takedown archive CSVs -> ``{screen_name: (ts, actions)}`` with coarse actions.

    action 2 = retweet or quote, 1 = reply, 0 = original post.
    """
    seq = defaultdict(list)
    use = ["user_screen_name", "tweet_time", "is_retweet", "in_reply_to_tweetid", "quoted_tweet_tweetid"]
    for path in paths:
        for c in pd.read_csv(path, dtype=str, usecols=use, chunksize=1_000_000,
                             lineterminator="\n", on_bad_lines="skip"):
            secs, ok = to_epoch_s(c["tweet_time"])
            act = np.zeros(len(c), dtype=np.int8)
            act[(c["is_retweet"].str.lower() == "true").values | c["quoted_tweet_tweetid"].notna().values] = 2
            act[c["in_reply_to_tweetid"].notna().values] = 1
            df = pd.DataFrame({"sn": c["user_screen_name"].str.lower().values[ok], "ts": secs[ok], "a": act[ok]})
            for name, g in df.groupby("sn", sort=False):
                seq[name].append(g)
        if log is not None:
            log(f"parsed {path.split('/')[-1]}")
    out = {}
    for name, parts in seq.items():
        g = pd.concat(parts)
        if len(g) < minev:
            continue
        ts = g["ts"].values.astype(np.int64)
        ac = g["a"].values.astype(np.int64)
        o = np.argsort(ts, kind="stable")
        out[name] = (ts[o], ac[o])
    return out
