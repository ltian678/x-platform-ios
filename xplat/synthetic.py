"""Synthetic stand-in data so the framework can be exercised without the raw archives.

Writes small, randomly generated files with the exact layout and columns the
experiments read (per-account sequence files, membership tables, gap edges,
language table, a stand-in archived split, Telegram and Facebook inputs) into
``IO_RESULTS_DIR``, and takedown-style tweet CSVs into ``TWITTER_TAKEDOWN_DIR``.
The generated populations differ in phase, schedule, and tempo so classifiers
have something to learn, but nothing about them resembles the real data and
no number computed from them is meaningful.

    python -m xplat.synthetic [--out-dir DIR] [--takedown-dir DIR] [--seed N] [--scale F]

Not generated (only the prior-work content baselines need them): the organic
timeline ``.json.bz2`` files and the Reddit recrawl folders.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import config
from .features import account_features

# population -> (n_accounts, hour profile, weekday weight, gap regime)
# gap regimes: "bursty" (sub-minute bursts within office hours) or "diffuse".
POPS = {
    "tw_ira": dict(n=120, hours=(6, 16), wk=0.85, regime="bursty", year=(2015, 2017), n_ev=(60, 400)),
    "tw_org": dict(n=150, hours=(0, 24), wk=0.72, regime="diffuse", year=(2016, 2016), n_ev=(60, 400)),
    "rd_ira": dict(n=40, hours=(6, 16), wk=0.85, regime="bursty", year=(2015, 2018), n_ev=(12, 300)),
    "rd_org": dict(n=120, hours=(0, 24), wk=0.72, regime="diffuse", year=(2015, 2018), n_ev=(12, 400)),
}
TAKEDOWN_POPS = {   # file base name -> (prefix, n accounts, year range)
    "iranian_tweets_csv_unhashed.csv": ("iran", 60, (2016, 2018)),
    "venezuela_201906_1_tweets_csv_unhashed.csv": ("venz", 40, (2017, 2018)),
    "venezuela_201901_1_tweets_csv_unhashed_1.csv": ("venz", 20, (2017, 2018)),
    "venezuela_201901_1_tweets_csv_unhashed_2.csv": ("venz", 20, (2017, 2018)),
    "venezuela_201901_1_tweets_csv_unhashed_3.csv": ("venz", 20, (2017, 2018)),
}
BLM = ["blacklivesmatter", "blm", "blacktwitter"]
MAGA = ["maga", "tcot", "trump2016"]
ACTORS = {"Russia": "TASS news", "Iran": "PressTV English", "Syria": "SANA Syria", "Yemen": "Al Masirah", "Palestine": "Quds News"}


def _hour_weights(hours):
    w = np.full(24, 0.02)
    lo, hi = hours
    w[lo:hi] = 1.0
    return w / w.sum()


def _timestamps(rng, n, spec, year):
    """n UTC seconds drawn from a day/hour/gap model."""
    y0, y1 = year
    start = pd.Timestamp(f"{y0}-01-01").value // 10**9
    end = pd.Timestamp(f"{y1}-12-31").value // 10**9
    hw = _hour_weights(spec["hours"])
    ts = []
    t = start + rng.integers(0, max(1, (end - start) // 2))
    while len(ts) < n:
        day = (t // 86400) * 86400
        dow = pd.Timestamp(day, unit="s").dayofweek
        if dow >= 5 and rng.random() < spec["wk"]:
            t = day + 86400
            continue
        hour = rng.choice(24, p=hw)
        k = rng.integers(1, 6) if spec["regime"] == "bursty" else rng.integers(1, 3)
        base = day + hour * 3600 + rng.integers(0, 3600)
        for _ in range(k):
            if spec["regime"] == "bursty":
                base += int(rng.choice([20, 45, 90, 600, 1800]))
            else:
                base += int(rng.exponential(5400)) + 60
            ts.append(base)
        t = base + int(rng.exponential(2 * 86400 if spec["regime"] == "diffuse" else 6 * 3600)) + 60
        if t > end:
            t = start + rng.integers(0, (end - start) // 2)
    ts = np.sort(np.array(ts[:n], dtype=np.int64))
    return ts


def make_population(rng, key, spec, prefix):
    rows = []
    accounts = {}
    n_ev = np.exp(rng.uniform(np.log(spec["n_ev"][0]), np.log(spec["n_ev"][1]), spec["n"])).astype(int)
    for i in range(spec["n"]):
        user = f"{prefix}_{i:04d}"
        ts = _timestamps(rng, int(n_ev[i]), spec, spec["year"])
        p_act = [0.35, 0.5, 0.15] if spec["regime"] == "bursty" else [0.5, 0.4, 0.1]
        ac = rng.choice(3, size=len(ts), p=p_act)
        accounts[user] = (ts, ac)
        rows += [(user, int(t), int(a)) for t, a in zip(ts, ac)]
    return pd.DataFrame(rows, columns=["user", "ts", "action"]), accounts


def write_membership(path, accounts, window=None, minev=10):
    rows = []
    for u, (ts, ac) in accounts.items():
        if window is not None:
            m = (ts >= window[0]) & (ts <= window[1])
            ts, ac = ts[m], ac[m]
        if len(ts) < minev:
            continue
        rows.append({"user": u, **account_features(ts, ac)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return [r["user"] for r in rows]


def takedown_frame(rng, users, accounts, hashtags=None):
    rows = []
    for u in users:
        ts, ac = accounts[u]
        for t, a in zip(ts, ac):
            tags = []
            if hashtags is not None and u in hashtags:
                tags = [str(t) for t in rng.choice(hashtags[u], size=rng.integers(1, 3))]
            rows.append({
                "user_screen_name": u,
                "tweet_time": pd.Timestamp(int(t), unit="s").strftime("%Y-%m-%d %H:%M"),
                "tweet_text": f"synthetic post {rng.integers(0, 10**6)} http://example.org/{rng.integers(0, 999)}",
                "hashtags": str(tags) if tags else "[]",
                "urls": "['http://example.org']",
                "is_retweet": "true" if a == 2 and rng.random() < 0.7 else "false",
                "in_reply_to_tweetid": str(rng.integers(10**9, 10**10)) if a == 1 else "",
                "quoted_tweet_tweetid": str(rng.integers(10**9, 10**10)) if a == 2 and rng.random() < 0.3 else "",
            })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=config.IO_RESULTS_DIR, help="working directory (IO_RESULTS_DIR)")
    ap.add_argument("--takedown-dir", default=config.TWITTER_TAKEDOWN_DIR, help="TWITTER_TAKEDOWN_DIR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every account count")
    ap.add_argument("--skip-takedown", action="store_true", help="do not write takedown-style tweet CSVs")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    pops = {k: dict(v, n=max(8, int(v["n"] * args.scale))) for k, v in POPS.items()}

    # --- per-account sequences and membership tables
    seq, acc = {}, {}
    for key, spec in pops.items():
        seq[key], acc[key] = make_population(rng, key, spec, key)
    seq["tw_ira"].to_csv(os.path.join(out, config.SEQ_FILES["tw_t"]), index=False)
    seq["tw_org"].to_csv(os.path.join(out, config.SEQ_FILES["tw_o"]), index=False)
    seq["rd_ira"].to_csv(os.path.join(out, config.SEQ_FILES["rd_t"]), index=False)
    seq["rd_org"].to_csv(os.path.join(out, config.SEQ_FILES["rd_o"]), index=False)
    members = {
        "tw_t": write_membership(os.path.join(out, config.FEAT_FILES["tw_t"]), acc["tw_ira"]),
        "tw_o": write_membership(os.path.join(out, config.FEAT_FILES["tw_o"]), acc["tw_org"]),
        "rd_t": write_membership(os.path.join(out, config.FEAT_FILES["rd_t"]), acc["rd_ira"], config.RD_WIN),
        "rd_o": write_membership(os.path.join(out, config.FEAT_FILES["rd_o"]), acc["rd_org"], config.RD_WIN),
    }

    # --- archived Twitter gap-decile edges (minutes) and IRA language table
    gaps = np.concatenate([np.diff(ts) / 60.0 for ts, _ in acc["tw_org"].values()])
    edges = np.unique(np.quantile(gaps, np.arange(1, 10) / 10))
    json.dump({"tw_edges_min": edges.tolist()}, open(os.path.join(out, "china_policy_edges.json"), "w"), indent=1)
    ru = rng.random(len(acc["tw_ira"]))
    pd.DataFrame({"user": list(acc["tw_ira"]), "ru_frac": np.where(ru < 0.5, 0.9, 0.1),
                  "en_frac": np.where(ru < 0.5, 0.1, 0.9)}).to_csv(os.path.join(out, "ira_twitter_account_langs.csv"), index=False)

    # --- takedown-style tweet CSVs (IRA + other operations + China eras)
    td_pops = {}
    if not args.skip_takedown:
        tc = os.path.join(args.takedown_dir, "tweet_csvs")
        os.makedirs(tc, exist_ok=True)
        ira_users = list(acc["tw_ira"])
        tags = {}
        for i, u in enumerate(ira_users):
            if i % 4 == 0:
                tags[u] = BLM
            elif i % 4 == 1:
                tags[u] = MAGA
        takedown_frame(rng, ira_users, acc["tw_ira"], tags).to_csv(os.path.join(tc, config.IRA_TWEETS_FILE), index=False)
        for fname, (prefix, n, year) in TAKEDOWN_POPS.items():
            spec = dict(pops["tw_ira"], n=max(6, int(n * args.scale)), year=year)
            _, a = make_population(rng, prefix, spec, f"{prefix}_{fname.split('_')[1]}")
            td_pops.setdefault(prefix, {}).update(a)
            takedown_frame(rng, list(a), a).to_csv(os.path.join(tc, fname), index=False)
        # China: accounts active across 2017..2020 so the era analysis has stayers
        china = {}
        for i in range(max(12, int(80 * args.scale))):
            spec = dict(pops["tw_ira"], n_ev=(150, 600))
            years = [y for y in (2017, 2018, 2019, 2020) if rng.random() < 0.75] or [2018]
            ts = np.sort(np.concatenate([_timestamps(rng, int(rng.integers(40, 150)), spec, (y, y)) for y in years]))
            china[f"china_{i:04d}"] = (ts, rng.choice(3, size=len(ts)))
        td_pops["china"] = china
        users = list(china)
        chunks = np.array_split(np.array(users), len(config.CHINA_FILES))
        for fname, chunk in zip(config.CHINA_FILES, chunks):
            takedown_frame(rng, list(chunk), china).to_csv(os.path.join(tc, fname), index=False)
        gru_dir = os.path.join(args.takedown_dir, "2020_12", "GRU_202012")
        os.makedirs(gru_dir, exist_ok=True)
        _, gru = make_population(rng, "gru", dict(pops["tw_ira"], n=max(6, int(15 * args.scale)), year=(2019, 2020)), "gru")
        td_pops["gru"] = gru
        takedown_frame(rng, list(gru), gru).to_csv(os.path.join(gru_dir, "GRU_202012_tweets_csv_unhashed.csv"), index=False)

    # --- stand-in for the archived split written by rep_task_matrix (pop, user, split, n_events)
    rows = []
    def add(pop, accounts, test_only=False):
        keys = np.array(sorted(accounts))
        perm = rng.permutation(len(keys))
        n_test = len(keys) if test_only else int(round(0.3 * len(keys)))
        test = set(keys[perm[:n_test]])
        for u in keys:
            rows.append({"pop": pop, "user": u, "split": "test" if u in test else "train", "n_events": len(accounts[u][0])})
    add("TW:IRA", acc["tw_ira"]); add("TW:organic", acc["tw_org"])
    for pop, prefix in (("TW:Iran", "iran"), ("TW:Venezuela", "venz"), ("TW:China", "china")):
        if prefix in td_pops:
            add(pop, td_pops[prefix])
    if "gru" in td_pops:
        add("TW:GRU", td_pops["gru"], test_only=True)
    add("RD:IRA", {u: acc["rd_ira"][u] for u in members["rd_t"]}, test_only=True)
    add("RD:organic", {u: acc["rd_org"][u] for u in members["rd_o"]}, test_only=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "rep_task_scores.csv"), index=False)

    # --- Telegram inputs (2024 window) and Facebook inputs for the point-process pipeline
    tg_rows, meta, disc, edges_rows = [], [], [], []
    n_tg = max(30, int(120 * args.scale))
    spec = dict(pops["tw_org"], year=(2024, 2024), n_ev=(120, 600))
    for i in range(n_tg):
        ch = f"tg{i:04d}"
        actor = list(ACTORS)[i % len(ACTORS)] if i < 15 else None
        ts = _timestamps(rng, int(rng.integers(120, 600)), spec if i % 3 else dict(spec, hours=(6, 16), regime="bursty"), (2024, 2024))
        ac = rng.choice(3, size=len(ts), p=[0.8, 0.1, 0.1])
        tg_rows += [(ch, int(t), int(a)) for t, a in zip(ts, ac)]
        title = ACTORS[actor] if actor else f"Channel {i}"
        if 15 <= i < 22:
            title = f"Tasnim affiliate {i}"
        meta.append({"channel": ch, "title": title, "username": f"user{i}", "broadcast": int(actor is None),
                     "verified": int(i % 5 == 0)})
        disc.append({"channel": ch, "political_share": float(rng.random())})
        for s in rng.choice(50, size=rng.integers(3, 8), replace=False):
            edges_rows.append({"src": f"src{s}", "dst": ch})
    pd.DataFrame(tg_rows, columns=["user", "ts", "action"]).to_csv(os.path.join(out, "seq_tg_sample.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(out, "tg_channels.csv"), index=False)
    pd.DataFrame(disc).to_csv(os.path.join(out, "tg_discovery_scores.csv"), index=False)
    pd.DataFrame(edges_rows).to_csv(os.path.join(out, "tg_forward_edges.csv"), index=False)
    fb_rows, labels = [], []
    for i in range(max(12, int(40 * args.scale))):
        page = f"fbpage{i:04d}"
        ts = _timestamps(rng, int(rng.integers(60, 300)), dict(spec, year=(2020, 2022)), (2020, 2022))
        fb_rows += [(page, int(t), 0) for t in ts]
        labels.append({"page": page, "label": "china_state" if i % 2 else "pacific_local"})
    pd.DataFrame(fb_rows, columns=["user", "ts", "action"]).to_csv(os.path.join(out, "seq_fb_din.csv"), index=False)
    pd.DataFrame(labels).to_csv(os.path.join(out, "din_labels.csv"), index=False)
    ct_rows = []
    for i in range(max(12, int(60 * args.scale))):
        ts = _timestamps(rng, int(rng.integers(60, 300)), dict(spec, year=(2020, 2022)), (2020, 2022))
        ct_rows += [(f"ctpage{i:04d}", int(t), 0) for t in ts]
    pd.DataFrame(ct_rows, columns=["user", "ts", "action"]).to_csv(os.path.join(out, "seq_fb_ctpages.csv"), index=False)

    print(f"synthetic data written to {out}" + ("" if args.skip_takedown else f" and {args.takedown_dir}"), flush=True)
    print({k: len(v) for k, v in members.items()}, flush=True)
    return 0


if __name__ == "__main__":
    main()
