"""Takedown-bridged organizations on Telegram vs matched controls.

Positive set = 2024 Telegram channels of organizations documented in the
takedown archives' amplified layer (Iranian agencies: Press TV, Tasnim, IRNA,
ISNA, SNN, Mehr News; 836 covert archive links), identified by title/username
match — selection external to the evaluated behavior — plus the co-forward
agency component as a graph-based variant.

Controls: (a) verified channels with no takedown linkage, (b) political
channels (share >= 0.5) volume-decile matched, (c) all-eligible volume-decile
matched.  Instruments: per-channel tempo and clock features (Mann-Whitney +
rank-biserial), and pooled-stream synchrony vs 100 matched control sets with
400 whole-week circular-shift nulls for the observed group (100 per control
set).  H2-2024-only evaluation is included as a time-split robustness check.

Inputs (IO_RESULTS_DIR): seq_tg_sample.csv, tg_channels.csv, tg_discovery_scores.csv,
        tg_forward_edges.csv
Run:    python -m xplat.analysis.tg_org_bridge
Output: IO_RESULTS_DIR/tg_org_bridge.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ..config import EPS, io_path

MINEV = 20
CAP_ACCT = 3000
WEEK10 = 7 * 24 * 6
WEEK1 = 7 * 24 * 60
AGENCY = re.compile(r"\b(press ?tv|tasnim|irna|isna|snn|mehrnews|mehr news)\b", re.I)
FEATS = ["med_gap_min", "burstiness", "sub1min", "top6h_mass", "hour_entropy", "weekday_ratio", "n"]


def epoch(s):
    return int(pd.Timestamp(s).value // 10**9)


LO, HI = epoch("2024-01-01"), epoch("2024-12-01")
LO2 = epoch("2024-07-01")  # H2 time-split evaluation window


def sync_metrics(pop, lo, hi, n_null):
    res = {}
    for bin_s, week_bins, tag in ((600, WEEK10, "10min"), (60, WEEK1, "1min")):
        nbins = (hi - lo) // bin_s
        n_weeks = nbins // week_bins
        if n_weeks < 4:
            return None
        nbins = n_weeks * week_bins
        acct_bins = []
        for u, ts in pop.items():
            t = ts[(ts >= lo) & (ts < lo + nbins * bin_s)]
            if len(t) < MINEV:
                continue
            acct_bins.append(((t - lo) // bin_s).astype(np.int64))
        if len(acct_bins) < 5:
            return None
        pooled = np.zeros(nbins)
        for b in acct_bins:
            pooled += np.bincount(b, minlength=nbins)
        obs_var = float(pooled.var())
        null_vars = []
        for r in range(n_null):
            nr = np.random.default_rng(1000 + r)
            pn = np.zeros(nbins)
            for b in acct_bins:
                shift = int(nr.integers(1, n_weeks)) * week_bins
                pn += np.bincount((b + shift) % nbins, minlength=nbins)
            null_vars.append(float(pn.var()))
        nv = np.array(null_vars)
        res[tag] = {"n_accounts": len(acct_bins), "n_weeks": int(n_weeks),
                    "var_ratio": round(obs_var / max(nv.mean(), 1e-9), 3),
                    "var_z": round(float((obs_var - nv.mean()) / max(nv.std(), 1e-9)), 1)}
    return res


def chan_feats(ts):
    di = pd.DatetimeIndex(ts.astype("datetime64[s]"))
    hh = np.bincount(di.hour, minlength=24).astype(float)
    hh /= hh.sum()
    dow = np.bincount(di.dayofweek, minlength=7).astype(float)
    wk = dow[:5].sum() / 5.0
    we = dow[5:].sum() / 2.0
    g = np.diff(ts) / 60.0
    return {"med_gap_min": float(np.median(g)) if len(g) else 0.0,
            "burstiness": float((np.std(g) - np.mean(g)) / (np.std(g) + np.mean(g) + EPS)) if len(g) else 0.0,
            "sub1min": float((g < 1).mean()) if len(g) else 0.0,
            "top6h_mass": float(np.sort(hh)[-6:].sum()),
            "hour_entropy": float(-(hh * np.log(hh + EPS)).sum()),
            "weekday_ratio": min(float(wk / (we + EPS)) if we > 0 else 5.0, 5.0),
            "n": int(len(ts))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)

    df = pd.read_csv(io_path("seq_tg_sample.csv"))
    df["user"] = df["user"].astype(str)
    orig = df[df["action"] == 0]
    seq = {u: np.sort(g["ts"].values.astype(np.int64)) for u, g in orig.groupby("user")}
    seq = {u: ts[(ts >= LO) & (ts < HI)] for u, ts in seq.items()}
    vol = {u: len(ts) for u, ts in seq.items()}
    elig = [u for u, v in vol.items() if v >= MINEV]
    meta = pd.read_csv(io_path("tg_channels.csv"))
    meta["channel"] = meta["channel"].astype(str)
    meta = meta.set_index("channel")
    disc = pd.read_csv(io_path("tg_discovery_scores.csv"))
    disc["channel"] = disc["channel"].astype(str)
    disc = disc.set_index("channel")
    pol = disc["political_share"].reindex(elig).fillna(0)
    print(f"eligible: {len(elig)}", flush=True)

    # ---- positive set: takedown-amplified Iranian agency organizations (name-based)
    def name_of(u):
        return f"{meta['title'].get(u, '')} {meta['username'].get(u, '')}"
    bridged_name = [u for u in elig if AGENCY.search(str(name_of(u)))]

    # graph variant: agency co-forward component (same construction as the synchrony analysis)
    e = pd.read_csv(io_path("tg_forward_edges.csv"))
    e["dst"] = e["dst"].astype(str)
    src_sets = e.groupby("dst")["src"].apply(set)
    ids = [c for c in elig if c in src_sets.index]
    G = nx.Graph()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            inter = len(src_sets[a] & src_sets[b])
            if inter >= 3:
                G.add_edge(a, b, weight=inter / len(src_sets[a] | src_sets[b]))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    agency_comp = [u for u in comps[2] if vol.get(u, 0) >= MINEV] if len(comps) > 2 else []
    bridged = sorted(set(bridged_name) | set(agency_comp))
    print(f"bridged: name={len(bridged_name)} comp={len(agency_comp)} union={len(bridged)}", flush=True)

    groups = {
        "bridged": bridged,
        "bridged_name_only": bridged_name,
        "verified_unbridged": [u for u in elig
                               if str(meta["verified"].get(u, 0)) in ("1", "True", "true")
                               and u not in set(bridged)],
        "political_pool": [u for u in elig if pol[u] >= 0.5 and u not in set(bridged)],
    }
    res = {"groups": {k: {"n": len(v)} for k, v in groups.items()},
           "bridged_members": [{"channel": u, "title": str(meta["title"].get(u, ""))[:60],
                                "source": ("name" if u in set(bridged_name) else "") +
                                          ("+comp" if u in set(agency_comp) else "")}
                               for u in bridged]}

    # ---- per-channel features (originals, 2024 window)
    F = {u: chan_feats(seq[u]) for u in elig}

    def compare(a_users, b_users, tag):
        out = {}
        for f in FEATS:
            a = np.array([F[u][f] for u in a_users])
            b = np.array([F[u][f] for u in b_users])
            if len(a) < 5 or len(b) < 5:
                continue
            U, p = mannwhitneyu(a, b, alternative="two-sided")
            rb = 2 * U / (len(a) * len(b)) - 1  # rank-biserial
            out[f] = {"median_a": round(float(np.median(a)), 3),
                      "median_b": round(float(np.median(b)), 3),
                      "rank_biserial": round(float(rb), 3), "p": round(float(p), 5)}
        res[tag] = out
        print(tag, json.dumps({k: v["rank_biserial"] for k, v in out.items()}), flush=True)

    compare(groups["bridged"], groups["verified_unbridged"], "bridged_vs_verified")
    compare(groups["bridged"], groups["political_pool"], "bridged_vs_political")

    # ---- synchrony: bridged vs matched control-set distributions
    lv = {u: np.log10(max(vol[u], 1)) for u in elig}
    edges = np.quantile([lv[u] for u in elig], np.linspace(0, 1, 11))
    edges[-1] += 1e-9

    def decile(u):
        return int(np.clip(np.searchsorted(edges, lv[u], side="right") - 1, 0, 9))

    def matched_set(members, pool, seed):
        r = np.random.default_rng(seed)
        out = []
        by = defaultdict(list)
        for u in pool:
            by[decile(u)].append(u)
        for u in members:
            cand = by[decile(u)] or list(pool)
            out.append(cand[r.integers(0, len(cand))])
        return list(dict.fromkeys(out))

    def sync_block(window_lo, tag):
        obs = sync_metrics({u: seq[u] for u in groups["bridged"]}, window_lo, HI, 400)
        block = {"observed": obs}
        for pool_name in ("political_pool", "all_eligible"):
            pool = groups["political_pool"] if pool_name == "political_pool" else \
                [u for u in elig if u not in set(groups["bridged"])]
            vals = {"10min": [], "1min": []}
            for s in range(100):
                m = sync_metrics({u: seq[u] for u in matched_set(groups["bridged"], pool, s)},
                                 window_lo, HI, 100)
                if m:
                    for k in vals:
                        vals[k].append(m[k]["var_ratio"])
            block[f"ctrl_{pool_name}"] = {
                k: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3),
                    "z_of_observed": round((obs[k]["var_ratio"] - np.mean(v)) / max(np.std(v), 1e-9), 1),
                    "n_sets": len(v)} for k, v in vals.items() if v and obs}
        res[tag] = block
        print(tag, json.dumps(block.get("observed", {})), flush=True)

    sync_block(LO, "sync_full2024")
    sync_block(LO2, "sync_H2_2024_timesplit")

    json.dump(res, open(io_path("tg_org_bridge.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
