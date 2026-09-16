"""China eras: within-account change vs cohort replacement.

Era construction (corrected run: NO timestamp dedup — minute-resolution
archives make identical ts real same-minute bursts; era = posting year; an
account contributes a cell to an era with >=30 events).

For each era pair of interest:
  1. overlap accounting: stayers / entrants / leavers
  2. paired within-account changes for stayers (median deltas of top6h_mass,
     weekday_ratio, median_gap_min, sub-minute share; per-account hour cosine
     across the pair), with 2,000-draw account-bootstrap CIs
  3. decomposition: population medians vs stayers-only medians vs
     leavers/entrants medians, so composition and within-account components
     are explicit
  4. sensitivities: (a) restrict the le2017 era to calendar 2017 (equal-ish
     windows), (b) drop the top event-decile of stayers

Inputs: the China takedown CSVs under TWITTER_TAKEDOWN_DIR/tweet_csvs
Run:    python -m xplat.analysis.china_within_account
Output: IO_RESULTS_DIR/china_within_account.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import CHINA_FILES, EPS, io_path, takedown_paths

N_BOOT = 2000
ERAS = ["le2017", "2018", "2019", "2020"]
Y2017 = (pd.Timestamp("2017-01-01").value // 10**9, pd.Timestamp("2017-12-31 23:59:59").value // 10**9)
KEYS = ["top6h_mass", "weekday_ratio", "median_gap_min", "frac_gap_under_1min"]


def feats(ts_i):
    """Timestamp-only era features (hour vector + four scalars); not the profile schema."""
    ts = np.sort(ts_i)
    di = pd.DatetimeIndex(ts.astype("datetime64[s]"))
    hh = np.bincount(di.hour, minlength=24).astype(float)
    hh /= hh.sum()
    dow = np.bincount(di.dayofweek, minlength=7).astype(float)
    wk = dow[:5].sum() / 5.0
    we = dow[5:].sum() / 2.0
    g = np.diff(ts) / 60.0
    return {"hour": hh,
            "top6h_mass": float(np.sort(hh)[-6:].sum()),
            "weekday_ratio": min(float(wk / (we + EPS)) if we > 0 else 5.0, 5.0),
            "median_gap_min": float(np.median(g)) if len(g) else 0.0,
            "frac_gap_under_1min": float((g < 1).mean()) if len(g) else 0.0,
            "n": int(len(ts))}


def era_of(y):
    return "le2017" if y <= 2017 else str(y)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def med(d, users, k):
    return round(float(np.median([d[u][k] for u in users])), 4)


def parse_china(paths):
    seq = defaultdict(list)
    use = ["user_screen_name", "tweet_time"]
    for path in paths:
        for c in pd.read_csv(path, dtype=str, usecols=use, chunksize=1_000_000,
                             lineterminator="\n", on_bad_lines="skip"):
            ts = pd.to_datetime(c["tweet_time"], errors="coerce")
            ok = ts.notna().values
            df = pd.DataFrame({"sn": c["user_screen_name"].str.lower().values[ok],
                               "ts": ts.values.astype("datetime64[s]").astype("int64")[ok]})
            for name, g in df.groupby("sn", sort=False):
                seq[name].append(g["ts"].values)
        print("parsed", path.split("/")[-1], flush=True)
    return seq


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    rng = np.random.default_rng(0)

    print("parsing china twitter", flush=True)
    seq = parse_china(takedown_paths(CHINA_FILES))

    cells = {e: {} for e in ERAS}          # era -> account -> feats
    cells17 = {}                            # calendar-2017-only variant of le2017
    for name, parts in seq.items():
        ts = np.sort(np.concatenate(parts))
        years = pd.DatetimeIndex(ts.astype("datetime64[s]")).year.values
        for era in ERAS:
            m = np.array([era_of(y) == era for y in years])
            if m.sum() >= 30:
                cells[era][name] = feats(ts[m])
        m17 = (ts >= Y2017[0]) & (ts <= Y2017[1])
        if m17.sum() >= 30:
            cells17[name] = feats(ts[m17])
    print("era cells:", {e: len(v) for e, v in cells.items()}, "| cal2017:", len(cells17), flush=True)

    def ci_boot(vals):
        vals = np.asarray(vals)
        meds = [float(np.median(vals[rng.choice(len(vals), len(vals), replace=True)])) for _ in range(N_BOOT)]
        return [round(float(np.percentile(meds, 2.5)), 4), round(float(np.percentile(meds, 97.5)), 4)]

    res = {"n_boot": N_BOOT}

    def pair_analysis(e1, d1, e2, d2, tag):
        u1, u2 = set(d1), set(d2)
        stay = sorted(u1 & u2)
        leave = sorted(u1 - u2)
        enter = sorted(u2 - u1)
        out = {"pair": f"{e1}->{e2}", "n": {"era1": len(u1), "era2": len(u2),
               "stayers": len(stay), "leavers": len(leave), "entrants": len(enter)}}
        if len(stay) >= 10:
            deltas = {k: [d2[u][k] - d1[u][k] for u in stay] for k in KEYS}
            out["stayers_paired_delta"] = {
                k: {"median": round(float(np.median(v)), 4), "ci": ci_boot(v),
                    "frac_increase": round(float(np.mean(np.array(v) > 0)), 3)}
                for k, v in deltas.items()}
            hc = [cos(d1[u]["hour"], d2[u]["hour"]) for u in stay]
            out["stayers_hour_cosine"] = {"median": round(float(np.median(hc)), 4), "ci": ci_boot(hc)}
            out["decomposition"] = {k: {
                "pop_era1": med(d1, u1, k), "pop_era2": med(d2, u2, k),
                "stayers_era1": med(d1, stay, k), "stayers_era2": med(d2, stay, k),
                "leavers_era1": med(d1, leave, k) if leave else None,
                "entrants_era2": med(d2, enter, k) if enter else None} for k in KEYS}
            nvals = np.array([d1[u]["n"] + d2[u]["n"] for u in stay])
            keep = [u for u, nv in zip(stay, nvals) if nv <= np.percentile(nvals, 90)]
            out["sens_drop_top_decile"] = {
                k: {"median": round(float(np.median([d2[u][k] - d1[u][k] for u in keep])), 4),
                    "n": len(keep)} for k in KEYS}
        res[tag] = out
        print(tag, json.dumps(out["n"]), flush=True)

    pair_analysis("le2017", cells["le2017"], "2018", cells["2018"], "p_le2017_2018")
    pair_analysis("2018", cells["2018"], "2019", cells["2019"], "p_2018_2019")
    pair_analysis("2019", cells["2019"], "2020", cells["2020"], "p_2019_2020")
    pair_analysis("le2017", cells["le2017"], "2019", cells["2019"], "p_le2017_2019")
    pair_analysis("cal2017", cells17, "2019", cells["2019"], "p_cal2017_2019_equalwin")

    json.dump(res, open(io_path("china_within_account.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
