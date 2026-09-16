"""Account-bootstrap CIs for the audience-adaptation contrasts.

Cohort construction follows the archived audience analysis (BLM/MAGA hashtag
cohorts in their matched windows; ru/en language cohorts; organic timeline
reference).  2,000 account-bootstrap draws recompute equal-weight hour
centroids and both cosines per draw, giving CIs for
  delta_C = cos(C, organic) - cos(C, ru_ira)
per cohort, the BLM-minus-MAGA difference-in-differences, and cohort median
gaps.  Within a draw, one resample per cohort is shared across all statistics,
preserving the dependence between the two cosines.

The hashtag source is the IRA takedown tweet file; the BLM / MAGA windows were
reconstructed from the archived public case summary
(results/public/case_summaries_public.json, adaptation.windows).

Inputs (IO_RESULTS_DIR): seq_twitter_ira.csv, seq_twitter_organic_timeline2016.csv,
        ira_twitter_account_langs.csv
Run:    python -m xplat.analysis.audience_bootstrap
Output: IO_RESULTS_DIR/audience_bootstrap.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from ..config import EPS, IRA_TWEETS_FILE, SEQ_FILES, io_path, takedown_path
from ..sequences import read_sequence_frame

N_BOOT = 2000
WIN = {"blm": (np.datetime64("2015-06-01"), np.datetime64("2016-10-31")),
       "maga": (np.datetime64("2015-06-01"), np.datetime64("2016-12-31"))}
BLM = {"blacklivesmatter", "policebrutality", "btp", "blacktolive", "blackskinisnotacrime", "blm",
       "blacktwitter", "myblackroots", "blackhistory", "donotshoot"}
MAGA = {"maga", "tcot", "pjnet", "trump2016", "trumpforpresident", "neverhillary", "ccot",
        "wakeupamerica", "hillaryforprison", "draintheswamp"}
COHORTS_OF_INTEREST = ("TW_blm", "TW_maga", "en_ira", "ru_ira")


def account_hours_gaps(df, users=None, win=None, minev=30):
    H, G = [], []
    for u, g in df.groupby("user"):
        if users is not None and u not in users:
            continue
        ts = np.sort(g["ts"].values.astype(np.int64))
        if win is not None:
            lo = win[0].astype("datetime64[s]").astype("int64")
            hi = win[1].astype("datetime64[s]").astype("int64")
            ts = ts[(ts >= lo) & (ts <= hi)]
        if len(ts) < minev:
            continue
        hh = np.bincount(pd.DatetimeIndex(pd.to_datetime(ts, unit="s")).hour, minlength=24).astype(float)
        H.append(hh / hh.sum())
        gg = np.diff(ts) / 60.0
        G.append(float(np.median(gg)) if len(gg) else 0.0)
    return np.array(H), np.array(G)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def ci(vals):
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    rng = np.random.default_rng(0)

    tweets = takedown_path(IRA_TWEETS_FILE)
    tot, nb, nm = Counter(), Counter(), Counter()
    for c in pd.read_csv(tweets, dtype=str, usecols=["user_screen_name", "hashtags"],
                         chunksize=2_000_000, lineterminator="\n", on_bad_lines="skip"):
        for s, h in zip(c["user_screen_name"].str.lower(), c["hashtags"].fillna("")):
            tot[s] += 1
            tags = {t.strip().lower() for t in re.split(r"[\[\]',]+", h) if t.strip()}
            if tags & BLM:
                nb[s] += 1
            if tags & MAGA:
                nm[s] += 1
    cohort = {"TW_blm": {s for s in tot if tot[s] >= 50 and nb[s] / tot[s] >= 0.10},
              "TW_maga": {s for s in tot if tot[s] >= 50 and nm[s] / tot[s] >= 0.10}}
    langs = pd.read_csv(io_path("ira_twitter_account_langs.csv"))
    cohort["ru_ira"] = set(langs[langs.ru_frac >= 0.6].user.str.lower())
    cohort["en_ira"] = set(langs[langs.en_frac >= 0.6].user.str.lower())
    print("cohorts:", {k: len(v) for k, v in cohort.items()}, flush=True)

    ira = read_sequence_frame(io_path(SEQ_FILES["tw_t"]), positive_only=False)
    ira["user"] = ira["user"].astype(str).str.lower()
    org = read_sequence_frame(io_path(SEQ_FILES["tw_o"]), positive_only=False)

    POPS = {"ru_ira": account_hours_gaps(ira, cohort["ru_ira"]),
            "en_ira": account_hours_gaps(ira, cohort["en_ira"]),
            "TW_blm": account_hours_gaps(ira, cohort["TW_blm"], WIN["blm"]),
            "TW_maga": account_hours_gaps(ira, cohort["TW_maga"], WIN["maga"]),
            "organic": account_hours_gaps(org)}
    print({k: len(v[0]) for k, v in POPS.items()}, flush=True)

    def stats(idx):
        """idx: dict pop -> row indices. Returns per-cohort (cosA, cosH, delta) + med gaps."""
        cents = {k: POPS[k][0][idx[k]].mean(axis=0) for k in POPS}
        out = {}
        for k in COHORTS_OF_INTEREST:
            ca, ch = cos(cents[k], cents["organic"]), cos(cents[k], cents["ru_ira"])
            out[k] = (ca, ch, ca - ch)
        out["gaps"] = {k: float(np.median(POPS[k][1][idx[k]])) for k in POPS}
        return out

    full = {k: np.arange(len(POPS[k][0])) for k in POPS}
    point = stats(full)

    draws = []
    for _ in range(N_BOOT):
        idx = {k: rng.choice(len(POPS[k][0]), len(POPS[k][0]), replace=True) for k in POPS}
        draws.append(stats(idx))

    res = {"n_boot": N_BOOT,
           "n_accounts": {k: int(len(POPS[k][0])) for k in POPS},
           "point": {k: {"cos_audience": round(point[k][0], 4),
                         "cos_home": round(point[k][1], 4),
                         "delta": round(point[k][2], 4)} for k in COHORTS_OF_INTEREST},
           "gaps_point": {k: round(v, 2) for k, v in point["gaps"].items()},
           "ci": {}}
    for k in COHORTS_OF_INTEREST:
        res["ci"][k] = {"cos_audience": ci([d[k][0] for d in draws]),
                        "cos_home": ci([d[k][1] for d in draws]),
                        "delta": ci([d[k][2] for d in draws])}
    did = [d["TW_blm"][2] - d["TW_maga"][2] for d in draws]
    res["blm_minus_maga_delta"] = {
        "point": round(point["TW_blm"][2] - point["TW_maga"][2], 4),
        "ci": ci(did),
        "excludes_zero": bool(np.percentile(did, 2.5) > 0 or np.percentile(did, 97.5) < 0)}
    res["gap_ci"] = {k: ci([d["gaps"][k] for d in draws]) for k in POPS}
    for k in ("TW_blm", "TW_maga"):
        ratio = [d["gaps"]["organic"] / max(d["gaps"][k], 1e-3) for d in draws]
        res[f"gap_ratio_organic_over_{k}"] = {
            "point": round(point["gaps"]["organic"] / max(point["gaps"][k], 1e-3), 1),
            "ci": ci(ratio)}

    json.dump(res, open(io_path("audience_bootstrap.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
