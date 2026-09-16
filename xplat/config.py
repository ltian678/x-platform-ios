"""Central configuration: data locations, shared constants, and result paths.

Every experiment reads inputs from the directories below and writes its
outputs into ``IO_RESULTS_DIR``.  Raw archives are never redistributed; see
README.md for how to obtain them.  The archived aggregates shipped with the
repository live under ``RESULTS_DIR`` and are never overwritten by a rerun.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# Working directory: parsed per-account sequences (seq_*.csv with columns
# user, ts, action), membership tables (feat_*.csv), caches, model artefacts,
# and every file an experiment writes.
IO_RESULTS_DIR = os.environ.get("IO_RESULTS_DIR", "output/analysis")

# Raw sources (see README.md, "Data access").
TWITTER_TAKEDOWN_DIR = os.environ.get("TWITTER_TAKEDOWN_DIR", "data/twitter-takedown")
TAKEDOWN_CSV_DIR = f"{TWITTER_TAKEDOWN_DIR}/tweet_csvs"
TWITTER_TIMELINE_DIR = os.environ.get("TWITTER_TIMELINE_DIR", "data/twitter-timelines-2016")
REDDIT_RECRAWL_DIR = os.environ.get("REDDIT_RECRAWL_DIR", "data/reddit-recrawl")

# Private archived result files consumed by the public export builders
# (xplat.public).  They are not part of the repository.
ARCHIVE_DIR = os.environ.get("XPLAT_ARCHIVE_DIR", str(REPO_ROOT / "archive"))

SEED = 0
EPS = 1e-9
N_BOOT = 500

# Reddit evaluation window (UTC seconds): 2015-01-02 .. 2018-04-11.
RD_WIN = (pd.Timestamp("2015-01-02").value // 10**9, pd.Timestamp("2018-04-11").value // 10**9)

# Canonical sequence and membership files inside IO_RESULTS_DIR.
SEQ_FILES = {
    "tw_t": "seq_twitter_ira.csv",
    "tw_o": "seq_twitter_organic_timeline2016.csv",
    "rd_t": "seq_reddit_troll.csv",
    "rd_o": "seq_reddit_organic.csv",
}
FEAT_FILES = {
    "tw_t": "feat_twitter_ira_full.csv",
    "tw_o": "feat_twitter_organic_timeline2016.csv",
    "rd_t": "feat_reddit_troll.csv",
    "rd_o": "feat_reddit_organic.csv",
}

# Takedown archive files (relative to TAKEDOWN_CSV_DIR).
CHINA_FILES = [
    "china_082019_1_tweets_csv_unhashed.csv",
    "china_082019_2_tweets_csv_unhashed.csv",
    "china_082019_3_tweets_csv_unhashed_part1.csv",
    "china_082019_3_tweets_csv_unhashed_part2.csv",
    "china_082019_3_tweets_csv_unhashed_part3.csv",
    "china_052020_tweets_csv_unhashed.csv",
]
IRAN_FILES = ["iranian_tweets_csv_unhashed.csv"]
IRA_TWEETS_FILE = "ira_tweets_csv_unhashed.csv"


def io_path(*parts: str) -> str:
    """Path inside the working directory."""
    return os.path.join(IO_RESULTS_DIR, *parts)


def takedown_path(name: str) -> str:
    return os.path.join(TAKEDOWN_CSV_DIR, name)


def takedown_paths(names) -> list:
    return [takedown_path(n) for n in names]


def ensure_io_dir() -> str:
    os.makedirs(IO_RESULTS_DIR, exist_ok=True)
    return IO_RESULTS_DIR


def configure(io_dir=None, takedown_dir=None, timeline_dir=None, recrawl_dir=None, archive_dir=None):
    """Override the data locations at runtime (tests, smoke runs).

    Experiments resolve paths through ``io_path`` / ``takedown_path`` at call
    time, so overriding here takes effect for everything run afterwards.
    """
    global IO_RESULTS_DIR, TWITTER_TAKEDOWN_DIR, TAKEDOWN_CSV_DIR, TWITTER_TIMELINE_DIR, REDDIT_RECRAWL_DIR, ARCHIVE_DIR
    if io_dir is not None:
        IO_RESULTS_DIR = str(io_dir)
    if takedown_dir is not None:
        TWITTER_TAKEDOWN_DIR = str(takedown_dir)
        TAKEDOWN_CSV_DIR = f"{TWITTER_TAKEDOWN_DIR}/tweet_csvs"
    if timeline_dir is not None:
        TWITTER_TIMELINE_DIR = str(timeline_dir)
    if recrawl_dir is not None:
        REDDIT_RECRAWL_DIR = str(recrawl_dir)
    if archive_dir is not None:
        ARCHIVE_DIR = str(archive_dir)
