"""Regression checks for the shared library against values computed with the
original (pre-refactor) helper functions on the deterministic synthetic data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xplat import config, evaluation, features, sequences, synthetic

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "library_regression.json").read_text())


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("work")
    synthetic.main(["--out-dir", str(d), "--skip-takedown", "--seed", "0"])
    return d


def test_account_features_match_original(work):
    members = set(pd.read_csv(work / config.FEAT_FILES["rd_t"])["user"])
    df = features.featurize_population(str(work / config.SEQ_FILES["rd_t"]), members, window=config.RD_WIN)
    fx = FIXTURE["features"]
    assert len(df) == fx["n_rows"]
    assert list(df.columns) == fx["columns"]
    got = json.loads(df.head(3).to_json(orient="records"))
    for a, b in zip(got, fx["first_rows"]):
        assert a["user"] == b["user"]
        for k in b:
            if k != "user":
                assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12), k


def test_load_accounts_is_sorted_and_filtered(work):
    acc = sequences.load_accounts(str(work / config.SEQ_FILES["tw_t"]), 50)
    assert acc and all(len(ts) >= 50 and np.all(np.diff(ts) >= 0) for ts, _ in acc.values())
    assert all(isinstance(u, str) for u in acc)


def test_bootstrap_auc_matches_original_draws():
    fx = FIXTURE["auc_ci"]
    assert evaluation.auc_ci(fx["pos"], fx["neg"]) == fx["expected"]


def test_volume_match_matches_original_draws():
    fx = FIXTURE["volume_match"]
    got = evaluation.volume_match(np.array(fx["n_events"]), np.array(fx["neg_idx"]), np.array(fx["pos_idx"]),
                                  np.random.default_rng(0))
    assert [int(i) for i in got] == fx["expected"]


def test_configure_changes_paths(tmp_path):
    old = config.IO_RESULTS_DIR
    try:
        config.configure(io_dir=tmp_path)
        assert config.io_path("x.csv") == str(tmp_path / "x.csv")
    finally:
        config.configure(io_dir=old)
