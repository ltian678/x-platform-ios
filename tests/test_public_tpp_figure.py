"""Regression checks for the public Figure 4 export (no model fitting).

The checks that need the private archived fits (``tpp_results_v4.json`` and
``tpp_fig4_diagnostic.json`` in ``config.ARCHIVE_DIR``) skip when those files
are absent; the midrank and exported-summary checks always run.
"""

import copy
import itertools
import json
import unittest
from pathlib import Path

import numpy as np

from xplat.config import ARCHIVE_DIR, RESULTS_DIR
from xplat.public.tpp_figure import (BASELINES, DIAGNOSTIC_FILE, DISPLAY, PUBLIC, RESULTS_FILE,
                                     average_ranks, build)

ARCHIVE = Path(ARCHIVE_DIR)
PUBLIC_DIR = RESULTS_DIR / "public"


def _archive_available():
    return (ARCHIVE / RESULTS_FILE).exists() and (ARCHIVE / DIAGNOSTIC_FILE).exists()


class MidrankTests(unittest.TestCase):
    def test_midranks(self):
        np.testing.assert_allclose(average_ranks([3, 1, 1, 5]), [3, 1.5, 1.5, 4])


class ExportedSummaryTests(unittest.TestCase):
    def test_exported_case_summaries(self):
        cases = json.loads((PUBLIC_DIR / "case_summaries_public.json").read_text())
        self.assertEqual(set(cases["china"]["eras"]),
                         {"tw_le2017_ifttt", "tw_2018", "tw_2019", "tw_2020"})
        self.assertEqual(set(cases["china"]["policy_populations"]),
                         {"C_le2017", "C_2019", "C_2020", "ORG"})
        for row in cases["china"]["unified_tw_symmetric_distance"].values():
            self.assertEqual(set(row), {"C_le2017", "C_2019", "C_2020", "ORG"})
        self.assertEqual(cases["reddit_matched"]["zero_shot"]["mean_tau_modelfree"]["n_neg"], 73)
        self.assertEqual(cases["reddit_matched"]["populations"]["RDm:IRA"]["split"], "entities 52/6/6")
        for forbidden in ("FB_state", "FB_pacific", "FB:Pacific", "FB:China-state", "FB:newsrooms"):
            self.assertNotIn(forbidden, json.dumps(cases))


@unittest.skipUnless(_archive_available(), f"archived TPP fits not present in {ARCHIVE}")
class PublicFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((ARCHIVE / RESULTS_FILE).read_text())
        cls.diagnostic = json.loads((ARCHIVE / DIAGNOSTIC_FILE).read_text())

    def test_public_allowlist_and_pair_counts(self):
        result, diagnostic = build(self.results, self.diagnostic)
        self.assertEqual(set(result["populations"]), set(PUBLIC))
        self.assertEqual(len(PUBLIC), 13)
        self.assertFalse(any(population.startswith("FB:") for population in PUBLIC))
        self.assertEqual(set(PUBLIC) - set(BASELINES), set(DISPLAY))
        self.assertEqual(diagnostic["public_with_baselines"]["n_pairs"], 78)
        self.assertEqual(diagnostic["displayed_without_baselines"]["n_pairs"], 45)
        for row in result["excess_nll_matrix"].values():
            self.assertEqual(set(row), set(PUBLIC) | {"Exp(1) null"})
        self.assertEqual(set(result["tempo_excess_nll_matrix"]), set(PUBLIC))
        for row in result["tempo_excess_nll_matrix"].values():
            self.assertEqual(set(row), set(PUBLIC))
        for label in set(self.results["populations"]) - set(PUBLIC):
            self.assertNotIn(label, json.dumps(result))
            self.assertNotIn(label, json.dumps(diagnostic))

    def test_unknown_population_is_never_exported(self):
        results = copy.deepcopy(self.results)
        results["populations"]["FUTURE:private"] = {"secret": 42}
        result, diagnostic = build(results, self.diagnostic)
        self.assertNotIn("FUTURE:private", json.dumps([result, diagnostic]))

    def test_missing_public_population_fails(self):
        results = copy.deepcopy(self.results)
        del results["populations"]["RD:IRA"]
        with self.assertRaises(ValueError):
            build(results, self.diagnostic)

    def test_disagreeing_archived_matrices_fail(self):
        results = copy.deepcopy(self.results)
        results["excess_nll_matrix"]["TW:IRA"]["RD:IRA"] += 0.1
        with self.assertRaises(ValueError):
            build(results, self.diagnostic)

    def test_public_correlation_regression(self):
        _, diagnostic = build(self.results, self.diagnostic)
        comparison = diagnostic["public_with_baselines"]
        self.assertAlmostEqual(comparison["pearson_r"], 0.8376290617, places=8)
        self.assertAlmostEqual(comparison["spearman_rho"], 0.8133243988, places=8)

    def test_cross_platform_display_ranges(self):
        result, _ = build(self.results, self.diagnostic)
        matrix = result["excess_nll_matrix"]
        twitter = ("TW:IRA", "TW:Iran", "TW:China", "TW:Venezuela")
        telegram = ("TG:Russia", "TG:Iran", "TG:Syria", "TG:Yemen", "TG:Palestine")

        def similarity(a, b):
            return float(np.exp(-0.5 * (matrix[a][b] + matrix[b][a])))

        rd_tg = [similarity("RD:IRA", target) for target in telegram]
        tw_rd = [similarity(source, "RD:IRA") for source in twitter]
        tw_tg = [similarity(source, target) for source, target in itertools.product(twitter, telegram)]
        np.testing.assert_allclose([min(rd_tg), max(rd_tg)], [0.9092, 0.9685], atol=5e-5)
        np.testing.assert_allclose([min(tw_rd), max(tw_rd)], [0.6041, 0.6923], atol=5e-5)
        np.testing.assert_allclose([min(tw_tg), max(tw_tg)], [0.5474, 0.7593], atol=5e-5)


if __name__ == "__main__":
    unittest.main()
