"""Smoke checks for the ``xplat`` command-line dispatcher (no experiment imports)."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from xplat.cli import EXPERIMENTS, main


class CliTests(unittest.TestCase):
    def test_usage_returns_zero(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main([]), 0)
        self.assertIn("usage:", out.getvalue())

    def test_list_returns_zero(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["list"]), 0)
        for name in EXPERIMENTS:
            self.assertIn(name, out.getvalue())

    def test_unknown_experiment_returns_two(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(main(["nope"]), 2)
        self.assertIn("unknown experiment", err.getvalue())

    def test_registry_paths(self):
        for name, module in EXPERIMENTS.items():
            self.assertTrue(module.startswith("xplat."), (name, module))
            self.assertRegex(module, r"^xplat(\.[A-Za-z_][A-Za-z0-9_]*)+$")


if __name__ == "__main__":
    unittest.main()
