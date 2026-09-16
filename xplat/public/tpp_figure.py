"""Build Figure 4 and reproducible diagnostics from explicitly public populations.

No fitting is performed: this filters archived population fits and recomputes
descriptive comparisons from the saved, rounded excess-NLL matrices.  The
internal inputs are never copied wholesale into the public outputs.

Inputs (private, from ``config.ARCHIVE_DIR`` or ``--archive-dir``):
    tpp_results_v4.json, tpp_fig4_diagnostic.json
Outputs (``config.RESULTS_DIR/public`` or ``--out-dir``):
    tpp_results_v4_public.json, tpp_fig4_diagnostic_public.json, tpp_public_macros.tex

Run: python -m xplat.public.tpp_figure [--archive-dir DIR] [--out-dir DIR]
                                       [--plot-script PATH --figure-out PATH]
Run this before ``xplat.public.case_summaries``, which reads the public JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import config

BASELINES = ("TW:organic", "RD:organic", "TG:broadcast")
COVERT = ("TW:IRA", "TW:Iran", "TW:China", "TW:Venezuela")
DISPLAY = COVERT + ("RD:IRA", "TG:Russia", "TG:Iran", "TG:Syria", "TG:Yemen", "TG:Palestine")
PUBLIC = BASELINES + DISPLAY

RESULTS_FILE = "tpp_results_v4.json"
DIAGNOSTIC_FILE = "tpp_fig4_diagnostic.json"
PUBLIC_RESULTS_FILE = "tpp_results_v4_public.json"
PUBLIC_DIAGNOSTIC_FILE = "tpp_fig4_diagnostic_public.json"
MACROS_FILE = "tpp_public_macros.tex"


def average_ranks(values):
    """One-based midranks, including ties, without a SciPy dependency."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2)[inverse]


def symmetric(matrix, a, b):
    return 0.5 * (matrix[a][b] + matrix[b][a])


def compare_matrices(names, exponential, learned):
    pairs = list(itertools.combinations(names, 2))
    x = np.array([symmetric(exponential, a, b) for a, b in pairs])
    y = np.array([symmetric(learned, a, b) for a, b in pairs])
    return {
        "populations": list(names),
        "n_populations": len(names),
        "n_pairs": len(pairs),
        "quantity": "symmetric excess NLL (not exponentiated similarity)",
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_rho": float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1]),
        "inference": "descriptive correlation; pairs share populations; no independent-pair significance test",
    }


def submatrix(matrix, names, *, include_null=False):
    columns = tuple(names) + (("Exp(1) null",) if include_null else ())
    return {a: {b: matrix[a][b] for b in columns} for a in names}


def build(results, diagnostic):
    for p in PUBLIC:
        if p not in results["populations"] or p not in diagnostic["account_lengths"]:
            raise ValueError(f"Missing required public population: {p}")
    learned = diagnostic["learned_matrix_recomputed"]
    for a in PUBLIC:
        for b in PUBLIC:
            if not np.isclose(learned[a][b], results["excess_nll_matrix"][a][b], atol=5e-5, rtol=0):
                raise ValueError(f"Archived matrix inputs disagree for {a}, {b}")
    exponential = diagnostic["one_param_exp"]["matrix"]
    public_result = {
        "populations": {p: results["populations"][p] for p in PUBLIC},
        "excess_nll_matrix": submatrix(results["excess_nll_matrix"], PUBLIC, include_null=True),
        "tempo_excess_nll_matrix": submatrix(exponential, PUBLIC),
    }
    typical = tuple(p for p in PUBLIC if p not in COVERT)
    similarities = [np.exp(-symmetric(learned, "RD:IRA", p)) for p in typical if p != "RD:IRA"]
    comparison = {
        "scope": "Public populations only, selected by a fixed allowlist",
        "public_with_baselines": compare_matrices(PUBLIC, exponential, learned),
        "displayed_without_baselines": compare_matrices(DISPLAY, exponential, learned),
        "account_lengths": {p: diagnostic["account_lengths"][p] for p in PUBLIC},
        "exponential_fit_rates": {p: diagnostic["one_param_exp"]["lambda"][p] for p in PUBLIC},
        "exponential_excess_nll": submatrix(exponential, PUBLIC),
        "learned_excess_nll": submatrix(learned, PUBLIC),
        "heldout_summary": {
            "baseline_ks_range": [min(results["populations"][p]["KS_vs_uniform_test"] for p in BASELINES),
                                  max(results["populations"][p]["KS_vs_uniform_test"] for p in BASELINES)],
            "typical_mean_tau_range": [min(results["populations"][p]["mean_tau_test"] for p in typical),
                                       max(results["populations"][p]["mean_tau_test"] for p in typical)],
            "reddit_ira_similarity_to_typical_range": [float(min(similarities)), float(max(similarities))],
            "covert_data_under_typical_policy_excess_range": [min(learned[a][b] for a in COVERT for b in typical),
                                                             max(learned[a][b] for a in COVERT for b in typical)],
            "typical_data_under_covert_policy_excess_range": [min(learned[a][b] for a in typical for b in COVERT),
                                                             max(learned[a][b] for a in typical for b in COVERT)],
        },
    }
    return public_result, comparison


def export(archive_dir, out_dir, plot_script=None, figure_out=None):
    """Read the archived fits, write the public files, optionally plot.  Returns ``comparison``."""
    archive_dir, out = Path(archive_dir), Path(out_dir)
    result_path = archive_dir / RESULTS_FILE
    diagnostic_path = archive_dir / DIAGNOSTIC_FILE
    results = json.loads(result_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    public_result, comparison = build(results, diagnostic)
    # Reproduce the historical calculation before filtering to guard against
    # accidentally changing the correlation's estimand during the public rebuild.
    old = compare_matrices(tuple(diagnostic["account_lengths"]), diagnostic["one_param_exp"]["matrix"],
                           diagnostic["learned_matrix_recomputed"])
    for metric in ("pearson_r", "spearman_rho"):
        assert abs(old[metric] - diagnostic["one_param_vs_learned"][metric]) < 0.001
    comparison["input_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (result_path, diagnostic_path)
    }
    out.mkdir(parents=True, exist_ok=True)
    public_path = out / PUBLIC_RESULTS_FILE
    public_path.write_text(json.dumps(public_result, indent=2) + "\n")
    (out / PUBLIC_DIAGNOSTIC_FILE).write_text(json.dumps(comparison, indent=2) + "\n")
    c = comparison["public_with_baselines"]
    (out / MACROS_FILE).write_text(
        "% Generated by scripts/build_public_tpp_figure.py; do not hand edit.\n"
        f"\\newcommand{{\\publicTPPPopulations}}{{{c['n_populations']}}}\n"
        f"\\newcommand{{\\publicTPPPairs}}{{{c['n_pairs']}}}\n"
        f"\\newcommand{{\\publicTPPPearson}}{{{c['pearson_r']:.2f}}}\n"
        f"\\newcommand{{\\publicTPPSpearman}}{{{c['spearman_rho']:.2f}}}\n"
    )
    if plot_script:
        if not figure_out:
            raise SystemExit("--figure-out is required with --plot-script")
        subprocess.run([sys.executable, str(plot_script), str(public_path), str(figure_out), "--summary-panel"],
                       check=True)
    else:
        print("plot skipped: pass --plot-script PATH --figure-out PATH to render the figure", flush=True)
    return comparison


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive-dir", default=None, help="directory holding the private archived fits")
    ap.add_argument("--out-dir", default=None, help="where the public files are written")
    ap.add_argument("--plot-script", default=None, help="optional plotting script (legacy tpp_v3/plot_fig24_v4.py)")
    ap.add_argument("--figure-out", default=None, help="figure path prefix passed to the plotting script")
    a = ap.parse_args(argv)
    a.archive_dir = a.archive_dir or config.ARCHIVE_DIR
    a.out_dir = a.out_dir or str(config.RESULTS_DIR / "public")
    comparison = export(a.archive_dir, a.out_dir, a.plot_script, a.figure_out)
    print(json.dumps({k: comparison[k] for k in ("public_with_baselines", "displayed_without_baselines",
                                                  "heldout_summary")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
