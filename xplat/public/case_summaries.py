"""Export public-only case summaries and table rows from archived result JSON.

These are descriptive summaries, not new model fits or uncertainty estimates.
China's Twitter summaries are selected from a mixed internal result file;
cross-platform comparisons are not exported.

Inputs (private, from ``config.ARCHIVE_DIR`` or ``--archive-dir``):
    ira_adaptation.json, din_era_drift.json, era_policy_unified_results.json,
    tpp_results_v5b_rdmatched.json
plus ``tpp_results_v4_public.json`` from the output directory, which
``xplat.public.tpp_figure`` must have written first.
Outputs (``config.RESULTS_DIR/public`` or ``--out-dir``):
    case_summaries_public.json, adaptation_rows.tex, china_era_rows.tex,
    tpp_cohort_rows.tex, case_table_macros.tex

Run: python -m xplat.public.case_summaries [--archive-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .. import config
from .tpp_figure import PUBLIC, PUBLIC_RESULTS_FILE


def load(path):
    path = Path(path)
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def export(archive_dir, out_dir):
    archive_dir, out = Path(archive_dir), Path(out_dir)
    adaptation, ah = load(archive_dir / "ira_adaptation.json")
    china, ch = load(archive_dir / "din_era_drift.json")
    policies, ph = load(archive_dir / "era_policy_unified_results.json")
    matched, mh = load(archive_dir / "tpp_results_v5b_rdmatched.json")
    tpp, th = load(out / PUBLIC_RESULTS_FILE)
    cohorts = ("ru_ira", "en_ira", "TW_blm", "TW_maga", "organic")
    fields = ("cohort", "n_accounts", "gap_min", "burst", "cos_audience", "cos_home")
    adaptation_rows = [{k: row[k] for k in fields} for row in adaptation["table"] if row["cohort"] in cohorts]
    assert {row["cohort"] for row in adaptation_rows} == set(cohorts)
    era_order = ("tw_le2017_ifttt", "tw_2018", "tw_2019", "tw_2020")
    era_fields = ("n_account_cells", "n_events", "top6h_mass", "weekday_ratio", "median_gap_min",
                  "frac_gap_under_1min", "burstiness", "hour_centroid")
    eras = {era: {k: china["china_twitter_by_era"][era][k] for k in era_fields} for era in era_order}
    a, b = [np.array(eras[era]["hour_centroid"]) for era in (era_order[0], era_order[-1])]
    policy_names = ("C_le2017", "C_2019", "C_2020", "ORG")
    policy = policies["unified_tw"]
    summary = {
        "scope": "Public Twitter and Reddit evidence; selected fields only; no new fits or bootstrap intervals",
        "input_sha256": {"ira_adaptation.json": ah, "din_era_drift.json": ch,
                         "era_policy_unified_results.json": ph, "tpp_results_v5b_rdmatched.json": mh,
                         "tpp_results_v4_public.json": th},
        "adaptation": {
            "table": adaptation_rows, "windows": adaptation["windows"],
            "estimand": "Cosine between equal-account-weight normalized hour-profile centroids; gaps are medians of account medians",
            "reference": "Organic Twitter timelines, not verified audience geolocation; home reference is ru-IRA",
        },
        "china": {
            "eras": eras,
            "estimand": "Per-account-era medians; at least 30 events; top6h sums the six largest hourly bins, not a contiguous window; same-timestamp events retained",
            "early_to_2020_hour_cosine_from_rounded_centroids": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "hour_cosine": {k: v for k, v in china["hour_cosine"].items()
                            if all(p in era_order for p in k.split(" vs "))},
            "policy_populations": {p: policies["populations"][p] for p in policy_names},
            "unified_tw_symmetric_distance": {p: {q: policy["symmetric_distance"][p][q] for q in policy_names} for p in policy_names},
            "unified_tw_pairwise_auc": {k: v for k, v in policy["pairwise_separability_auc"].items()
                                        if all(p in policy_names for p in k.split(" vs "))},
        },
        "reddit_matched": {
            "populations": {p: matched["populations"][p] for p in ("RDm:organic", "RDm:IRA")},
            "zero_shot": matched["reddit_zero_shot"],
            "scope": "Zero-shot Twitter-policy scores use 64 Reddit IRA accounts and 73 held-out organic controls; own Reddit-policy test uses six IRA accounts",
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "case_summaries_public.json").write_text(json.dumps(summary, indent=2) + "\n")
    labels = {"ru_ira": "ru-IRA", "en_ira": "en-IRA", "TW_blm": "BLM", "TW_maga": "MAGA"}
    rows = [f"{labels[r['cohort']]} & {r['n_accounts']:,} & {r['gap_min']:g} & {r['cos_audience']:.2f} & {r['cos_home']:.2f} \\\\"
            for r in adaptation_rows if r["cohort"] in labels]
    (out / "adaptation_rows.tex").write_text("% Generated public case summary.\n" + "\n".join(rows) + "\n")
    macros = "\\newcommand{\\publicAdaptationRows}{%\n" + "\n".join(rows) + "\n}\n"
    rows = []
    for label, era in zip((r"$\leq$2017", "2018", "2019", "2020"), era_order):
        r = eras[era]
        rows.append(f"{label} & {r['n_account_cells']:,} & {r['median_gap_min']:g} & {r['top6h_mass']:.2f} & {r['weekday_ratio']:.2f} \\\\")
    (out / "china_era_rows.tex").write_text("% Generated public case summary.\n" + "\n".join(rows) + "\n")
    macros += "\\newcommand{\\publicChinaEraRows}{%\n" + "\n".join(rows) + "\n}\n"
    rows = []
    for p in PUBLIC:
        r = tpp["populations"][p]
        split = r["split"].replace("entities ", "").replace("time60/20/20", r"time 60/20/20\%")
        rows.append(f"{p} & {r['n_entities']:,} & {split} & {r['n_test_windows']:,} \\\\")
    (out / "tpp_cohort_rows.tex").write_text("% Generated public cohort accounting.\n" + "\n".join(rows) + "\n")
    macros += "\\newcommand{\\publicTPPCohortRows}{%\n" + "\n".join(rows) + "\n}\n"
    (out / "case_table_macros.tex").write_text("% Generated by scripts/build_public_case_summaries.py.\n" + macros)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive-dir", default=None, help="directory holding the private archived result JSON")
    ap.add_argument("--out-dir", default=None,
                    help="public output directory (must already contain tpp_results_v4_public.json)")
    a = ap.parse_args(argv)
    a.archive_dir = a.archive_dir or config.ARCHIVE_DIR
    a.out_dir = a.out_dir or str(config.RESULTS_DIR / "public")
    summary = export(a.archive_dir, a.out_dir)
    print("Exported public case summaries and three table fragments.")
    print("China early-to-2020 hour cosine:", summary["china"]["early_to_2020_hour_cosine_from_rounded_centroids"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
