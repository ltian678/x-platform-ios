# Which Behavioral Traces Transfer Across Platforms?

Code accompanying the ICWSM submission *"Which Behavioral Traces Transfer
Across Platforms? Evidence from State-Linked Information Operations"*.

The repository is a Python package, `xplat`, with a small shared library
(sequence loading, production-signature features, bootstrap evaluation,
cohorts, the learned action-gap policy, the interval-censored point process)
and one importable module per experiment.  It contains code and derived
aggregates only: no raw archives, no account-level records.

## Layout

```
xplat/                      the package
├── config.py               data locations (environment variables), shared constants, result paths
├── log.py                  elapsed-time logger
├── sequences.py            seq_*.csv loading, timestamp conversion, gaps, takedown parsing
├── features.py             production-signature features, feature groups, design matrices
├── evaluation.py           account-bootstrap AUC intervals, volume matching
├── cohorts.py              archived 70/30 split, Reddit transfer cohort, matched controls, shared CV folds
├── policy.py               action-gap tokenizer, 2-layer Mamba policy, training, log-likelihood scoring
├── tpp/                    interval-censored Mamba-TPP
│   ├── time_resolution.py  completed-minute gaps and conservative censoring bounds
│   ├── store.py            packed timestamp caches
│   ├── model.py            windows, network, mixture likelihoods, training, PIT rescaling
│   └── pipeline.py         stages A-D with the v3 / v4 / v5 population variants
├── adaptation/             every Table 2 row fitted in the paper (one module per row family)
├── analysis/               component ablation, content baselines, audience, China eras, Telegram bridge
├── public/                 builders for the public figure, case summaries, LaTeX macros (no fitting)
├── synthetic.py            synthetic stand-in data with the real file layout
├── fake_mamba.py           CPU stand-in for mamba_ssm (smoke runs only)
├── smoke.py                end-to-end smoke run on synthetic data
└── cli.py                  `xplat list` / `xplat run <experiment>`
tests/                      unit, regression (against the pre-refactor helpers), and public-export tests
results/                    archived derived aggregates consumed by the paper
├── adaptation/             Table 2 cells and row result files
├── analysis/               ablation, audience, China, error analysis, text baseline, Telegram
└── public/                 public figure data, case summaries, LaTeX macros
environment.txt             package versions of the archived runs
```

Every experiment module exposes `main(argv=None)` and does no work at
import time, so each one can be run as `python -m xplat.<group>.<name>`,
through the `xplat` command (`python -m xplat` without installing), or
called from Python.

## Install

```sh
python -m pip install -e ".[dev]"        # core + pytest
python -m pip install -e ".[seq]"        # + torch, mamba_ssm (learned policies, point process; one CUDA device)
python -m pip install -e ".[text]"       # + transformers (content baselines)
```

`environment.txt` lists the exact versions used for the archived runs.

## Check that the framework runs (no data needed)

```sh
make test          # 21 tests; the 6 that need the private archived fits skip cleanly
make smoke         # synthetic data -> component ablation, error analysis, audience, China eras, Telegram, TW profile
make smoke-seq     # the same plus rep-task matrix, policy rows, BLOC/LSTM baselines, and the TPP pipeline
                   # using a CPU stand-in for mamba_ssm (numbers are meaningless; code paths are real)
```

`xplat.smoke` writes synthetic inputs with the real file layout into a
scratch directory, points the configuration at them, and runs the
experiments in dependency order, printing a PASS/FAIL table.  The
experiments it cannot run (content baselines and the Alizadeh
re-implementation need the organic timeline archive and the Reddit recrawl;
the public builders need the private archived fits) are listed as SKIP.

## Data access

Raw data are not redistributed.  Every experiment reads from the
directories below; set them as environment variables or call
`xplat.config.configure(...)` before running.

| Variable | Default | Contents | How to obtain |
|---|---|---|---|
| `IO_RESULTS_DIR` | `output/analysis` | Parsed per-account sequences (`seq_*.csv`: `user, ts, action`), membership tables (`feat_*.csv`), caches, model artefacts, and every experiment output | Produced by the parsing step described in the paper; synthetic stand-ins from `python -m xplat.synthetic` |
| `TWITTER_TAKEDOWN_DIR` | `data/twitter-takedown` | Platform information-operations archives (IRA, Iran, Venezuela, China, GRU) under `tweet_csvs/*.csv` | Apply to the platform's moderation-research release program |
| `TWITTER_TIMELINE_DIR` | `data/twitter-timelines-2016` | Organic 2016 timelines, hourly `.json.bz2` | Public streaming collection described in the paper |
| `REDDIT_RECRAWL_DIR` | `data/reddit-recrawl` | One folder per Reddit account with submission and comment CSVs | Roster from the platform's April 2018 transparency report; activity from a recrawl of public pages |
| `XPLAT_ARCHIVE_DIR` | `archive` | Private archived fits consumed by `xplat.public` (`tpp_results_v4.json`, `tpp_fig4_diagnostic.json`, `ira_adaptation.json`, `din_era_drift.json`, `era_policy_unified_results.json`, `tpp_results_v5b_rdmatched.json`) | Not distributed |

Telegram channels come from the CC BY 4.0 election dataset cited in the
paper (`seq_tg_sample.csv`, `tg_channels.csv`, `tg_discovery_scores.csv`,
`tg_forward_edges.csv` in `IO_RESULTS_DIR`).  The point-process pipeline
additionally needs the Facebook inputs (`seq_fb_din.csv`, `din_labels.csv`,
`seq_fb_ctpages.csv`) that its archived baselines were fitted on; those
populations are never exported.

## Running the experiments

```sh
xplat list                                   # every experiment and its module
xplat run rep-task-matrix                    # or: python -m xplat.adaptation.rep_task_matrix
xplat run tpp --variant v5                   # point-process pipeline (v3, v4, v5)
xplat run xplat-seq-baselines --models "BLOC|tfidf_lr"
```

`rep-task-matrix` must run first: it writes the archived account split
(`rep_task_scores.csv`) and the Twitter policies (`rep_task_policy_*.pt`)
that `tw-inplatform-profile`, `policy-adapted-transfer`,
`xplat-seq-baselines`, and `alizadeh-reimpl` reuse.  `component-ablation`
must precede `error-analysis` and the text baselines (they join
`ablation_scores.csv`).  Sequence models need one CUDA device and
`mamba_ssm`; the profile, BLOC, and random-forest rows run on CPU.  Every
AUC carries a 500-draw account-bootstrap interval; hyperparameters are
stated in each module docstring and in the paper's reproducibility
appendix.  Outputs are written to `IO_RESULTS_DIR`; the archived copies
under `results/` are never overwritten.

### Table 2: which experiment produces which row

| Block | Row | Experiment | Archived result |
|---|---|---|---|
| TW→TW | PS | `tw-inplatform-profile` | `results/adaptation/tw_inplatform_profile.json` (`full_profile`) |
| TW→TW | Policy | `rep-task-matrix` | `results/adaptation/rep_task_matrix_cells.csv` (T1, `policy\|llr`, TW:IRA→TW:IRA) |
| TW→TW | BLOC | `xplat-seq-baselines` | `results/adaptation/xplat_seq_baselines.json` |
| TW→TW | MPNet, XLM-R | `text-baseline`, `text-baseline-v2` | `results/analysis/text_baseline_results.json` |
| RD→RD | PS, MPNet, Schneider et al. | `rd-inplatform` | `results/adaptation/rd_inplatform_results.json` |
| RD→RD | Policy | `rd-policy-inplatform` | `results/adaptation/rd_policy_inplatform.json` (`pooled_fold_centered_llr`) |
| RD→RD | BLOC | `xplat-seq-baselines` | `results/adaptation/xplat_seq_baselines.json` (`RD->RD\|5fold`) |
| TW→RD | PS and component ablation | `component-ablation` | `results/analysis/ablation_results.json` (cohort A, matched) |
| TW→RD | MPNet, XLM-R | `text-baseline*` | `results/analysis/text_baseline_results.json` (matched) |
| TW→RD | BLOC | `xplat-seq-baselines` | `results/adaptation/xplat_seq_baselines.json` (`TW->RD\|matched`) |
| TW→RD | Alizadeh et al. (reimpl.) | `alizadeh-reimpl` | `results/adaptation/alizadeh_reimpl.json` (`TW->RD\|account_auc\|matched`) |
| Appendix | policy transfer, T2, T3 | `rep-task-matrix`, `policy-adapted-transfer` | `results/adaptation/rep_task_matrix_cells.csv` |

### Other analyses

| Experiment | Archived result |
|---|---|
| `error-analysis` | `results/analysis/error_analysis.json` |
| `audience-bootstrap` | `results/analysis/audience_bootstrap.json` |
| `china-within-account` | `results/analysis/china_within_account.json` |
| `tg-org-bridge` | `results/analysis/tg_org_bridge.json` |
| `tpp --variant v4 / v5` | public projections in `results/public/` |

`results/analysis/tg_synchrony_controls_hi.json` is an archived aggregate
whose producing script is not part of this release.

### Public exports (no refitting)

```sh
XPLAT_ARCHIVE_DIR=/path/to/archived/fits xplat run public-tpp-figure
XPLAT_ARCHIVE_DIR=/path/to/archived/fits xplat run public-case-summaries
xplat run public-tex manuscript.tex manuscript_public.tex
```

These select public fields from the archived result files, record source
hashes, verify the archived calculation is reproduced, and write the JSON
and LaTeX macros in `results/public/`.  `tests/test_public_tpp_figure.py`
regression-tests them when the archived fits are available.
