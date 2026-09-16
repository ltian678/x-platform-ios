"""xplat: cross-platform transfer of behavioral traces in state-linked information operations.

Library modules (shared, side-effect free):
    config       data locations, shared constants, result paths
    log          elapsed-time logger
    sequences    per-account sequence loading and timestamp conversion
    features     production-signature (profile) features and feature groups
    evaluation   account-bootstrap AUC intervals and volume matching
    cohorts      archived splits, Reddit cohort, matched-control replay, shared CV folds
    policy       action-gap next-token policy (2-layer Mamba) and log-likelihood scoring
    tpp          interval-censored temporal point-process model, store, and pipeline

Experiment packages (each module exposes ``main(argv=None)``):
    adaptation   every Table 2 row fitted in the paper
    analysis     component ablation, content baselines, audience, China eras, Telegram bridge
    public       builders for the public figure, case summaries, and LaTeX macros
"""

__version__ = "1.0.0"
