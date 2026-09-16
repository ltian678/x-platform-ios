"""Interval-censored Mamba-TPP pipeline (Figure 4): stages A-D with three population variants.

Protocol (identical across variants; see ``xplat.tpp.model`` for the machinery):
  A  populations: per-entity sorted distinct UTC-second timestamps, converted to
     completed-minute gaps with conservative minute-censoring intervals.
  B  one censored-likelihood baseline TPP per platform, fitted on the training
     entities of that platform's reference population (``BASE_POP``).
  C  randomized-PIT rescaling of every entity through its platform baseline:
     u ~ Uniform(F(a), F(b)) (seeded) and tau = -log(1 - u), which is Exp(1)
     under the baseline on every platform.  A KS gate prints the baselines'
     held-out uniformity.
  D  one density policy (log-normal mixture on tau) per population, trained with
     early stopping on validation NLL (small populations: per-entity 60/20/20
     time split; large populations: held-out entities split into val/test
     halves); test-only excess-NLL matrix, KS, and operation-vs-organic AUCs.

Variants (``--variant``):
  v3   Twitter (organic, IRA, Iran, Venezuela, China), Facebook, Telegram.
       Output ``tpp_results_v3.json``.
  v4   v3 + Reddit (RD:organic, RD:IRA >= 30 events) as a fourth platform, with
       zero-shot cross-platform readouts on Reddit.  Output ``tpp_results_v4.json``
       (the public Figure 4 export is built from this file).
  v5   v4 with a VOLUME-MATCHED Reddit baseline (RDm:organic = organics with
       30-1000 events), a step-budget epoch rule for that small baseline, and a
       separate rescale cache.  Output ``tpp_results_v5b_rdmatched.json`` (the
       paper's matched Reddit rows).  Default.

Refitting needs the non-public Facebook and Telegram sequence files
(seq_fb_din.csv, din_labels.csv, seq_fb_ctpages.csv, tg_channels.csv,
seq_tg_sample.csv) in the working directory in addition to the Twitter and
Reddit sequences, and the Iran / Venezuela / China takedown archives.  Stages
are checkpointed to the working directory (tpp_base2_*.pt, tpp_resc2*_*.npz,
tpp_pop3_*.pt + curve JSON) and every stage reloads its artefacts if present,
so a run can be relaunched.

Run:  python -m xplat.tpp.pipeline --variant v5
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest
from sklearn.metrics import roc_auc_score

from .. import config
from .model import (BASE_EPOCHS, BS, EVAL_EVERY, LR, MAX_STEPS, PATIENCE, device, eval_nll, r_windows,
                    rescale_entity, time_split, train_population_policy, train_tpp, windows_of)
from .store import MINEV, acc_ts, load_seq_csv, parse_takedown_ts
from .time_resolution import completed_gap_minutes as gap_minutes

TW_COVERT = ("TW:IRA", "TW:Iran", "TW:China", "TW:Venezuela")
VENEZUELA_FILES = ["venezuela_201906_1_tweets_csv_unhashed.csv"]
FB_NEWSROOM_SAMPLE = 1500
TG_MINEV = 100
ORGANIC_NEG_CAP = 300          # TW:organic test accounts used as negatives in the AUC readout

ACTOR_KW = {"Russia": ["tass", "sputnik", " rt ", "донецк", "russia"],
            "Iran": ["mehr", "tasnim", "presstv", "press tv", "hispantv", "irna", "خبرگزاری", "تسنيم"],
            "Syria": ["sana", "syrian arab news", "سوريا", "syria"],
            "Yemen": ["almasirah", "al masirah", "ansar", "أنصار", "المسيرة", "yemen"],
            "Palestine": ["quds", "gaza", "غزة", "قدس", "فلسطين"]}


@dataclass(frozen=True)
class RedditPop:
    label: str
    file: str
    minev: int
    max_events: Optional[int] = None   # keep entities with <= max_events timestamps


@dataclass(frozen=True)
class Variant:
    name: str
    output: str
    reddit: Tuple[RedditPop, ...]                  # appended after Telegram, in this order
    base_pop: Dict[str, str]                       # platform prefix -> reference population
    resc_tag: str                                  # rescale cache prefix (tpp_<tag>_<pop>.npz)
    zero_shot: Optional[Tuple[str, str]] = None    # (Reddit covert label, Reddit organic label)
    long_baseline: Optional[str] = None            # platform whose baseline uses the step budget
    long_baseline_tag: str = ""
    long_baseline_min_steps: int = 1500


VARIANTS = {
    "v3": Variant(
        name="v3", output="tpp_results_v3.json", reddit=(),
        base_pop={"TW": "TW:organic", "FB": "FB:newsrooms", "TG": "TG:broadcast"},
        resc_tag="resc2"),
    "v4": Variant(
        name="v4", output="tpp_results_v4.json",
        reddit=(RedditPop("RD:organic", "seq_reddit_organic.csv", MINEV),
                RedditPop("RD:IRA", "seq_reddit_troll.csv", 30)),
        base_pop={"TW": "TW:organic", "FB": "FB:newsrooms", "TG": "TG:broadcast", "RD": "RD:organic"},
        resc_tag="resc2", zero_shot=("RD:IRA", "RD:organic")),
    "v5": Variant(
        name="v5", output="tpp_results_v5b_rdmatched.json",
        reddit=(RedditPop("RDm:organic", "seq_reddit_organic.csv", 30, max_events=1000),
                RedditPop("RDm:IRA", "seq_reddit_troll.csv", 30)),
        base_pop={"TW": "TW:organic", "FB": "FB:newsrooms", "TG": "TG:broadcast", "RDm": "RDm:organic"},
        resc_tag="resc2b", zero_shot=("RDm:IRA", "RDm:organic"),
        long_baseline="RDm", long_baseline_tag="base2_rdm_long"),
}


def actor_of(title):
    t = str(title).lower()
    for a, kws in ACTOR_KW.items():
        if any(k in t for k in kws):
            return a
    return None


def _p(out_dir, name):
    return os.path.join(out_dir, name)


# ---------------- Stage A: populations ----------------
def build_populations(variant: Variant, out_dir: str, rng) -> Dict[str, Dict[str, np.ndarray]]:
    """Ordered ``{label: {entity: sorted distinct timestamps}}``.  Consumes ``rng`` once (FB newsrooms)."""
    print("=== Stage A: sequences ===", flush=True)
    pops = {}
    pops["TW:organic"] = acc_ts(load_seq_csv(_p(out_dir, "seq_twitter_organic_timeline2016.csv")))
    pops["TW:IRA"] = acc_ts(load_seq_csv(_p(out_dir, "seq_twitter_ira.csv")))
    pops["TW:Iran"] = parse_takedown_ts(config.takedown_paths(config.IRAN_FILES), _p(out_dir, "tpp_cache_tw_iran.npz"))
    pops["TW:Venezuela"] = parse_takedown_ts(config.takedown_paths(VENEZUELA_FILES), _p(out_dir, "tpp_cache_tw_venez.npz"))
    pops["TW:China"] = parse_takedown_ts(config.takedown_paths(config.CHINA_FILES), _p(out_dir, "tpp_cache_tw_china.npz"))

    din = load_seq_csv(_p(out_dir, "seq_fb_din.csv"))
    lab = pd.read_csv(_p(out_dir, "din_labels.csv"))
    lmap = dict(zip(lab[lab.columns[0]].astype(str), lab["label"].astype(str)))
    din["lab"] = din["user"].astype(str).map(lmap)
    pops["FB:China-state"] = acc_ts(din[din.lab == "china_state"], minev=MINEV)
    pops["FB:Pacific"] = acc_ts(din[din.lab == "pacific_local"], minev=MINEV)
    ctp = acc_ts(load_seq_csv(_p(out_dir, "seq_fb_ctpages.csv")), minev=MINEV)
    ct_keys = sorted(ctp)
    rng.shuffle(ct_keys)
    pops["FB:newsrooms"] = {u: ctp[u] for u in ct_keys[:FB_NEWSROOM_SAMPLE]}

    ro = pd.read_csv(_p(out_dir, "tg_channels.csv"))
    ro["channel"] = ro["channel"].astype(str)
    ro["actor"] = ro["title"].apply(actor_of)
    tg = acc_ts(load_seq_csv(_p(out_dir, "seq_tg_sample.csv")), minev=TG_MINEV)
    amap = dict(zip(ro.channel, ro.actor))
    bmap = dict(zip(ro.channel, ro.broadcast))
    for a in ACTOR_KW:
        d = {u: ts for u, ts in tg.items() if amap.get(u) == a}
        if d:
            pops[f"TG:{a}"] = d
    pops["TG:broadcast"] = {u: ts for u, ts in tg.items()
                            if not isinstance(amap.get(u), str) and bmap.get(u, 0) == 1}

    for rp in variant.reddit:
        d = acc_ts(load_seq_csv(_p(out_dir, rp.file)), minev=rp.minev)
        if rp.max_events is not None:
            d = {u: t for u, t in d.items() if len(t) <= rp.max_events}
            print(f"  {rp.label} baseline: {len(d)} accounts with {rp.minev}-{rp.max_events} events", flush=True)
        pops[rp.label] = d
    for l, d in pops.items():
        print(f"  {l}: {len(d)} entities, {sum(len(v) for v in d.values())} events", flush=True)
    return pops


def make_splits(pops, rng):
    """Held-out entities per population (20%, none below 10 entities); consumes ``rng`` per population."""
    splits = {}
    for l, d in pops.items():
        ks = sorted(d)
        rng.shuffle(ks)
        nh = max(1, len(ks) // 5) if len(ks) >= 10 else 0
        splits[l] = {"hold": ks[:nh], "train": ks[nh:] if nh else ks}
    return splits


# ---------------- Stage B: platform baselines (censored) ----------------
def fit_baselines(variant: Variant, pops, splits, out_dir, dev):
    print("=== Stage B: baselines ===", flush=True)
    base_models = {}
    for plat, bl in variant.base_pop.items():
        tr = [gap_minutes(pops[bl][u]) for u in splits[bl]["train"]]
        wins = windows_of(tr)
        if plat == variant.long_baseline:
            # small baselines get a step budget comparable to the large ones instead of 5 epochs
            ep = max(BASE_EPOCHS, int(np.ceil(variant.long_baseline_min_steps / max(1, len(wins) // BS))))
            tag = variant.long_baseline_tag
            print(f"  baseline {plat}: {len(wins)} windows, {ep} epochs", flush=True)
        else:
            ep, tag = BASE_EPOCHS, f"base2_{plat.lower()}"
        base_models[plat] = train_tpp(wins, tag, ep, out_dir, dev=dev)
    return base_models


# ---------------- Stage C: randomized-PIT rescale ----------------
def rescale_all(variant: Variant, pops, splits, base_models, out_dir, pit_rng, dev):
    """``{label: {entity: (tau, u)}}``; cached per population as tpp_<resc_tag>_<label>.npz."""
    print("=== Stage C: randomized-PIT rescale ===", flush=True)
    resc = {}
    for l, d in pops.items():
        cache = _p(out_dir, f"tpp_{variant.resc_tag}_{l.replace(':', '_')}.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            resc[l] = {str(u): (t, uu) for u, t, uu in zip(z["users"], z["tau"], z["u"])}
        else:
            model = base_models[l.split(":")[0]]
            resc[l] = {u: rescale_entity(model, gap_minutes(ts), pit_rng, dev=dev) for u, ts in d.items()}
            np.savez_compressed(cache, users=np.array(list(resc[l]), dtype=object),
                                tau=np.array([v[0] for v in resc[l].values()], dtype=object),
                                u=np.array([v[1] for v in resc[l].values()], dtype=object))
        print(f"  rescaled {l}", flush=True)
    # calibration gate: baselines' held-out u must be ~Uniform
    for plat, bl in variant.base_pop.items():
        uu = np.concatenate([resc[bl][u][1] for u in (splits[bl]["hold"] or splits[bl]["train"])])
        print(f"  GATE {bl}: heldout KS={kstest(uu, 'uniform').statistic:.4f} (want ~0)", flush=True)
    return resc


# ---------------- Stage D: density policies on tau + metrics ----------------
def build_sets(pops, splits, resc):
    """Train / val / test window sets per population."""
    sets = {}
    for l in pops:
        small = len(splits[l]["hold"]) == 0
        if small:
            tr, va, te, te_u = [], [], [], []
            for u in pops[l]:
                tau, uu = resc[l][u]
                a, b, c = time_split(tau)
                tr.append((a, None)); va.append((b, None)); te.append((c, None))
                n = len(tau)
                te_u.append(uu[int(0.8 * n):])
            sets[l] = dict(train=r_windows(tr, maxw=10**9), val=r_windows(va, maxw=10**9),
                           test=r_windows(te, maxw=10**9), test_u=te_u,
                           test_tau=[c for c, _ in te], split="time60/20/20", test_users=list(pops[l]))
        else:
            hold = splits[l]["hold"]
            h = len(hold) // 2
            va_k, te_k = hold[:h], hold[h:]
            sets[l] = dict(train=r_windows([resc[l][u] for u in splits[l]["train"]]),
                           val=r_windows([resc[l][u] for u in va_k]),
                           test=r_windows([resc[l][u] for u in te_k]),
                           test_u=[resc[l][u][1] for u in te_k], test_tau=[resc[l][u][0] for u in te_k],
                           split=f"entities {len(splits[l]['train'])}/{len(va_k)}/{len(te_k)}", test_users=te_k)
        print(f"  {l}: windows train={len(sets[l]['train'])} val={len(sets[l]['val'])} "
              f"test={len(sets[l]['test'])} [{sets[l]['split']}]", flush=True)
    return sets


def acct_llr(l, model, users, resc, dev, tail=False):
    """Per-account Exp(1)-null minus policy NLL on rescaled time (higher = more policy-like)."""
    out = []
    for u in users:
        tau, _ = resc[l][u]
        if tail:
            tau = tau[int(0.8 * len(tau)):]
        w = r_windows([(tau, None)], maxw=10**9)
        if not w:
            continue
        null = float(np.maximum(tau, 1e-8).mean())
        out.append(null - eval_nll(model, w, dens=True, dev=dev))
    return out


def fit_policies_and_metrics(pops, splits, sets, out_dir, dev):
    print("=== Stage D: policies & metrics ===", flush=True)
    results = {"populations": {}, "auc_llr": {}, "excess_nll_matrix": {}, "training": {},
               "config": {"MAX_STEPS": MAX_STEPS, "EVAL_EVERY": EVAL_EVERY, "PATIENCE": PATIENCE, "LR": LR, "BS": BS}}
    pop_models = {}
    for l in pops:
        pop_models[l], info = train_population_policy(sets[l]["train"], sets[l]["val"],
                                                      f"pop3_{l.replace(':', '_')}", out_dir, dev=dev)
        results["training"][l] = {k: v for k, v in info.items() if k != "curve"}
        uu = np.concatenate(sets[l]["test_u"])
        tt = np.concatenate(sets[l]["test_tau"])
        results["populations"][l] = {
            "n_entities": len(pops[l]), "split": sets[l]["split"],
            "n_test_windows": len(sets[l]["test"]),
            "KS_vs_uniform_test": round(float(kstest(uu, "uniform").statistic), 4),
            "mean_tau_test": round(float(tt.mean()), 3),
            "small_n_flag": len(pops[l]) < 10}
        print(f"  {l}: KS_test={results['populations'][l]['KS_vs_uniform_test']} "
              f"best_step={info['best_step']} best_val={info['best_val']}", flush=True)
    return results, pop_models


def twitter_auc_readout(results, pops, splits, sets, resc, pop_models, dev):
    org_test = sets["TW:organic"]["test_users"]
    for op in TW_COVERT:
        m = pop_models[op]
        pos = acct_llr(op, m, sets[op]["test_users"], resc, dev, tail=(len(splits[op]["hold"]) == 0))
        neg = acct_llr("TW:organic", m, org_test[:ORGANIC_NEG_CAP], resc, dev)
        y = [1] * len(pos) + [0] * len(neg)
        results["auc_llr"][f"{op}_vs_organic_test"] = {
            "auc": round(float(roc_auc_score(y, pos + neg)), 3), "n_pos": len(pos), "n_neg": len(neg)}
        print(f"  AUC {op}: {results['auc_llr'][f'{op}_vs_organic_test']}", flush=True)


def excess_nll_matrix(results, pops, sets, pop_models, dev):
    labels = list(pops)
    self_nll = {l: eval_nll(pop_models[l], sets[l]["test"], dens=True, dev=dev) for l in labels}
    M = {}
    for a in labels:
        M[a] = {b: round(eval_nll(pop_models[b], sets[a]["test"], dens=True, dev=dev) - self_nll[a], 4) for b in labels}
        M[a]["Exp(1) null"] = round(results["populations"][a]["mean_tau_test"] - self_nll[a], 4)
    results["excess_nll_matrix"] = M
    results["self_nll_test"] = {l: round(v, 4) for l, v in self_nll.items()}


def reddit_zero_shot(variant: Variant, results, pops, splits, sets, resc, pop_models, dev):
    """Zero-shot cross-platform readouts for Reddit in rescaled time (v4 / v5)."""
    rd_ira, rd_org = variant.zero_shot
    results["reddit_zero_shot"] = {}
    rd_pos_users = list(pops[rd_ira])             # no Reddit-IRA policy involved -> all accounts are held-out
    rd_neg_users = splits[rd_org]["hold"]         # not used to fit the Reddit baseline

    def mean_tau_scores(l, users):
        return [float(np.maximum(resc[l][u][0], 1e-8).mean()) for u in users]

    mt_pos, mt_neg = mean_tau_scores(rd_ira, rd_pos_users), mean_tau_scores(rd_org, rd_neg_users)
    y = [1] * len(mt_pos) + [0] * len(mt_neg)
    results["reddit_zero_shot"]["mean_tau_modelfree"] = {
        "auc": round(float(roc_auc_score(y, [-v for v in mt_pos + mt_neg])), 3),
        "n_pos": len(mt_pos), "n_neg": len(mt_neg),
        "median_tau_pos": round(float(np.median(mt_pos)), 3), "median_tau_neg": round(float(np.median(mt_neg)), 3)}
    print("  RD zero-shot mean-tau:", results["reddit_zero_shot"]["mean_tau_modelfree"], flush=True)
    for op in TW_COVERT + ("TW:organic", rd_ira):
        m = pop_models[op]
        pos_u = rd_pos_users if op != rd_ira else sets[rd_ira]["test_users"]
        pos = acct_llr(rd_ira, m, pos_u, resc, dev)
        neg = acct_llr(rd_org, m, rd_neg_users, resc, dev)
        y = [1] * len(pos) + [0] * len(neg)
        results["reddit_zero_shot"][f"{op}_policy_on_RD"] = {
            "auc": round(float(roc_auc_score(y, pos + neg)), 3),
            "n_pos": len(pos), "n_neg": len(neg),
            "median_llr_pos": round(float(np.median(pos)), 3), "median_llr_neg": round(float(np.median(neg)), 3)}
        print(f"  RD zero-shot {op}:", results["reddit_zero_shot"][f"{op}_policy_on_RD"], flush=True)


def run(variant: Variant, out_dir: Optional[str] = None) -> dict:
    out_dir = out_dir or config.IO_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    dev = device()
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)        # FB newsroom sample, then per-population splits
    pit_rng = np.random.default_rng(1)    # randomized PIT, consumed in population order

    pops = build_populations(variant, out_dir, rng)
    splits = make_splits(pops, rng)
    base_models = fit_baselines(variant, pops, splits, out_dir, dev)
    resc = rescale_all(variant, pops, splits, base_models, out_dir, pit_rng, dev)
    sets = build_sets(pops, splits, resc)
    results, pop_models = fit_policies_and_metrics(pops, splits, sets, out_dir, dev)
    twitter_auc_readout(results, pops, splits, sets, resc, pop_models, dev)
    excess_nll_matrix(results, pops, sets, pop_models, dev)
    if variant.zero_shot is not None:
        reddit_zero_shot(variant, results, pops, splits, sets, resc, pop_models, dev)
    json.dump(results, open(_p(out_dir, variant.output), "w"), indent=1)
    print("DONE", flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="v5",
                    help="population variant (default v5: volume-matched Reddit baseline)")
    ap.add_argument("--out-dir", default=None,
                    help=f"working directory for inputs, artefacts, and results (default IO_RESULTS_DIR={config.IO_RESULTS_DIR})")
    args = ap.parse_args(argv)
    run(VARIANTS[args.variant], args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
