"""Alizadeh et al. (2020, Sci. Adv.) content-based detector, reimplemented on the
Table 2 cohorts to obtain an account-level AUC (they report only macro-F1).

Their design, followed here: every post is a "post-URL pair" represented by
human-interpretable features in five groups, classified by an out-of-the-box
random forest (1000 trees, max_features=sqrt, no tuning), trained on troll vs
control posts at a 1:2 ratio, with posts as the evaluation unit.  For task 5
(Twitter -> Reddit) they use only the features shared by both platforms.
  (i)   content:        word / char counts, URL / hashtag / mention counts,
                        punctuation and case ratios, 20 LDA topic proportions
  (ii)  meta-content:   share of tokens in the top-25 troll words / bigrams and
                        top-25 control words / bigrams of the training month
  (iii) URL domain:     TLD, news / social / platform-internal domain flags
  (iv)  meta URL domain: first-URL domain in the top-25 troll / control domains
  (v)   timing:         hour of day and day of week of the post
What we cannot reproduce: LIWC (proprietary), their curated political / news
domain lists, URL expansion, and their exact 859 mutual features.  What we
add for comparability with Table 2: an account score = mean post probability,
and AUC on the same account cohorts as the policy and BLOC rows (rep-task
70/30 Twitter split; 96 Reddit IRA vs 168 matched / 643 organic).  We also
report their own statistic, post-level macro-F1 at threshold 0.5.

Stage A caches posts per population to IO_RESULTS_DIR/alz_posts_{pop}.jsonl
(user, ts, text, urls, is_reply); <= POST_CAP posts per account (seeded reservoir).
Stage B fits and evaluates.

Run:    python -m xplat.adaptation.alizadeh_reimpl
Output: alizadeh_reimpl.json (in IO_RESULTS_DIR)
"""
from __future__ import annotations

import argparse
import bz2
import glob
import json
import os
import re
import zlib
from collections import Counter
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, roc_auc_score

from ..cohorts import load_archived_splits, matched_reddit_controls, reddit_cv_splits
from .. import config
from ..config import IRA_TWEETS_FILE, RD_WIN, SEED, io_path, takedown_path
from ..evaluation import auc_ci
from ..log import log

POST_CAP, N_TREES, N_TOPICS, N_JOBS = 100, 1000, 20, 32

# ------------------------------------------------------------------ stage A: posts
URL_RE = re.compile(r"https?://\S+")


def reservoir_add(store, user, post, seen):
    """Seeded per-account reservoir of POST_CAP posts."""
    n = seen[user] = seen.get(user, 0) + 1
    lst = store.setdefault(user, [])
    if len(lst) < POST_CAP:
        lst.append(post)
    else:
        j = np.random.default_rng(zlib.crc32(f"{user}|{n}".encode())).integers(0, n)
        if j < POST_CAP:
            lst[j] = post


def to_epoch(x):
    t = pd.to_datetime(x, errors="coerce", utc=True)
    return None if pd.isna(t) else int(t.value // 10**9)


def parse_list(s):
    if not isinstance(s, str) or not s.strip():
        return []
    s = s.strip().strip("[]")
    return [x.strip().strip("'\"") for x in s.split(",") if x.strip().strip("'\"")]


def collect_tw_ira(members):
    store, seen = {}, {}
    use = ["user_screen_name", "tweet_text", "tweet_time", "urls", "in_reply_to_tweetid"]
    for c in pd.read_csv(takedown_path(IRA_TWEETS_FILE), dtype=str, usecols=use, chunksize=500_000,
                         lineterminator="\n", on_bad_lines="skip"):
        sn = c["user_screen_name"].str.lower()
        m = sn.isin(members)
        tsv = pd.to_datetime(c["tweet_time"], errors="coerce", utc=True)
        tsv = (tsv.astype("int64", errors="ignore") // 10**9).where(tsv.notna(), -1) if hasattr(tsv, "where") else tsv
        for u, txt, ts, urls, rep in zip(sn[m], c["tweet_text"][m].fillna(""), tsv[m], c["urls"][m], c["in_reply_to_tweetid"][m]):
            ts = int(ts)
            if ts <= 0 or len(txt) < 5:
                continue
            reservoir_add(store, u, {"ts": ts, "text": txt, "urls": parse_list(urls) or URL_RE.findall(txt),
                                     "is_reply": int(isinstance(rep, str) and rep != "")}, seen)
    return store


def _scan_bz2(args):
    path, members = args
    out = []
    try:
        with bz2.open(path, "rt", errors="replace") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                uid = str((o.get("user") or {}).get("id_str", ""))
                if uid not in members:
                    continue
                txt = o.get("full_text") or o.get("text") or ""
                if len(txt) < 5:
                    continue
                try:
                    ts = int(datetime.strptime(o.get("created_at", ""), "%a %b %d %H:%M:%S %z %Y").timestamp())
                except Exception:
                    continue
                ents = o.get("entities") or {}
                urls = [e.get("expanded_url") or e.get("url") for e in ents.get("urls", []) if e]
                out.append((uid, {"ts": ts, "text": txt, "urls": [x for x in urls if x],
                                  "is_reply": int(o.get("in_reply_to_status_id") is not None)}))
    except Exception as e:
        print("skip", path, e, flush=True)
    return out


def collect_tw_org(members):
    files = sorted(glob.glob(f"{config.TWITTER_TIMELINE_DIR}/**/*.bz2", recursive=True))
    log(f"organic timelines: {len(files)} files, {N_JOBS} workers")
    store, seen = {}, {}
    with Pool(N_JOBS) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_bz2, [(p, members) for p in files], chunksize=8)):
            for u, post in res:
                reservoir_add(store, u, post, seen)
            if i % 1000 == 0:
                log(f"  {i}/{len(files)} files, {len(store)} accounts")
    return store


def read_csv_safe(p, cols):
    if not os.path.exists(p):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(p, dtype=str, on_bad_lines="skip", lineterminator="\n", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(p, dtype=str, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def _rd_user(args):
    u, folder = args
    posts = []
    sub = read_csv_safe(os.path.join(folder, "all_submission.csv"), ["created_utc", "title", "selftext", "url", "is_self", "domain"])
    for ts, title, self_, url, is_self, dom in sub.itertuples(index=False):
        ts = pd.to_numeric(ts, errors="coerce")
        if pd.isna(ts):
            continue
        txt = (title or "") if not isinstance(title, float) else ""
        if isinstance(self_, str) and self_ not in ("[deleted]", "[removed]"):
            txt = (txt + " " + self_).strip()
        urls = [] if (str(is_self).lower() == "true" or not isinstance(url, str)) else [url]
        urls += URL_RE.findall(txt)
        posts.append({"ts": int(ts), "text": txt, "urls": urls, "is_reply": 0})
    for fn in ("all_user_comment_submission.csv", "all_user_comment_reply_w_agreement.csv"):
        com = read_csv_safe(os.path.join(folder, fn), ["created_utc", "body"])
        for ts, body in com.itertuples(index=False):
            ts = pd.to_numeric(ts, errors="coerce")
            if pd.isna(ts) or not isinstance(body, str) or body in ("[deleted]", "[removed]"):
                continue
            posts.append({"ts": int(ts), "text": body, "urls": URL_RE.findall(body), "is_reply": 1})
    posts = [p for p in posts if RD_WIN[0] <= p["ts"] <= RD_WIN[1] and len(p["text"]) >= 5]
    r = np.random.default_rng(zlib.crc32(u.encode()))
    if len(posts) > POST_CAP:
        posts = [posts[i] for i in sorted(r.choice(len(posts), POST_CAP, replace=False))]
    return u, posts


def collect_rd(members):
    folders = {f.lower(): f for f in os.listdir(config.REDDIT_RECRAWL_DIR)}
    jobs = [(u, os.path.join(config.REDDIT_RECRAWL_DIR, folders[u.lower()])) for u in members if u.lower() in folders]
    log(f"reddit: {len(jobs)} of {len(members)} accounts have folders")
    store = {}
    with Pool(N_JOBS) as pool:
        for u, posts in pool.imap_unordered(_rd_user, jobs, chunksize=4):
            if posts:
                store[u] = posts
    return store


def cached(pop, fn, members):
    path = io_path(f"alz_posts_{pop}.jsonl")
    if os.path.exists(path):
        d = {}
        for line in open(path):
            o = json.loads(line)
            d[o["user"]] = o["posts"]
        log(f"{pop}: cache hit, {len(d)} accounts")
        return d
    d = fn(members)
    with open(path, "w") as f:
        for u, posts in d.items():
            f.write(json.dumps({"user": u, "posts": posts}) + "\n")
    log(f"{pop}: collected {len(d)} accounts, {sum(len(v) for v in d.values())} posts")
    return d


# ------------------------------------------------------------------ stage B: features
TOK_RE = re.compile(r"[a-z][a-z']+")
STOP = set(CountVectorizer(stop_words="english").get_stop_words()) | {"rt", "amp", "http", "https", "co"}
NEWS = {"nytimes.com", "washingtonpost.com", "cnn.com", "foxnews.com", "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "theguardian.com",
        "wsj.com", "usatoday.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com", "politico.com", "thehill.com", "huffpost.com", "huffingtonpost.com",
        "breitbart.com", "dailycaller.com", "nypost.com", "latimes.com", "chicagotribune.com", "bloomberg.com", "time.com", "newsweek.com",
        "buzzfeednews.com", "buzzfeed.com", "vox.com", "salon.com", "slate.com", "theatlantic.com", "npr.org", "rt.com", "sputniknews.com",
        "dailymail.co.uk", "independent.co.uk", "telegraph.co.uk", "aljazeera.com", "msnbc.com", "cnbc.com", "yahoo.com", "news.yahoo.com",
        "infowars.com", "truthfeed.com", "washingtontimes.com", "thegatewaypundit.com", "rawstory.com", "motherjones.com", "theblaze.com"}
SOCIAL = {"twitter.com", "t.co", "facebook.com", "fb.me", "youtube.com", "youtu.be", "instagram.com", "reddit.com", "redd.it", "imgur.com",
          "i.redd.it", "v.redd.it", "i.imgur.com", "vk.com", "tumblr.com", "pinterest.com", "linkedin.com", "twitch.tv", "gfycat.com", "giphy.com"}
PLATFORM = {"twitter.com", "t.co", "reddit.com", "redd.it", "i.redd.it", "v.redd.it"}
TLDS = ["com", "org", "net", "ru", "uk", "info", "co", "us"]


def domain_of(url):
    m = re.match(r"https?://([^/?#]+)", url or "")
    if not m:
        return ""
    d = m.group(1).lower()
    return d[4:] if d.startswith("www.") else d


def tokens(text):
    return [t for t in TOK_RE.findall(URL_RE.sub(" ", text.lower())) if t not in STOP]


class Vocab:
    """Training-set statistics: top-25 words / bigrams / domains per class, LDA."""

    def __init__(self, train_docs, train_y, seed=SEED):
        cw = {0: Counter(), 1: Counter()}
        cb = {0: Counter(), 1: Counter()}
        cd = {0: Counter(), 1: Counter()}
        for d, y in zip(train_docs, train_y):
            tk = tokens(d["text"])
            cw[y].update(tk)
            cb[y].update(zip(tk, tk[1:]))
            dom = domain_of(d["urls"][0]) if d["urls"] else ""
            if dom:
                cd[y][dom] += 1
        self.top = {}
        for y in (0, 1):
            self.top[("w", y)] = set(w for w, _ in cw[y].most_common(25))
            self.top[("b", y)] = set(b for b, _ in cb[y].most_common(25))
            self.top[("d", y)] = set(x for x, _ in cd[y].most_common(25))
        r = np.random.default_rng(seed)
        sub = [train_docs[i]["text"] for i in r.choice(len(train_docs), min(len(train_docs), 40000), replace=False)]
        self.cv = CountVectorizer(max_features=5000, min_df=5, stop_words="english", token_pattern=r"[a-zA-Z][a-zA-Z']+",
                                  preprocessor=lambda s: URL_RE.sub(" ", s.lower()))
        X = self.cv.fit_transform(sub)
        self.lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=seed, max_iter=10, learning_method="online",
                                             batch_size=2048, n_jobs=N_JOBS).fit(X)

    def features(self, docs):
        base = []
        for d in docs:
            t = d["text"]
            tk = tokens(t)
            words = re.findall(r"\S+", URL_RE.sub(" ", t))
            n_w = max(1, len(words))
            bg = list(zip(tk, tk[1:]))
            n_t = max(1, len(tk))
            n_b = max(1, len(bg))
            dom = domain_of(d["urls"][0]) if d["urls"] else ""
            tld = dom.rsplit(".", 1)[-1] if dom else ""
            di = pd.Timestamp(d["ts"], unit="s")
            f = [len(words), len(t), len(d["urls"]), t.count("#"), t.count("@"), t.count("!"), t.count("?"),
                 sum(c.isupper() for c in t) / max(1, len(t)), sum(c.isdigit() for c in t) / max(1, len(t)),
                 np.mean([len(w) for w in words]) if words else 0.0, int(bool(d["urls"])), d["is_reply"],
                 sum(w in self.top[("w", 1)] for w in tk) / n_t, sum(w in self.top[("w", 0)] for w in tk) / n_t,
                 sum(b in self.top[("b", 1)] for b in bg) / n_b, sum(b in self.top[("b", 0)] for b in bg) / n_b,
                 int(dom in NEWS), int(dom in SOCIAL), int(dom in PLATFORM), int(dom != "" and dom not in NEWS and dom not in SOCIAL),
                 int(dom in self.top[("d", 1)]), int(dom in self.top[("d", 0)])]
            f += [int(tld == x) for x in TLDS]
            f += [int(di.hour == h) for h in range(24)] + [int(di.dayofweek == w) for w in range(7)]
            base.append(f)
        base = np.array(base, dtype=float)
        topics = self.lda.transform(self.cv.transform([d["text"] for d in docs]))
        return np.hstack([base, topics])


def docs_of(POSTS, pop, users):
    docs, owners = [], []
    for u in users:
        for p in POSTS[pop].get(u, []):
            docs.append(p)
            owners.append(u)
    return docs, owners


def account_mean(prob, owners):
    s = pd.Series(prob).groupby(pd.Series(owners)).mean()
    return s.to_dict()


def fit_rf(train_pos_docs, train_neg_docs, seed=SEED):
    r = np.random.default_rng(seed)
    if len(train_neg_docs) > 2 * len(train_pos_docs):   # their 1:2 troll:control training ratio
        train_neg_docs = [train_neg_docs[i] for i in r.choice(len(train_neg_docs), 2 * len(train_pos_docs), replace=False)]
    docs = train_pos_docs + train_neg_docs
    y = np.r_[np.ones(len(train_pos_docs)), np.zeros(len(train_neg_docs))].astype(int)
    log(f"  fitting vocab/LDA on {len(docs)} posts")
    V = Vocab(docs, y)
    X = V.features(docs)
    log(f"  fitting RF ({N_TREES} trees) on {X.shape}")
    rf = RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt", n_jobs=N_JOBS, random_state=seed).fit(X, y)
    return V, rf


def score(V, rf, docs):
    return rf.predict_proba(V.features(docs))[:, 1] if docs else np.array([])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    return p.parse_args(argv)


def main(argv=None):
    parse_args(argv)

    # ------------------------------------------------------------------ cohorts (same as xplat_seq_baselines)
    arch = load_archived_splits(usecols=("pop", "user", "split", "n_events"))
    TW = {p: {s: set(arch[(arch["pop"] == p) & (arch["split"] == s)]["user"].str.lower()) for s in ("train", "test")}
          for p in ("TW:IRA", "TW:organic")}
    rd_pos = sorted(arch[arch["pop"] == "RD:IRA"]["user"])
    rd_neg = sorted(arch[arch["pop"] == "RD:organic"]["user"])
    nev = dict(zip(arch["user"], arch["n_events"]))
    rd_neg_matched = matched_reddit_controls(arch, rd_pos, rd_neg, nev)
    log(f"cohorts: TW IRA {len(TW['TW:IRA']['train'])}/{len(TW['TW:IRA']['test'])}, "
        f"organic {len(TW['TW:organic']['train'])}/{len(TW['TW:organic']['test'])}; "
        f"RD {len(rd_pos)} IRA, {len(rd_neg)} organic, {len(rd_neg_matched)} matched (archive: 168)")

    # ------------------------------------------------------------------ stage A: posts
    POSTS = {}
    POSTS["TW:IRA"] = cached("tw_ira", collect_tw_ira, TW["TW:IRA"]["train"] | TW["TW:IRA"]["test"])
    POSTS["TW:organic"] = cached("tw_org", collect_tw_org, TW["TW:organic"]["train"] | TW["TW:organic"]["test"])
    POSTS["RD:IRA"] = cached("rd_ira", collect_rd, set(rd_pos))
    POSTS["RD:organic"] = cached("rd_org", collect_rd, set(rd_neg))

    results = {"post_cap": POST_CAP, "n_trees": N_TREES, "n_topics": N_TOPICS,
               "coverage": {p: [len(POSTS[p]), int(sum(len(v) for v in POSTS[p].values()))] for p in POSTS}}

    # ---- fit on Twitter (train split), evaluate TW test and RD
    log("=== fit on Twitter ===")
    tp, _ = docs_of(POSTS, "TW:IRA", sorted(TW["TW:IRA"]["train"]))
    tn, _ = docs_of(POSTS, "TW:organic", sorted(TW["TW:organic"]["train"]))
    V, rf = fit_rf(tp, tn)
    res = {}
    d_p, o_p = docs_of(POSTS, "TW:IRA", sorted(TW["TW:IRA"]["test"]))
    d_n, o_n = docs_of(POSTS, "TW:organic", sorted(TW["TW:organic"]["test"]))
    s_p, s_n = score(V, rf, d_p), score(V, rf, d_n)
    res["TW->TW|post_auc"] = round(float(roc_auc_score(np.r_[np.ones(len(s_p)), np.zeros(len(s_n))], np.r_[s_p, s_n])), 4)
    res["TW->TW|post_macroF1@0.5"] = round(float(f1_score(np.r_[np.ones(len(s_p)), np.zeros(len(s_n))], np.r_[s_p, s_n] >= 0.5, average="macro")), 4)
    ap, an = account_mean(s_p, o_p), account_mean(s_n, o_n)
    res["TW->TW|account_auc"] = auc_ci(list(ap.values()), list(an.values()))
    log(f"TW->TW: post AUC {res['TW->TW|post_auc']}, post macro-F1 {res['TW->TW|post_macroF1@0.5']}, "
        f"account AUC {res['TW->TW|account_auc']} (n={len(ap)}/{len(an)})")

    d_p, o_p = docs_of(POSTS, "RD:IRA", rd_pos)
    d_n, o_n = docs_of(POSTS, "RD:organic", rd_neg)
    s_p, s_n = score(V, rf, d_p), score(V, rf, d_n)
    res["TW->RD|post_auc"] = round(float(roc_auc_score(np.r_[np.ones(len(s_p)), np.zeros(len(s_n))], np.r_[s_p, s_n])), 4)
    res["TW->RD|post_macroF1@0.5"] = round(float(f1_score(np.r_[np.ones(len(s_p)), np.zeros(len(s_n))], np.r_[s_p, s_n] >= 0.5, average="macro")), 4)
    ap, an = account_mean(s_p, o_p), account_mean(s_n, o_n)
    res["TW->RD|account_auc|unmatched"] = auc_ci(list(ap.values()), list(an.values()))
    res["TW->RD|account_auc|matched"] = auc_ci(list(ap.values()), [an[u] for u in rd_neg_matched if u in an])
    res["TW->RD|n_accounts"] = [len(ap), len(an), len([u for u in rd_neg_matched if u in an])]
    log(f"TW->RD: post AUC {res['TW->RD|post_auc']}, post macro-F1 {res['TW->RD|post_macroF1@0.5']}, "
        f"account AUC matched {res['TW->RD|account_auc|matched']} unmatched {res['TW->RD|account_auc|unmatched']} (n={res['TW->RD|n_accounts']})")
    results["fit_on_twitter"] = res

    # ---- within Reddit: 5-fold by account (first repeat of the rd_inplatform splits)
    log("=== 5-fold CV within Reddit ===")
    rd_users = rd_pos + rd_neg
    rd_y = np.r_[np.ones(len(rd_pos)), np.zeros(len(rd_neg))].astype(int)
    splits = reddit_cv_splits(rd_y)[:5]
    oof = {}
    for k, (tr, te) in enumerate(splits):
        trp, _ = docs_of(POSTS, "RD:IRA", [rd_users[i] for i in tr if rd_y[i] == 1])
        trn, _ = docs_of(POSTS, "RD:organic", [rd_users[i] for i in tr if rd_y[i] == 0])
        Vk, rfk = fit_rf(trp, trn, seed=SEED + k)
        for pop, lab in (("RD:IRA", 1), ("RD:organic", 0)):
            d, o = docs_of(POSTS, pop, [rd_users[i] for i in te if rd_y[i] == lab])
            oof.update(account_mean(score(Vk, rfk, d), o))
        log(f"  fold {k} done")
    pos = [oof[u] for u in rd_pos if u in oof]
    neg = [oof[u] for u in rd_neg if u in oof]
    results["RD->RD|account_auc|5fold"] = auc_ci(pos, neg)
    results["RD->RD|n_accounts"] = [len(pos), len(neg)]
    log(f"RD->RD: account AUC {results['RD->RD|account_auc|5fold']} (n={len(pos)}/{len(neg)})")
    json.dump(results, open(io_path("alizadeh_reimpl.json"), "w"), indent=1)
    log("DONE -> alizadeh_reimpl.json")
    return 0


if __name__ == "__main__":
    main()
