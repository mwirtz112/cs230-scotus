"""
Walk-forward evaluation: the test over terms 2018-2025, and its rehearsal on 2012-2017.

For every scored term T, each method is refit on the cases of earlier terms that had been
decided before term T opened (1 October of year T), and then predicts every case of term
T. Nothing from term T or later is used: no label, vocabulary, scaling statistic, class
weight or cutoff. The network recipe is fixed (model v1: train from 1980, 6 epochs, no
early stopping, 5 seeds averaged, cutoff 0.5), so no step needs a validation set.

    python -m scotus.evaluation.walk_forward              # rehearsal on terms 2012-2017: train and validation
                                        # terms only, the test labels are never read
    python -m scotus.evaluation.walk_forward --report     # rebuild the tables from the saved predictions
    python -m scotus.evaluation.walk_forward --terms 2017 --methods v1 --run diag_min30 --extra --min-justice-votes 30
                                        # a diagnostic: a changed recipe on rehearsal terms,
                                        # in its own output folder (never with --final)
    python -m scotus.evaluation.walk_forward --dry-run    # the test-run code path, end to end, on PLACEHOLDER
                                        # labels for 2018-25: a crash test, nothing is scored
Run from the repository root:
    python -m scotus.evaluation.walk_forward --final      # terms 2018-2025: THE test run. Once, for the final
                                        # report. First: python -I -m scotus.data.prepare_scdb --final
                                        #        then:  python -m scotus.features.meta_features

The test run was made once (results/walk_forward_final.md); the model, the recipe and the
baselines were fixed before it.

Two rows per method:
    each term   refit before every term (the primary result)
    once        the first step's fit applied unchanged to every later term: what a model
                trained once would do. The gap between the rows is the cost of going stale.

Held-over cases. Oyez files a reargued case under the term of its first argument, so a few
cases "of" an earlier term were decided after term T opened (Citizens United: term 2008,
decided January 2010). They are left out of step T's training set and enter at the first
step after their decision. Decision days come from the SCDB history table; the 20 old
cases without one are taken as decided within their own term.

The test lock. Without --final, only rows of the train and validation splits are read
from data/, and data/ext/ (terms 2020-25) is not opened. --dry-run opens the locked rows
without their label columns and fills in coin-flip placeholders, writes to a temporary
folder and prints no score.

Outputs
    data/preds/wf_<run>/              per-step predictions (gitignored)
    results/wf_runs/<run>/            one record per trained network (loss curve, settings)
    results/walk_forward_<run>.md     the tables
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score, roc_auc_score

from scotus.baselines import baselines as B
from scotus.evaluation import eval as ev
from scotus.baselines import flat_baseline as FB
from scotus.baselines import meta_baselines as MB
from scotus.features import meta_features as M
from scotus.baselines import qc_plus as Q
from scotus.evaluation import summarize_v1 as S
from scotus.models import train_v1 as T
from scotus.features.features import WIDE_FEATURES

REHEARSAL_TERMS = list(range(2012, 2018))
TEST_TERMS = list(range(2018, 2026))
FIRST_LOCKED = 2018
SEEDS = [0, 1, 2, 3, 4]
# A recipe is a train_v1.py command line. "v1" is the main run (v1_main), "v1_fusion" adds
# the metadata prior on the logit.
RECIPES = {"v1": ["--epochs", "6", "--train-start", "1980", "--affinity", "4"],
           "v1_fusion": ["--epochs", "6", "--train-start", "1980", "--affinity", "4",
                         "--prior", "1"]}
GROUPS = {"base": ["always_petitioner", "justice_rate", "question_count", "qc_plus", "qc_bloc"],
          "flat": ["flat_mb", "flat_mb_all"],     # frozen ModernBERT document vectors
          "meta": ["katz_rf", "qc_bloc_meta"], "v1": ["v1"], "v1_fusion": ["v1_fusion"]}
BLENDS = {"v1+qc_bloc": ["v1", "qc_bloc"], "v1+qc_bloc_meta": ["v1", "qc_bloc_meta"]}
SAME_EVERY_STEP = ("always_petitioner", "question_count")       # nothing is fitted
LABEL = {"always_petitioner": "Always petitioner", "justice_rate": "Justice vote rate",
         "question_count": "Question count",
         "flat_mb": "Flat ModernBERT, first 8k tokens (frozen)",
         "flat_mb_all": "Flat ModernBERT, whole transcript (frozen)",
         "katz_rf": "Case profile only, Katz-style forest (katz_rf)",
         "qc_plus": "Question features, LR (qc_plus)",
         "qc_bloc": "+ named colleagues, LR (qc_bloc)",
         "qc_bloc_meta": "+ case profile history, LR (qc_bloc_meta)",
         "v1": "Model v1", "v1_fusion": "Model v1 + metadata prior",
         "v1+qc_bloc": "Model v1 + qc_bloc (mean of logits)",
         "v1+qc_bloc_meta": "Model v1 + qc_bloc_meta (mean of logits)"}
ORDER = ["always_petitioner", "justice_rate", "question_count", "flat_mb", "flat_mb_all", "katz_rf",
         "qc_plus",
         "qc_bloc",
         "qc_bloc_meta", "v1", "v1_fusion", "v1+qc_bloc", "v1+qc_bloc_meta"]
# the same methods in the standard folds (data/preds/<stem>_{dev,val}...), for reference
FOLD_STEM = {"v1": "v1_main", "v1_fusion": "v1_fusion", "v1+qc_bloc": "blend:v1_main+qc_bloc",
             "v1+qc_bloc_meta": "blend:v1_main+qc_bloc_meta"}
TURN_COLS = ["case_id", "convo_id", "turn_idx", "speaker_id", "role"]
KEY = ["case_id", "justice"]


# ---------------------------------------------------------------- data

def read_split(path, splits, drop=()):
    """Rows of the given splits. Columns in `drop` are not read from the file at all."""
    cols = [c for c in pq.read_schema(path).names if c not in drop]
    return pd.read_parquet(path, columns=cols, filters=[("split", "in", splits)])


def placeholder(keys):
    """A coin flip per row that depends only on the row's ids (dry run)."""
    return (pd.util.hash_pandas_object(keys, index=False).to_numpy() % 2).astype(np.int64)


def load_vectors(paths, lengths, device):
    """Standardized fp16 turn vectors of the corpora, one after the other, and a final
    all-zero padding row. Scaling: the statistics every v1 run uses (training split of the
    main corpus, 1955-2015), a fixed constant here."""
    stats = Path(paths[0]).with_name(Path(paths[0]).stem + "_train_stats.npz")
    s = np.load(stats)
    mean, std = s["mean"], s["std"]
    out = torch.zeros((sum(lengths) + 1, len(mean)), dtype=torch.float16)
    at = 0
    for path, n in zip(paths, lengths):
        E = np.load(path, mmap_mode="r")
        assert len(E) == n, f"{path}: {len(E)} vectors for {n} turns"
        for i in range(0, n, 200_000):
            z = (E[i:i + 200_000].astype(np.float32) - mean) / std
            out[at + i:at + i + len(z)] = torch.from_numpy(z.astype(np.float16))
        at += n
    return out.to(device)


def load_corpus(args, locked, fake, device=None):
    """The tables every step works on. device=None: no turn vectors (report only).

    locked=False (rehearsal): the train and validation splits of the main corpus.
    locked=True (--final, --dry-run): plus the 2018-19 test split and the 2020-25
    extension, which become one test period. With fake=True their vote and winner
    columns are never read; placeholders stand in for them.
    """
    data, ext = Path(args.data), Path(args.ext)
    cases = read_split(data / "cases.parquet", ["train", "val"])
    votes = read_split(data / "votes.parquet", ["train", "val"])
    turns = pd.read_parquet(data / "turns.parquet", columns=TURN_COLS)
    feats = pd.read_parquet(data / "turn_feats.parquet")
    vote_feats = pd.read_parquet(data / "vote_feats.parquet")
    emb, lengths = [args.emb], [len(turns)]
    if locked:
        hide_c, hide_v = (("win_side",), ("vote",)) if fake else ((), ())
        more_c = [read_split(data / "cases.parquet", ["test"], hide_c),
                  read_split(ext / "cases.parquet", ["ext"], hide_c)]
        more_v = [read_split(data / "votes.parquet", ["test"], hide_v),
                  read_split(ext / "votes.parquet", ["ext"], hide_v)]
        if fake:
            for c in more_c:
                c["win_side"] = placeholder(c.case_id)
            for v in more_v:
                v["vote"] = placeholder(v.case_id + "|" + v.justice)
        assert not set(more_c[1].case_id) & (set(cases.case_id) | set(more_c[0].case_id))
        xt = pd.read_parquet(ext / "turns.parquet", columns=TURN_COLS)
        xf = pd.read_parquet(ext / "turn_feats.parquet")
        ix = pd.read_parquet(Path(args.emb_ext).with_name("index.parquet"))
        assert len(xt) == len(xf) == len(ix) and (xt.case_id.to_numpy() == ix.case_id.to_numpy()).all() \
            and (xt.turn_idx.to_numpy() == ix.turn_idx.to_numpy()).all(), "extension tables out of step"
        xt.index = xf.index = np.arange(len(turns), len(turns) + len(xt))
        turns, feats = pd.concat([turns, xt]), pd.concat([feats, xf])
        vote_feats = pd.concat([vote_feats, pd.read_parquet(ext / "vote_feats.parquet")],
                               ignore_index=True)
        cases = pd.concat([cases, *more_c], ignore_index=True)
        votes = pd.concat([votes, *more_v], ignore_index=True)
        emb.append(args.emb_ext); lengths.append(len(xt))
    cols = ["case_id", "year", "justice", "vote"]
    D = {"cases": cases[["case_id", "year", "win_side", "convo_ids"]].reset_index(drop=True),
         "votes": votes[cols].reset_index(drop=True), "turns": turns, "feats": feats,
         "vote_feats": vote_feats, "stance": None,
         "E": load_vectors(emb, lengths, device) if device else None}
    assert not D["votes"].duplicated(KEY).any() and D["cases"].case_id.is_unique
    return D


def decision_dates(args, cases, locked, final):
    """Day each case was decided (a Series by case id) and how many had to be assumed."""
    data = Path(args.data)
    sc = [pd.read_parquet(data / "scdb_cases.parquet", columns=["case_id", "scdb_case_id"])]
    if locked and (Path(args.ext) / "scdb_cases.parquet").exists():
        sc.append(pd.read_parquet(Path(args.ext) / "scdb_cases.parquet",
                                  columns=["case_id", "scdb_case_id"]))
    hv = pd.read_parquet(data / "scdb_votes.parquet", columns=["scdb_case_id", "date_decision"])
    hv = hv.dropna(subset=["scdb_case_id"]).drop_duplicates("scdb_case_id")
    d = cases[["case_id", "year"]].merge(pd.concat(sc), on="case_id", how="left").merge(
        hv, on="scdb_case_id", how="left")
    missing = d.date_decision.isna()
    if final and (missing & (d.year >= 2000)).any():
        raise SystemExit("Decision dates are missing for recent cases: run "
                         "`python -I prepare_scdb.py --final` and `python meta_features.py` first.")
    own_term = pd.to_datetime((d.year + 1).astype(str) + "-06-30")
    return pd.Series(d.date_decision.fillna(own_term).to_numpy(), index=d.case_id.to_numpy()), \
        int(missing.sum())


def plan(D, decided, terms):
    """One entry per step: what may be trained on and what is predicted."""
    cases = D["cases"]
    dec = decided.reindex(cases.case_id).to_numpy()
    steps = []
    for i, term in enumerate(terms):
        earlier = (cases.year < term).to_numpy()
        known = earlier & (dec < np.datetime64(f"{term}-10-01"))
        steps.append({"term": term, "fold": f"wf{term}", "hi": terms[-1] if i == 0 else term,
                      "known": set(cases.case_id[known]),
                      "held_over": sorted(cases.case_id[earlier & ~known])})
    return steps


# ---------------------------------------------------------------- one step

def save(preds, path):
    preds[["case_id", "justice", "p", "pred"]].to_parquet(path, index=False)


def meta_fold(term, fake):
    """Which history protocol in data/meta_hist.parquet belongs to this step."""
    have = set(pq.read_table("data/meta_hist.parquet", columns=["protocol"])
               .column("protocol").unique().to_pylist())
    if f"frozen_wf{term}" in have:
        return f"wf{term}"
    if not fake:
        raise SystemExit(f"No voting history as of the {term} term in data/meta_hist.parquet: "
                         "run `python -I prepare_scdb.py --final` and `python meta_features.py`.")
    return max(p[7:] for p in have if p.startswith("frozen_wf") and int(p[9:]) <= term)


def run_step(step, D, args, groups, seeds, recipes, preds_dir, runs_dir, device, fake):
    term, fold, hi = step["term"], step["fold"], step["hi"]
    votes, cases, vf = D["votes"], D["cases"], D["vote_feats"]
    usable = votes.case_id.isin(step["known"]) | (votes.year >= term)
    v = votes[usable].reset_index(drop=True)
    train = v[v.year < term]
    held = v[(v.year >= term) & (v.year <= hi)].reset_index(drop=True)
    done = lambda name: (preds_dir / f"{name}_{fold}.parquet").exists()
    t0 = time.time()

    if "base" in groups and not all(done(m) for m in GROUPS["base"]):
        ids = set(held.case_id)
        counts = B.count_questions(B.ordered_turns(D["turns"][D["turns"].case_id.isin(ids)],
                                                   cases[cases.case_id.isin(ids)]))
        tr, he = train.merge(vf, on=KEY), held.merge(vf, on=KEY)
        out = {"always_petitioner": B.always_petitioner(held),
               "justice_rate": B.justice_rate(held, train)[0],
               "question_count": B.question_count(held, counts),
               "qc_plus": Q.fit_predict(tr, [he])[0][0],
               "qc_bloc": Q.fit_predict_bloc(tr, [he])[0]}
        for name, p in out.items():
            assert len(p) == len(held), name
            save(p, preds_dir / f"{name}_{fold}.parquet")

    for name in (GROUPS["flat"] if "flat" in groups else []):
        if name in D["flat"] and not done(name):
            # like the networks: fitted on terms from 1980, five seeds averaged
            save(FB.fit_predict(train[train.year >= FB.TRAIN_START], held, D["flat"][name],
                                FB.chosen(name)), preds_dir / f"{name}_{fold}.parquet")

    if "meta" in groups and not all(done(m) for m in GROUPS["meta"]):
        prof, hist = M.load(args.data, meta_fold(term, fake), "frozen")
        jcode = {j: i for i, j in enumerate(sorted(votes.justice.unique()))}
        df = v.merge(hist, on=KEY).merge(prof, on="case_id").merge(vf, on=KEY)
        df["justice_code"] = df.justice.map(jcode)
        tr, te = df[df.year < term], df[(df.year >= term) & (df.year <= hi)]
        assert len(te) == len(held), "history features are missing for scored votes"
        p_rf, _ = MB.forest(tr, te, ["justice_code"] + MB.PROFILE + MB.H)
        save(te[KEY].assign(p=p_rf, pred=(p_rf >= 0.5).astype(int)),
             preds_dir / f"katz_rf_{fold}.parquet")
        save(Q.fit_predict_bloc(tr, [te], pool_feats=WIDE_FEATURES + MB.H)[0],
             preds_dir / f"qc_bloc_meta_{fold}.parquet")

    T.FOLDS[fold] = (term - 1, (term, hi))
    for name in [n for g in groups for n in GROUPS[g] if n in recipes]:
        a = T.make_parser().parse_args(["--name", name, *recipes[name]])
        a.data = args.data
        step_D = {**D, "votes": v, "cases": cases[cases.case_id.isin(set(v.case_id))]}
        if a.prior or a.hist or a.meta:
            step_D["meta_fold"] = meta_fold(term, fake)
        for seed in seeds:
            if not (preds_dir / f"{name}_{fold}_s{seed}.parquet").exists():
                T.train_one(a, fold, seed, step_D, device, runs_dir=runs_dir,
                            preds_dir=preds_dir, track=False)
    print(f"step {term}: fit on {train.case_id.nunique():,} cases "
          f"({(cases.year[cases.case_id.isin(step['known'])] >= 1980).sum():,} since 1980), "
          f"{len(step['held_over'])} held over; predicted {held.case_id.nunique()} cases of "
          f"{term}{'' if hi == term else f'-{hi}'}; {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- collect and score

def step_preds(name, term, preds_dir, seeds):
    one = preds_dir / f"{name}_wf{term}.parquet"
    if one.exists():
        return pd.read_parquet(one)
    paths = [preds_dir / f"{name}_wf{term}_s{s}.parquet" for s in seeds]
    return S.ensemble(paths) if all(p.exists() for p in paths) else None


def assemble(name, terms, votes, preds_dir, seeds):
    """(each-term, once) prediction tables over the scored terms; None if a step is missing."""
    if name in BLENDS:
        parts = [assemble(n, terms, votes, preds_dir, seeds) for n in BLENDS[name]]
        if any(p is None for p in parts):
            return None
        return tuple(S.blend_many([p[i] for p in parts]) for i in (0, 1))
    steps = [step_preds(name, t, preds_dir, seeds) for t in terms]
    if any(s is None for s in steps):
        return None
    keys = votes[votes.year.between(terms[0], terms[-1])][["case_id", "year", "justice"]]
    walk = pd.concat([keys[keys.year == t].merge(s, on=KEY) for t, s in zip(terms, steps)],
                     ignore_index=True)
    once = keys.merge(steps[0], on=KEY)
    assert len(walk) == len(once) == len(keys), f"{name}: predictions do not cover every vote"
    return walk, once


def paired(a, b, v, n=1000, seed=0):
    """b minus a on the same votes: AUC and macro-F1 differences with 95% CIs from a
    bootstrap over cases (the same resamples for both)."""
    d = v.merge(a[KEY + ["p", "pred"]], on=KEY).merge(b[KEY + ["p", "pred"]], on=KEY,
                                                      suffixes=("_a", "_b"))
    code, cases = pd.factorize(d.case_id)
    diff = lambda w: (roc_auc_score(d.vote, d.p_b, sample_weight=w)
                      - roc_auc_score(d.vote, d.p_a, sample_weight=w),
                      f1_score(d.vote, d.pred_b, average="macro", sample_weight=w)
                      - f1_score(d.vote, d.pred_a, average="macro", sample_weight=w))
    rng = np.random.default_rng(seed)
    draws = np.array([diff(rng.multinomial(len(cases), np.full(len(cases), 1 / len(cases)))[code])
                      for _ in range(n)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    point = diff(None)
    return [f"{point[i]:+.3f} [{lo[i]:+.3f}, {hi[i]:+.3f}]" for i in (0, 1)]


def labels(D, lo, hi):
    v, c = D["votes"], D["cases"]
    v = v[v.year.between(lo, hi)].reset_index(drop=True)
    return v, c[c.case_id.isin(set(v.case_id))][["case_id", "year", "win_side"]].reset_index(drop=True)


def at_rate(p, v, rates, step=None):
    """The second cutoff rule (the train-rate cutoff, per term): within each
    predicted term, cut where the share of predicted petitioner votes equals the
    pro-petitioner share of the votes the model was fitted on (`rates[term]`, or
    `rates[step]` for a model fitted once). Uses the term's predictions, not its labels."""
    q = v[KEY + ["year"]].merge(p[KEY + ["p"]], on=KEY)
    prob, pred = q.p.to_numpy(), np.zeros(len(q), dtype=int)
    for term, idx in q.groupby("year").indices.items():
        pred[idx] = prob[idx] >= np.quantile(prob[idx], 1 - rates[step or term])
    return q.assign(pred=pred)


NOT_BALANCED = {"always_petitioner", "justice_rate", "question_count"}   # no class-balanced p


def restored(p, v, rates, step=None):
    """Votes called with the base rate put back (fixed before the test run).

    The class-balanced loss takes the petitioner's base rate out of p: true odds = model
    odds * pi / (1 - pi), pi = the pro-petitioner share of the votes the model was fitted
    on. So a vote counts for the petitioner when p > 1 - pi. No fitted number, and nothing
    of the predicted term is used. Only the case call built on these votes is reported."""
    q = v[KEY + ["year"]].merge(p[KEY + ["p"]], on=KEY)
    pi = q.year.map(lambda term: rates[step or term])
    return q.assign(pred=(q.p > 1 - pi).astype(int))


def score_table(tables, v, c, rates, fitted_at, n_boot=ev.N_BOOT):
    """fitted_at: fit label -> the step whose training votes a once-fitted model saw."""
    rows = []
    for (name, fit), p in tables.items():
        r = ev.score(p, v, c, n_boot=n_boot)
        q = v[KEY].merge(p, on=KEY)
        rr = ev.score(at_rate(p, v, rates, fitted_at.get(fit)), v, c, n_boot=0)
        back = "–" if name in NOT_BALANCED else "{:.3f}".format(
            ev.score(restored(p, v, rates, fitted_at.get(fit)), v, c, n_boot=0)["case_acc"][0])
        rows.append({"Method": LABEL[name], "Fit": fit, "Macro-F1": ev.fmt_ci(*r["macro_f1"]),
                     "AUC": ev.fmt_ci(*r["auc"]), "Vote acc": f"{r['vote_acc'][0]:.3f}",
                     "Case acc": f"{r['case_acc'][0]:.3f}",
                     "Pred. petitioner %": f"{100 * q.pred.mean():.1f}",
                     "Macro-F1 @rate": f"{rr['macro_f1'][0]:.3f}",
                     "Case acc @rate": f"{rr['case_acc'][0]:.3f}",
                     "Case acc, base rate restored": back})
    return pd.DataFrame(rows).to_markdown(index=False)


def report(D, steps, terms, seeds, preds_dir, path, mode, n_assumed):
    """All tables of one walk-forward run -> a markdown file. Returns the text."""
    votes, first = D["votes"], terms[0]
    each, once = "each term", f"once (terms ≤ {first - 1})"
    got = {n: assemble(n, terms, votes, preds_dir, seeds) for n in ORDER}
    got = {n: g for n, g in got.items() if g is not None}
    tables = {}
    for n, (walk, frozen) in got.items():
        if n in SAME_EVERY_STEP:
            tables[(n, "–")] = walk
        else:
            tables[(n, each)], tables[(n, once)] = walk, frozen
    lo, hi = terms[0], terms[-1]
    v_all, c_all = labels(D, lo, hi)
    # pro-petitioner share of the votes each step was fitted on, for the second cutoff rule
    rates = {s["term"]: float(votes.vote[votes.case_id.isin(s["known"])].mean()) for s in steps}
    fitted_at = {once: first, "dev fold (terms ≤ 2011)": 2012,
                 "validation fold (terms ≤ 2015)": 2016}
    what = {"rehearsal": "Rehearsal on train and validation terms. The test terms (2018–25) "
                         "were not read.",
            "final": "THE TEST RUN (terms 2018–2025), made once for the final report.",
            "custom": "A partial run on train and validation terms."}[mode]
    md = [f"# Walk-forward evaluation, terms {lo}–{hi}", "",
          f"{what} For each term every method is refit on the cases of earlier terms decided "
          "before that term opened (1 October), then predicts the term. 'each term' = that "
          f"refit; 'once' = the fit made before {first}, applied unchanged to every term "
          "(the two agree on the first term by construction). Network rows are the "
          f"mean-probability ensemble of {len(seeds)} seeds of the fixed recipe (train from "
          "1980, 6 epochs, no early stopping). Cutoff 0.5 (justice vote rate: the training "
          "rate). Point estimate [95% bootstrap CI over cases]. '@rate' columns: the same "
          "predictions cut, within each term, where the share of predicted petitioner "
          "votes equals the pro-petitioner share of the votes the model was fitted on "
          f"({100 * min(rates.values()):.1f}–{100 * max(rates.values()):.1f}%); this uses "
          "the term's predictions but none of its labels. 'Case acc, base rate restored': "
          "the case call when a vote counts for the petitioner if p > 1 − π, π being that "
          "same training share. The class-balanced loss takes the base rate out of p and "
          "this puts it back, with no fitted number and nothing of the predicted term; a "
          "secondary column, fixed before the test run (notes §19.7). '–' for the methods "
          "without a class-balanced probability.", "",
          f"## All scored terms, {lo}–{hi} ({len(c_all)} cases, {len(v_all):,} votes)", "",
          score_table(tables, v_all, c_all, rates, fitted_at), ""]

    blocks = {tuple(REHEARSAL_TERMS): [("2012–15, the dev terms", 2012, 2015),
                                       ("2016–17, the validation terms", 2016, 2017)],
              tuple(TEST_TERMS): [("2018–19, the test set of the approved design", 2018, 2019),
                                  ("2020–25, the new argument format", 2020, 2025)]}
    for title, b_lo, b_hi in blocks.get(tuple(terms), []):
        v, c = labels(D, b_lo, b_hi)
        block = dict(tables)
        if mode == "rehearsal":               # the standard folds, for reference
            from scotus.report import report_tables as RT
            fold, name_fit = ("dev", "dev fold (terms ≤ 2011)") if b_hi == 2015 else \
                ("val", "validation fold (terms ≤ 2015)")
            for n in got:
                if n not in SAME_EVERY_STEP:
                    p = RT.preds(FOLD_STEM.get(n, n), fold)
                    if p is not None and len(v[KEY].merge(p, on=KEY)) == len(v):
                        block[(n, name_fit)] = p
            block = dict(sorted(block.items(), key=lambda kv: ORDER.index(kv[0][0])))
        md += [f"## {title} ({len(c)} cases, {len(v):,} votes)", "",
               score_table(block, v, c, rates, fitted_at), ""]

    # per term: every row to a csv (make_figures.py draws from it), the main rows as tables
    per = pd.concat([ev.score_by_term(p, v_all, c_all).drop(columns="petitioner win %")
                     .assign(method=n, fit="once" if f == once else "each")
                     for (n, f), p in tables.items()], ignore_index=True)
    for m in ("macro_f1", "auc"):
        per[f"{m}_lo"] = per[f"{m}_hi"] = np.nan
    if "v1" in got:                                   # CI band for the main row only
        for term in terms:
            r = ev.score(tables[("v1", each)], *labels(D, term, term))
            row = (per.method == "v1") & (per.fit == "each") & (per.term == term)
            for m in ("macro_f1", "auc"):
                per.loc[row, [f"{m}_lo", f"{m}_hi"]] = r[m][1:]
    per.round(4).to_csv(Path(path).with_suffix("").as_posix() + "_terms.csv", index=False)
    show = [(n, f) for n in ("qc_bloc", "katz_rf", "v1", "v1+qc_bloc") if n in got
            for f in ("each", "once")]
    if show:
        pick = lambda n, f: per[(per.method == n) & (per.fit == f)].sort_values("term")
        base = pick(*show[0])
        for metric, title in [("auc", "AUC"), ("macro_f1", "Macro-F1")]:
            tab = pd.DataFrame({"Term": base.term.to_numpy(), "Cases": base.cases.to_numpy()})
            for n, f in show:
                tab[f"{n}, {'each term' if f == 'each' else 'once'}"] = \
                    pick(n, f)[metric].round(3).to_numpy()
            md += [f"## Per term: {title}", "", tab.to_markdown(index=False), ""]

    # paired differences
    v_late, _ = labels(D, terms[min(1, len(terms) - 1)], hi)
    rows = []
    if len(terms) > 1:
        for n in got:
            if n not in SAME_EVERY_STEP:
                auc, f1 = paired(tables[(n, once)], tables[(n, each)], v_late)
                rows.append({"Comparison": f"{LABEL[n]}: each term − once",
                             "Terms": f"{terms[1]}–{hi}", "AUC": auc, "Macro-F1": f1})
    for a, b in [("flat_mb", "flat_mb_all"), ("flat_mb_all", "v1"), ("qc_bloc", "v1"),
                 ("qc_bloc", "v1+qc_bloc"), ("v1", "v1+qc_bloc"), ("v1", "v1_fusion"),
                 ("v1+qc_bloc", "v1+qc_bloc_meta")]:
        if a in got and b in got:
            auc, f1 = paired(tables[(a, each)], tables[(b, each)], v_all)
            rows.append({"Comparison": f"{LABEL[b]} − {LABEL[a]} (both refit each term)",
                         "Terms": f"{lo}–{hi}", "AUC": auc, "Macro-F1": f1})
    if rows:
        md += ["## Differences (paired bootstrap over cases, 95% CI)", "",
               pd.DataFrame(rows).to_markdown(index=False), ""]

    # the steps
    cases = D["cases"]
    log = [{"Term predicted": s["term"],
            "Fitted on (cases, all terms)": int(cases.case_id.isin(s["known"]).sum()),
            "of which since 1980 (network)": int((cases.year[cases.case_id.isin(s["known"])]
                                                  >= 1980).sum()),
            "Held over, not yet decided": ", ".join(s["held_over"]) or "–",
            "Cases predicted": int((cases.year == s["term"]).sum())} for s in steps]
    md += ["## The steps", "", pd.DataFrame(log).to_markdown(index=False), "",
           f"Cases without an SCDB decision day, taken as decided within their own term: "
           f"{n_assumed}.", ""]

    if 2016 in terms and mode != "final":
        # Step 2016 is fitted on exactly the validation fold's training set (terms <= 2015,
        # nothing held over), so it must give the validation fold's predictions for 2016.
        keys = votes[votes.year == 2016][KEY]
        pairs = [(m, f"data/preds/{m}_val.parquet", preds_dir / f"{m}_wf2016.parquet")
                 for m in GROUPS["base"] + GROUPS["flat"] + GROUPS["meta"]]
        pairs += [(f"{n}, seed {s}", f"data/preds/{FOLD_STEM[n]}_val_s{s}.parquet",
                   preds_dir / f"{n}_wf2016_s{s}.parquet") for n in RECIPES for s in seeds]
        rows = []
        for name, old, new in pairs:
            if Path(old).exists() and Path(new).exists():
                d = keys.merge(pd.read_parquet(old)[KEY + ["p"]], on=KEY).merge(
                    pd.read_parquet(new)[KEY + ["p"]], on=KEY, suffixes=("_fold", "_wf"))
                gap = (d.p_fold - d.p_wf).abs()
                rows.append({"Method": name, "Votes compared": len(d),
                             "Largest gap in p": f"{gap.max():.2g}",
                             "Mean gap in p": f"{gap.mean():.2g}"})
        md += ["## Harness check: step 2016 against the validation fold", "",
               "Step 2016 is fitted on exactly the validation fold's training terms (≤ 2015, "
               "no case held over), so its predictions for the 2016 term should equal the "
               "ones already on disk from the standard runs.", "",
               pd.DataFrame(rows).to_markdown(index=False), ""]
    text = "\n".join(md)
    Path(path).write_text(text)
    return text


# ---------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ext", default="data/ext", help="the 2020-25 tables (prepare_oyez.py)")
    ap.add_argument("--emb", default="embeddings/turns_last.npy")
    ap.add_argument("--emb-ext", default="embeddings/ext/turns_last.npy")
    ap.add_argument("--terms", nargs="+", type=int, help="rehearsal terms (default 2012-2017)")
    ap.add_argument("--methods", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--run", help="name of the output folders (default: rehearsal / final)")
    ap.add_argument("--report", action="store_true", help="only rebuild the tables")
    ap.add_argument("--dry-run", action="store_true",
                    help="test-run code path on placeholder labels; nothing kept or scored")
    ap.add_argument("--final", action="store_true", help="unlock the test terms (2018-2025)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="further train_v1.py options for the network recipes (diagnostics on "
                         "rehearsal terms only; give it last, together with --run)")
    args = ap.parse_args()
    if args.final and args.dry_run:
        raise SystemExit("--final and --dry-run exclude each other.")
    if args.extra and (args.final or args.dry_run or not args.run):
        raise SystemExit("--extra changes the recipe: rehearsal terms only, under its own --run.")
    locked = args.final or args.dry_run
    terms = TEST_TERMS if locked else sorted(args.terms or REHEARSAL_TERMS)
    if not locked and max(terms) >= FIRST_LOCKED:
        raise SystemExit(f"Terms from {FIRST_LOCKED} on are the test set: locked until the "
                         "final report (pass --final).")
    mode = "final" if args.final else "rehearsal" if terms == REHEARSAL_TERMS else "custom"
    run = args.run or ("final" if args.final else "dry run" if args.dry_run else "rehearsal")
    seeds, recipes = args.seeds, {n: r + args.extra for n, r in RECIPES.items()}
    tmp = None
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="wf_dryrun_"))
        preds_dir, runs_dir, out = tmp / "preds", tmp / "runs", tmp / "report.md"
        seeds, recipes = [0], {n: r + ["--epochs", "1"] for n, r in RECIPES.items()}
    else:
        preds_dir = Path(args.data) / "preds" / f"wf_{run}"
        runs_dir, out = Path("results/wf_runs") / run, Path(f"results/walk_forward_{run}.md")
    preds_dir.mkdir(parents=True, exist_ok=True)

    if args.final:
        # Nothing locked is read unless the test run can go through: the SCDB history
        # table must reach the last test term and the per-step history must be built.
        last = pd.read_parquet(Path(args.data) / "scdb_votes.parquet",
                               columns=["date_decision"]).date_decision.max()
        if last < pd.Timestamp(f"{TEST_TERMS[-1]}-06-01") or meta_fold(TEST_TERMS[-1], True) \
                != f"wf{TEST_TERMS[-1]}":
            raise SystemExit("Not ready for the test run: first `python -I prepare_scdb.py "
                             "--final`, then `python meta_features.py`. Nothing was read.")

    device = None if args.report else ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    D = load_corpus(args, locked, fake=args.dry_run, device=device)
    if "flat" in args.methods and not args.report:
        folders = [Path(args.emb).parent] + ([Path(args.emb_ext).parent] if locked else [])
        D["flat"] = {name: FB.load_vectors(folders, FB.chosen(name)["pool"])
                     for name in GROUPS["flat"] if all(FB.available(name, f) for f in folders)}
        if len(D["flat"]) < len(GROUPS["flat"]):
            print(f"flat ModernBERT vectors found for: {list(D['flat']) or 'none'} "
                  "(flat_encode.py makes them); the others are skipped", flush=True)
    decided, n_assumed = decision_dates(args, D["cases"], locked, args.final)
    steps = plan(D, decided, terms)
    print(f"walk-forward '{run}': terms {terms[0]}-{terms[-1]}, {len(D['cases']):,} cases "
          f"loaded in {time.time() - t0:.0f}s" + (f"; device {device}" if device else ""),
          flush=True)
    if not args.report:
        for step in steps:
            run_step(step, D, args, args.methods, seeds, recipes, preds_dir, runs_dir, device,
                     fake=args.dry_run)
    text = report(D, steps, terms, seeds, preds_dir, out, mode, n_assumed)
    if args.dry_run:
        n = sum(len(pd.read_parquet(f)) for f in preds_dir.glob("*.parquet"))
        bad = sum(int((~np.isfinite(pd.read_parquet(f).p)).sum()) for f in preds_dir.glob("*.parquet"))
        shutil.rmtree(tmp)
        print(f"dry run finished: {len(steps)} steps, {n:,} placeholder-scored predictions "
              f"written and deleted, {bad} not finite. Nothing was scored against real labels.")
    else:
        print(text)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
