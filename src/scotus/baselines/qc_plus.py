"""
Question-feature baselines: logistic regressions on counts of the questioning.

qc_plus.

The question-count baseline uses one number per justice: the share of their turns aimed
at each side. This baseline keeps the same idea but uses every simple count we can read
off the transcript without looking at the words:
    turns, words and "?" marks aimed at each side, for the justice, for the whole bench,
    and for the other justices; the two advocates' words and turns; a silent flag
(features.WIDE_FEATURES, 41 numbers). A class-balanced logistic regression is fit on the
training votes; the cutoff is 0.5 (balanced classes), nothing is tuned on validation.

Why it matters: it is a much stronger floor than the plain question count (validation AUC
0.78 against 0.70), it uses no deep learning, and the milestone model was below it. Model v1 gets these same
features as its "wide" input, so v1 must beat this to show the text adds anything.

qc_bloc: the same, plus named colleagues. Stage 1 is
qc_plus, and a second regression on the justice's own features only gives every justice an
"own evidence" score per case. Stage 2 is one small regression per justice: that justice's
qc_plus score plus the own-evidence score of each named colleague. It learns, from
training votes, whose questioning predicts whose vote (blocs). A justice with fewer than
300 training votes keeps the qc_plus prediction. Still no text and no deep learning; it is
the linear version of the bloc layer in model v1.

Run from the repository root:
    python -m scotus.baselines.qc_plus   -> results/qc_plus.md, data/preds/qc_{plus,bloc}_{train,val,dev}.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scotus.evaluation import eval as ev
from scotus.features.features import WIDE_FEATURES

DEV = (2011, 2012, 2015)        # rolling-origin dev fold: train <= 2011, score 2012-15


def fit_predict(train, held, C=1.0):
    Xtr = train[WIDE_FEATURES].to_numpy(float)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    clf.fit((Xtr - mean) / std, train.vote)
    out = []
    for df in held:
        p = clf.predict_proba((df[WIDE_FEATURES].to_numpy(float) - mean) / std)[:, 1]
        out.append(df[["case_id", "year", "justice", "vote"]]
                   .assign(p=p, pred=(p >= 0.5).astype(int)))
    return out, pd.Series(clf.coef_[0], index=WIDE_FEATURES)


OWN = [f for f in WIDE_FEATURES if f.startswith("o_")] + ["silent"]
MIN_VOTES, MIN_TOGETHER, C_BLOC = 300, 0.15, 0.03


def _logit(train, feats, C):
    X = train[feats].to_numpy(float)
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    clf.fit((X - mean) / std, train.vote)
    return lambda df: clf.decision_function((df[feats].to_numpy(float) - mean) / std)


def fit_predict_bloc(train, held, C=1.0, return_coef=False, pool_feats=WIDE_FEATURES,
                     own_feats=OWN):
    """Two-stage regression with named colleagues. Returns predictions for each `held`.

    pool_feats / own_feats: the columns of the stage-1 regression and of each justice's
    "own evidence" score (defaults: the count features; meta_baselines.py adds the SCDB
    history features).

    With return_coef: also a table of the stage-2 weights, rows = the voting justice,
    columns = the colleague whose own-evidence score is weighted (standardized inputs),
    plus the weight on the justice's own qc_plus score in column "own"."""
    pool, own = _logit(train, pool_feats, C), _logit(train, own_feats, C)
    frames = [df.assign(pool=pool(df), ell=own(df)) for df in [train, *held]]
    ell = {i: f.pivot(index="case_id", columns="justice", values="ell")
           for i, f in enumerate(frames)}
    tr = frames[0]
    out = [f.assign(z=f.pool) for f in frames[1:]]
    coefs = {}
    for j, g in tr.groupby("justice"):
        if len(g) < MIN_VOTES:
            continue                                    # too few votes: keep stage 1
        sat = ell[0].loc[g.case_id].notna().mean()
        mates = [k for k in sat.index[sat > MIN_TOGETHER] if k != j]

        def X(f, table):
            E = table.reindex(index=f.case_id, columns=mates).to_numpy()
            return np.hstack([f[["pool"]].to_numpy(), np.nan_to_num(E)])   # absent -> 0
        Xtr = X(g, ell[0])
        mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
        clf = LogisticRegression(C=C_BLOC, max_iter=5000, class_weight="balanced")
        clf.fit((Xtr - mean) / std, g.vote)
        coefs[j] = dict(zip(["own", *mates], clf.coef_[0]))
        for i, f in enumerate(out):
            m = (f.justice == j).to_numpy()
            if m.any():
                f.loc[m, "z"] = clf.decision_function((X(f[m], ell[i + 1]) - mean) / std)
    res = []
    for f in out:
        p = 1 / (1 + np.exp(-f.z.to_numpy()))
        res.append(f[["case_id", "year", "justice", "vote"]]
                   .assign(p=p, pred=(p >= 0.5).astype(int)))
    if return_coef:
        return res, pd.DataFrame(coefs).T
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()
    data = Path(args.data)
    vf = pd.read_parquet(data / "vote_feats.parquet")
    lab = {s: ev.load_labels(s, args.data) for s in ["train", "val"]}       # test locked
    tr = lab["train"][0].merge(vf, on=["case_id", "justice"])
    va = lab["val"][0].merge(vf, on=["case_id", "justice"])

    rows, terms = [], []
    (p_tr, p_va), coef = fit_predict(tr, [tr, va], args.C)
    b_tr, b_va = fit_predict_bloc(tr, [tr, va], args.C)
    end, lo, hi = DEV                                   # dev fold, refit on <= 2011 only
    dev_tr, dev_te = tr[tr.year <= end], tr[(tr.year >= lo) & (tr.year <= hi)]
    (p_dev,), _ = fit_predict(dev_tr, [dev_te], args.C)
    (b_dev,) = fit_predict_bloc(dev_tr, [dev_te], args.C)
    dv = lab["train"][0]; dv = dv[(dv.year >= lo) & (dv.year <= hi)].reset_index(drop=True)
    dc = lab["train"][1]; dc = dc[dc.case_id.isin(dv.case_id)].reset_index(drop=True)
    for method, by_split in [("qc_plus", {"train": p_tr, "val": p_va, "dev": p_dev}),
                             ("qc_bloc", {"train": b_tr, "val": b_va, "dev": b_dev})]:
        for split, preds in by_split.items():
            preds.to_parquet(data / "preds" / f"{method}_{split}.parquet", index=False)
            v, c = (dv, dc) if split == "dev" else lab[split]
            res = ev.score(preds, v, c, n_boot=200 if split == "train" else ev.N_BOOT)
            rows.append({"method": method,
                         "split": f"dev {lo}-{hi}" if split == "dev" else split,
                         "fit on": f"1955-{end}" if split == "dev" else "1955-2015",
                         **{k: ev.fmt_ci(*res[k]) for k in ev.METRICS},
                         "pred petitioner %": round(100 * preds.pred.mean(), 1)})
            if split != "train":
                terms.append(ev.score_by_term(preds, v, c).assign(method=method))

    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10).round(3)
    md = ["# qc_plus and qc_bloc: logistic regressions on question features", "",
          f"{len(WIDE_FEATURES)} count features (features.py), class-balanced, C={args.C}, "
          "cutoff 0.5. qc_bloc adds named colleagues' own-evidence scores in a second, "
          f"per-justice regression (C={C_BLOC}). Train rows are in-sample. Test is locked.", "",
          pd.DataFrame(rows).to_markdown(index=False), "", "## Per term", "",
          pd.concat(terms).round(3).to_markdown(index=False), "",
          "## qc_plus: largest coefficients (standardized; + means pro-petitioner)", "",
          top.to_frame("coef").to_markdown(), ""]
    Path("results/qc_plus.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
