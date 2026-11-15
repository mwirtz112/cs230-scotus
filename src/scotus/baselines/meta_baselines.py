"""
Metadata baselines: what the case profile predicts without, and together with, the
question counts.

    python -m scotus.baselines.meta_baselines   -> results/meta_baselines.md,
                                  data/preds/<method>_{dev,val}.parquet

No transcript text and no deep learning anywhere in this file.

    katz_rf       Katz-style rebuild (Katz, Bommarito & Blackman 2017): a random forest
                  on the justice, the SCDB case profile and the voting-history rates.
                  No transcript at all: "how far does the case's profile get you?"
    katz_rf_rec   the same without the three fields SCDB reads off the Court's opinion
                  (lower-court direction, lower-court dissent, cert reason) and without
                  every history rate that conditions on the direction. A lower bound that
                  uses only facts of record.
    meta_lr       logistic regression on the history rates alone (the transparent version)
    qc_meta       qc_plus + the history rates: counts and metadata, one regression
    qc_bloc_meta  qc_bloc with the same addition in its first stage: the strongest
                  non-deep fusion, and the baseline the fusion network has to beat

Folds as everywhere: val = fit on terms <= 2015, score 2016-17; dev = fit on <= 2011,
score 2012-15. Class-balanced fits, cutoff 0.5, nothing tuned on validation. The voting
history of a held-out case stops at the fold's training cutoff ("frozen"); the "rolling"
rows let it run up to the day of the argument (meta_features.py). Test is not read.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from scotus.evaluation import eval as ev
from scotus.features import meta_features as M
from scotus.baselines import qc_plus as Q
from scotus.evaluation import summarize_v1 as S
from scotus.features.features import WIDE_FEATURES

H = M.HIST_FEATURES
# history rates that do not condition on the lower-court direction
H_REC = ["h_exp", "h_pet", "h_lib", "h_pet_area", "b_pet", "c_base", "c_source", "c_pet",
         "c_resp"]
PROFILE = M.CAT_FIELDS
PROFILE_REC = [f for f in PROFILE if f not in M.OPINION_CODED]


def forest(train, held, cols, seed=0):
    clf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=5, max_features="sqrt",
                                 class_weight="balanced_subsample", n_jobs=-1,
                                 random_state=seed)
    clf.fit(train[cols].to_numpy(float), train.vote)
    imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
    return clf.predict_proba(held[cols].to_numpy(float))[:, 1], imp


def logistic(train, held, cols, C=0.3):
    X = train[cols].to_numpy(float)
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    clf.fit((X - mean) / std, train.vote)
    return clf.predict_proba((held[cols].to_numpy(float) - mean) / std)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    votes = pd.concat([ev.load_labels(s, args.data)[0] for s in ["train", "val"]],
                      ignore_index=True)                       # test labels never loaded
    vf = pd.read_parquet(data / "vote_feats.parquet")
    jcode = {j: i for i, j in enumerate(sorted(votes.justice.unique()))}

    rows, importances = [], None
    for protocol in ["frozen", "rolling"]:
        for fold in ["dev", "val"]:
            end, (lo, hi) = M.FOLD_CUTOFF[fold], S.FOLD_YEARS[fold]
            prof, hist = M.load(args.data, fold, protocol)
            df = (votes.merge(hist, on=["case_id", "justice"]).merge(prof, on="case_id")
                  .merge(vf, on=["case_id", "justice"]))
            df["justice_code"] = df.justice.map(jcode)
            tr, te = df[df.year <= end], df[(df.year >= lo) & (df.year <= hi)]
            lab = S.labels(fold, args.data)
            p_rf, imp = forest(tr, te, ["justice_code"] + PROFILE + H)
            p_rec, _ = forest(tr, te, ["justice_code"] + PROFILE_REC + H_REC)
            (p_bloc,) = Q.fit_predict_bloc(tr, [te], pool_feats=WIDE_FEATURES + H)
            methods = {"katz_rf": p_rf, "katz_rf_rec": p_rec,
                       "meta_lr": logistic(tr, te, H),
                       "qc_meta": logistic(tr, te, WIDE_FEATURES + H),
                       "qc_bloc_meta": p_bloc.p.to_numpy()}
            if protocol == "frozen" and fold == "val":
                importances = imp
            for name, p in methods.items():
                preds = te[["case_id", "year", "justice", "vote"]].assign(
                    p=p, pred=(p >= 0.5).astype(int))
                stem = name if protocol == "frozen" else f"{name}_roll"
                preds.to_parquet(data / "preds" / f"{stem}_{fold}.parquet", index=False)
                res = ev.score(preds, *lab, n_boot=ev.N_BOOT if fold == "val" else 0)
                sc = S.scores(preds, lab[0])
                rows.append({"method": name, "history": protocol, "fold": fold,
                             **{k: (ev.fmt_ci(*res[k]) if fold == "val" and k in
                                    ("macro_f1", "auc") else f"{res[k][0]:.3f}")
                                for k in ev.METRICS},
                             "AUC 2017": f"{sc['auc17']:.3f}" if "auc17" in sc else "",
                             "pred petitioner %": round(100 * preds.pred.mean(), 1)})

    df = pd.DataFrame(rows)
    md = ["# Metadata baselines (no text, no deep learning)", "",
          "SCDB case profile and voting-history rates (meta_features.py). katz_rf = random "
          "forest, 1,000 trees, on the justice, the profile and the history rates; "
          "katz_rf_rec = the same on facts of record only (no lower-court direction, "
          "lower-court dissent or cert reason); meta_lr = logistic regression on the "
          "history rates; qc_meta = qc_plus + history rates; qc_bloc_meta = qc_bloc with the "
          "history rates in its first stage. Class-balanced, cutoff 0.5. 'frozen': a "
          "held-out case sees voting history up to the fold's training cutoff; 'rolling': "
          "up to its argument day. Validation rows carry 95% bootstrap CIs over cases. "
          "Test untouched.", ""]
    for fold, title in [("val", "Validation 2016-17 (fit on terms <= 2015)"),
                        ("dev", "Dev 2012-15 (fit on terms <= 2011)")]:
        md += [f"## {title}", "",
               df[df.fold == fold].drop(columns="fold").to_markdown(index=False), ""]
    md += ["## katz_rf: largest impurity importances (validation fold, frozen)", "",
           importances.head(12).round(4).to_frame("importance").to_markdown(), ""]
    Path("results/meta_baselines.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
