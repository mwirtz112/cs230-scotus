"""
Collect the v1 runs into one comparison table.

For every configuration (the --name given to train_v1.py) and fold:
    single  mean and standard deviation over seeds of each run's own score
    ens     the seed ensemble: average the seeds' probabilities, cutoff 0.5
The ensemble is what we would deploy; "single" shows how noisy one run is.

Folds: dev = train <= 2011, score 2012-15 (256 cases); val = train <= 2015, score
2016-17 (123 cases, the official validation set). A change counts as an improvement only
if it helps on dev and on val 2017
alone (2016 was an unusual eight-justice term).

Second cutoff ("f1 rate", "acc rate", "case rate" on val rows): instead of 0.5, the cutoff
is set on the dev fold's predictions so that the share of predicted petitioner votes on
dev equals the training vote rate (TRAIN_RATE), then applied unchanged to val. Nothing is
tuned on val.

Run from the repository root:
    python -m scotus.evaluation.summarize_v1                 -> results/v1_experiments.md
    python -m scotus.evaluation.summarize_v1 --ci a3_full    # also bootstrap CIs for these configurations
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scotus.evaluation import eval as ev

FOLD_YEARS = {"dev": (2012, 2015), "val": (2016, 2017), "dev2": (2008, 2011),
              "nogap": (2014, 2015), "newj_gap": (2010, 2011), "newj_nogap": (2010, 2011)}
TRAIN_RATE = 0.599          # pro-petitioner share of training votes (1955-2015)


def rate_cutoff(dev_preds, rate=TRAIN_RATE):
    """Cutoff at which the share of predicted petitioner votes on dev equals `rate`."""
    return float(np.quantile(dev_preds.p, 1 - rate))


def scores_at(preds, v, c, cutoff):
    p = preds.assign(pred=(preds.p >= cutoff).astype(int))
    res = ev.score(p, v, c, n_boot=0)
    return {"cut": round(cutoff, 3), "f1 rate": res["macro_f1"][0],
            "acc rate": res["vote_acc"][0], "case rate": res["case_acc"][0]}


def labels(fold, data="data"):
    lo, hi = FOLD_YEARS[fold]
    v = pd.concat([ev.load_labels(s, data)[0] for s in ["train", "val"]])
    c = pd.concat([ev.load_labels(s, data)[1] for s in ["train", "val"]])
    v = v[(v.year >= lo) & (v.year <= hi)].reset_index(drop=True)
    return v, c[c.case_id.isin(v.case_id)].reset_index(drop=True)


def scores(preds, v):
    df = v.merge(preds[["case_id", "justice", "p"]], on=["case_id", "justice"])
    assert len(df) == len(v), "missing predictions"
    pred = (df.p >= 0.5).astype(int)
    out = {"auc": roc_auc_score(df.vote, df.p), "f1": f1_score(df.vote, pred, average="macro"),
           "acc": float((df.vote == pred).mean()), "pet": float(pred.mean())}
    m = df.year == 2017
    if m.any():
        out["auc17"] = roc_auc_score(df.vote[m], df.p[m])
        out["f117"] = f1_score(df.vote[m], pred[m], average="macro")
    return out


def ensemble(paths):
    """Average probabilities over seeds (same votes in every file)."""
    dfs = [pd.read_parquet(p).set_index(["case_id", "justice"]).p for p in paths]
    p = pd.concat(dfs, axis=1).mean(axis=1).rename("p").reset_index()
    return p.assign(pred=(p.p >= 0.5).astype(int))


def blend_many(preds):
    """Mean of several models' logits, per vote (equal weights, nothing fitted)."""
    logit = lambda p: np.log(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6)))
    d = preds[0][["case_id", "justice"]].copy()
    z = np.zeros(len(d))
    for q in preds:
        m = d.merge(q[["case_id", "justice", "p"]], on=["case_id", "justice"], how="left")
        assert m.p.notna().all(), "the models do not cover the same votes"
        z += logit(m.p.to_numpy())
    p = 1 / (1 + np.exp(-z / len(preds)))
    return d.assign(p=p, pred=(p >= 0.5).astype(int))


def blend_with(a, b):
    """Mean of the two models' logits, per vote."""
    return blend_many([a, b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ci", nargs="*", default=[], help="configurations to bootstrap")
    ap.add_argument("--only", nargs="*", help="name prefixes to include")
    ap.add_argument("--blend", nargs="*", default=[], help="configurations to also average "
                    "with the qc_bloc regression (mean of logits): the two make different "
                    "errors, so the blend is a candidate final system")
    args = ap.parse_args()
    preds_dir = Path(args.data) / "preds"

    runs = defaultdict(list)                       # (name, fold) -> [seed, ...]
    meta = {}
    for f in sorted(Path("results/v1_runs").glob("*.json")):
        r = json.loads(f.read_text())
        if args.only and not any(r["name"].startswith(p) for p in args.only):
            continue
        runs[(r["name"], r["fold"])].append(r["seed"])
        meta[r["name"]] = r

    lab = {f: labels(f, args.data) for f in {k[1] for k in runs} | {"dev", "val"}}
    rows = []
    for name, stem in [("question_count", "question_count"), ("qc_plus (LR on counts)", "qc_plus"),
                       ("qc_bloc (LR + named colleagues)", "qc_bloc")]:
        dev_file = preds_dir / (f"{stem}_train.parquet" if stem == "question_count"
                                else f"{stem}_dev.parquet")
        if not dev_file.exists():
            continue
        dev = lab["dev"][0][["case_id", "justice"]].merge(pd.read_parquet(dev_file),
                                                         on=["case_id", "justice"])
        val = pd.read_parquet(preds_dir / f"{stem}_val.parquet")
        rows.append({"config": name, "fold": "dev", "seeds": "-", **scores(dev, lab["dev"][0])})
        rows.append({"config": name, "fold": "val", "seeds": "-", **scores(val, lab["val"][0]),
                     **scores_at(val, *lab["val"], rate_cutoff(dev))})
    for (name, fold), seeds in sorted(runs.items()):
        paths = [preds_dir / f"{name}_{fold}_s{s}.parquet" for s in sorted(seeds)]
        single = pd.DataFrame([scores(pd.read_parquet(p), lab[fold][0]) for p in paths])
        ens_preds = ensemble(paths)
        ens = scores(ens_preds, lab[fold][0])
        row = {"config": name, "fold": fold, "seeds": len(seeds), **ens,
               "single auc": f"{single.auc.mean():.3f} ± {single.auc.std(ddof=0):.3f}",
               "single f1": f"{single.f1.mean():.3f} ± {single.f1.std(ddof=0):.3f}"}
        if fold == "val" and (name, "dev") in runs:
            dev_paths = [preds_dir / f"{name}_dev_s{s}.parquet" for s in sorted(runs[(name, "dev")])]
            row.update(scores_at(ens_preds, *lab["val"], rate_cutoff(ensemble(dev_paths))))
        if name in args.ci:
            v, c = lab[fold]
            res = ev.score(ensemble(paths), v, c)
            row["f1 CI"] = "[{:.3f}, {:.3f}]".format(*res["macro_f1"][1:])
            row["auc CI"] = "[{:.3f}, {:.3f}]".format(*res["auc"][1:])
            row["case acc"] = round(res["case_acc"][0], 3)
        rows.append(row)
        if name in args.blend and fold in ("dev", "val"):
            blend = blend_with(ens_preds, pd.read_parquet(preds_dir / f"qc_bloc_{fold}.parquet"))
            brow = {"config": f"{name} + qc_bloc", "fold": fold, "seeds": len(seeds),
                    **scores(blend, lab[fold][0])}
            if fold == "val" and (name, "dev") in runs:
                dev_b = blend_with(ensemble(dev_paths),
                                   pd.read_parquet(preds_dir / "qc_bloc_dev.parquet"))
                brow.update(scores_at(blend, *lab["val"], rate_cutoff(dev_b)))
            res = ev.score(blend, *lab[fold])
            brow["f1 CI"] = "[{:.3f}, {:.3f}]".format(*res["macro_f1"][1:])
            brow["auc CI"] = "[{:.3f}, {:.3f}]".format(*res["auc"][1:])
            brow["case acc"] = round(res["case_acc"][0], 3)
            rows.append(brow)

    df = pd.DataFrame(rows)
    num = [c for c in ["auc", "f1", "acc", "pet", "auc17", "f117", "f1 rate", "acc rate",
                       "case rate"] if c in df]
    df[num] = df[num].round(3)
    out = ["# Model v1 experiments", "",
           "Seed-ensemble scores (auc, f1, acc, pet = share predicted petitioner; auc17 and "
           "f117 = the 2017 term alone) and single-run mean ± sd over seeds. Columns "
           "ending in 'rate' use the cutoff that matches the train vote rate on dev "
           "instead of 0.5 (cut = that cutoff). "
           "dev = train ≤ 2011, scored on 2012–15; val = train ≤ 2015, scored on 2016–17. "
           "Test is locked.", ""]
    for fold in ["dev", "val", "dev2", "nogap", "newj_gap", "newj_nogap"]:
        part = df[df.fold == fold].drop(columns="fold").dropna(axis=1, how="all")
        if len(part):
            out += [f"## {fold}", "", part.fillna("").to_markdown(index=False), ""]
    Path("results/v1_experiments.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
