"""Checks on the finished test run (terms 2018-2025). Reads saved predictions only.

    python -m scotus.evaluation.check_walk_final        -> results/walk_forward_final_check.md

1. An independent recomputation. The scores of every method and fit are rebuilt from the
   prediction files in data/preds/wf_final with this file's own assembly (seed files
   averaged; logits averaged for the blends) and scikit-learn, without the harness's
   scoring code (`eval.py`, `walk_forward.assemble`), and compared, term by term, with
   results/walk_forward_final_terms.csv, which the test run wrote.
2. Paired differences that the test report does not list. They were computed after the
   run, so they describe the result and were not fixed in advance. Same bootstrap as the
   report's own table (`walk_forward.paired`: cases resampled, 1,000 draws, seed 0); the
   first row repeats a row of the report to show that the procedure is the same.

Nothing is fitted and no prediction is changed. Written after the test run.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scotus.evaluation import walk_forward as W          # paired() and the methods' labels; nothing is run

PREDS = Path("data/preds/wf_final")
REF = "results/walk_forward_final_terms.csv"
OUT = "results/walk_forward_final_check.md"
TERMS = list(range(2018, 2026))
BLOCKS = {"2018–2025": (2018, 2025), "2018–19": (2018, 2019), "2020–25": (2020, 2025)}
KEY = ["case_id", "justice"]
SEEDS = range(5)
BLENDS = {"v1+qc_bloc": ["v1", "qc_bloc"], "v1+qc_bloc_meta": ["v1", "qc_bloc_meta"]}
METRICS = ["macro_f1", "auc", "vote_acc", "case_acc"]
EXTRA = [("v1", "qc_bloc", "2018–2025", "in the test report: +0.026 [+0.007, +0.045] and +0.025 [+0.006, +0.046]"),
         ("v1", "qc_bloc", "2018–19", ""), ("v1", "qc_bloc", "2020–25", ""),
         ("v1", "qc_bloc_meta", "2018–2025", ""), ("v1", "qc_bloc_meta", "2018–19", ""),
         ("v1", "qc_bloc_meta", "2020–25", ""), ("v1", "katz_rf", "2018–2025", ""),
         ("v1", "question_count", "2018–2025", ""), ("v1+qc_bloc_meta", "v1", "2018–2025", ""),
         ("v1+qc_bloc_meta", "qc_bloc_meta", "2018–2025", "same inputs on both sides: transcript and case profile")]


def labels():
    v = pd.concat([pd.read_parquet("data/votes.parquet"),
                   pd.read_parquet("data/ext/votes.parquet")], ignore_index=True)
    c = pd.concat([pd.read_parquet("data/cases.parquet"),
                   pd.read_parquet("data/ext/cases.parquet")], ignore_index=True)
    v = v[v.year.between(TERMS[0], TERMS[-1])][KEY + ["year", "vote"]].reset_index(drop=True)
    c = c[c.case_id.isin(set(v.case_id))][["case_id", "year", "win_side"]].reset_index(drop=True)
    assert not v.duplicated(KEY).any() and c.case_id.is_unique
    return v, c


def step(name, term):
    """What the model fitted before `term` predicts, for every vote it was given."""
    if name in BLENDS:
        a, b = (step(n, term) for n in BLENDS[name])
        m = a.merge(b, on=KEY, suffixes=("_a", "_b"), validate="one_to_one")
        assert len(m) == len(a) == len(b), "the blended models do not cover the same votes"
        z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        p = 1 / (1 + np.exp(-(z(m.p_a) + z(m.p_b)) / 2))
        return m[KEY].assign(p=p, pred=(p >= 0.5).astype(int))
    one = PREDS / f"{name}_wf{term}.parquet"
    if one.exists():                                # baselines: one file, their own cutoff
        return pd.read_parquet(one)[KEY + ["p", "pred"]]
    seeds = pd.concat([pd.read_parquet(PREDS / f"{name}_wf{term}_s{s}.parquet").set_index(KEY).p
                       for s in SEEDS], axis=1)
    assert seeds.notna().all().all(), "the seeds do not cover the same votes"
    p = seeds.mean(axis=1).rename("p").reset_index()
    return p.assign(pred=(p.p >= 0.5).astype(int))


def table(name, fit, v):
    """Predictions for all test votes: refit before each term, or the fit made before 2018."""
    out = []
    for t in TERMS:
        d = v[v.year == t].merge(step(name, t if fit == "each" else TERMS[0]), on=KEY,
                                 how="left", validate="one_to_one")
        assert d.p.notna().all() and np.isfinite(d.p).all(), (name, fit, t)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def metrics(d, c):
    g = d.groupby("case_id", sort=False).agg(n=("pred", "size"), k=("pred", "sum")) \
         .join(c.set_index("case_id").win_side)
    call = (g.k > g.n / 2).astype(int)              # a tie goes to the respondent
    return {"macro_f1": f1_score(d.vote, d.pred, average="macro"),
            "auc": roc_auc_score(d.vote, d.p),
            "vote_acc": float((d.pred == d.vote).mean()),
            "case_acc": float((call == g.win_side).mean())}


def main():
    v, c = labels()
    ref = pd.read_csv(REF)
    tabs, rows, cells, worst, bad = {}, [], 0, 0.0, []
    for (name, fit), r in ref.groupby(["method", "fit"], sort=False):
        d = tabs[(name, fit)] = table(name, fit, v)
        for x in r.itertuples():
            dt = d[d.year == x.term]
            assert (len(dt), dt.case_id.nunique()) == (x.votes, x.cases), (name, fit, x.term)
            m = metrics(dt, c)
            for k in METRICS:
                gap = abs(m[k] - getattr(x, k))
                cells, worst = cells + 1, max(worst, gap)
                if gap > 5.1e-5:                    # the csv is rounded to 4 decimals
                    bad.append((name, fit, x.term, k, round(m[k], 4), getattr(x, k)))
        row = {"Method": W.LABEL.get(name, name), "Fit": fit}
        for b, (lo, hi) in BLOCKS.items():
            m = metrics(d[d.year.between(lo, hi)], c)
            row.update({f"Macro-F1 {b}": f"{m['macro_f1']:.3f}", f"AUC {b}": f"{m['auc']:.3f}"})
            if b == "2018–2025":
                row.update({"Vote acc": f"{m['vote_acc']:.3f}", "Case acc": f"{m['case_acc']:.3f}",
                            "Pred. petitioner %": f"{100 * d.pred.mean():.1f}"})
        rows.append(row)
    finite = sum(int(np.isfinite(d.p).all()) for d in tabs.values())
    md = ["# The test run, checked (terms 2018–2025)", "",
          f"Reads the saved predictions of the test run (`{PREDS}`) and the labels; fits nothing. "
          "`check_walk_final.py`, written on Nov 14 after the run.", "",
          "## 1. Independent recomputation", "",
          f"{len(c)} cases, {len(v):,} votes, {len(tabs)} method × fit tables, all predictions "
          f"finite in {finite} of them. Per-term macro-F1, AUC, vote accuracy and case accuracy, "
          f"rebuilt without the harness's scoring code and compared with `{REF}`: "
          f"**{cells:,} cells, largest difference {worst:.1e}, {len(bad)} beyond rounding "
          "(the file is rounded to 4 decimals).**", ""]
    if bad:
        md += ["Cells that differ: " + "; ".join(map(str, bad)), ""]
    md += ["Recomputed over all test terms and the two blocks (compare with "
           "`results/walk_forward_final.md`; 'each' = refit before each term, 'once' = the "
           "fit made before 2018):", "", pd.DataFrame(rows).to_markdown(index=False), "",
           "## 2. Differences the test report does not list (computed after the run)", "",
           "First method minus second, both refit before each term; 95% interval from a "
           "bootstrap over cases, paired (`walk_forward.paired`, 1,000 draws, seed 0). These "
           "rows were not fixed before the test run: they describe the result.", ""]
    extra = []
    for a, b, blk, note in EXTRA:
        lo, hi = BLOCKS[blk]
        vb = v[v.year.between(lo, hi)].reset_index(drop=True)
        auc, f1 = W.paired(tabs[(b, "each")], tabs[(a, "each")], vb)
        extra.append({"Comparison": f"{W.LABEL.get(a, a)} − {W.LABEL.get(b, b)}", "Terms": blk,
                      "Cases": vb.case_id.nunique(), "AUC": auc, "Macro-F1": f1, "Note": note})
    md += [pd.DataFrame(extra).to_markdown(index=False), ""]
    Path(OUT).write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {OUT}")
    if bad:
        raise SystemExit("the recomputation does not match the test run's per-term file")


if __name__ == "__main__":
    main()
