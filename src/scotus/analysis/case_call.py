"""Does the model know which side wins a case? Reads walk-forward predictions already made.

Reads the cached walk-forward predictions of the rehearsal (terms 2012-17, every method
refit before each term; `walk_forward.py`) and writes results/case_call.md:

1. vote accuracy and case accuracy of each method next to always-petitioner;
2. how well the mean probability over the justices sorts cases by winner (thirds,
   case-level AUC) and how often the "respondent" calls are right;
3. rules for calling the winner of a case: the standard rule (majority of the votes
   called at 0.5), two rules that undo the class balance of the training loss with the
   training vote rate and no fitted number, and two rules fitted on earlier terms.

Why rule 3 exists: every model is trained with a class-balanced loss, so p = 0.5 means
"as petitioner-leaning as the average vote". That suits macro-F1. A case is won by the
petitioner two times in three, and calling the winner needs that base rate back:
true odds = model odds * pi / (1 - pi), pi = pro-petitioner share of the training votes.

Terms from 2018 on are the test set and are refused unless --final is given. --final
(added after the test run had been made) builds the same tables from the saved
predictions of that run (data/preds/wf_final, terms 2018-2025) and writes
results/case_call_final.md. It changes no prediction and no model.

    python -m scotus.analysis.case_call
    python -m scotus.analysis.case_call --final
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from scotus.evaluation import walk_forward as W

KEY = W.KEY
BALANCED = ["katz_rf", "qc_bloc", "v1", "v1+qc_bloc", "v1+qc_bloc_meta"]   # class-balanced fits
OTHER = ["always_petitioner", "question_count"]
RULES = {"R0": "Standard: majority of the votes called at 0.5",
         "C1": "C1: majority of the votes, class balance undone (p > 1 − π)",
         "C2": "C2: mean probability with the class balance undone, above 0.5",
         "A": "A: cutoff on the mean probability, best on earlier terms",
         "B": "B: logistic fit of the winner on the mean probability, earlier terms",
         "ALW": "Always petitioner"}
N_BOOT, SEED = 2000, 0


def load(terms, preds_dir, final=False):
    votes = pd.read_parquet("data/votes.parquet")
    cases = pd.read_parquet("data/cases.parquet")
    if final:                                  # terms 2020-25 are kept apart, in data/ext
        votes = pd.concat([votes, pd.read_parquet("data/ext/votes.parquet")], ignore_index=True)
        cases = pd.concat([cases, pd.read_parquet("data/ext/cases.parquet")], ignore_index=True)
    v = votes[votes.year.between(terms[0], terms[-1])][KEY + ["year", "vote"]].reset_index(drop=True)
    c = cases[cases.case_id.isin(set(v.case_id))][["case_id", "win_side"]].set_index("case_id")
    # pro-petitioner share of the votes of all earlier terms: what a step's fit has seen
    pi = {t: float(votes[votes.year < t].vote.mean()) for t in terms}
    tabs = {}
    for name in OTHER + BALANCED:
        got = W.assemble(name, terms, votes, preds_dir, W.SEEDS)
        if got is None:
            raise SystemExit(f"no walk-forward predictions for {name} in {preds_dir}")
        d = v.merge(got[0][KEY + ["p", "pred"]], on=KEY)
        w = d.year.map(pi)
        d["pt"] = d.p * w / (d.p * w + (1 - d.p) * (1 - w))        # class balance undone
        d["pred_t"] = (d.pt > 0.5).astype(int)                     # the same as p > 1 - pi
        tabs[name] = d
    return v, c, pi, tabs


def case_table(d, c, terms, fitted=True):
    g = d.groupby("case_id", sort=False).agg(year=("year", "first"), n=("pred", "size"),
                                             s=("p", "mean"), st=("pt", "mean"),
                                             k=("pred", "sum"), kt=("pred_t", "sum")).join(c)
    g["R0"] = (g.k > g.n / 2).astype(int)                  # a tie counts as respondent
    g["C1"] = (g.kt > g.n / 2).astype(int)
    g["C2"] = (g.st > 0.5).astype(int)
    g["ALW"], g["A"], g["B"] = 1, np.nan, np.nan
    if fitted:
        grid = np.round(np.arange(0.20, 0.71, 0.01), 2)
        for t in terms[1:]:                                # fitted on earlier scored terms
            past, now = g[g.year < t], g[g.year == t]
            acc = lambda x: ((past.s > x).astype(int) == past.win_side).mean()
            cut = max(grid, key=lambda x: (acc(x), -abs(x - 0.5)))
            g.loc[now.index, "A"] = (now.s > cut).astype(int)
            logit = lambda x: np.log(x / (1 - x)).to_numpy()[:, None]
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(past.s), past.win_side)
            g.loc[now.index, "B"] = lr.predict(logit(now.s))
    return g


def boot_diff(a, b, rng):
    """Mean of a - b over cases with a 95% bootstrap interval (paired)."""
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    lo, hi = np.percentile((a[idx] - b[idx]).mean(1), [2.5, 97.5])
    return f"{(a - b).mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", nargs=2, type=int, default=[2012, 2017])
    ap.add_argument("--preds", default=None, help="default: data/preds/wf_rehearsal (wf_final with --final)")
    ap.add_argument("--out", default=None, help="default: results/case_call.md (case_call_final.md with --final)")
    ap.add_argument("--final", action="store_true",
                    help="the test terms 2018-2025, from the saved predictions of the test run")
    args = ap.parse_args()
    if args.final:
        args.terms = [W.TEST_TERMS[0], W.TEST_TERMS[-1]]
    elif args.terms[1] >= W.FIRST_LOCKED:
        raise SystemExit(f"Terms from {W.FIRST_LOCKED} on are the test set; this script reads "
                         "rehearsal terms only (or the finished test run with --final).")
    preds = Path(args.preds or ("data/preds/wf_final" if args.final else "data/preds/wf_rehearsal"))
    out = args.out or ("results/case_call_final.md" if args.final else "results/case_call.md")
    terms = list(range(args.terms[0], args.terms[1] + 1))
    v, c, pi, tabs = load(terms, preds, args.final)
    rng = np.random.default_rng(SEED)
    lo, hi = terms[0], terms[-1]
    if args.final:
        blocks = {"2018–19": (2018, 2019), "2020–25": (2020, 2025), f"{lo}–{hi}": (lo, hi)}
        kind, run, note = "test", "test run", ("The test run was made once; this table only "
                                               "reads its saved predictions.")
    else:
        blocks = {"2012–15": (2012, 2015), "2016–17": (2016, 2017), f"{lo}–{hi}": (lo, hi)}
        kind, run, note = "rehearsal", "rehearsal", "The test terms (2018–25) were not read."
    G = {n: case_table(d, c, terms, fitted=n in BALANCED) for n, d in tabs.items()}
    label = lambda n: W.LABEL.get(n, n)
    pct = lambda x: f"{100 * x:.1f}"
    md = [f"# Calling the winner of a case, {kind} terms {lo}–{hi}", "",
          f"Walk-forward predictions of the {run} ({len(c)} cases, {len(v):,} votes; every "
          f"method refit before each term). {note} π = "
          "pro-petitioner share of the votes of all earlier terms "
          f"({min(pi.values()):.3f}–{max(pi.values()):.3f}). Intervals: 95%, bootstrap over cases, "
          f"paired, {N_BOOT:,} resamples.", ""]

    md += ["## 1. Against always-petitioner (standard rule, cutoff 0.5)", ""]
    rows = []
    for n in OTHER + BALANCED:
        row = {"Method": label(n)}
        for b, (a, z) in blocks.items():
            d, g = tabs[n][tabs[n].year.between(a, z)], G[n][G[n].year.between(a, z)]
            row[f"Vote acc {b}"] = pct((d.vote == d.pred).mean())
            row[f"Case acc {b}"] = pct((g.R0 == g.win_side).mean())
        rows.append(row)
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]
    d, a = tabs["v1"], tabs["always_petitioner"]
    code, _ = pd.factorize(d.case_id)
    okm, oka = (d.vote == d.pred).to_numpy(float), (a.vote == a.pred).to_numpy(float)
    draws = []
    for _ in range(1000):
        w = rng.multinomial(code.max() + 1, np.full(code.max() + 1, 1 / (code.max() + 1)))[code]
        draws.append(np.average(okm, weights=w) - np.average(oka, weights=w))
    l, h = np.percentile(draws, [2.5, 97.5])
    g = G["v1"]
    md += [f"- Model v1 − always petitioner, vote accuracy, {lo}–{hi}: "
           f"{okm.mean() - oka.mean():+.3f} [{l:+.3f}, {h:+.3f}]."]
    for b, (a_, z) in blocks.items():
        gb = g[g.year.between(a_, z)]
        md += [f"- Model v1 − always petitioner, case accuracy, {b} ({len(gb)} cases): "
               + boot_diff((gb.R0 == gb.win_side).to_numpy(float), (gb.win_side == 1).to_numpy(float), rng) + "."]
    per = []
    for t in terms:
        dt, at, gt = d[d.year == t], a[a.year == t], g[g.year == t]
        per.append({"Term": t, "Cases": len(gt), "Vote acc, v1": pct((dt.vote == dt.pred).mean()),
                    "Vote acc, always petitioner": pct((at.vote == at.pred).mean()),
                    "Case acc, v1": pct((gt.R0 == gt.win_side).mean()),
                    "Case acc, always petitioner": pct(gt.win_side.mean())})
    md += ["", pd.DataFrame(per).to_markdown(index=False), ""]

    md += ["## 2. Does the model know which side wins?", "",
           "Cases sorted by the mean probability over the justices and cut in three equal groups: "
           "the share of cases the petitioner wins in each group. Always-petitioner gives every "
           f"case the same answer ({pct(g.win_side.mean())}% of these cases are won by the petitioner).", ""]
    rows = []
    for n in BALANCED:
        gn = G[n]
        row = {"Method": label(n), "Case-level AUC": f"{roc_auc_score(gn.win_side, gn.s):.3f}"}
        q = pd.qcut(gn.s.rank(method="first"), 3, labels=["Lowest third", "Middle third", "Top third"])
        for k, x in gn.groupby(q, observed=True).win_side.mean().items():
            row[f"{k}: petitioner wins %"] = pct(x)
        row["Calls respondent, % of cases"] = pct(1 - gn.R0.mean())
        row["Respondent calls that are right %"] = pct(1 - gn[gn.R0 == 0].win_side.mean())
        row["Petitioner calls that are right %"] = pct(gn[gn.R0 == 1].win_side.mean())
        rows.append(row)
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]
    rows = []
    for b, (a_, z) in blocks.items():
        gb = g[g.year.between(a_, z)]
        q = pd.qcut(gb.s.rank(method="first"), 3, labels=["Lowest third", "Middle third", "Top third"])
        row = {"Model v1, terms": b, "Cases": len(gb), "Case-level AUC": f"{roc_auc_score(gb.win_side, gb.s):.3f}"}
        row.update({f"{k}: petitioner wins %": pct(x)
                    for k, x in gb.groupby(q, observed=True).win_side.mean().items()})
        rows.append(row)
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]

    md += ["## 3. Rules for calling the winner", "",
           "C1 and C2 have no fitted number: they put the base rate back that the class-balanced "
           "loss takes out. A and B are fitted on the earlier scored terms, so they start in "
           f"{terms[1]}. Case accuracy, %.", ""]
    late = (terms[1], hi)
    cols = {**blocks, **({} if args.final else {"2017": (2017, 2017)}), f"{late[0]}–{late[1]}": late}
    for n in BALANCED:
        gn = G[n]
        rows = []
        for r, text in RULES.items():
            row = {label(n): text}
            for b, (a_, z) in cols.items():
                gb = gn[gn.year.between(a_, z)]
                row[b] = "–" if gb[r].isna().any() else pct((gb[r] == gb.win_side).mean())
            row["Calls petitioner %"] = pct(gn[gn.year.between(*late)][r].mean())
            rows.append(row)
        md += [pd.DataFrame(rows).to_markdown(index=False), ""]
        ok = lambda gg, r: (gg[r] == gg.win_side).to_numpy(float)
        gl = gn[gn.year.between(*late)]
        md += [f"- {lo}–{hi}: C1 − standard {boot_diff(ok(gn, 'C1'), ok(gn, 'R0'), rng)}; "
               f"C1 − always petitioner {boot_diff(ok(gn, 'C1'), ok(gn, 'ALW'), rng)}; "
               f"C2 − standard {boot_diff(ok(gn, 'C2'), ok(gn, 'R0'), rng)}; "
               f"C2 − always petitioner {boot_diff(ok(gn, 'C2'), ok(gn, 'ALW'), rng)}.",
               f"- {late[0]}–{late[1]}: A − always petitioner {boot_diff(ok(gl, 'A'), ok(gl, 'ALW'), rng)}; "
               f"B − always petitioner {boot_diff(ok(gl, 'B'), ok(gl, 'ALW'), rng)}; "
               f"B − standard {boot_diff(ok(gl, 'B'), ok(gl, 'R0'), rng)}.", ""]
    per = []
    for t in terms:
        gt = g[g.year == t]
        per.append({"Model v1, term": t, "Cases": len(gt), **{RULES[r].split(":")[0]: pct((gt[r] == gt.win_side).mean())
                                                              for r in ("R0", "C1", "C2", "ALW")}})
    md += [pd.DataFrame(per).to_markdown(index=False), ""]
    d = tabs["v1"]
    md += ["Votes of model v1 with the class balance undone (p > 1 − π), "
           f"{lo}–{hi}: vote accuracy {pct((d.pred_t == d.vote).mean())}% (at 0.5: "
           f"{pct((d.pred == d.vote).mean())}%), macro-F1 "
           f"{f1_score(d.vote, d.pred_t, average='macro'):.3f} (at 0.5: "
           f"{f1_score(d.vote, d.pred, average='macro'):.3f}), predicted petitioner "
           f"{pct(d.pred_t.mean())}% (at 0.5: {pct(d.pred.mean())}%; true {pct(d.vote.mean())}%).", ""]
    Path(out).write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
