"""
Shared evaluation for every method (baselines and models).

A method hands over one row per (case, justice) vote:
    case_id | justice | p (probability of a pro-petitioner vote) | pred (0/1)
The method picks its own cutoff (from train) to turn p into pred; eval only scores.

Metrics:
    macro_f1  primary. Mean of F1 on pro-petitioner and pro-respondent votes
    auc       secondary. Ranking quality of p; ignores cutoff and base rate
    vote_acc  share of individual votes predicted correctly
    case_acc  majority of predicted votes vs true winner; a tie counts as respondent
              (a tied Court affirms the lower court)

95% confidence intervals come from a bootstrap over CASES, not votes: votes in one case
are correlated, so resampling votes would make intervals too narrow. Resampling is done
with weights (how many times each case was drawn), which is equivalent and fast.

Run from the repository root:
    python -m scotus.evaluation.eval data/preds/question_count_val.parquet --split val
    python -m scotus.evaluation.eval ... --split test --final        # test is locked without --final
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

METRICS = ["macro_f1", "auc", "vote_acc", "case_acc"]
N_BOOT, SEED = 1000, 0


def load_labels(split, data="data", final=False):
    """True votes and case winners for one split. Test needs final=True."""
    if split == "test" and not final:
        raise SystemExit("Test split is locked until the final report (pass --final).")
    data = Path(data)
    votes = pd.read_parquet(data / "votes.parquet")
    cases = pd.read_parquet(data / "cases.parquet")
    v = votes[votes.split == split][["case_id", "year", "justice", "vote"]]
    c = cases[cases.split == split][["case_id", "year", "win_side"]]
    return v.reset_index(drop=True), c.reset_index(drop=True)


def join(preds, votes):
    """Attach predictions to true votes; every true vote must have a prediction."""
    df = votes.merge(preds[["case_id", "justice", "p", "pred"]],
                     on=["case_id", "justice"], how="left")
    missing = df.p.isna().sum()
    if missing:
        raise ValueError(f"{missing} votes have no prediction")
    return df


def case_correct(df, cases):
    """Per case (in cases order): 1 if the majority of predicted votes picks the winner."""
    g = df.groupby("case_id", sort=False).pred.agg(["sum", "size"])
    case_pred = (g["sum"] > g["size"] / 2).astype(int)   # strict majority, tie -> 0
    truth = cases.set_index("case_id").win_side
    return (case_pred == truth.loc[case_pred.index]).reindex(cases.case_id).to_numpy()


def _metrics(df, correct, w_vote=None, w_case=None):
    """All metrics, optionally weighted (weights = bootstrap draw counts)."""
    out = {
        "macro_f1": f1_score(df.vote, df.pred, average="macro", sample_weight=w_vote,
                             zero_division=0),
        "vote_acc": np.average(df.vote == df.pred, weights=w_vote),
    }
    try:
        out["auc"] = roc_auc_score(df.vote, df.p, sample_weight=w_vote)
    except ValueError:                   # only one class present in a resample
        out["auc"] = np.nan
    out["case_acc"] = np.average(correct, weights=w_case)
    return out


def score(preds, votes, cases, n_boot=N_BOOT, seed=SEED):
    """Point estimates plus 95% bootstrap CIs. Returns {metric: (est, lo, hi)}."""
    df = join(preds, votes)
    correct = case_correct(df, cases)
    point = _metrics(df, correct)
    if n_boot == 0:
        return {m: (point[m], np.nan, np.nan) for m in METRICS}

    rng = np.random.default_rng(seed)
    case_pos = pd.Series(np.arange(len(cases)), index=cases.case_id)
    row_case = case_pos.loc[df.case_id].to_numpy()       # which case each vote belongs to
    draws = {m: [] for m in METRICS}
    for _ in range(n_boot):
        counts = rng.multinomial(len(cases), np.full(len(cases), 1 / len(cases)))
        m = _metrics(df, correct, w_vote=counts[row_case], w_case=counts)
        for k in METRICS:
            draws[k].append(m[k])
    return {k: (point[k], np.nanpercentile(draws[k], 2.5), np.nanpercentile(draws[k], 97.5))
            for k in METRICS}


def score_by_term(preds, votes, cases):
    """Point estimates per term (no CIs; per-term samples are small)."""
    rows = []
    for year, c in cases.groupby("year"):
        v = votes[votes.case_id.isin(c.case_id)]
        m = score(preds, v, c, n_boot=0)
        rows.append({"term": year, "cases": len(c), "votes": len(v),
                     "petitioner win %": round(100 * c.win_side.mean(), 1),
                     **{k: m[k][0] for k in METRICS}})
    return pd.DataFrame(rows)


def fmt_ci(est, lo, hi):
    if np.isnan(lo):
        return f"{est:.3f}"
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", help="parquet with case_id, justice, p, pred")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="unlock the test split")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    votes, cases = load_labels(args.split, final=args.final)
    preds = pd.read_parquet(args.preds)
    res = score(preds, votes, cases, n_boot=args.n_boot)
    for k in METRICS:
        print(f"{k:9s} {fmt_ci(*res[k])}")
    print()
    print(score_by_term(preds, votes, cases).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
