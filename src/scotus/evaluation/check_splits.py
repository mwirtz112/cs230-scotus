"""
Leakage and distribution checks across the train / val / test splits.

Run from the repository root:
    python -m scotus.evaluation.check_splits   # reads data/*.parquet, writes data/split_checks.md

Two kinds of checks:
  1. Leakage: nothing about a test case may appear in train or val. A case must sit in
     exactly one split, no transcript may belong to two cases, and a reargued case (same
     docket number in neighbouring terms) must not straddle a split boundary.
  2. Distribution: val and test should look like the same distribution (the recent Court).
     Train may differ on purpose (older eras); we document the differences.
"""

import argparse
from pathlib import Path

import pandas as pd

SPLITS = ["train", "val", "test"]
ROLES = ["justice", "petitioner", "respondent", "amicus", "advocate_unknown", "unknown"]


def load(data):
    cases = pd.read_parquet(data / "cases.parquet")
    votes = pd.read_parquet(data / "votes.parquet")
    turns = pd.read_parquet(data / "turns.parquet",
                            columns=["case_id", "convo_id", "speaker_id", "role", "text"])
    return cases, votes, turns


# ---------------------------------------------------------------- 1. leakage

def leakage_checks(cases, votes, turns):
    """Return a list of (check, n_problems, detail) rows. Every n_problems should be 0."""
    rows = []

    n = (cases.groupby("case_id").split.nunique() > 1).sum()
    rows.append(("case_id in more than one split", n, ""))

    convo = cases[["case_id", "convo_ids"]].explode("convo_ids")
    n = convo.convo_ids.duplicated().sum()
    rows.append(("transcript shared by two cases", n, ""))

    n = (turns.groupby("convo_id").case_id.nunique() > 1).sum()
    rows.append(("transcript's turns tagged with two cases", n, ""))

    merged = votes.merge(cases[["case_id", "split"]], on="case_id", suffixes=("", "_case"))
    n = (merged.split != merged.split_case).sum()
    rows.append(("vote's split differs from its case's split", n, ""))

    # Reargued cases: same docket number in neighbouring terms, landing in different
    # splits. Old docket numbers are small integers that get reused each term, so only
    # terms at most 2 years apart count as "the same case".
    c = cases.assign(docket=cases.case_id.str.split("_", n=1).str[1])
    hits = []
    for docket, g in c.groupby("docket"):
        if g.split.nunique() < 2:
            continue
        g = g.sort_values("year")
        for (_, a), (_, b) in zip(g.iterrows(), g.iloc[1:].iterrows()):
            if a.split != b.split and b.year - a.year <= 2:
                hits.append(f"{a.case_id} ({a.split}) / {b.case_id} ({b.split})")
    rows.append(("reargued docket straddles a split boundary", len(hits), "; ".join(hits)))
    return rows


# ---------------------------------------------------------------- 2. distribution

def profile(cases, votes, turns, train_justices):
    """One column of descriptive statistics for a subset of cases."""
    v = votes[votes.case_id.isin(cases.case_id)]
    t = turns[turns.case_id.isin(cases.case_id)]
    per_case = v.groupby("case_id").vote.agg(["mean", "size"])
    spoke = t[t.role == "justice"].groupby("case_id").speaker_id.apply(set)
    silent = [j not in spoke.get(c, set()) for c, j in zip(v.case_id, v.justice)]
    words = t.text.str.split().str.len()
    roles = t.role.value_counts(normalize=True)
    unseen = sorted(set(v.justice) - train_justices)

    out = {
        "terms": f"{cases.year.min()}–{cases.year.max()}",
        "cases": len(cases),
        "votes": len(v),
        "petitioner win %": 100 * cases.win_side.mean(),
        "pro-petitioner vote %": 100 * v.vote.mean(),
        "unanimous case %": 100 * per_case["mean"].isin([0, 1]).mean(),
        "justices voting / case (mean)": per_case["size"].mean(),
        "cases with <= 8 votes %": 100 * (per_case["size"] <= 8).mean(),
        "votes by a justice silent in that case %": 100 * pd.Series(silent).mean(),
        "median turns / case": t.groupby("case_id").size().median(),
        "median words / turn": words.median(),
        "mean words / turn": words.mean(),
    }
    for r in ROLES:
        out[f"role: {r} %"] = 100 * roles.get(r, 0.0)
    out["distinct justices"] = v.justice.nunique()
    out["justices unseen in train"] = ", ".join(j.replace("j__", "") for j in unseen) or "none"
    return out


def fmt(x):
    if isinstance(x, float):
        return f"{x:.1f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    cases, votes, turns = load(data)
    train_justices = set(votes[votes.split == "train"].justice)

    leak = pd.DataFrame(leakage_checks(cases, votes, turns),
                        columns=["check", "problems", "detail"])

    by_split = pd.DataFrame({s: profile(cases[cases.split == s], votes, turns, train_justices)
                             for s in SPLITS})
    recent = cases[cases.year >= 2014]
    by_term = pd.DataFrame({y: profile(g, votes, turns, train_justices)
                            for y, g in recent.groupby("year")})

    report = [
        "# Split checks", "",
        "## 1. Leakage (every count should be 0)",
        leak.to_markdown(index=False), "",
        "## 2. Train vs val vs test",
        "Val and test should look alike; train is allowed to differ.", "",
        by_split.map(fmt).to_markdown(), "",
        "## 3. Per term, 2014–2019", "",
        by_term.map(fmt).to_markdown(), "",
    ]
    text = "\n".join(report)
    (data / "split_checks.md").write_text(text)
    print(text)
    print(f"\nWrote {data}/split_checks.md")


if __name__ == "__main__":
    main()
