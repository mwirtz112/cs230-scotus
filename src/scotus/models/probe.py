"""
Embedding probe: do the frozen Legal-BERT turn vectors carry vote-relevant content?

A diagnostic (and extra baseline), not the model. For each (case, justice) vote:
    x = [ mean vector of j's turns aimed at the PETITIONER,      768
          mean vector of j's turns aimed at the RESPONDENT,      768
          log(1 + # turns to petitioner), log(1 + # to respondent), silent flag ]
Vectors are standardized with TRAIN mean/std first (the anisotropy fix). "Aimed at" uses
the same rule as the question-count baseline (baselines.question_targets). A logistic
regression is fit on train votes; the cutoff is the train pro-petitioner vote rate
(as for the justice-rate baseline).

If this beats the question count, the vectors carry content beyond "how many questions";
the transformer's job is then to add context on top.

Model v1 reuses train_stats, the standardization of the turn vectors.

Run from the repository root:
    python -m scotus.models.probe [--emb embeddings/turns_last.npy]    -> results/embedding_probe*.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from scotus.baselines import baselines as bl
from scotus.evaluation import eval as ev

SPLITS = ["train", "val"]


def train_stats(E, rows, block=200_000):
    """Mean and std of the embedding dims over the given (train) rows, in blocks."""
    s = np.zeros(E.shape[1]); s2 = np.zeros(E.shape[1])
    for i in range(0, len(rows), block):
        x = E[np.sort(rows[i:i + block])].astype(np.float64)
        s += x.sum(0); s2 += (x ** 2).sum(0)
    mean = s / len(rows)
    return mean.astype(np.float32), np.sqrt(s2 / len(rows) - mean ** 2).astype(np.float32)


def vote_features(votes, credited, target, E, mean, std):
    """Feature matrix with one row per vote (same order as `votes`)."""
    key = pd.MultiIndex.from_frame(votes[["case_id", "justice"]])
    X = np.zeros((len(votes), 2 * E.shape[1] + 3), dtype=np.float32)
    n = np.zeros((len(votes), 2))
    for side in (0, 1):
        t = credited[target[credited.index] == side]
        vi = key.get_indexer(pd.MultiIndex.from_frame(t[["case_id", "speaker_id"]]))
        t, vi = t[vi >= 0], vi[vi >= 0]            # drop turns by justices with no vote
        rows = t.index.to_numpy()
        order = np.argsort(rows)                   # sorted reads from the memmap
        Z = (E[rows[order]].astype(np.float32) - mean) / std
        G = sparse.csr_matrix((np.ones(len(rows)), (vi[order], np.arange(len(rows)))),
                              shape=(len(votes), len(rows)))
        cnt = np.asarray(G.sum(1)).ravel()
        X[:, side * E.shape[1]:(side + 1) * E.shape[1]] = (G @ Z) / np.maximum(cnt, 1)[:, None]
        n[:, side] = cnt
    X[:, -3:-1] = np.log1p(n)
    X[:, -1] = (n.sum(1) == 0)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--emb", default="embeddings/turns_last.npy")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    args = ap.parse_args()
    data = Path(args.data)
    stem = Path(args.emb).stem                      # turns_last, turns_last4, turns_bge, ...
    name = "embedding_probe" + ("" if stem == "turns_last" else "_" + stem.replace("turns_", ""))

    E = np.load(args.emb, mmap_mode="r")          # row i = row i of turns.parquet
    cases = pd.read_parquet(data / "cases.parquet")
    turns = pd.read_parquet(data / "turns.parquet",
                            columns=["case_id", "convo_id", "turn_idx", "speaker_id", "role"])
    assert len(turns) == len(E), "embeddings don't match turns.parquet"

    keep = cases[cases.split.isin(SPLITS)]
    ordered = bl.ordered_turns(turns[turns.case_id.isin(keep.case_id)], keep)
    target = bl.question_targets(ordered)
    credited = ordered[target >= 0]

    train_rows = turns.index[turns.case_id.isin(cases.case_id[cases.split == "train"])].to_numpy()
    print("Computing train mean/std...")
    mean, std = train_stats(E, train_rows)

    Xs, Vs, Cs = {}, {}, {}
    for split in SPLITS:
        votes, split_cases = ev.load_labels(split, args.data)
        print(f"Building {split} features for {len(votes):,} votes...")
        Xs[split], Vs[split], Cs[split] = (
            vote_features(votes, credited, target, E, mean, std), votes, split_cases)

    print("Fitting logistic regression...")
    clf = LogisticRegression(C=args.C, max_iter=3000)
    clf.fit(Xs["train"], Vs["train"].vote)
    cutoff = Vs["train"].vote.mean()

    rows, terms = [], []
    for split in SPLITS:
        p = clf.predict_proba(Xs[split])[:, 1]
        preds = Vs[split].assign(p=p, pred=(p >= cutoff).astype(int))
        preds.to_parquet(data / "preds" / f"{name}_{split}.parquet", index=False)
        res = ev.score(preds, Vs[split], Cs[split], n_boot=ev.N_BOOT if split == "val" else 200)
        rows.append({"method": name, "split": split,
                     **{k: ev.fmt_ci(*res[k]) for k in ev.METRICS},
                     "pred petitioner %": round(100 * preds.pred.mean(), 1)})
        if split == "val":
            terms.append(ev.score_by_term(preds, Vs[split], Cs[split]))
        print(f"  {split:5s} " + "  ".join(f"{k}={res[k][0]:.3f}" for k in ev.METRICS))

    md = [f"# {name}", "", f"Embeddings: `{args.emb}`; logistic regression C={args.C}; "
          f"cutoff = train pro-petitioner rate {cutoff:.3f}.", "",
          pd.DataFrame(rows).to_markdown(index=False), "", "## Val, per term", "",
          pd.concat(terms).round(3).to_markdown(index=False), ""]
    Path(f"results/{name}.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
