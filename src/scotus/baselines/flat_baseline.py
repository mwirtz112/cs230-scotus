"""
Flat-transcript baseline, step 2: votes from one frozen ModernBERT vector per case.

Run from the repository root:
    python -m scotus.baselines.flat_baseline          -> data/preds/flat_mb*_{dev,val}.parquet,
                                                         results/flat_modernbert.md
    python -m scotus.baselines.flat_baseline --grid   # first rerun the dev-fold grid and save
                                                      # the choice (see below)

What it tests: how far "hand the transcript to a long-context model" gets with no turn
structure, no question counting and no bloc layer. The transcript's first 8,192 tokens go
through ModernBERT once (flat_encode.py); everything here works on that one vector.

    z       = the case's document vector, standardized with the fold's training cases
    g       = ReLU(W dropout(z))                    one hidden layer, shared by all justices
    logit_j = (v + a_j) . g + b + b_j               a_j, b_j: a small vector and an offset
                                                    per justice

A justice the fold has not seen gets a_j = 0, b_j = 0 (the shared part only), and in
training 10% of the justices of a case are hidden the same way, as in model v1. Class-
balanced loss, cutoff 0.5, AdamW, a fixed number of epochs, five seeds averaged. Trained
on terms from 1980, like the networks.

Choices that are not fixed by the design (pooling, width, dropout, epochs) are made on
the dev fold only (train <= 2011, score 2012-15), by AUC, from the small grid below, and
saved in results/flat_modernbert_chosen.json; the validation fold is then scored once
with the chosen setting. Test is not read.

Two versions (POOLS): flat_mb reads the first 8,192 tokens, as proposed; flat_mb_all
reads the whole transcript in several windows (flat_encode.py --rest).
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from scotus.evaluation import eval as ev
from scotus.evaluation import summarize_v1 as S

TRAIN_START = 1980
FOLDS = {"dev": (2011, (2012, 2015)), "val": (2015, (2016, 2017))}
SEEDS = [0, 1, 2, 3, 4]
# Two versions of the baseline, by what the document vector covers:
#   flat_mb      the first 8,192 tokens (the baseline as proposed)
#   flat_mb_all  the whole transcript, in windows of 8,192 tokens whose means are averaged
POOLS = {"flat_mb": ["mean", "cls"], "flat_mb_all": ["meanall", "clsall"]}
GRID = {"hidden": [64, 256], "dropout": [0.2, 0.5], "epochs": [10, 30]}
DEFAULT = {"hidden": 64, "dropout": 0.5, "epochs": 10}
CHOSEN_FILE = Path("results/flat_modernbert_chosen.json")     # written by --grid (dev fold only)
KEY = ["case_id", "justice"]


def chosen(method):
    """The setting picked on the dev fold for this method (results/flat_modernbert_chosen.json)."""
    if CHOSEN_FILE.exists():
        saved = json.loads(CHOSEN_FILE.read_text())
        if method in saved:
            return saved[method]
    return {"pool": POOLS[method][0], **DEFAULT}


def available(method, folder="embeddings"):
    return all((Path(folder) / f"flat_mb_{pool}.npy").exists() for pool in POOLS[method])


def load_vectors(folders=("embeddings",), pool="mean"):
    """case_id -> document vector, from one or more flat_encode.py output folders."""
    ids, mats = [], []
    for f in folders:
        f = Path(f)
        ids.append(pd.read_parquet(f / "flat_mb_index.parquet", columns=["case_id"]).case_id)
        mats.append(np.load(f / f"flat_mb_{pool}.npy"))
    return pd.Series(np.arange(sum(map(len, ids))), index=pd.concat(ids).to_numpy()), \
        np.concatenate(mats)


class Head(nn.Module):
    def __init__(self, n_justices, hidden, dropout, dim=768):
        super().__init__()
        self.drop, self.lin = nn.Dropout(dropout), nn.Linear(dim, hidden)
        self.v = nn.Parameter(torch.zeros(hidden))
        self.b = nn.Parameter(torch.zeros(1))
        self.a = nn.Embedding(n_justices + 1, hidden, padding_idx=0)     # 0 = unknown: zero
        self.bj = nn.Embedding(n_justices + 1, 1, padding_idx=0)
        nn.init.normal_(self.v, std=hidden ** -0.5)
        nn.init.zeros_(self.a.weight); nn.init.zeros_(self.bj.weight)

    def forward(self, z, jid):
        g = F.relu(self.lin(self.drop(z)))                               # [B, H]
        return ((self.v + self.a(jid)) * g[:, None, :]).sum(-1) + self.b + self.bj(jid).squeeze(-1)


def pack(votes, row_of, vocab):
    """Votes -> per-case arrays, cases in a fixed (sorted) order."""
    v = votes.sort_values(KEY)
    groups = v.groupby("case_id", sort=True)
    cases = list(groups.groups)
    n, k = len(cases), int(groups.size().max())
    jid, y, mask = (np.zeros((n, k), dtype=d) for d in (np.int64, np.float32, bool))
    for i, (_, g) in enumerate(groups):
        m = len(g)
        jid[i, :m] = [vocab.get(j, 0) for j in g.justice]
        y[i, :m], mask[i, :m] = g.vote.to_numpy(), True
    return v, row_of.loc[cases].to_numpy(), jid, y, mask


def fit_head(ztr, tj, ty, tm, n_justices, cfg, seed, id_drop=0.1, lr=1e-3, wd=0.1, bs=64):
    """One head (one seed) trained on standardized document vectors ztr [N, D]; tj, ty, tm
    [N, K] = justice ids, votes, mask. Returned in eval mode. Also the starting point of
    the fine-tuned version (flat_finetune.py)."""
    pos_weight = torch.tensor(float((tm.sum() - ty.sum()) / ty.sum()))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head(n_justices, cfg["hidden"], cfg["dropout"], ztr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(ztr))
        for i in range(0, len(order), bs):
            idx = torch.from_numpy(order[i:i + bs])
            j = tj[idx].clone()
            j[torch.from_numpy(rng.random(j.shape) < id_drop)] = 0       # identity dropout
            loss = F.binary_cross_entropy_with_logits(
                model(ztr[idx], j), ty[idx], pos_weight=pos_weight, reduction="none")
            loss = (loss * tm[idx]).sum() / tm[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def fit_predict(train, held, vectors, cfg, seeds=SEEDS, id_drop=0.1, lr=1e-3, wd=0.1, bs=64):
    """Seed-averaged predictions for `held` (case_id, justice, p, pred). `train` and `held`
    need case_id, justice, vote; `vectors` = load_vectors(...) for cfg's pooling."""
    row_of, X = vectors
    train = train[train.case_id.isin(row_of.index)]
    vocab = {j: i + 1 for i, j in enumerate(sorted(train.justice.unique()))}
    _, tr_rows, tr_j, tr_y, tr_m = pack(train, row_of, vocab)
    he_v, he_rows, he_j, _, he_m = pack(held, row_of, vocab)
    mean, std = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
    Z = lambda rows: torch.from_numpy(((X[rows] - mean) / std).astype(np.float32))
    ztr, zhe = Z(tr_rows), Z(he_rows)
    tj, ty, tm = map(torch.from_numpy, (tr_j, tr_y, tr_m))
    probs = []
    for seed in seeds:
        model = fit_head(ztr, tj, ty, tm, len(vocab), cfg, seed, id_drop, lr, wd, bs)
        with torch.no_grad():
            probs.append(torch.sigmoid(model(zhe, torch.from_numpy(he_j))).numpy()[he_m])
    p = np.mean(probs, axis=0)
    return he_v[KEY].assign(p=p, pred=(p >= 0.5).astype(int)).reset_index(drop=True)


def fold_votes(fold, votes):
    end, (lo, hi) = FOLDS[fold]
    return (votes[(votes.year >= TRAIN_START) & (votes.year <= end)],
            votes[(votes.year >= lo) & (votes.year <= hi)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--grid", action="store_true",
                    help="rerun the dev-fold grid (3 seeds per setting) and save the choice")
    args = ap.parse_args()
    votes = pd.concat([ev.load_labels(s, args.data)[0] for s in ["train", "val"]],
                      ignore_index=True)                           # test labels never loaded
    methods = [m for m in POOLS if available(m)]
    md = ["# Flat ModernBERT baseline (frozen encoder)", "",
          "The transcript as one text (one line per turn: a justice by surname, an advocate "
          "by side) through ModernBERT-base, one vector per case (flat_encode.py); then a "
          "one-hidden-layer head with a small vector and offset per justice "
          "(flat_baseline.py). flat_mb = the first 8,192 tokens only, as proposed; "
          "flat_mb_all = the whole transcript, read in windows of 8,192 tokens whose "
          "vectors are averaged. Trained on terms from 1980, class-balanced, cutoff 0.5, "
          "five seeds averaged. Settings were chosen on the dev fold by AUC; validation was "
          "scored once with them. Test untouched.", ""]

    ix = pd.read_parquet("embeddings/flat_mb_index.parquet")
    ix = ix[ix.year <= 2017]                                       # the open terms
    cover = ix.assign(share=ix.n_tokens / ix.n_tokens_full,
                      period=np.where(ix.year <= 2015, "train 1980-2015", "validation 2016-17"))
    md += ["## What the first window of 8,192 tokens covers", "",
           cover.groupby("period").agg(
               cases=("case_id", "size"),
               longer_than_window=("n_tokens_full", lambda s: f"{100 * (s > 8192).mean():.1f}%"),
               median_tokens_in_transcript=("n_tokens_full", "median"),
               median_share_kept=("share", lambda s: f"{100 * s.median():.0f}%"),
               median_turns_seen=("n_turns_seen", "median"),
               median_turns=("n_turns", "median"),
               respondent_inside_window=("resp_seen", lambda s: f"{100 * s.mean():.1f}%"),
           ).to_markdown(), ""]

    if args.grid:
        tr, he = fold_votes("dev", votes)
        lab = S.labels("dev", args.data)
        rows, pick = [], {}
        for method in methods:
            for pool in POOLS[method]:
                vectors = load_vectors(pool=pool)
                for combo in itertools.product(*GRID.values()):
                    cfg = {"pool": pool, **dict(zip(GRID, combo))}
                    sc = S.scores(fit_predict(tr, he, vectors, cfg, seeds=[0, 1, 2]), lab[0])
                    rows.append({"method": method, **cfg, "dev AUC": round(sc["auc"], 4),
                                 "dev macro-F1": round(sc["f1"], 4),
                                 "pred petitioner %": round(100 * sc["pet"], 1)})
                    print(rows[-1], flush=True)
        grid = pd.DataFrame(rows).sort_values(["method", "dev AUC"], ascending=[True, False])
        grid.to_csv("results/flat_modernbert_grid.csv", index=False)
        for method, g in grid.groupby("method"):
            pick[method] = {k: (v.item() if hasattr(v, "item") else v) for k, v in
                            g.iloc[0][["pool", *GRID]].items()}
        keep = json.loads(CHOSEN_FILE.read_text()) if CHOSEN_FILE.exists() else {}
        CHOSEN_FILE.write_text(json.dumps({**keep, **pick}, indent=1))
        print("chosen on dev:", pick)
    if Path("results/flat_modernbert_grid.csv").exists():
        grid = pd.read_csv("results/flat_modernbert_grid.csv")
        md += ["## Dev-fold grid (3 seeds per setting; per method, the top row is the one used)",
               "", grid.to_markdown(index=False), ""]

    rows, terms = [], []
    for method in methods:
        cfg = chosen(method)
        vectors = load_vectors(pool=cfg["pool"])
        for fold in ["dev", "val"]:
            tr, he = fold_votes(fold, votes)
            preds = fit_predict(tr, he, vectors, cfg)
            preds.to_parquet(Path(args.data) / "preds" / f"{method}_{fold}.parquet", index=False)
            lab = S.labels(fold, args.data)
            res = ev.score(preds, *lab)
            rows.append({"method": method,
                         "fold": {"dev": "dev 2012-15 (fit on 1980-2011)",
                                  "val": "validation 2016-17 (fit on 1980-2015)"}[fold],
                         **{k: ev.fmt_ci(*res[k]) for k in ev.METRICS},
                         "pred petitioner %": round(100 * preds.pred.mean(), 1)})
            terms.append(ev.score_by_term(preds, *lab).assign(method=method))
    md += ["## Results", "", "Settings used: " + "; ".join(f"{m}: {chosen(m)}" for m in methods),
           "", pd.DataFrame(rows).to_markdown(index=False), "",
           "## Per term", "", pd.concat(terms).round(3).to_markdown(index=False), ""]
    Path("results/flat_modernbert.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
