"""
Train model v1 (model_v1.py) on the dev and validation folds.

Protocol:
  * Two evaluation folds. "val" is the official one (train <= 2015, score 2016-17).
    "dev" is a rolling-origin copy inside the training years (train <= 2011, score
    2012-15, 256 cases). Val has 123 cases and half of it is the odd 2016 term, so
    comparing many ideas on val alone would overfit it. A change should help on dev too.
  * No early stopping on val. The number of epochs is fixed and the learning rate follows
    warmup + cosine decay, so val is never used to pick a checkpoint. The milestone model
    picked its best epoch by val macro-F1, which swung with the predicted-petitioner rate.
  * Several seeds per configuration; their probabilities are averaged by summarize_v1.py.
  * The embedding matrix lives on the GPU, so an epoch takes a few seconds.

Run from the repository root:
    python -m scotus.models.train_v1 --name v1                      # dev + val, seeds 0 1 2
    python -m scotus.models.train_v1 --name v1_nowide --no-wide     # ablations: see the options below
The main model, v1_main (one command):
    python -m scotus.models.train_v1 --name v1_main --epochs 6 --train-start 1980 --affinity 4
        --seeds 0 1 2 3 4 --save-model --eval-hide-ids

Outputs:
    data/preds/<name>_<fold>_s<seed>.parquet      predictions (gitignored)
    results/v1_runs/<name>_<fold>_s<seed>.json    settings, per-epoch history, final scores
    checkpoints/<name>_<fold>_s<seed>.pt          trained weights (with --save-model)
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from scotus.evaluation import eval as ev
from scotus.features import meta_features
from scotus.models import probe
from scotus.features.features import STANCE_FEATURES, WIDE_FEATURES, stance_features
from scotus.models.model_v1 import N_SCAL, TurnModelV1, build_cases, collate, n_params

FOLDS = {"val": (2015, (2016, 2017)),      # name: (last train term, scored terms)
         "dev": (2011, (2012, 2015)),
         "dev2": (2007, (2008, 2011)),
         # "nogap": the same 2014-15 terms that the dev fold scores after a two-term gap,
         # but trained right up to them. Compared with the dev fold's own 2014-15 scores
         # it measures what retraining on the latest two terms is worth: the same
         # question as "retrain on train + val before the 2018-19 test run?".
         "nogap": (2013, (2014, 2015)),
         # The same question with a change of bench, the closest past analogue of the test
         # set: score 2010-11 (Kagan's first two terms; Sotomayor joined in 2009) after
         # training to 2008 (both unknown, like training to 2015 for the 2018-19 test) or
         # to 2009 (Sotomayor known from one term, like training to 2017).
         "newj_gap": (2008, (2010, 2011)),
         "newj_nogap": (2009, (2010, 2011))}


def load_embeddings(path, turns, train_ids, device):
    """Standardized fp16 turn vectors on the device, plus a final all-zero padding row."""
    E = np.load(path, mmap_mode="r")
    assert len(E) == len(turns), "embeddings don't match turns.parquet"
    stats = Path(path).with_name(Path(path).stem + "_train_stats.npz")
    if stats.exists():
        s = np.load(stats); mean, std = s["mean"], s["std"]
    else:
        print("Computing train mean/std...")
        mean, std = probe.train_stats(E, turns.index[turns.case_id.isin(train_ids)].to_numpy())
        np.savez(stats, mean=mean, std=std)
    out = torch.zeros((len(E) + 1, E.shape[1]), dtype=torch.float16)
    for i in range(0, len(E), 200_000):
        z = (E[i:i + 200_000].astype(np.float32) - mean) / std
        out[i:i + len(z)] = torch.from_numpy(z.astype(np.float16))
    return out.to(device)


def batches(cases, bs, rng=None):
    """Index batches. Training: shuffled, but grouped by similar length to cut padding."""
    n = len(cases)
    lens = np.array([len(c.rows) for c in cases])
    if rng is None:
        order = np.argsort(lens)
        return [order[i:i + bs] for i in range(0, n, bs)]
    order = rng.permutation(n)
    chunk = bs * 50
    order = np.concatenate([o[np.argsort(lens[o])] for o in
                            (order[i:i + chunk] for i in range(0, n, chunk))])
    out = [order[i:i + bs] for i in range(0, n, bs)]
    rng.shuffle(out)
    return out


@torch.no_grad()
def predict(model, cases, device, bs=32, hide_ids=False):
    """hide_ids: score every justice as an unknown justice (no identity anywhere)."""
    model.eval()
    rows = []
    for idx in batches(cases, bs):
        batch = [cases[i] for i in idx]
        b = collate(batch, device)
        if hide_ids:
            b["jid"], b["vjid"] = torch.zeros_like(b["jid"]), torch.zeros_like(b["vjid"])
        p = torch.sigmoid(model(b)["logit"]).cpu().numpy()
        for b, c in enumerate(batch):
            rows += [(c.case_id, name, float(p[b, k])) for k, name in enumerate(c.names)]
    preds = pd.DataFrame(rows, columns=["case_id", "justice", "p"])
    return preds.assign(pred=(preds.p >= 0.5).astype(int))        # balanced loss -> 0.5


def quick_scores(preds, votes):
    df = votes.merge(preds, on=["case_id", "justice"])
    out = {"auc": roc_auc_score(df.vote, df.p),
           "f1": f1_score(df.vote, df.pred, average="macro"),
           "pred_pet": float(df.pred.mean())}
    for year, g in df.groupby("year"):
        out[f"auc_{year}"] = roc_auc_score(g.vote, g.p)
        out[f"f1_{year}"] = f1_score(g.vote, g.pred, average="macro")
    return out


def hist_logodds(hist):
    """History rates as log-odds (h_exp and the d_ differences stay as they are)."""
    h = hist[["case_id", "justice"]].copy()
    for c in meta_features.HIST_FEATURES:
        x = hist[c].to_numpy()
        h[c] = x if c == "h_exp" or c.startswith("d_") else np.log(
            np.clip(x, 0.02, 0.98) / (1 - np.clip(x, 0.02, 0.98)))
    return h


def metadata_prior(hist, train_votes, held_votes, k=5, C=0.3):
    """Log-odds of a pro-petitioner vote from the history rates alone.

    Class-balanced logistic regression (the meta_lr baseline). Returns case_id, justice,
    prior: cross-fitted over k folds of cases for `train_votes`, full fit for `held_votes`.
    """
    from sklearn.linear_model import LogisticRegression
    cols = meta_features.HIST_FEATURES
    h = hist_logodds(hist)
    tr = train_votes[["case_id", "justice", "vote"]].merge(h, on=["case_id", "justice"])
    he = held_votes[["case_id", "justice"]].merge(h, on=["case_id", "justice"])
    X = tr[cols].to_numpy(float)
    mean, std = X.mean(0), X.std(0) + 1e-6
    fit = lambda rows: LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(
        (X[rows] - mean) / std, tr.vote.to_numpy()[rows])
    case_fold = pd.util.hash_pandas_object(tr.case_id, index=False).to_numpy() % k
    prior = np.zeros(len(tr))
    for i in range(k):
        m = case_fold == i
        prior[m] = fit(~m).decision_function((X[m] - mean) / std)
    full = fit(np.ones(len(tr), dtype=bool))
    out = [tr[["case_id", "justice"]].assign(prior=prior),
           he[["case_id", "justice"]].assign(
               prior=full.decision_function((he[cols].to_numpy(float) - mean) / std))]
    return pd.concat(out, ignore_index=True)


def regression_prior(source, hist, vote_feats, train_votes, held_votes, k=5):
    """Log-odds from the two-stage count regression, as a prior for the network.

    source "qc_bloc": question counts + named colleagues (no metadata);
    "qc_bloc_meta": the same with the history rates in its first stage (meta_baselines.py).
    Cross-fitted over k folds of cases for `train_votes`, full fit for `held_votes`.
    """
    from scotus.baselines import qc_plus as Q
    pool = list(WIDE_FEATURES)
    cols = ["case_id", "year", "justice", "vote"]
    df = pd.concat([train_votes[cols], held_votes[cols]], ignore_index=True).merge(
        vote_feats[["case_id", "justice"] + pool], on=["case_id", "justice"])
    if source == "qc_bloc_meta":
        df = df.merge(hist[["case_id", "justice"] + meta_features.HIST_FEATURES],
                      on=["case_id", "justice"])
        pool += meta_features.HIST_FEATURES
    tr = df[df.case_id.isin(set(train_votes.case_id))].reset_index(drop=True)
    he = df[df.case_id.isin(set(held_votes.case_id))].reset_index(drop=True)
    case_fold = pd.util.hash_pandas_object(tr.case_id, index=False).to_numpy() % k
    parts = [Q.fit_predict_bloc(tr[case_fold != i], [tr[case_fold == i]], pool_feats=pool)[0]
             for i in range(k)]
    parts.append(Q.fit_predict_bloc(tr, [he], pool_feats=pool)[0])
    out = pd.concat(parts, ignore_index=True)
    p = out.p.clip(1e-4, 1 - 1e-4)
    return out[["case_id", "justice"]].assign(prior=np.log(p / (1 - p)))


def train_one(args, fold, seed, D, device, runs_dir="results/v1_runs", preds_dir=None,
              track=True):
    """Train one seed on one fold and save its held-out predictions.

    runs_dir / preds_dir: where the run record and the predictions go (walk_forward.py
    keeps its own folders). track=False: no held-out score is computed during or after
    training; the predictions are saved and scored in one place (walk_forward.py), so a
    test run never shows a per-epoch test score."""
    train_end, (ev_lo, ev_hi) = FOLDS[fold]
    preds_dir = Path(preds_dir) if preds_dir else Path(args.data) / "preds"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    votes, cases_df = D["votes"], D["cases"]
    tr_votes = votes[(votes.year >= args.train_start) & (votes.year <= train_end)]
    ev_votes = votes[(votes.year >= ev_lo) & (votes.year <= ev_hi)].reset_index(drop=True)
    ev_cases = cases_df[cases_df.case_id.isin(ev_votes.case_id)].reset_index(drop=True)

    # Fold-specific vocabulary and feature scaling: fitted on this fold's train terms only.
    tr_ids = set(tr_votes.case_id)
    turns = D["turns"]
    seen = turns.speaker_id[(turns.role == "justice") & turns.case_id.isin(tr_ids)]
    known = set(tr_votes.justice) | set(seen)
    if args.min_justice_votes:
        # Diagnostic: a justice with fewer training votes than this stays unknown.
        n_votes = tr_votes.justice.value_counts()
        known = {j for j in known if n_votes.get(j, 0) >= args.min_justice_votes}
    vocab = {j: i + 1 for i, j in enumerate(sorted(known))}
    feats, vote_feats, wide_cols = D["feats"], D["vote_feats"], list(WIDE_FEATURES)
    if D["stance"] is not None:
        # Turn-level stance scores (cross-fitted, from a side experiment). Centre and
        # scale them with this fold's training turns, then use them twice: as a fourth
        # per-turn scalar, and summed per justice / per bench as extra wide features.
        raw = D["stance"]
        in_train = turns.case_id.isin(tr_ids) & raw.notna()
        centre, scale = raw[in_train].mean(), raw[in_train].std()
        z = (raw - centre) / scale
        feats = feats.assign(stance=z.fillna(0.0).astype(np.float32))
        vote_feats = vote_feats.merge(stance_features(D["votes"], turns, feats, z),
                                      on=["case_id", "justice"], how="left")
        wide_cols += STANCE_FEATURES
    if args.no_amicus:
        wide_cols = [c for c in wide_cols if "am_" not in c]
    meta, meta_sizes = None, None
    if args.hist or args.meta or args.prior:
        # Metadata fusion. Frozen protocol: a held-out case sees voting history only
        # up to this fold's training cutoff; rolling: up to its own argument day.
        # (walk_forward.py names the history protocol of its step in D["meta_fold"])
        prof, hist = meta_features.load(args.data, D.get("meta_fold", fold), args.meta_protocol)
    if args.hist:
        # History rates as log-odds, appended to the wide features (linear path + MLP).
        vote_feats = vote_feats.merge(hist_logodds(hist), on=["case_id", "justice"],
                                      how="left")
        wide_cols += meta_features.HIST_FEATURES
    prior_col = None
    if args.prior:
        # Prior on the logit: log-odds from a regression fitted outside the network
        # on every training vote of this fold (all terms from 1955, not just
        # --train-start). Training rows get cross-fitted values (5 folds by case), so the
        # network never sees an in-sample prior; held-out rows get the full fit.
        key = ("prior", args.prior_source, fold, args.meta_protocol)
        if key not in D:                               # the same for every seed
            tv = votes[votes.year <= train_end]
            D[key] = (metadata_prior(hist, tv, ev_votes) if args.prior_source == "meta" else
                      regression_prior(args.prior_source, hist, D["vote_feats"], tv, ev_votes))
        vote_feats = vote_feats.merge(D[key], on=["case_id", "justice"], how="left")
        vote_feats["prior"] = vote_feats.prior.fillna(0.0)
        prior_col = "prior"
    if args.meta:
        # Categorical case profile -> embedding indices. The vocabulary of each field is
        # this fold's training cases: values seen in at least --meta-min of them get their
        # own index, the rest share index 1 ("rare"); 0 = unknown.
        fields = args.meta_fields or meta_features.CAT_FIELDS
        p_tr = prof[prof.case_id.isin(tr_ids)]
        codes = {}
        for f in fields:
            n = p_tr[f][p_tr[f] > 0].value_counts()
            index = {v: i + 2 for i, v in enumerate(sorted(n.index[n >= args.meta_min]))}
            codes[f] = prof[f].map(lambda v, index=index: 0 if v == 0 else index.get(v, 1))
        table = np.stack([codes[f].to_numpy() for f in fields], axis=1)
        sign = prof.lc_dir.map({1: 1.0, 2: -1.0}).fillna(0.0).to_numpy()
        if "lc_dir" not in fields:
            sign = np.zeros(len(prof))                 # ablation without the direction
        meta = {c: (table[i], float(sign[i])) for i, c in enumerate(prof.case_id)}
        meta_sizes = [int(table[:, i].max()) + 1 for i in range(len(fields))]
    wf = tr_votes[["case_id", "justice"]].merge(vote_feats, on=["case_id", "justice"])
    wide_stats = (wf[wide_cols].mean().to_numpy(np.float32),
                  wf[wide_cols].std().to_numpy(np.float32) + 1e-6)
    build = lambda v: build_cases(set(v.case_id), turns, feats, v, vote_feats, vocab,
                                  wide_stats, wide_cols, meta, prior_col)
    train, held = build(tr_votes), build(ev_votes)

    # Recency weights: a case from `tau` terms before the last train term counts 1/e.
    year = np.array([c.year for c in train])
    case_w = np.exp((year - train_end) / args.tau) if args.tau > 0 else np.ones(len(train))
    case_w = case_w / case_w.mean()
    n_votes = np.array([len(c.labels) for c in train])
    n_pos = np.array([c.labels.sum() for c in train])
    pos_weight = float((case_w * (n_votes - n_pos)).sum() / (case_w * n_pos).sum())
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    model = TurnModelV1(D["E"], len(vocab), d=args.d, layers=args.layers, heads=args.heads,
                        ff=args.ff, dropout=args.dropout, in_drop=args.in_drop,
                        counters=args.counters, hidden=args.hidden, struct=not args.no_struct,
                        wide=not args.no_wide, content=not args.no_content,
                        identity=not args.no_identity, stance_aux=args.stance_aux > 0,
                        text=not args.no_text, bottleneck=args.bottleneck,
                        bench_layers=args.bench_layers, affinity=args.affinity,
                        aff_rank=args.aff_rank, aff_mode=args.aff_mode,
                        aff_unknown=args.aff_unknown, mf_iters=args.mf_iters,
                        case_aux=args.case_aux > 0, meta_sizes=meta_sizes,
                        d_meta=args.d_meta, meta_concat=not args.no_meta_concat,
                        ideal=args.ideal, prior=args.prior, prior_learn=args.prior_learn,
                        center_text=args.center_text,
                        n_wide=len(wide_cols),
                        n_scal=N_SCAL + (D["stance"] is not None)).to(device)
    # Per-justice tables (affinity, gain, lean) see few examples per entry, so they get a
    # larger learning rate and no weight decay.
    tables = ("aff_pair", "aff_recv", "aff_send", "gain", "lean", "ip_justice")
    fast = [p for n, p in model.named_parameters() if n.split(".")[0] in tables]
    slow = [p for n, p in model.named_parameters() if n.split(".")[0] not in tables]
    opt = torch.optim.AdamW([{"params": slow, "lr": args.lr, "weight_decay": args.wd},
                             {"params": fast, "lr": args.lr * args.table_lr_mult,
                              "weight_decay": 0.0}])
    steps = args.epochs * math.ceil(len(train) / args.bs)
    warm = math.ceil(len(train) / args.bs)                         # one epoch of warmup
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) /
                                                                  max(1, steps - warm))))
    probe_set = [train[i] for i in rng.choice(len(train), min(500, len(train)), replace=False)]
    probe_votes = tr_votes[tr_votes.case_id.isin({c.case_id for c in probe_set})]
    cw = torch.tensor(case_w, dtype=torch.float32, device=device)

    hist, t0 = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for idx in batches(train, args.bs, rng):
            b = collate([train[i] for i in idx], device, args.turn_drop, args.id_drop, rng)
            out = model(b, count_scale=1.0 / (1.0 - args.turn_drop))
            bce = F.binary_cross_entropy_with_logits(out["logit"], b["y"], pos_weight=pw,
                                                     reduction="none")
            w = b["vmask"].float() * cw[torch.as_tensor(idx, device=device)][:, None]
            loss = (bce * w).sum() / w.sum()
            if args.case_aux > 0:
                # Auxiliary case-level task: does the petitioner get a majority of the
                # recorded votes? (the case winner, from the same labels)
                n_v = b["vmask"].float().sum(1)
                won = ((b["y"] * b["vmask"]).sum(1) > n_v / 2).float()
                loss = loss + args.case_aux * F.binary_cross_entropy_with_logits(out["case"], won)
            if args.stance_aux > 0:
                # Auxiliary turn-level task: does the speaker of this justice turn end up
                # voting AGAINST the advocate at the lectern? One label per justice turn.
                ok = (b["turn_vote"] >= 0) & (b["target"] >= 0)
                hostile = torch.where(b["target"] == 1, b["turn_vote"], 1 - b["turn_vote"])
                if ok.any():
                    loss = loss + args.stance_aux * F.binary_cross_entropy_with_logits(
                        out["stance"][ok], hostile[ok].float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot, n = tot + loss.item() * len(idx), n + len(idx)
        h = {"epoch": epoch, "train_loss": tot / n}
        if track:
            h.update({f"ev_{k}": v for k, v in quick_scores(predict(model, held, device),
                                                            ev_votes).items()})
            if epoch % 3 == 0 or epoch == args.epochs:
                tr = quick_scores(predict(model, probe_set, device), probe_votes)
                h.update(train_auc=tr["auc"], train_f1=tr["f1"])
        hist.append(h)
        print(f"  [{fold} s{seed}] epoch {epoch:2d}  loss {h['train_loss']:.3f}"
              + (f"  AUC {h['ev_auc']:.3f}  F1 {h['ev_f1']:.3f}  pred-pet "
                 f"{h['ev_pred_pet']:.2f}" if track else "")
              + (f"  train AUC {h['train_auc']:.3f}" if "train_auc" in h else "")
              + f"  {time.time() - t0:.0f}s", flush=True)

    preds = predict(model, held, device)                           # last epoch, no selection
    tag = f"{args.name}_{fold}_s{seed}"
    preds.to_parquet(preds_dir / f"{tag}.parquet", index=False)
    if args.eval_hide_ids:
        # Held-out-justice check: the same trained model, but every justice is scored as
        # a newcomer. Saved as data/preds/<name>_noid_<fold>_s<seed>.parquet.
        predict(model, held, device, hide_ids=True).to_parquet(
            preds_dir / f"{args.name}_noid_{fold}_s{seed}.parquet", index=False)
    final = {}
    if track:
        res = ev.score(preds, ev_votes[["case_id", "year", "justice", "vote"]],
                       ev_cases[["case_id", "year", "win_side"]], n_boot=0)
        final = {k: float(res[k][0]) for k in ev.METRICS}
        final.update({k: v for k, v in hist[-1].items() if k.startswith("ev_")})
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    Path(runs_dir, f"{tag}.json").write_text(json.dumps(
        {"name": args.name, "fold": fold, "seed": seed, "args": vars(args),
         "params": n_params(model), "train_cases": len(train), "eval_cases": len(held),
         "pos_weight": pos_weight, "secs": time.time() - t0, "final": final,
         "history": hist}, indent=1))
    if args.save_model:
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "args": vars(args), "vocab": vocab, "wide_stats": wide_stats},
                   f"checkpoints/{tag}.pt")
    return final


def make_parser():
    """Every option of a v1 run. walk_forward.py builds its recipes from the same parser,
    so a recipe is literally a train_v1.py command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--emb", nargs="+", default=["embeddings/turns_last.npy"],
                    help="turn-vector file(s); several files are concatenated per turn "
                         "(encoder comparison, docs/writeup_notes.md §17)")
    ap.add_argument("--folds", nargs="+", default=["dev", "val"], choices=list(FOLDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train-start", type=int, default=1955)
    ap.add_argument("--tau", type=float, default=0, help="recency half-life-like scale in "
                    "terms (weight = exp((year - last train year) / tau)); 0 = uniform")
    # model
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--counters", type=int, default=8, help="learned counters (0 = off)")
    ap.add_argument("--no-struct", action="store_true", help="drop lectern/segment/length inputs")
    ap.add_argument("--no-wide", action="store_true", help="drop the hand-built count features")
    ap.add_argument("--no-content", action="store_true", help="drop the attention readout")
    ap.add_argument("--no-identity", action="store_true", help="drop justice identities")
    ap.add_argument("--no-amicus", action="store_true", help="drop the amicus-slot features")
    ap.add_argument("--stance", help="turn-level stance scores, e.g. data/stance_scores.parquet")
    ap.add_argument("--no-text", action="store_true", help="drop the turn embeddings: "
                    "the network sees who spoke, to whom and how long, but not what was said")
    ap.add_argument("--bottleneck", type=int, default=0, help="rank of the text projection")
    ap.add_argument("--center-text", action="store_true",
                    help="subtract each case's mean turn vector from its turn vectors")
    ap.add_argument("--bench-layers", type=int, default=0, help="self-attention over justices")
    ap.add_argument("--affinity", type=int, default=0, help="justice-affinity channels (0 = off)")
    ap.add_argument("--aff-rank", type=int, default=16)
    ap.add_argument("--aff-mode", default="pair", choices=["pair", "belief", "lowrank"])
    ap.add_argument("--aff-unknown", action="store_true",
                    help="learn affinity entries for an unknown justice (via identity dropout)")
    ap.add_argument("--mf-iters", type=int, default=1, help="mean-field steps (belief mode)")
    ap.add_argument("--table-lr-mult", type=float, default=10.0,
                    help="learning-rate multiplier for the per-justice tables")
    # metadata fusion (SCDB; run prepare_scdb.py and meta_features.py first)
    ap.add_argument("--hist", action="store_true",
                    help="add the SCDB voting-history rates to the wide features")
    ap.add_argument("--meta", action="store_true",
                    help="add the categorical case profile as learned embeddings")
    ap.add_argument("--meta-fields", nargs="+", choices=meta_features.CAT_FIELDS,
                    help="subset of the profile fields (default: all)")
    ap.add_argument("--meta-min", type=int, default=20,
                    help="a field value needs this many training cases for its own embedding")
    ap.add_argument("--d-meta", type=int, default=32)
    ap.add_argument("--no-meta-concat", action="store_true",
                    help="keep the profile out of the per-justice MLP (ideal-point term only)")
    ap.add_argument("--ideal", type=int, default=0,
                    help="dimensions of the justice ideal-point term (0 = off; needs --meta)")
    ap.add_argument("--prior", type=float, default=0.0,
                    help="weight of the metadata-only log-odds added to the logit (0 = off)")
    ap.add_argument("--prior-learn", action="store_true", help="learn that weight")
    ap.add_argument("--prior-source", default="meta", choices=["meta", "qc_bloc", "qc_bloc_meta"],
                    help="what the prior is: the metadata-only regression (meta), or the "
                         "count regression with named colleagues, without (qc_bloc) or with "
                         "(qc_bloc_meta) the history rates. The last two make the network a "
                         "residual model on top of the strongest baseline.")
    ap.add_argument("--meta-protocol", default="frozen", choices=["frozen", "rolling"],
                    help="voting history for held-out cases: up to the training cutoff "
                         "(frozen) or up to the case's argument day (rolling)")
    # regularization and optimization
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--in-drop", type=float, default=0.0, help="dropout on the turn vectors")
    ap.add_argument("--turn-drop", type=float, default=0.0, help="drop whole turns in training")
    ap.add_argument("--id-drop", type=float, default=0.1, help="hide a justice's identity")
    ap.add_argument("--min-justice-votes", type=int, default=0,
                    help="a justice needs this many training votes to get an identity "
                         "(0 = any; a diagnostic, not part of the recipe)")
    ap.add_argument("--stance-aux", type=float, default=0.0, help="weight of the turn-level loss")
    ap.add_argument("--case-aux", type=float, default=0.0,
                    help="weight of the case-winner loss (also adds a pooled case vector)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--eval-hide-ids", action="store_true",
                    help="also save predictions with every justice's identity hidden")
    return ap


def main():
    args = make_parser().parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = Path(args.data)
    (data / "preds").mkdir(exist_ok=True)
    cases = pd.read_parquet(data / "cases.parquet")
    turns = pd.read_parquet(data / "turns.parquet", columns=["case_id", "speaker_id", "role"])
    votes = pd.concat([ev.load_labels(s, args.data)[0] for s in ["train", "val"]],
                      ignore_index=True)                           # test labels never loaded
    t0 = time.time()
    D = {"cases": cases[cases.split.isin(["train", "val"])], "turns": turns, "votes": votes,
         "feats": pd.read_parquet(data / "turn_feats.parquet"),
         "vote_feats": pd.read_parquet(data / "vote_feats.parquet"),
         "E": torch.cat([load_embeddings(e, turns, cases.case_id[cases.split == "train"],
                                         device) for e in args.emb], dim=1),
         "stance": None}
    if args.stance:
        sc = pd.read_parquet(args.stance)
        D["stance"] = pd.Series(np.nan, index=turns.index, dtype=np.float64)
        D["stance"].iloc[sc.row.to_numpy()] = sc.s.to_numpy()
    print(f"{args.name}: data loaded in {time.time() - t0:.0f}s; device {device}")
    for fold in args.folds:
        for seed in args.seeds:
            final = train_one(args, fold, seed, D, device)
            print(f"{args.name} [{fold} s{seed}] " +
                  "  ".join(f"{k}={final[k]:.3f}" for k in ev.METRICS), flush=True)


if __name__ == "__main__":
    main()
