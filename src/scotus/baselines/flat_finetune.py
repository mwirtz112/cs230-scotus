"""
Flat-transcript baseline, fine-tuned: flat_mb with the top layers of ModernBERT trained too.

This is the frozen baseline flat_mb with the top layers of ModernBERT unfrozen. Everything
else is taken from flat_mb: the input (the transcript's first 8,192 tokens as one text),
the pooled vector, width and dropout of the head that flat_mb chose on dev, and the same
standardization. The two rows therefore differ in one thing: whether the encoder's top
layers are trained on the votes. (Not exactly one: in stage 2 the head also trains for
three more epochs. check_flat_finetune.py repeats stage 2 with the encoder frozen to
measure that part; on validation it changes nothing measurable.)

    caffeinate -i python -I -m scotus.baselines.flat_finetune --fold val      # about 6 hours on the M2 Max
    caffeinate -i python -I -m scotus.baselines.flat_finetune --fold dev      # about 6 hours
    python -I -m scotus.baselines.flat_finetune --fold dev --limit 48 --epochs 1 --name smoke   # short test
(-I: isolated mode, because it reads downloaded model files.)

    h       = ModernBERT(first 8,192 tokens), top K layers trainable
    z       = (pooled h - m) / s       pooled as flat_mb (the [CLS] vector); m, s = mean
                                       and sd of the frozen vectors of the fold's training
                                       cases, per dimension (fixed numbers)
    logit_j = (v + a_j) . ReLU(W dropout(z)) + b + b_j        the head of flat_baseline.py

Two stages:
  1. (seconds) The head is trained on the cached frozen vectors exactly as flat_baseline.py
     does it (FB.fit_head, seed --seed). The model is now one seed of flat_mb.
  2. (hours) The top K layers and the head are trained together on the votes, one case of
     8,192 tokens per step.
Starting from a trained head gives the encoder a useful gradient from the first step and
keeps the pretrained features from being pulled around by a random head ("linear probing,
then fine-tuning": Kumar et al. 2022).

Why z is standardized: an early run without it was void. Two
transcripts' ModernBERT vectors are almost the same vector (the common part has norm 39,
a case's own part 8.7; cosine 0.95), so a head that reads the raw vector sees nearly the
same input for every case. That run's training loss ended at 0.567, the level of a model
that knows only the justices (0.568; a constant 0.5 gives 0.571): it never used the
transcript. The frozen baseline always subtracted the mean.

Two checks are built in so that this cannot pass unnoticed again:
  - before training, the live model's held-out predictions must agree with the head on
    the cached vectors (same tokens, pooling and standardization), or the run stops
    (mean logit difference above 0.05; measured 0.013 on validation, where one case of
    123 differs by up to 0.30 and 99% of the votes by less than 0.06);
  - the log prints the losses that mean "knows nothing" and "the frozen baseline" next to
    the running training loss.

What is affordable on an M2 Max laptop (one case of 8,192 tokens per step):
    top 2 of 22 layers trainable   1.2 s per step
    top 4                          1.9 s          <- used (2.1 s on real transcripts)
    top 8                          3.3 s
    all 22                         25 s, 90 GB    (not workable)
The lower layers stay frozen. Settings were not tuned for accuracy (each run costs
hours): 3 epochs, AdamW, linear warm-up over the first 5% of updates then cosine decay,
gradients accumulated over 8 cases, half-precision autocast (its gradients were compared
with full precision on three cases: cosine 1.00), one seed. The two learning rates were
set by stability checks on the dev fold's training cases, not by held-out scores:
    head 1e-4     with the encoder frozen, three more epochs at this rate leave the
                  stage-1 head's dev AUC where it was; 1e-3 unsettles it
    encoder 5e-6  on 320 cases the usual 2e-5 makes the training loss jump above the
                  "knows nothing" level while the rate peaks (0.62 against 0.57); at 5e-6
                  it stays at its starting level (0.42-0.47)
CAUTION: that 320-case test was harsher than a full run. It had
40 updates, so its warm-up was 2 updates and the rate stood at its peak from case 16 on; a
full run warms up over 64 updates (508 cases). And 5e-6 is so low that in the full run the
unfrozen weights moved by 0.07% and the scores equal the frozen baseline's. So the default
--lr 5e-6 is a cautious setting, not a tested optimum; 2e-5 under the full schedule has
not been tried. After any run: python -I -m scotus.evaluation.check_flat_finetune --name NAME --fold FOLD.

No early stopping: the row is the model after the last epoch. The held-out predictions
are also saved before fine-tuning (epoch 0 = one seed of flat_mb) and after each epoch;
on the validation fold that curve is a diagnostic and must not be used to pick an epoch.

Outputs: data/preds/<name>_<fold>.parquet (the row), data/preds/<name>_<fold>_epochs.parquet
(epoch 0..E), results/flat_runs/<name>_<fold>.json (settings, reference losses, loss
curve, time), checkpoints/<name>_<fold>.pt (the trained weights). A checkpoint is written
every 200 steps (to a temporary file first, then renamed), so an interrupted run resumes
where it stopped.
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
from torch import nn
from transformers import AutoModel, AutoTokenizer

from scotus.evaluation import eval as ev
from scotus.baselines import flat_baseline as FB
from scotus.features import flat_encode as FE

FROZEN = "flat_mb"                     # the frozen baseline this model starts from
MAX_GAP = 0.05                         # allowed mean logit difference, live vs cached vectors
                                       # (measured 0.012-0.014; a wiring error gives 0.5 or more)


class FlatModel(nn.Module):
    def __init__(self, head, mean, std, unfreeze, pool):
        super().__init__()
        self.enc = AutoModel.from_pretrained(FE.MODEL, revision=FE.REVISION, dtype=torch.float32)
        for p in self.enc.parameters():
            p.requires_grad_(False)
        top = list(self.enc.layers[len(self.enc.layers) - unfreeze:]) + [self.enc.final_norm]
        for module in top:
            for p in module.parameters():
                p.requires_grad_(True)
        self.pool, self.head = pool, head
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))

    def forward(self, ids, jid):
        """ids [1, T] token ids of one case; jid [1, K] justice ids -> logits [1, K]."""
        h = self.enc(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        h = h.float()
        v = h[:, 0] if self.pool == "cls" else h.mean(1)
        return self.head((v - self.mean) / self.std, jid)

    def trainable(self):
        return {n: p for n, p in self.named_parameters() if p.requires_grad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="val", choices=list(FB.FOLDS))
    ap.add_argument("--name", default="flat_mb_ft")
    ap.add_argument("--data", default="data")
    ap.add_argument("--unfreeze", type=int, default=4, help="trainable top layers")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-6, help="encoder layers")
    ap.add_argument("--head-lr", type=float, default=1e-4, help="head, in stage 2")
    ap.add_argument("--accum", type=int, default=8, help="cases per optimizer step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int,
                    help="short test: fine-tune on this many training cases (needs its own --name)")
    ap.add_argument("--eval-limit", type=int, default=10,
                    help="held-out cases predicted in a --limit run")
    ap.add_argument("--ckpt-every", type=int, default=200,
                    help="steps between checkpoints (a multiple of --accum)")
    args = ap.parse_args()
    assert not (args.limit and args.name == "flat_mb_ft"), "a --limit run needs its own --name"
    assert args.unfreeze >= 1 and args.ckpt_every % args.accum == 0
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    cfg = FB.chosen(FROZEN)            # pooling, width, dropout, epochs of the frozen head
    assert cfg["pool"] in ("cls", "mean")
    tag = f"{args.name}_{args.fold}"

    votes = pd.concat([ev.load_labels(s, args.data)[0] for s in ["train", "val"]],
                      ignore_index=True)                           # test labels never loaded
    train, held = FB.fold_votes(args.fold, votes)
    vec_row, X = FB.load_vectors(pool=cfg["pool"])                 # the cached frozen vectors
    train = train[train.case_id.isin(vec_row.index)]
    if args.limit:
        held = held[held.case_id.isin(set(sorted(held.case_id.unique())[:args.eval_limit]))]
    vocab = {j: i + 1 for i, j in enumerate(sorted(train.justice.unique()))}
    _, tr_rows, tr_j, tr_y, tr_m = FB.pack(train, vec_row, vocab)
    he_v, he_rows, he_j, _, he_m = FB.pack(held, vec_row, vocab)
    tr_cases, he_cases = sorted(train.case_id.unique()), sorted(held.case_id.unique())
    assert (vec_row.loc[tr_cases].to_numpy() == tr_rows).all()
    assert (vec_row.loc[he_cases].to_numpy() == he_rows).all()

    # Stage 1: the frozen baseline's head on the cached vectors (one seed of flat_mb)
    mean, std = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
    Z = lambda rows: torch.from_numpy(((X[rows] - mean) / std).astype(np.float32))
    tj, ty, tm = map(torch.from_numpy, (tr_j, tr_y, tr_m))
    head = FB.fit_head(Z(tr_rows), tj, ty, tm, len(vocab), cfg, args.seed)
    w = float((tm.sum() - ty.sum()) / ty.sum())

    def case_loss(logit):                                         # mean over cases, as logged
        per = F.binary_cross_entropy_with_logits(logit, ty, pos_weight=torch.tensor(w),
                                                 reduction="none")
        return float(((per * tm).sum(1) / tm.sum(1)).mean())
    with torch.no_grad():
        frozen_logit = head(Z(he_rows), torch.from_numpy(he_j)).numpy()[he_m]
        hidden = tj.clone()
        hidden[torch.from_numpy(np.random.default_rng(args.seed + 1).random(tj.shape) < 0.1)] = 0
        ref = {"constant": case_loss(torch.zeros_like(ty)),
               "head_frozen": case_loss(head(Z(tr_rows), tj)),
               "head_frozen_dropout": case_loss(head.train()(Z(tr_rows), hidden))}
    head.eval()
    print(f"{tag}: reference losses on the training cases: {ref['constant']:.4f} for a constant "
          f"p = 0.5; the head on frozen vectors (stage 1) {ref['head_frozen']:.4f}, and "
          f"{ref['head_frozen_dropout']:.4f} with dropout as in training", flush=True)

    ft = np.arange(len(tr_rows))                                   # cases used in stage 2
    if args.limit:
        ft = np.sort(rng.choice(len(tr_rows), args.limit, replace=False))
    need = {tr_cases[i] for i in ft} | set(he_cases)
    turns = pd.read_parquet(Path(args.data) / "turns.parquet",
                            columns=["case_id", "speaker_id", "role", "text"])
    turns["pos"] = pd.read_parquet(Path(args.data) / "turn_feats.parquet",
                                   columns=["pos"]).pos.to_numpy()
    turns = turns[turns.case_id.isin(need)]
    tok = AutoTokenizer.from_pretrained(FE.MODEL, revision=FE.REVISION)
    ids = FE.tokenize(FE.flat_texts(turns), tok).set_index("case_id").ids
    case_ids = lambda cid: torch.from_numpy(ids.loc[cid])[None].to(device)

    # Stage 2: top layers and head together
    model = FlatModel(head, mean, std, args.unfreeze, cfg["pool"]).to(device)
    pos_weight = torch.tensor(w, device=device)
    enc_params = [p for n, p in model.trainable().items() if n.startswith("enc.")]
    head_params = [p for n, p in model.trainable().items() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr, "weight_decay": 0.01},
                             {"params": head_params, "lr": args.head_lr, "weight_decay": 0.1}])
    n = len(ft)
    total = args.epochs * n
    updates = math.ceil(total / args.accum)
    warm = max(1, round(0.05 * updates))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: (u + 1) / warm if u < warm else
                                              0.5 * (1 + math.cos(math.pi * (u - warm) /
                                                                  max(1, updates - warm))))
    # held-out predictions are taken at the first update boundary after each epoch's end
    eval_at = {min(total, math.ceil(k * n / args.accum) * args.accum): k
               for k in range(1, args.epochs + 1)}
    recipe = json.dumps({"cfg": cfg, "unfreeze": args.unfreeze, "lr": args.lr,
                         "head_lr": args.head_lr, "accum": args.accum, "seed": args.seed,
                         "total": total}, sort_keys=True)
    ckpt = Path("checkpoints") / f"{tag}_partial.pt"
    order = np.concatenate([rng.permutation(n) for _ in range(args.epochs)])
    step, losses, spent, epoch_logit = 0, [], 0.0, {}
    if ckpt.exists():
        try:
            c = torch.load(ckpt, map_location=device, weights_only=True)
        except Exception as e:                       # unreadable: start over, say so
            print(f"checkpoint {ckpt} could not be read ({type(e).__name__}); starting at step 0")
            c = {"recipe": None}
        if c.get("recipe") == recipe:
            model.load_state_dict(c["weights"], strict=False)
            opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
            step, losses, spent = c["step"], c["losses"], c["spent"]
            epoch_logit = {int(k): v.cpu().numpy() for k, v in c["epoch_logit"].items()}
            print(f"resuming at step {step:,} of {total:,}", flush=True)
        else:
            print(f"checkpoint {ckpt} is from another setting; starting at step 0")
    print(f"{tag}: fine-tuning on {n:,} training cases, {len(he_rows)} held out, top "
          f"{args.unfreeze} layers trainable ({sum(p.numel() for p in enc_params) / 1e6:.1f}M "
          f"encoder + {sum(p.numel() for p in head_params) / 1e3:.0f}K head parameters), "
          f"{total:,} steps; device {device}", flush=True)

    he_jt = [torch.from_numpy(j)[None].to(device) for j in he_j]

    def predict():
        """Logits of the held-out votes (rows as he_v), model in eval mode."""
        model.eval()
        out = []
        with torch.no_grad():
            for cid, j in zip(he_cases, he_jt):
                with torch.autocast(device, dtype=torch.float16, enabled=device == "mps"):
                    out.append(model(case_ids(cid), j).float()[0].cpu().numpy())
        model.train()
        return np.stack(out)[he_m]

    def save():
        ckpt.parent.mkdir(exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"weights": {k: p.detach().cpu() for k, p in model.trainable().items()},
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "losses": losses, "spent": spent + time.time() - t0, "recipe": recipe,
                    "epoch_logit": {k: torch.from_numpy(v) for k, v in epoch_logit.items()}}, tmp)
        tmp.replace(ckpt)

    t0 = time.time()
    if 0 not in epoch_logit:           # check the wiring before spending hours
        epoch_logit[0] = predict()
        gap = np.abs(epoch_logit[0] - frozen_logit)
        print(f"before fine-tuning: live model against the head on the cached vectors, "
              f"{len(gap):,} held-out votes: logit difference mean {gap.mean():.4f}, largest "
              f"{gap.max():.4f}", flush=True)
        assert gap.mean() < MAX_GAP, ("the live encoder does not reproduce the cached vectors: "
                                      "check tokens, pooling and standardization")

    model.train()
    run_loss, run_n = 0.0, 0
    opt.zero_grad()
    while step < total:
        i = ft[order[step]]
        j = torch.from_numpy(tr_j[i])[None].clone()
        j[torch.from_numpy(rng.random(j.shape) < 0.1)] = 0                 # identity dropout
        y, m = torch.from_numpy(tr_y[i])[None].to(device), torch.from_numpy(tr_m[i])[None].to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "mps"):
            logit = model(case_ids(tr_cases[i]), j.to(device))
        loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_weight,
                                                  reduction="none")
        loss = (loss * m).sum() / m.sum()
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        (loss / args.accum).backward()
        losses.append(loss.item())
        run_loss, run_n, step = run_loss + loss.item(), run_n + 1, step + 1
        if step % args.accum == 0 or step == total:
            torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if step % 100 == 0 or step == total:
            el = spent + time.time() - t0
            print(f"  step {step:6,}/{total:,}  loss {run_loss / run_n:.4f}  "
                  f"{el / step:.2f}s per case  {el / 3600:.2f} h elapsed, about "
                  f"{el / step * (total - step) / 3600:.1f} h left", flush=True)
            run_loss, run_n = 0.0, 0
        if step in eval_at:
            k = eval_at[step]
            epoch_logit[k] = predict()
            print(f"  epoch {k} done: training loss {np.mean(losses[(k - 1) * n:k * n]):.4f}; "
                  f"held-out predictions saved (sd of the logit {epoch_logit[k].std():.3f})",
                  flush=True)
            if step < total:
                save()
        elif step % args.ckpt_every == 0 and step < total:
            save()

    secs = spent + time.time() - t0
    prob = lambda a: 1 / (1 + np.exp(-a))
    p = prob(epoch_logit[args.epochs])
    assert np.isfinite(p).all(), "non-finite predictions"
    keys = he_v[FB.KEY].reset_index(drop=True)
    out = Path(args.data) / "preds"
    keys.assign(p=p, pred=(p >= 0.5).astype(int)).to_parquet(out / f"{tag}.parquet", index=False)
    pd.concat([keys.assign(epoch=k, p=prob(v)) for k, v in sorted(epoch_logit.items())],
              ignore_index=True).to_parquet(out / f"{tag}_epochs.parquet", index=False)
    Path("results/flat_runs").mkdir(parents=True, exist_ok=True)
    Path(f"results/flat_runs/{tag}.json").write_text(json.dumps(
        {"name": args.name, "fold": args.fold, "args": vars(args), "model": FE.MODEL,
         "revision": FE.REVISION, "frozen_start": {"method": FROZEN, **cfg},
         "train_cases": int(n), "eval_cases": int(len(he_rows)), "secs": secs,
         "reference_loss": ref,
         "epoch_loss": [float(np.mean(losses[k * n:(k + 1) * n])) for k in range(args.epochs)],
         "curve": [{"step": s, "loss": float(np.mean(losses[max(0, s - 100):s]))}
                   for s in list(range(100, total, 100)) + [total]]}, indent=1))
    torch.save({k: v.detach().cpu() for k, v in model.trainable().items()},
               Path("checkpoints") / f"{tag}.pt")
    ckpt.unlink(missing_ok=True)
    print(f"{tag}: wrote data/preds/{tag}.parquet ({len(keys):,} votes) after {secs / 3600:.2f} h")


if __name__ == "__main__":
    main()
