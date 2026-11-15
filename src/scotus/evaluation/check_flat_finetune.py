"""
Checks for a finished run of flat_finetune.py: did it train, and what did training change?

    python -I -m scotus.evaluation.check_flat_finetune                 -> results/flat_runs/flat_mb_ft_val_check.md
    python -I -m scotus.evaluation.check_flat_finetune --name NAME --fold dev
    python -I -m scotus.evaluation.check_flat_finetune --name flat_mb_ft_lr1e5     (also compares it with the
                                                                  run --against, flat_mb_ft)
(-I: isolated mode, it loads the downloaded ModernBERT weights to compare them. About two
minutes, CPU only.)

A fine-tuning run that ends without an error has not necessarily trained: an early run
ended cleanly and had learned nothing. This script answers, in this order:

  1. Did it train? Training loss by epoch against the reference losses and against a
     CONTROL: stage 2 repeated with the encoder frozen (cached vectors; the same stage-1
     head, schedule, case order and identity dropout). What the control cannot do is the
     encoder's share.
  2. How far did the weights move? ||w - w0|| / ||w0|| for the unfrozen layers (w0 = the
     pretrained weights) and for the head (w0 = the stage-1 head).
  3. What do the held-out predictions vary with, and what does the encoder's change do to
     them? The run's final head is applied to the cached FROZEN vectors; its difference
     from the run's own predictions is the encoder's change alone.
  4. Scores, only if step 1 passes: the run after each epoch (a diagnostic, not a way to
     pick an epoch), the control, the final head on frozen vectors, the frozen rows, model
     v1, with paired differences (bootstrap over cases).

Step 1 fails, and nothing is scored, when the last epoch's training loss is nearer to the
loss of a constant p = 0.5 than to the frozen baseline's.

Reads training and validation terms only.
"""

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel

from scotus.evaluation import eval as ev
from scotus.baselines import flat_baseline as FB
from scotus.features import flat_encode as FE
from scotus.baselines import flat_finetune as FF
from scotus.report import report_tables as RT
from scotus.evaluation import summarize_v1 as S

KEY = FB.KEY
logit_of = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
prob = lambda z: 1 / (1 + np.exp(-z))


def setup(fold, seed):
    """The fold's cached vectors and votes, packed as flat_finetune.py packs them, and the
    stage-1 head (deterministic on the CPU)."""
    votes = pd.concat([ev.load_labels(s, "data")[0] for s in ["train", "val"]],
                      ignore_index=True)                           # test labels never loaded
    cfg = FB.chosen(FF.FROZEN)
    row_of, X = FB.load_vectors(pool=cfg["pool"])
    tr, he = FB.fold_votes(fold, votes)
    tr = tr[tr.case_id.isin(row_of.index)]
    vocab = {j: i + 1 for i, j in enumerate(sorted(tr.justice.unique()))}
    _, tr_rows, tr_j, tr_y, tr_m = FB.pack(tr, row_of, vocab)
    he_v, he_rows, he_j, _, he_m = FB.pack(he, row_of, vocab)
    mean, std = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
    Z = lambda rows: torch.from_numpy(((X[rows] - mean) / std).astype(np.float32))
    d = SimpleNamespace(cfg=cfg, n_justices=len(vocab), ztr=Z(tr_rows), zhe=Z(he_rows),
                        tj=torch.from_numpy(tr_j), ty=torch.from_numpy(tr_y),
                        tm=torch.from_numpy(tr_m), he_j=torch.from_numpy(he_j), he_m=he_m,
                        keys=he_v[KEY].reset_index(drop=True), dim=X.shape[1])
    d.w = torch.tensor(float((d.tm.sum() - d.ty.sum()) / d.ty.sum()))
    d.head0 = FB.fit_head(d.ztr, d.tj, d.ty, d.tm, d.n_justices, cfg, seed)
    return d


def held_out(head, d):
    head.eval()
    with torch.no_grad():
        return head(d.zhe, d.he_j).numpy()[d.he_m]


def control(d, a):
    """Stage 2 with the encoder frozen: training loss by epoch and held-out logits after
    each epoch (0 = the stage-1 head)."""
    head = copy.deepcopy(d.head0)
    logits = {0: held_out(head, d)}
    rng = np.random.default_rng(a["seed"])                 # the same stream as flat_finetune.py
    n, epochs, accum = len(d.ztr), a["epochs"], a["accum"]
    opt = torch.optim.AdamW(head.parameters(), lr=a["head_lr"], weight_decay=0.1)
    total = epochs * n
    updates = math.ceil(total / accum)
    warm = max(1, round(0.05 * updates))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: (u + 1) / warm if u < warm else
        0.5 * (1 + math.cos(math.pi * (u - warm) / max(1, updates - warm))))
    order = np.concatenate([rng.permutation(n) for _ in range(epochs)])
    eval_at = {min(total, math.ceil(k * n / accum) * accum): k for k in range(1, epochs + 1)}
    losses = []
    head.train()
    opt.zero_grad()
    for step, i in enumerate(order, 1):
        j = d.tj[i][None].clone()
        j[torch.from_numpy(rng.random(j.shape) < 0.1)] = 0                 # identity dropout
        loss = F.binary_cross_entropy_with_logits(head(d.ztr[i][None], j), d.ty[i][None],
                                                  pos_weight=d.w, reduction="none")
        loss = (loss * d.tm[i][None]).sum() / d.tm[i].sum()
        (loss / accum).backward()
        losses.append(loss.item())
        if step % accum == 0 or step == total:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if step in eval_at:
            logits[eval_at[step]] = held_out(head, d)
            head.train()
    losses = np.array(losses)
    return [float(losses[k * n:(k + 1) * n].mean()) for k in range(epochs)], logits


def weight_change(weights, d):
    """Relative change of the trained tensors: encoder against the pretrained model, head
    against the stage-1 head."""
    enc0 = AutoModel.from_pretrained(FE.MODEL, revision=FE.REVISION,
                                     dtype=torch.float32).state_dict()
    rows = []
    for name, v in weights.items():
        if not name.startswith("enc."):
            continue
        v, v0 = v.float(), enc0[name[4:]].float()
        part = name.split(".")
        rows.append({"part": f"layer {part[2]}" if part[1] == "layers" else part[1],
                     "values": v.numel(), "d2": float(((v - v0) ** 2).sum()),
                     "w2": float((v0 ** 2).sum()), "abs": float((v - v0).abs().sum()),
                     "abs0": float(v0.abs().sum())})
    df = pd.DataFrame(rows)
    g = df.groupby("part", sort=False)[["values", "d2", "w2"]].sum()
    g["relative change"] = np.sqrt(g.d2 / g.w2)
    enc = {"relative": float(np.sqrt(df.d2.sum() / df.w2.sum())),
           "mean_abs_change": float(df["abs"].sum() / df["values"].sum()),
           "mean_abs_weight": float(df.abs0.sum() / df["values"].sum())}
    h0 = d.head0.state_dict()
    hd2 = sum(float(((v.float() - h0[k[5:]]) ** 2).sum()) for k, v in weights.items()
              if k.startswith("head."))
    hw2 = sum(float((h0[k[5:]] ** 2).sum()) for k in weights if k.startswith("head."))
    return g[["values", "relative change"]].reset_index(), enc, math.sqrt(hd2 / hw2)


def structure(p, keys, base=None):
    """What a set of held-out predictions varies with (no labels)."""
    z = pd.Series(logit_of(np.asarray(p)))
    out = {"logit sd": z.std(),
           "variance between justices": z.groupby(keys.justice.to_numpy()).transform("mean").var() / z.var(),
           "variance between cases": z.groupby(keys.case_id.to_numpy()).transform("mean").var() / z.var(),
           "predicted petitioner %": 100 * float((np.asarray(p) >= 0.5).mean())}
    if base is not None:
        zb = logit_of(np.asarray(base))
        out.update({"correlation with epoch 0": np.corrcoef(z, zb)[0, 1],
                    "sd of the change": float((z - zb).std())})
    return out


def parts(preds, v, c):
    """Case-level AUC and within-case AUC (as make_figures.parts)."""
    d = v.merge(preds[["case_id", "justice", "p"]], on=KEY)
    mean_p = d.groupby("case_id").p.mean()
    case_auc = roc_auc_score(c.set_index("case_id").win_side.loc[mean_p.index], mean_p)
    num = den = 0
    for _, g in d.groupby("case_id"):
        a, b = g.p[g.vote == 1].to_numpy(), g.p[g.vote == 0].to_numpy()
        num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        den += len(a) * len(b)
    return case_auc, num / den


def paired(a, b, v, c, n=1000, seed=0):
    """b - a for every metric, with a 95% CI from the same resamples of cases for both."""
    da, db = ev.join(a, v), ev.join(b, v)
    ca, cb = ev.case_correct(da, c), ev.case_correct(db, c)
    pa, pb = ev._metrics(da, ca), ev._metrics(db, cb)
    rng = np.random.default_rng(seed)
    row_case = pd.Series(np.arange(len(c)), index=c.case_id).loc[da.case_id].to_numpy()
    draws = {m: [] for m in ev.METRICS}
    for _ in range(n):
        counts = rng.multinomial(len(c), np.full(len(c), 1 / len(c)))
        ma = ev._metrics(da, ca, counts[row_case], counts)
        mb = ev._metrics(db, cb, counts[row_case], counts)
        for m in ev.METRICS:
            draws[m].append(mb[m] - ma[m])
    return {m: (pb[m] - pa[m], *np.nanpercentile(draws[m], [2.5, 97.5])) for m in ev.METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="flat_mb_ft")
    ap.add_argument("--fold", default="val", choices=list(FB.FOLDS))
    ap.add_argument("--against", default="flat_mb_ft",
                    help="another finished fine-tuning run to compare with, on the same fold "
                         "(skipped when it is this run or has no predictions)")
    args = ap.parse_args()
    tag = f"{args.name}_{args.fold}"
    torch.set_num_threads(8)
    r = json.loads(Path(f"results/flat_runs/{tag}.json").read_text())
    a, ref, n, E = r["args"], r["reference_loss"], r["train_cases"], r["args"]["epochs"]
    d = setup(args.fold, a["seed"])
    assert len(d.ztr) == n, "the run was not trained on this fold's training cases"
    e = pd.read_parquet(f"data/preds/{tag}_epochs.parquet")
    run_p = {k: g.p.to_numpy() for k, g in e.groupby("epoch")}
    assert (e[e.epoch == 0][KEY].to_numpy() == d.keys.to_numpy()).all()
    final = pd.read_parquet(f"data/preds/{tag}.parquet")
    assert np.allclose(final.p, run_p[E]) and np.isfinite(e.p).all()

    md = [f"# Checks of the fine-tuned flat baseline: `{tag}`", "",
          f"Written by check_flat_finetune.py. Run: top {a['unfreeze']} layers of ModernBERT-base "
          f"and the head, {E} epochs, encoder rate {a['lr']:g}, head rate {a['head_lr']:g}, "
          f"{a['accum']} cases per update, seed {a['seed']}; {n:,} training cases, "
          f"{r['eval_cases']} held out; {r['secs'] / 3600:.1f} hours. It starts from one seed of "
          f"`{FF.FROZEN}` (the head trained on the frozen vectors).", ""]

    # 1. did it train?
    ctrl_loss, ctrl_logit = control(d, a)
    ctrl_p = {k: prob(z) for k, z in ctrl_logit.items()}
    rows = [{"epoch": k + 1, "run (top layers + head)": r["epoch_loss"][k],
             "control (encoder frozen)": ctrl_loss[k],
             "run - control": r["epoch_loss"][k] - ctrl_loss[k]} for k in range(E)]
    trained = r["epoch_loss"][-1] < 0.5 * (ref["constant"] + ref["head_frozen_dropout"])
    md += ["## 1. Did it train?", "",
           f"Reference losses on the training cases: {ref['constant']:.4f} for a constant "
           f"p = 0.5 (knows nothing); the stage-1 head on the frozen vectors "
           f"{ref['head_frozen_dropout']:.4f} with dropout as in training "
           f"({ref['head_frozen']:.4f} with dropout off). Control: stage 2 repeated with the "
           "encoder frozen (same head, schedule, case order and identity dropout).", "",
           "Training loss by epoch (mean over cases, dropout on):", "",
           pd.DataFrame(rows).round(4).to_markdown(index=False), "",
           ("**Passed**: the last epoch's training loss is nearer to the frozen baseline's "
            "than to the know-nothing level." if trained else
            "**FAILED: the training loss is at the know-nothing level. Nothing is scored; "
            "do not report this run.**"), ""]

    # 2. how far did the weights move?
    weights = torch.load(f"checkpoints/{tag}.pt", map_location="cpu", weights_only=True)
    layers, enc, head_rel = weight_change(weights, d)
    md += ["## 2. How far did the weights move?", "",
           "Relative change ||w - w0|| / ||w0|| of the unfrozen encoder weights against the "
           "pretrained model:", "", layers.round(5).to_markdown(index=False), "",
           f"All unfrozen encoder weights: {100 * enc['relative']:.3f}% (mean absolute change "
           f"{enc['mean_abs_change']:.1e} against a mean absolute weight of "
           f"{enc['mean_abs_weight']:.1e}). The head, against the stage-1 head: "
           f"{100 * head_rel:.1f}%.", ""]

    # 3. the held-out predictions, without labels
    head = FB.Head(d.n_justices, d.cfg["hidden"], d.cfg["dropout"], d.dim)
    head.load_state_dict({k[5:]: v.float() for k, v in weights.items() if k.startswith("head.")})
    hf_p = prob(held_out(head, d))                         # the final head on frozen vectors
    rows = []
    for name, P in [("run", run_p), ("control", ctrl_p)]:
        for k in sorted(P):
            rows.append({"model": name, "epoch": k, **structure(P[k], d.keys, P[0] if k else None)})
    gap = np.abs(logit_of(run_p[0]) - logit_of(ctrl_p[0]))
    dz = logit_of(run_p[E]) - logit_of(hf_p)
    flips = int(((run_p[E] >= 0.5) != (hf_p >= 0.5)).sum())
    md += ["## 3. The held-out predictions (no labels used)", "",
           f"Before fine-tuning the live model reproduces the head on the cached vectors: mean "
           f"logit difference {gap.mean():.4f}, largest {gap.max():.4f}, over {len(gap):,} votes.",
           "", pd.DataFrame(rows).round(3).to_markdown(index=False), "",
           "The encoder's change alone (the run's final head on the fine-tuned encoder against "
           f"the same head on the frozen vectors): logit correlation "
           f"{np.corrcoef(logit_of(run_p[E]), logit_of(hf_p))[0, 1]:.3f}, sd of the difference "
           f"{dz.std():.3f} (sd of the logit {logit_of(run_p[E]).std():.3f}), {flips} of "
           f"{len(hf_p):,} votes ({100 * flips / len(hf_p):.1f}%) change side at cutoff 0.5.", ""]

    if trained:
        v, c = S.labels(args.fold)
        frame = lambda p: d.keys.assign(p=p, pred=(p >= 0.5).astype(int))
        ft, start, ctrl, hf = frame(run_p[E]), frame(run_p[0]), frame(ctrl_p[E]), frame(hf_p)
        P = {f"{args.name} (the row: after {E} epochs)": ft,
             "its start: the stage-1 head, one seed of flat_mb (epoch 0)": start}
        P.update({f"the run after epoch {k} (diagnostic)": frame(run_p[k]) for k in range(1, E)})
        P.update({f"control: encoder frozen, same schedule, after {E} epochs": ctrl,
                  "the run's final head on the frozen vectors": hf})
        other, other_p = None, Path(f"data/preds/{args.against}_{args.fold}.parquet")
        if args.against != args.name and other_p.exists():       # the same recipe, another rate
            ro = json.loads(Path(f"results/flat_runs/{args.against}_{args.fold}.json").read_text())
            other = f"{args.against} (encoder rate {ro['args']['lr']:g})"
            P[other] = v[KEY].merge(pd.read_parquet(other_p), on=KEY)
        for label, stem in [("flat_mb (5 seeds)", "flat_mb"), ("flat_mb_all (5 seeds)", "flat_mb_all"),
                            ("question_count", "question_count"), ("qc_bloc", "qc_bloc"),
                            ("v1_main (5 seeds)", "v1_main")]:
            p = RT.preds(stem, args.fold)
            if p is not None:
                P[label] = v[KEY].merge(p, on=KEY)
        rows = []
        for label, q in P.items():
            s = ev.score(q, v, c)
            ca, wi = parts(q, v, c)
            rows.append({"method": label, "macro-F1": ev.fmt_ci(*s["macro_f1"]),
                         "AUC": ev.fmt_ci(*s["auc"]), "vote acc": f"{s['vote_acc'][0]:.3f}",
                         "case acc": f"{s['case_acc'][0]:.3f}",
                         "predicted petitioner %": f"{100 * ev.join(q, v).pred.mean():.0f}",
                         "case-level AUC": f"{ca:.3f}", "within-case AUC": f"{wi:.3f}"})
        pairs = [("fine-tuned - its start (all of stage 2)", start, ft),
                 ("fine-tuned - control (unfreezing the encoder, same schedule)", ctrl, ft),
                 ("fine-tuned - its final head on frozen vectors (the encoder's change alone)", hf, ft),
                 ("control - its start (three more epochs of the head alone)", frame(ctrl_p[0]), ctrl)]
        pairs += [(f"fine-tuned - {k}", P[k], ft)
                  for k in (other, "flat_mb (5 seeds)", "question_count") if k in P]
        if "v1_main (5 seeds)" in P:
            pairs.append(("v1_main - fine-tuned", ft, P["v1_main (5 seeds)"]))
        fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
        diffs = []
        for label, x, y in pairs:
            q = paired(x, y, v, c)
            diffs.append({"difference": label, "AUC": fmt(q["auc"]), "macro-F1": fmt(q["macro_f1"]),
                          "vote acc": fmt(q["vote_acc"]), "case acc": fmt(q["case_acc"])})
        terms = pd.concat([ev.score_by_term(P[k], v, c).assign(method=k)
                           for k in list(P)[:2] + [k for k in P if k.startswith("control")]])
        lo, hi = FB.FOLDS[args.fold][1]
        md += [f"## 4. Scores ({args.fold} fold, terms {lo}-{hi}: {len(c)} cases, {len(v):,} votes)", "",
               "Cutoff 0.5; 95% bootstrap CI over cases. The rows after epochs before the last "
               "are a diagnostic, never a way to choose an epoch.", "",
               pd.DataFrame(rows).to_markdown(index=False), "",
               "Paired differences (the same 1,000 resamples of cases for both models):", "",
               pd.DataFrame(diffs).to_markdown(index=False), "", "Per term:", "",
               terms[["method", "term", "cases", "votes", *ev.METRICS]].round(3).to_markdown(index=False), ""]
    out = Path(f"results/flat_runs/{tag}_check.md")
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
