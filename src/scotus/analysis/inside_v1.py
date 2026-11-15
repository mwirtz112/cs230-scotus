"""Inside the trained network: which of its parts carry the prediction. Fits nothing.

    python -m scotus.analysis.inside_v1        -> results/v1_inside.md
                                  results/figures/fig_inside_{paths,readout,blocs}.pdf (+ .csv)
    python -m scotus.analysis.inside_v1 --figures-only     redraws the figures from those .csv files

Reads the saved fits of the main recipe (checkpoints/v1_main_{val,dev}_s0..4.pt) and runs
them on their own held-out terms: val = terms 2016-17 (the primary analysis), dev = terms
2012-15 (the repeat). model_v1.py and train_v1.py are not changed: intermediate values are
read or replaced with forward hooks.

  M1  which path carries the prediction (knock-outs at inference, mean ablation)
  M2  what the readout per justice looks at
  M3  what the learned counters count
  M4  what the bloc layer learned
  M5  the untrained identity rows of a justice who argued a training case as an advocate

The eight fits of the test run were not saved, so the test terms play no part in M1-M4.
They are read in one place only: the last table of M5, which breaks the SAVED predictions
of the test run down by justice. Written after the test run; it changes no
model and no prediction.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from scotus.evaluation import check_walk_final as CW     # labels() and table() for the saved test predictions
from scotus.evaluation import eval as ev
from scotus.report import make_figures as MF         # parts(), short() and the plot style; nothing is drawn by it
from scotus.evaluation import walk_forward as W          # paired(); nothing is run
from scotus.models.model_v1 import TurnModelV1, build_cases, collate

plt = MF.plt
FOLDS = {"val": (2015, (2016, 2017)), "dev": (2011, (2012, 2015))}   # last train term, held-out terms
SEEDS = range(5)
KEY = ["case_id", "justice"]
OUT, FIG = "results/v1_inside.md", Path("results/figures")
BENCH = ["j__john_g_roberts_jr", "j__samuel_a_alito_jr", "j__antonin_scalia", "j__clarence_thomas",
         "j__anthony_m_kennedy", "j__stephen_g_breyer", "j__ruth_bader_ginsburg",
         "j__sonia_sotomayor", "j__elena_kagan"]                       # sat together 2010-15
KAV = "j__brett_m_kavanaugh"
GROUPS = {"own": "The justice's own turns", "colleagues": "The other justices",
          "petitioner": "Petitioner's advocate", "respondent": "Respondent's advocate",
          "other": "Anyone else"}
KNOCK = {  # name in the tables: what is replaced
    "content readout": "the attention readout's content vector → its average",
    "learned counters": "the learned counters → their average",
    "count features": "the hand-built count features (MLP and direct path) → their average",
    "turn text": "every turn's Legal-BERT vector → the training mean",
    "identity": "every justice identity hidden (the bloc term goes with it)",
    "bloc term": "the bloc term removed from the logit",
    "uniform attention": "the readout's attention made uniform over the turns",
    "no own-turn bonus": "the bonus on the justice's own turns set to 0",
    "no identity in the query": "the identity removed from the readout's query only",
}
EDITS = {"uniform attention": ("q", "query_id", "own_bias"), "no own-turn bonus": ("own_bias",),
         "no identity in the query": ("query_id",)}


# ---------------------------------------------------------------- loading

def load_data():
    lab = [ev.load_labels(s) for s in ("train", "val")]                # test labels are not loaded
    return dict(turns=pd.read_parquet("data/turns.parquet", columns=["case_id", "speaker_id", "role"]),
                votes=pd.concat([x[0] for x in lab], ignore_index=True),
                cases=pd.concat([x[1] for x in lab], ignore_index=True),
                feats=pd.read_parquet("data/turn_feats.parquet"),
                vf=pd.read_parquet("data/vote_feats.parquet"))


def blank_model(a, E, n):
    """The network as train_v1.py builds it for the main recipe (weights at their start)."""
    off = ("no_struct", "no_wide", "no_content", "no_identity", "no_text", "no_amicus", "stance",
           "bottleneck", "bench_layers", "case_aux", "stance_aux")
    assert not any(a[k] for k in off) and a["aff_mode"] == "pair", "not the main recipe"
    return TurnModelV1(E, n, d=a["d"], layers=a["layers"], heads=a["heads"], ff=a["ff"],
                       dropout=a["dropout"], in_drop=a["in_drop"], counters=a["counters"],
                       hidden=a["hidden"], affinity=a["affinity"], aff_rank=a["aff_rank"],
                       aff_mode=a["aff_mode"], aff_unknown=a["aff_unknown"], mf_iters=a["mf_iters"])


def make_model(ck, E):
    m = blank_model(ck["args"], E, len(ck["vocab"]))
    m.load_state_dict(ck["state"])
    return m.eval()


def setup(fold, D):
    """The five saved fits of a fold and its held-out cases, with the turn vectors they need."""
    cks = [torch.load(f"checkpoints/v1_main_{fold}_s{s}.pt", weights_only=False, map_location="cpu")
           for s in SEEDS]
    vocab, ws = cks[0]["vocab"], cks[0]["wide_stats"]
    for ck in cks:
        assert ck["vocab"] == vocab and all(np.array_equal(x, y) for x, y in zip(ck["wide_stats"], ws))
    lo, hi = FOLDS[fold][1]
    v = D["votes"][D["votes"].year.between(lo, hi)].reset_index(drop=True)
    c = D["cases"][D["cases"].case_id.isin(set(v.case_id))].reset_index(drop=True)
    cs = build_cases(set(v.case_id), D["turns"], D["feats"], v, D["vf"], vocab, ws)
    assert len(cs) == v.case_id.nunique() == len(c)
    rows = np.unique(np.concatenate([x.rows for x in cs]))
    raw = np.load("embeddings/turns_last.npy", mmap_mode="r")
    st = np.load("embeddings/turns_last_train_stats.npz")
    E = torch.zeros((len(rows) + 1, raw.shape[1]), dtype=torch.float16)        # last row: padding
    E[:len(rows)] = torch.from_numpy(((raw[rows].astype(np.float32) - st["mean"]) / st["std"])
                                     .astype(np.float16))                      # fp16, as in training
    for x in cs:
        x.rows = np.searchsorted(rows, x.rows)
    return dict(fold=fold, v=v, c=c, cases=cs, vocab=vocab, models=[make_model(ck, E) for ck in cks],
                args=cks[0]["args"], E=E)


def slices(m):
    """Where each source of evidence sits in the input of the head's MLP."""
    a = m.d
    b = a + 6 * m.k
    return {"content": slice(0, a), "counters": slice(a, b),
            "wide": slice(b, b + m.wide_lin.in_features)}


# ---------------------------------------------------------------- one pass over the cases

@torch.no_grad()
def run(model, cases, hide=False, wide_mean=None, patch=None, keep=(), bs=16):
    """One forward pass over `cases`. Returns one row per vote (logit, and `base` = the logit
    before the bloc term) and, per case, the arrays named in `keep`: 'attn' [K, T],
    'm' [T, counters], 'msg' [K, channels], 'pre' [K, MLP inputs], 'out' [K, 2].
    hide: every identity hidden. wide_mean: every vote gets these count features.
    patch: {source: vector} written into the MLP's input (see slices())."""
    store, sl = {}, slices(model)

    def pre_hook(_, args):
        x = args[0]
        store["pre"] = x
        if patch:
            x = x.clone()
            for name, val in patch.items():
                x[..., sl[name]] = val
            return (x,)

    hooks = [model.pre.register_forward_pre_hook(pre_hook),
             model.aff_msg.register_forward_hook(lambda _, i, o: store.__setitem__("msg", o))]
    rows, extra = [], {k: {} for k in keep}
    order = np.argsort([len(c.rows) for c in cases], kind="stable")
    try:
        for a in range(0, len(order), bs):
            batch = [cases[j] for j in order[a:a + bs]]
            b = collate(batch, "cpu")
            if hide:
                b["jid"], b["vjid"] = torch.zeros_like(b["jid"]), torch.zeros_like(b["vjid"])
            if wide_mean is not None:
                b["wide"] = wide_mean.expand_as(b["wide"]).clone()
            out = model(b)
            got = {"attn": out["attn"], "m": out["m"], "msg": store["msg"], "pre": store["pre"],
                   "out": torch.stack([out["logit"], out["base"]], -1)}
            for i, c in enumerate(batch):
                k, t = len(c.vjid), len(c.rows)
                z, zb = out["logit"][i, :k].tolist(), out["base"][i, :k].tolist()
                rows += [(c.case_id, n, z[j], zb[j]) for j, n in enumerate(c.names)]
                for name in keep:
                    x = got[name][i]
                    x = x[:k, :t] if name == "attn" else x[:t] if name == "m" else x[:k]
                    extra[name][c.case_id] = x.numpy().copy()
    finally:
        for h in hooks:
            h.remove()
    return pd.DataFrame(rows, columns=KEY + ["logit", "base"]), extra


def ens(tabs, col="logit"):
    """Mean probability over the seeds, cutoff 0.5 (as in the reported model)."""
    p = pd.concat([t.set_index(KEY)[col].map(expit) for t in tabs], axis=1).mean(axis=1)
    p = p.rename("p").reset_index()
    return p.assign(pred=(p.p >= 0.5).astype(int))


def scores(p, v, c):
    d = v.merge(p, on=KEY)
    assert len(d) == len(v)
    case, within = MF.parts(p, v, c)
    return {"AUC": roc_auc_score(d.vote, d.p), "Macro-F1": f1_score(d.vote, d.pred, average="macro"),
            "Case-level AUC": case, "Within-case AUC": within, "Pred. petitioner %": 100 * d.pred.mean()}


# ---------------------------------------------------------------- M2: the readout

def attn_table(cases, attn, ref=None):
    """Per vote: attention mass and share of the turns by who speaks; the effective number
    of turns attended (exp of the entropy); with `ref`, the total-variation distance to it."""
    rows = []
    for c in cases:
        a, justice = attn[c.case_id], c.role == 1
        for i, name in enumerate(c.names):
            own = c.own[i]
            g = {"own": own, "colleagues": justice & ~own, "petitioner": (c.role == 2) & ~own,
                 "respondent": (c.role == 3) & ~own}
            g["other"] = ~np.any(list(g.values()), axis=0)
            r = {"case_id": c.case_id, "justice": name, "speaks": bool(own.any()),
                 "turns": len(own), "eff": float(np.exp(-(a[i] * np.log(a[i] + 1e-12)).sum()))}
            for k, mask in g.items():
                r[f"mass_{k}"], r[f"share_{k}"] = float(a[i, mask].sum()), float(mask.mean())
            if ref is not None:
                r["tv"] = float(0.5 * np.abs(a[i] - ref[c.case_id][i]).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def readout_summary(A):
    """A: {variant: [per-seed attn_table]} -> the tables of M2 (seeds averaged per vote)."""
    mean = {k: pd.concat(t).groupby(KEY + ["speaks"], as_index=False).mean(numeric_only=True)
            for k, t in A.items()}
    out = {}
    for speaks in (True, False):
        d = {k: t[t.speaks == speaks] for k, t in mean.items()}
        tab = pd.DataFrame({"Who speaks": list(GROUPS.values()),
                            "Share of the turns %": [100 * d["intact"][f"share_{g}"].mean() for g in GROUPS]})
        for k, label in [("intact", "Attention %"), ("no own-turn bonus", "Attention %, no own-turn bonus"),
                         ("no identity in the query", "Attention %, no identity in the query")]:
            tab[label] = [100 * d[k][f"mass_{g}"].mean() for g in GROUPS]
        out[speaks] = (len(d["intact"]), tab)
    sp = mean["intact"][mean["intact"].speaks]
    facts = {"turns": mean["intact"].turns.mean(), "eff": mean["intact"].eff.mean(),
             "lift_own": sp.mass_own.mean() / sp.share_own.mean(),
             "tv_identity": mean["no identity in the query"].tv.mean(),
             "tv_bonus": mean["no own-turn bonus"][mean["no own-turn bonus"].speaks].tv.mean()}
    return out, facts


def moved(S, D):
    """How far each learned part moved from its start. train_v1.py seeds PyTorch with the
    seed number and then builds the network, so the starting weights can be rebuilt here.
    The rebuild is checked on the rows that never get a gradient (identities of speakers
    without a training vote): they must still point exactly where they started."""
    vocab, end = S["vocab"], FOLDS[S["fold"]][0]
    tr = D["votes"][(D["votes"].year >= 1980) & (D["votes"].year <= end)]
    trained = sorted(vocab[j] for j in set(tr.justice))
    idle = sorted(set(vocab.values()) - set(trained))
    cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
    rel = lambda a, b: float((a - b).norm() / b.norm())
    rows = []
    for s, m in enumerate(S["models"]):
        torch.manual_seed(s)
        m0 = blank_model(S["args"], S["E"], len(vocab))
        get = lambda mod, name: dict(mod.named_parameters())[name].detach()
        enc = lambda mod: torch.cat([p.detach().flatten() for p in mod.encoder.parameters()])
        parts = {"Shared query q": (m.q.detach(), m0.q.detach()),
                 "Identity in the query, justices with training votes":
                     (m.query_id.weight.detach()[trained], m0.query_id.weight.detach()[trained]),
                 "Identity in the query, speakers without a training vote (check)":
                     (m.query_id.weight.detach()[idle], m0.query_id.weight.detach()[idle]),
                 "Identity vector in the head, justices with training votes":
                     (m.head_id.weight.detach()[trained], m0.head_id.weight.detach()[trained]),
                 "Text projection W": (get(m, "proj.weight"), get(m0, "proj.weight")),
                 "The two transformer layers": (enc(m), enc(m0)),
                 "Counter weights": (get(m, "count.weight"), get(m0, "count.weight")),
                 "Head MLP, first layer": (get(m, "pre.1.weight"), get(m0, "pre.1.weight"))}
        for k, (a, b) in parts.items():
            rows.append({"Part": k, "Size at the start": float(b.norm()), "Size after training": float(a.norm()),
                         "Relative change": rel(a, b), "Cosine with the start": cos(a, b)})
        for k, a in [("Speaker embeddings (start: 0)", m.justice.weight.detach()[trained]),
                     ("Pair table of the bloc layer (start: 0)", m.aff_pair.weight.detach()),
                     ("Gain and lean per justice (start: 0)",
                      torch.cat([m.gain.weight.detach(), m.lean.weight.detach()]))]:
            rows.append({"Part": k, "Size at the start": 0.0, "Size after training": float(a.norm())})
        rows.append({"Part": "Own-turn bonus (one number)", "Size at the start": 1.0,
                     "Size after training": float(m.own_bias.detach())})
    t = pd.DataFrame(rows).groupby("Part", sort=False).mean().reset_index()
    check = t.loc[t.Part.str.contains("check"), "Cosine with the start"].iloc[0]
    assert check > 0.9999, f"the starting weights are not rebuilt correctly (cosine {check:.4f})"
    return t


# ---------------------------------------------------------------- M3: the counters

def turn_frame(cases):
    return pd.concat([pd.DataFrame({
        "case_id": c.case_id, "role": c.role, "target": c.target,
        "words": np.rint(np.expm1(4 * c.scal[:, 0])).astype(int),      # undo model_v1's scaling
        "q": c.scal[:, 1] > 0, "cut": c.scal[:, 2] > 0}) for c in cases], ignore_index=True)


def count_feature(cases, weight=None):
    """log(1 + weight of the justice's turns aimed at the respondent) - the same for the
    petitioner; votes of justices who question at least one side. weight None = 1 a turn."""
    rows = []
    for c in cases:
        w = np.ones(len(c.rows)) if weight is None else weight[c.case_id]
        for i, name in enumerate(c.names):
            own = c.own[i] & (c.role == 1)
            if (own & (c.target >= 0)).any():
                f = np.log1p(w[own & (c.target == 1)].sum()) - np.log1p(w[own & (c.target == 0)].sum())
                rows.append((c.case_id, name, f, int(f > 0)))
    return pd.DataFrame(rows, columns=KEY + ["p", "pred"])


def counters(cases, v, M):
    """M: per seed {case_id: [T, counters]}. Weights are scaled so the mean justice turn = 1."""
    t = turn_frame(cases)
    is_j = (t.role == 1).to_numpy()
    stack = [np.concatenate([m[c.case_id] for c in cases]) for m in M]            # [turns, C] a seed
    total = np.stack([s.sum(1) / s.sum(1)[is_j].mean() for s in stack])           # [seeds, turns]
    t["w"] = total.mean(0)
    j = t[is_j].assign(length=pd.cut(t.words[is_j], [0, 5, 20, 50, 10 ** 6],
                                     labels=["1–5 words", "6–20 words", "21–50 words", "51+ words"]))
    per_seed = pd.DataFrame(total[:, is_j].T, index=j.index)
    rows = []
    for title, col, order in [("Length", "length", None), ("Question mark", "q", [False, True]),
                              ("Cut off", "cut", [False, True]), ("Aimed at", "target", [0, 1, -1])]:
        for key in (order if order is not None else j[col].cat.categories):
            idx = j.index[j[col] == key]
            s = per_seed.loc[idx].mean(axis=0)
            name = {"q": {False: "none", True: "at least one"}, "cut": {False: "no", True: "yes"},
                    "target": {0: "the petitioner", 1: "the respondent", -1: "nobody (e.g. after an amicus)"}
                    }.get(col, {}).get(key, key)
            rows.append({"Justice turns by": title, "Group": name, "Turns": len(idx),
                         "Share %": 100 * len(idx) / len(j), "Mean weight": j.w[idx].mean(),
                         "Seeds, min–max": f"{s.min():.2f}–{s.max():.2f}"})
    spread = {"cv": float(j.w.std() / j.w.mean()),
              "top20": float(j.w.sort_values(ascending=False).iloc[:len(j) // 5].sum() / j.w.sum()),
              "advocate": float(t.w[t.role.isin([2, 3])].mean()),
              "corr_words": float(spearmanr(j.w, j.words)[0])}
    # does the learned weighting sharpen the question count?
    cuts = np.cumsum([0] + [len(c.rows) for c in cases])
    as_dict = lambda w: {c.case_id: w[a:b] for c, a, b in zip(cases, cuts[:-1], cuts[1:])}
    plain, learned = count_feature(cases), count_feature(cases, as_dict(t.w.to_numpy()))
    words = count_feature(cases, as_dict((t.words / t.words[is_j].mean()).to_numpy()))
    vv = v.merge(plain[KEY], on=KEY)
    auc = lambda f: roc_auc_score(vv.merge(f, on=KEY).vote, vv.merge(f, on=KEY).p)
    single = [auc(count_feature(cases, as_dict(s[:, k] / s[is_j, k].mean())))
              for s in stack for k in range(s.shape[1])]
    by_seed = [auc(count_feature(cases, as_dict(w))) for w in total]
    sharpen = {"votes": len(vv), "plain": auc(plain), "learned": auc(learned),
               "diff": W.paired(plain, learned, vv)[0], "by_seed": (min(by_seed), max(by_seed)),
               "words": auc(words), "diff_words": W.paired(words, learned, vv)[0],
               "single": (min(single), float(np.median(single)), max(single))}
    return pd.DataFrame(rows), spread, sharpen


# ---------------------------------------------------------------- M4: the bloc layer

def blocs(S, X, D):
    """X: per seed the 'msg' and 'out' arrays of the intact pass. Effective pair weights
    W[j, k] = sum_h A[j, k, h] * beta_h, beta_h = slope of message channel h on the sender's
    own bloc-free logit; per colleague of a nine-justice bench (divided by 8)."""
    n, idx = len(S["vocab"]), [S["vocab"][j] for j in BENCH]
    mats, r2, size = [], [], []
    for m, x in zip(S["models"], X):
        A = m.aff_pair.weight.detach().view(n + 1, n + 1, -1).numpy()
        known = [c.vjid > 0 for c in S["cases"]]
        U = np.concatenate([x["msg"][c.case_id][k] for c, k in zip(S["cases"], known)])
        base = np.concatenate([x["out"][c.case_id][k, 1] for c, k in zip(S["cases"], known)])
        beta = ((U - U.mean(0)) * (base - base.mean())[:, None]).mean(0) / base.var()
        Wm = A @ beta
        mats.append(Wm[np.ix_(idx, idx)] / 8)
        const = A @ (U.mean(0) - beta * base.mean())           # the part that does not depend on the case
        term, lin = [], []
        for c, k in zip(S["cases"], known):
            o, vid = x["out"][c.case_id], c.vjid
            others = max(len(vid) - 1, 1)
            for j in np.flatnonzero(k):
                kk = np.flatnonzero(k & (np.arange(len(vid)) != j))
                term.append(o[j, 0] - o[j, 1])
                lin.append((const[vid[j], vid[kk]] + Wm[vid[j], vid[kk]] * o[kk, 1]).sum() / others)
        r2.append(np.corrcoef(term, lin)[0, 1] ** 2)
        size.append((np.std(term), np.std(base)))
    off = ~np.eye(len(BENCH), dtype=bool)
    mean = np.mean(mats, 0)
    seeds = [np.corrcoef(a[off], b[off])[0, 1] for i, a in enumerate(mats) for b in mats[i + 1:]]
    vv = D["votes"][D["votes"].justice.isin(BENCH)]
    corr = lambda lo, hi: (vv[vv.year.between(lo, hi)].pivot(index="case_id", columns="justice",
                                                             values="vote")[BENCH].corr().to_numpy())
    qc = pd.read_csv(FIG / "fig_affinity.csv", index_col=0)
    names = [MF.short(j) for j in BENCH]
    qc = qc.loc[names, names].to_numpy()
    ok = off & np.isfinite(qc)
    sym = np.where(off, (mean + mean.T) / 2, 0.0)
    axis = np.linalg.eigh(sym)[1][:, -1]
    order = np.argsort(axis)
    # the same line-up from the actual votes: the first axis of the vote correlations is
    # general agreement (all justices on one side), the second one divides the Court
    votes_axis = np.linalg.eigh(corr(2010, 2015))[1][:, -2]
    votes_axis = votes_axis * np.sign(np.corrcoef(axis, votes_axis)[0, 1])
    lineup = pd.DataFrame({"justice": [MF.short(j) for j in BENCH],
                           "table": axis / np.abs(axis).max(),
                           "votes": votes_axis / np.abs(votes_axis).max()}).iloc[order]
    same = np.sign(corr(2010, 2015)[off] - np.median(corr(2010, 2015)[off]))
    col = np.array([mean[off[:, k], k].mean() for k in range(len(BENCH))])      # a colleague's average weight
    centred = mean - col[None, :]
    facts = {"rho_votes": spearmanr(mean[off], corr(2010, 2015)[off])[0],
             "rho_votes_late": spearmanr(mean[off], corr(2012, 2015)[off])[0],
             "rho_qc": spearmanr(mean[ok], qc[ok])[0], "seed_corr": float(np.mean(seeds)),
             "rho_centred": spearmanr(centred[off], corr(2010, 2015)[off])[0],
             "axis_corr": float(np.corrcoef(axis, votes_axis)[0, 1]), "lineup": lineup,
             "col": dict(zip(names, col)),
             "r2": float(np.mean(r2)), "sd_term": float(np.mean([s[0] for s in size])),
             "sd_base": float(np.mean([s[1] for s in size])),
             "mean_close": float(mean[off][same > 0].mean()), "mean_far": float(mean[off][same < 0].mean()),
             "order": [names[i] for i in order]}
    np.fill_diagonal(mean, np.nan)                             # a justice sends itself nothing
    table = pd.DataFrame(mean, index=names, columns=names).iloc[order, order]
    votes = pd.DataFrame(corr(2010, 2015), index=names, columns=names).iloc[order, order]
    return table, votes, facts


# ---------------------------------------------------------------- M5: untrained identity rows

def advocate_facts():
    """Speakers tagged as justices in cases heard before their first vote on the Court.
    Ten turns or more in a case: the later justice argued it as an advocate. A few turns:
    more likely a sitting justice's turns filed under the wrong name in the source."""
    t = pd.read_parquet("data/turns.parquet", columns=["case_id", "speaker_id", "role"])
    c = pd.read_parquet("data/cases.parquet")[["case_id", "year", "split"]]
    v = pd.concat([pd.read_parquet("data/votes.parquet"), pd.read_parquet("data/ext/votes.parquet")])
    first = v.groupby("justice").year.min()                            # terms only, no outcome is read
    j = (t[t.role == "justice"].groupby(["case_id", "speaker_id"]).size().rename("turns")
         .reset_index().merge(c, on="case_id"))
    j = j[j.year < j.speaker_id.map(first)].assign(big=lambda d: d.turns >= 10)
    by = (j.groupby("speaker_id").agg(cases=("case_id", "nunique"), big=("big", "sum"),
                                      median=("turns", "median"), first=("year", "min"),
                                      last=("year", "max"))
          .sort_values(["big", "cases"], ascending=False))
    late = j[j.big & (j.year >= 1980)]
    return {"cases": j.case_id.nunique(), "argued": j[j.big].case_id.nunique(),
            "since1980": j[j.year >= 1980].case_id.nunique(), "argued1980": late.case_id.nunique(),
            "who1980": late.groupby("speaker_id").case_id.nunique().sort_values(ascending=False),
            "splits": sorted(set(j.split)), "by": by}


@torch.no_grad()
def one_changed(model, cases, new_id, bs=32):
    """Every known voter in turn: its identity becomes `new_id` (0 = unknown; None = kept, a
    check) on its own turns and as a voter, the colleagues keep theirs. Returns its logit."""
    items = []
    for c in cases:
        for i in np.flatnonzero(c.vjid > 0):
            c2 = copy.copy(c)
            if new_id is not None:
                c2.vjid, c2.jid = c.vjid.copy(), c.jid.copy()
                c2.jid[c.own[i] & (c.role == 1)] = new_id
                c2.vjid[i] = new_id
            items.append((c2, i))
    order = np.argsort([len(c.rows) for c, _ in items], kind="stable")
    rows = []
    for a in range(0, len(order), bs):
        chunk = [items[j] for j in order[a:a + bs]]
        z = model(collate([c for c, _ in chunk], "cpu"))["logit"]
        rows += [(c.case_id, c.names[i], z[b, i].item()) for b, (c, i) in enumerate(chunk)]
    return pd.DataFrame(rows, columns=KEY + ["logit"])


def untrained_rows(S, D, intact):
    """The same held-out votes of known justices, three ways (one justice changed at a time)."""
    vocab, kav = S["vocab"], S["vocab"][KAV]
    end = FOLDS[S["fold"]][0]
    tr = D["votes"][(D["votes"].year >= 1980) & (D["votes"].year <= end)]
    assert KAV not in set(tr.justice), "the row is supposed to be untrained"
    trained = [vocab[j] for j in set(tr.justice)]
    tabs, norms = {"as trained": [], "unknown justice (designed fallback)": [],
                   "untrained rows (as in the test run)": []}, []
    for s, m in enumerate(S["models"]):
        n = len(vocab)
        A = m.aff_pair.weight.view(n + 1, n + 1, -1)
        zero = max(m.gain.weight[kav].abs().max().item(), m.lean.weight[kav].abs().max().item(),
                   A[kav].abs().max().item(), A[:, kav].abs().max().item())
        assert zero == 0, "gain, lean and the pair-table entries of the untrained justice are not 0"
        norms.append({"query row, untrained": m.query_id.weight[kav].norm().item(),
                      "query rows, trained": m.query_id.weight[trained].norm(dim=1).mean().item(),
                      "shared query": m.q.norm().item(),
                      "head row, untrained": m.head_id.weight[kav].norm().item(),
                      "head rows, trained": m.head_id.weight[trained].norm(dim=1).mean().item(),
                      "speaker row, untrained": m.justice.weight[kav].norm().item(),
                      "speaker rows, trained": m.justice.weight[trained].norm(dim=1).mean().item(),
                      "own-turn bonus": m.own_bias.item()})
        if s == 0:                                             # the copies leave a prediction unchanged
            same = one_changed(m, S["cases"][:20], None).merge(intact[0], on=KEY)
            assert (same.logit_x - same.logit_y).abs().max() < 1e-4
        tabs["unknown justice (designed fallback)"].append(one_changed(m, S["cases"], 0))
        tabs["untrained rows (as in the test run)"].append(one_changed(m, S["cases"], kav))
        tabs["as trained"].append(intact[s].merge(tabs["unknown justice (designed fallback)"][-1][KEY],
                                                  on=KEY))
    P = {k: ens(t) for k, t in tabs.items()}
    vk = S["v"].merge(P["as trained"][KEY], on=KEY)
    rows = []
    for k, p in P.items():
        d = vk.merge(p, on=KEY)
        rows.append({"How the justice is identified": k, "Votes": len(d),
                     "AUC": roc_auc_score(d.vote, d.p),
                     "Macro-F1": f1_score(d.vote, d.pred, average="macro"),
                     "Vote acc": (d.vote == d.pred).mean(), "Pred. petitioner %": 100 * d.pred.mean()})
    a, b, c = P.values()
    both = b.merge(c, on=KEY, suffixes=("_f", "_u"))
    diffs = {"untrained − fallback": W.paired(b, c, vk), "fallback − as trained": W.paired(a, b, vk),
             "untrained − as trained": W.paired(a, c, vk),
             "mean_abs_p": float((both.p_u - both.p_f).abs().mean()),
             "flips": float((both.pred_u != both.pred_f).mean())}
    return pd.DataFrame(rows), diffs, pd.DataFrame(norms).mean().to_dict()


def test_by_justice():
    """From the SAVED predictions of the test run: the newcomers' first terms, and the fit
    made once before 2018, by justice. Descriptive; the samples are small."""
    v, _ = CW.labels()
    T = {(n, f): CW.table(n, f, v) for n in ("v1", "qc_bloc", "qc_plus") for f in ("each", "once")}

    def sc(d):
        return (f"{roc_auc_score(d.vote, d.p):.3f}" if d.vote.nunique() == 2 else "–",
                f"{100 * (d.vote == d.pred).mean():.1f}")

    first = []
    for name, term, how in [(KAV, 2018, "untrained identity rows"),
                            ("j__amy_coney_barrett", 2020, "unknown justice (as designed)"),
                            ("j__ketanji_brown_jackson", 2022, "unknown justice (as designed)")]:
        for who, pick in [(MF.short(name), lambda d: d.justice == name),
                          ("the other justices", lambda d: d.justice != name)]:
            row = {"Term": term, "Votes of": who, "Handled as": how if who != "the other justices" else "",
                   "Votes": int((pick(v) & (v.year == term)).sum())}
            for n, label in [("v1", "Model v1"), ("qc_bloc", "qc_bloc"), ("qc_plus", "qc_plus")]:
                d = T[(n, "each")]
                row[f"{label} AUC"], row[f"{label} acc %"] = sc(d[pick(d) & (d.year == term)])
            first.append(row)
    new = [KAV, "j__amy_coney_barrett", "j__ketanji_brown_jackson", "j__neil_gorsuch"]
    groups = [("Kavanaugh (untrained identity rows in the fit made once)", lambda d: d.justice == KAV),
              ("Barrett, Jackson (unknown justices, as designed)", lambda d: d.justice.isin(new[1:3])),
              ("Gorsuch (71 training votes before 2018)", lambda d: d.justice == new[3]),
              ("The justices who sat before 2016", lambda d: ~d.justice.isin(new))]
    once = []
    for label, pick in groups:
        row = {"Votes of": label, "Votes": int(pick(v).sum())}
        for n, f, col in [("v1", "once", "v1 once"), ("v1", "each", "v1 each term"),
                          ("qc_plus", "each", "qc_plus each term")]:
            row[f"{col} AUC"], row[f"{col} acc %"] = sc(T[(n, f)][pick(T[(n, f)])])
        once.append(row)
    return pd.DataFrame(first), pd.DataFrame(once)


# ---------------------------------------------------------------- one fold

def analyse(fold, D):
    t0 = time.time()
    S = setup(fold, D)
    cs, v, c = S["cases"], S["v"], S["c"]
    T = {k: [] for k in ["intact", *KNOCK]}
    A = {k: [] for k in ("intact", "no own-turn bonus", "no identity in the query")}
    X, repro = [], []
    for s, m in enumerate(S["models"]):
        t, x = run(m, cs, keep=("attn", "m", "msg", "pre", "out"))
        saved = pd.read_parquet(f"data/preds/v1_main_{fold}_s{s}.parquet").merge(t, on=KEY)
        assert len(saved) == len(t) == len(v)
        repro.append((saved.p - expit(saved.logit)).abs().max())
        mean = torch.from_numpy(np.concatenate([x["pre"][k.case_id] for k in cs]).mean(0))
        sl = slices(m)
        T["intact"].append(t)
        T["content readout"].append(run(m, cs, patch={"content": mean[sl["content"]]})[0])
        T["learned counters"].append(run(m, cs, patch={"counters": mean[sl["counters"]]})[0])
        T["count features"].append(run(m, cs, wide_mean=mean[sl["wide"]])[0])
        E = m.E
        m.E = torch.zeros_like(E)                               # 0 = the training mean of a turn vector
        T["turn text"].append(run(m, cs)[0])
        m.E = E
        T["identity"].append(run(m, cs, hide=True)[0])
        noid = Path(f"data/preds/v1_main_noid_{fold}_s{s}.parquet")
        if noid.exists():
            d = pd.read_parquet(noid).merge(T["identity"][-1], on=KEY)
            repro.append((d.p - expit(d.logit)).abs().max())
        T["bloc term"].append(t.assign(logit=t.base))
        for name, edit in EDITS.items():
            m2 = copy.deepcopy(m)
            with torch.no_grad():
                for e in edit:
                    (m2.query_id.weight if e == "query_id" else getattr(m2, e)).zero_()
            t2, x2 = run(m2, cs, keep=("attn",))
            T[name].append(t2)
            if name in A:
                A[name].append(attn_table(cs, x2["attn"], ref=x["attn"]))
        A["intact"].append(attn_table(cs, x["attn"]))
        X.append({k: x[k] for k in ("m", "msg", "out")})
        print(f"  [{fold} s{s}] passes done, {time.time() - t0:.0f}s", flush=True)
    assert max(repro) < 1e-3, f"the saved predictions are not reproduced: {max(repro):.1e}"

    P = {k: ens(t) for k, t in T.items()}
    paths = [{"Knocked out": "nothing (the intact network)", **scores(P["intact"], v, c)}]
    for k, what in KNOCK.items():
        auc, f1 = W.paired(P["intact"], P[k], v)
        paths.append({"Knocked out": f"{k}: {what}", **scores(P[k], v, c),
                      "AUC change": auc, "Macro-F1 change": f1})
    print(f"  [{fold}] M1 done, {time.time() - t0:.0f}s", flush=True)
    readout, rfacts = readout_summary(A)
    mtab = moved(S, D)
    ctab, cspread, csharp = counters(cs, v, [x["m"] for x in X])
    btab, bvotes, bfacts = blocs(S, X, D)
    utab, udiff, unorm = untrained_rows(S, D, T["intact"])
    print(f"  [{fold}] all done, {time.time() - t0:.0f}s", flush=True)
    return dict(fold=fold, cases=len(cs), votes=len(v), repro=max(repro), paths=pd.DataFrame(paths),
                readout=readout, rfacts=rfacts, mtab=mtab, ctab=ctab, cspread=cspread, csharp=csharp,
                btab=btab, bvotes=bvotes, bfacts=bfacts, utab=utab, udiff=udiff, unorm=unorm)


# ---------------------------------------------------------------- report and figures

def fmt(df, digits=3):
    d = df.copy()
    for col in d.columns:
        if d[col].dtype.kind == "f":
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else
                                f"{x:.1f}" if ("%" in col or col == "Turns") else f"{x:.{digits}f}")
    return d.fillna("").to_markdown(index=False, disable_numparse=True)


def report(R, adv, first, once):
    terms = {"val": "2016–17", "dev": "2012–15"}
    train = {"val": "1980–2015", "dev": "1980–2011"}
    md = ["# Inside the trained network (analysis C)", "",
          "`inside_v1.py`, written on Nov 14 after the test run; design fixed first in "
          "docs/writeup_notes.md §23.1. The saved fits of the main recipe (`v1_main`, five seeds a "
          "fold) are run on their own held-out terms. Nothing is fitted; `model_v1.py` and "
          "`train_v1.py` are unchanged. The fits of the test run were not saved, so the test "
          "terms play no part in M1–M4.", ""]
    for r in R:
        md += [f"- **{r['fold']} fold**: trained on terms {train[r['fold']]}, analysed on "
               f"{terms[r['fold']]} ({r['cases']} cases, {r['votes']:,} votes). The loaded networks "
               f"reproduce their saved predictions: largest difference in a probability "
               f"{r['repro']:.1e}."]
    md += ["", "## M1. Which path carries the prediction (knock-outs at inference)", "",
           "One thing at a time is replaced by its average over the held-out votes, in the trained "
           "network; nothing is retrained. Mean probability over the five seeds, cutoff 0.5. "
           "Change = knocked-out minus intact, 95% interval from a paired bootstrap over cases "
           "(1,000 draws). Case-level AUC = a case's mean probability against the winner; "
           "within-case AUC = ranking the justices inside split cases. A knock-out shows what the "
           "trained network relies on; the retrained ablations (`results/v1_ablations.md`) show "
           "what it needs.", ""]
    for r in R:
        md += [f"**{r['fold']} fold, terms {terms[r['fold']]}**", "", fmt(r["paths"]), ""]
    md += ["## M2. What the readout per justice looks at", "",
           "Attention of a justice's readout over the turns of the case, summed by who speaks, "
           "next to that group's share of the turns (what a uniform readout would give). Averages "
           "over the held-out votes and the five seeds.", ""]
    for r in R:
        f = r["rfacts"]
        for speaks in (True, False):
            n, tab = r["readout"][speaks]
            md += [f"**{r['fold']} fold, {'justices who speak in the case' if speaks else 'silent justices'}"
                   f" ({n:,} votes)**", "", fmt(tab), ""]
        md += [f"- {r['fold']}: a case has {f['turns']:.0f} turns on average; the readout spreads its "
               f"attention over the equivalent of {f['eff']:.0f} turns. A justice's own turns get "
               f"{f['lift_own']:.2f} times their share of the turns.",
               f"- {r['fold']}: taking the identity out of the query moves {100 * f['tv_identity']:.1f}% "
               f"of the attention (total-variation distance); setting the own-turn bonus to 0 moves "
               f"{100 * f['tv_bonus']:.1f}% (justices who speak).", "",
               f"**{r['fold']} fold: how far the weights moved from their start** (mean of five "
               "seeds; the starting weights are rebuilt from the seed; relative change = size of "
               "the change divided by the size at the start)", "", fmt(r["mtab"]), ""]
    md += ["## M3. What the learned counters count", "",
           "Weight of a turn = sum of its eight counters, scaled so that the mean justice turn "
           "is 1 (a plain count gives every turn 1). Mean over the five seeds; the last column "
           "is the range of the five seeds' own means.", ""]
    for r in R:
        s, a = r["cspread"], r["csharp"]
        md += [f"**{r['fold']} fold**", "", fmt(r["ctab"], 2), "",
               f"- Spread over justice turns: coefficient of variation {s['cv']:.2f}; the heaviest fifth "
               f"of the turns carries {100 * s['top20']:.0f}% of the weight (20% under a plain count). "
               f"Rank correlation of weight and length: {s['corr_words']:.2f}. An advocate's turn "
               f"weighs {s['advocate']:.2f} on the same scale.",
               f"- Does the learned weighting sharpen the question count? Feature: log(1 + weight of "
               f"the justice's turns aimed at the respondent) − the same for the petitioner; "
               f"{a['votes']:,} votes of justices who question at least one side. AUC for the vote: "
               f"plain count {a['plain']:.3f}, learned weights {a['learned']:.3f} "
               f"(difference {a['diff']}; single seeds {a['by_seed'][0]:.3f}–{a['by_seed'][1]:.3f}). "
               f"For reference, added after the first results: weight = the turn's number of words, "
               f"{a['words']:.3f} (learned − words: {a['diff_words']}). "
               f"The 40 single counters (8 × 5 seeds): {a['single'][0]:.3f} to {a['single'][2]:.3f}, "
               f"median {a['single'][1]:.3f}; the best one is a choice made after the fact.", ""]
    md += ["## M4. What the bloc layer learned", "",
           "Effective weight W[j, k]: how far justice j's logit moves when colleague k's own "
           "evidence moves by one logit toward the petitioner, through the bloc layer, on a full "
           "bench (the table's entry, times the slope of the messages on the sender's own logit, "
           "divided by 8 colleagues). Rows: the justice whose vote is predicted; columns: the "
           "colleague. Mean of five seeds; order = leading eigenvector of the table, no outside "
           "ideology score.", ""]
    for r in R:
        f = r["bfacts"]
        md += [f"**{r['fold']} fold: the network's table**", "",
               r["btab"].round(3).to_markdown(floatfmt="+.3f"), "",
               "**For comparison: correlation of the same justices' actual votes, terms 2010–15**", "",
               r["bvotes"].round(2).to_markdown(floatfmt=".2f"), "",
               f"- Order from the table alone: {', '.join(f['order'])}.",
               "- Position on that axis (farthest justice = 1), and for comparison the position "
               "on the axis that divides the Court in the actual votes of 2010–15 (the second "
               "principal axis of the vote correlations; the first is general agreement): "
               + "; ".join(f"{r_.justice} {r_.table:+.2f} / {r_.votes:+.2f}"
                           for r_ in f["lineup"].itertuples())
               + f". Correlation of the two line-ups: {f['axis_corr']:.2f}.",
               f"- Rank correlation over the 72 ordered pairs: with the correlation of the actual "
               f"votes in 2010–15, {f['rho_votes']:.2f} (2012–15 only: {f['rho_votes_late']:.2f}); "
               f"with the weights of the `qc_bloc` regression, {f['rho_qc']:.2f}.",
               "- A colleague's average weight over the eight others (the column mean): "
               + ", ".join(f"{k} {x:+.3f}" for k, x in f["col"].items())
               + f". With that average removed (added after the first results), the rank "
               f"correlation with the actual votes is {f['rho_centred']:.2f}.",
               f"- Mean weight for pairs whose votes agree more than the median pair: "
               f"{f['mean_close']:+.3f}; for the others: {f['mean_far']:+.3f}.",
               f"- Two seeds' tables correlate at {f['seed_corr']:.2f} on average. The bloc term has a "
               f"standard deviation of {f['sd_term']:.2f} logits over the votes (the bloc-free logit: "
               f"{f['sd_base']:.2f}); pair constants plus colleagues' own logits weighted by W "
               f"explain {100 * f['r2']:.0f}% of its variance.", ""]
    md += ["## M5. Untrained identity rows (a defect of the tested recipe)", "",
           f"ConvoKit marks a speaker as a justice in every case, also in a case argued as an "
           f"advocate before joining the Court; `prepare_data.py` takes the role from that mark. "
           f"{adv['cases']} cases (all in the {', '.join(adv['splits'])} split) have a speaker tagged "
           f"as a justice before that justice's first vote. In {adv['argued']} of them the speaker "
           f"has ten turns or more, an advocate's part; since 1980 that is every one of the "
           f"{adv['since1980']} cases ("
           + ", ".join(f"{MF.short(k)} {n}" for k, n in adv["who1980"].items())
           + "). In the others the speaker has a few turns: more likely a sitting justice's turns "
           "filed under the wrong name in the source (in 1964–67 the Chief Justice's calling of "
           "the case is filed under Burger). In an argued case that side's advocate turns are "
           "missing and are counted as turns of the bench.", "",
           adv["by"].rename(index=MF.short).rename_axis("Later justice").reset_index()
           .rename(columns={"cases": "Cases", "big": "With 10+ turns", "median": "Median turns a case",
                            "first": "First term", "last": "Last term"}).to_markdown(index=False), "",
           "`train_v1.py` gives an identity row to every speaker seen with the justice role in a "
           "training case. Kavanaugh therefore had rows at every step of the test run, without a "
           "training vote before the 2019 step: his query and head rows were the untrained random "
           "start; his gain, lean and pair-table entries are exactly 0 (the script checks it), so those "
           "parts fell back as designed. Affected: his 61 votes of the 2018 term in the primary "
           "row, and all 472 of his votes in the rows fitted once.", "",
           "**The check on held-out data.** Each held-out vote of a known justice, with only that "
           "justice changed. Mean probability over five seeds, as in the test run.", ""]
    for r in R:
        d, n = r["udiff"], r["unorm"]
        md += [f"**{r['fold']} fold, terms {terms[r['fold']]}**", "", fmt(r["utab"]), "",
               f"- Untrained rows − designed fallback: AUC {d['untrained − fallback'][0]}, macro-F1 "
               f"{d['untrained − fallback'][1]}. Mean absolute change of a probability "
               f"{d['mean_abs_p']:.3f}; {100 * d['flips']:.1f}% of the calls change.",
               f"- Designed fallback − as trained: AUC {d['fallback − as trained'][0]}, macro-F1 "
               f"{d['fallback − as trained'][1]}. Untrained rows − as trained: AUC "
               f"{d['untrained − as trained'][0]}, macro-F1 {d['untrained − as trained'][1]}.",
               f"- Size of the rows (vector norms, mean of five seeds): readout query, untrained "
               f"{n['query row, untrained']:.1f} against trained {n['query rows, trained']:.1f} "
               f"(the shared query: {n['shared query']:.1f}); head, {n['head row, untrained']:.1f} "
               f"against {n['head rows, trained']:.1f}; speaker embedding, "
               f"{n['speaker row, untrained']:.2f} against {n['speaker rows, trained']:.2f}. "
               f"Learned own-turn bonus: {n['own-turn bonus']:.2f} (start: 1).", ""]
    md += ["**From the saved predictions of the test run (descriptive; small samples).** "
           "Newcomers' first terms, models refit before each term:", "", first.to_markdown(index=False, disable_numparse=True),
           "", "The fit made once before 2018, terms 2018–2025, by justice:", "",
           once.to_markdown(index=False, disable_numparse=True), ""]
    Path(OUT).write_text("\n".join(md))
    print(f"Wrote {OUT}")


def figures(r=None):
    """Three figures from the primary (validation) fold, numbers alongside as .csv.
    Without `r` (--figures-only) they are redrawn from the .csv files of the last run."""
    INK2, GRID, BLUE, GREY = MF.INK2, MF.GRID, MF.BLUE, MF.GREY
    if r is not None:
        p = r["paths"].iloc[1:].copy()
        num = p["AUC change"].str.extract(r"([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]").astype(float)
        p["d"], p["lo"], p["hi"] = num[0], num[1], num[2]
        p["name"] = p["Knocked out"].str.split(":").str[0]
        p.sort_values("d", ascending=False)[["name", "AUC", "d", "lo", "hi"]].to_csv(
            FIG / "fig_inside_paths.csv", index=False)
        r["readout"][True][1].to_csv(FIG / "fig_inside_readout.csv", index=False)
        r["btab"].round(4).to_csv(FIG / "fig_inside_blocs.csv")
        r["bfacts"]["lineup"].round(3).to_csv(FIG / "fig_inside_lineup.csv", index=False)
    # which path carries the prediction
    p = pd.read_csv(FIG / "fig_inside_paths.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    y = np.arange(len(p))
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.hlines(y, p.lo, p.hi, color=BLUE, lw=1.2)
    MF.dot(ax, p.d, y, BLUE, size=5)
    ax.set_yticks(y, p.name)
    ax.set_xticks([-0.08, -0.04, 0], ["−0.08", "−0.04", "0"])
    ax.set_xlabel("change in AUC when knocked out")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_inside_paths.pdf")
    plt.close(fig)
    # what the readout looks at
    t = pd.read_csv(FIG / "fig_inside_readout.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    y, h = np.arange(len(t))[::-1], 0.36
    ax.barh(y + h / 2 + 0.02, t["Share of the turns %"], h, color=GREY, label="share of the turns")
    ax.barh(y - h / 2 - 0.02, t["Attention %"], h, color=BLUE, label="share of the attention")
    for yy, a, b in zip(y, t["Share of the turns %"], t["Attention %"]):
        ax.text(a + 0.8, yy + h / 2 + 0.02, f"{a:.0f}%", va="center", fontsize=7, color=INK2)
        ax.text(b + 0.8, yy - h / 2 - 0.02, f"{b:.0f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, t["Who speaks"])
    ax.set_xlim(0, max(t["Share of the turns %"].max(), t["Attention %"].max()) * 1.18)
    ax.set_xticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.legend(loc="upper center", ncol=2, fontsize=7, handlelength=1.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig_inside_readout.pdf")
    plt.close(fig)
    # the bloc layer's table
    A = pd.read_csv(FIG / "fig_inside_blocs.csv", index_col=0)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", BLUE])
    cmap.set_bad("#ffffff")
    M = A.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(M, np.nan)
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(A)), A.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(A)), A.index)
    ax.set_xticks(np.arange(-0.5, len(A)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(A)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("colleague whose questioning is observed")
    ax.set_ylabel("justice whose vote is predicted")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7)
    cb.set_label("weight (+ with, − against)", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_inside_blocs.pdf")
    plt.close(fig)
    print("Wrote results/figures/fig_inside_{paths,readout,blocs}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["val", "dev"], choices=list(FOLDS))
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from the .csv files of the last run")
    args = ap.parse_args()
    if args.figures_only:
        return figures()
    torch.manual_seed(0)
    D = load_data()
    R = [analyse(f, D) for f in args.folds]
    report(R, advocate_facts(), *test_by_justice())
    if R[0]["fold"] == "val":
        figures(R[0])


if __name__ == "__main__":
    main()
