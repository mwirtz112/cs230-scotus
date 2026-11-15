"""
Sanity checks for model v1 (run on CPU, about a minute):

    python -m scotus.evaluation.check_model_v1

  1. Padding invariance: a case's logits are the same alone or batched with longer cases.
  2. Justice-order equivariance: reordering the voters of a case reorders the logits.
  3. Unknown justices: with every identity hidden the bloc term is exactly 0 (pair mode).
  4. The model can overfit 16 cases (loss -> 0), so the training path is wired up.
  5. Metadata fusion (profile embeddings, ideal-point term, metadata prior): checks 1-2
     again; the ideal-point term is 0 for hidden justices and for a case without a
     profile; the prior moves the logit by exactly its own value.

Uses recent training cases only (2013-15); nothing from validation or test.
"""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scotus.evaluation import eval as ev
from scotus.features.features import WIDE_FEATURES
from scotus.models.model_v1 import TurnModelV1, build_cases, collate, n_params
torch.manual_seed(0)
dev = "cpu"
cases = pd.read_parquet("data/cases.parquet")
turns = pd.read_parquet("data/turns.parquet", columns=["case_id", "speaker_id", "role"])
votes = ev.load_labels("train")[0]; votes = votes[votes.year >= 2013]
feats = pd.read_parquet("data/turn_feats.parquet"); vf = pd.read_parquet("data/vote_feats.parquet")
vocab = {j: i + 1 for i, j in enumerate(sorted(set(votes.justice)))}
wf = votes[["case_id", "justice"]].merge(vf, on=["case_id", "justice"])
ws = (wf[WIDE_FEATURES].mean().to_numpy(np.float32), wf[WIDE_FEATURES].std().to_numpy(np.float32) + 1e-6)
cs = build_cases(set(votes.case_id), turns, feats, votes, vf, vocab, ws)
rows = np.unique(np.concatenate([c.rows for c in cs])); remap = {r: i for i, r in enumerate(rows)}
E_np = np.load("embeddings/turns_last.npy", mmap_mode="r"); st = np.load("embeddings/turns_last_train_stats.npz")
E = torch.zeros((len(rows) + 1, 768)); E[:len(rows)] = torch.from_numpy(((E_np[rows].astype(np.float32) - st["mean"]) / st["std"]))
for c in cs: c.rows = np.array([remap[r] for r in c.rows])
print(len(cs), "cases;", "turns per case min/max:", min(len(c.rows) for c in cs), max(len(c.rows) for c in cs))
for mode, kw in [("pair", dict(affinity=4)), ("belief", dict(affinity=4, aff_mode="belief", mf_iters=2, aff_unknown=True)), ("pair+case_aux", dict(affinity=4, case_aux=True)), ("pair+center", dict(affinity=4, center_text=True))]:
    m = TurnModelV1(E, len(vocab), **kw).eval()
    with torch.no_grad():
        for p in (m.aff_pair.weight, m.gain.weight, m.lean.weight):      # give the zero-initialized tables values so the checks bite
            p.normal_(0, 0.5)
        short, long_ = min(cs, key=lambda c: len(c.rows)), max(cs, key=lambda c: len(c.rows))
        alone = m(collate([short], dev))["logit"][0, :len(short.vjid)]
        batch = m(collate([long_, short, cs[3]], dev))["logit"][1, :len(short.vjid)]
        pad_err = (alone - batch).abs().max().item()
        perm = np.random.default_rng(0).permutation(len(short.vjid))
        c2 = copy.copy(short); c2.vjid, c2.labels, c2.own, c2.wide, c2.names = short.vjid[perm], short.labels[perm], short.own[perm], short.wide[perm], [short.names[i] for i in perm]
        permuted = m(collate([c2], dev))["logit"][0, :len(short.vjid)]
        perm_err = (alone[perm] - permuted).abs().max().item()
        hid = collate([short], dev); hid["vjid"] = torch.zeros_like(hid["vjid"]); hid["jid"] = torch.zeros_like(hid["jid"])
        o = m(hid); bloc_unknown = (o["logit"] - o["base"]).abs().max().item()
    print(f"{mode:14s} params {n_params(m):,} | padding invariance err {pad_err:.2e} | justice-order equivariance err {perm_err:.2e} | bloc term with all identities hidden {bloc_unknown:.3f}")
m = TurnModelV1(E, len(vocab), affinity=4, dropout=0.0); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
tiny = cs[:16]
for step in range(151):
    b = collate(tiny, dev); out = m(b)
    loss = (F.binary_cross_entropy_with_logits(out["logit"], b["y"], reduction="none") * b["vmask"]).sum() / b["vmask"].sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (0, 50, 100, 150): print(f"overfit 16 cases: step {step:3d} loss {loss.item():.4f}")

# ---- 5. metadata fusion options (model_v1: meta, ideal, prior) ----
from scotus.features import meta_features as M
prof = pd.read_parquet("data/meta_cat.parquet").set_index("case_id")
fields = M.CAT_FIELDS
codes = {f: {v: i + 1 for i, v in enumerate(sorted(set(prof.loc[[c.case_id for c in cs], f]) - {0}))} for f in fields}
rng = np.random.default_rng(1)
for c in cs:
    r = prof.loc[c.case_id]
    c.meta = np.array([codes[f].get(r[f], 0) for f in fields], dtype=np.int64)
    c.lc_sign = {1: 1.0, 2: -1.0}.get(int(r.lc_dir), 0.0)
    c.prior = rng.normal(0, 1, len(c.vjid)).astype(np.float32)
sizes = [len(codes[f]) + 1 for f in fields]
m = TurnModelV1(E, len(vocab), affinity=4, meta_sizes=sizes, ideal=4, prior=1.0).eval()
with torch.no_grad():
    for p in (m.aff_pair.weight, m.gain.weight, m.lean.weight, m.ip_justice.weight):
        p.normal_(0, 0.5)
    m.ip_justice.weight[0] = 0                                    # padding row: unknown justice
    known = [c for c in cs if c.lc_sign != 0]
    short, long_ = min(known, key=lambda c: len(c.rows)), max(cs, key=lambda c: len(c.rows))
    alone = m(collate([short], dev))
    k = len(short.vjid)
    batch = m(collate([long_, short, cs[3]], dev))["logit"][1, :k]
    pad_err = (alone["logit"][0, :k] - batch).abs().max().item()
    perm = rng.permutation(k)
    c2 = copy.copy(short)
    c2.vjid, c2.labels, c2.own, c2.wide, c2.prior = (short.vjid[perm], short.labels[perm], short.own[perm],
                                                     short.wide[perm], short.prior[perm])
    c2.names = [short.names[i] for i in perm]
    perm_err = (alone["logit"][0, :k][perm] - m(collate([c2], dev))["logit"][0, :k]).abs().max().item()
    hid = collate([short], dev); hid["vjid"] = torch.zeros_like(hid["vjid"]); hid["jid"] = torch.zeros_like(hid["jid"])
    ideal_hidden = m(hid)["ideal"].abs().max().item()
    blank = collate([short], dev); blank["meta"] = torch.zeros_like(blank["meta"]); blank["lc_sign"] = torch.zeros_like(blank["lc_sign"])
    ideal_blank = m(blank)["ideal"].abs().max().item()
    zero = collate([short], dev); shift = zero["prior"].clone(); zero["prior"] = torch.zeros_like(zero["prior"])
    prior_err = ((alone["logit"] - m(zero)["logit"]) - shift)[0, :k].abs().max().item()
print(f"fusion         params {n_params(m):,} | padding invariance err {pad_err:.2e} | justice-order equivariance err {perm_err:.2e} | "
      f"ideal-point term: {alone['ideal'].abs().max().item():.3f} known, {ideal_hidden:.3f} identities hidden, {ideal_blank:.3f} no profile | "
      f"prior offset err {prior_err:.2e}")
