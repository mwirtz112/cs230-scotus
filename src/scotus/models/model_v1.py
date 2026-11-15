"""
Model v1: a transformer over frozen turn vectors, with structure, counting and a voting-bloc layer.

What the milestone model (v0) lacked: it could not tell which side a question was aimed at
except by learning it from the neighbouring turns, and its readout was a softmax average,
which cannot count. A logistic regression on simple counts beats it by 8 AUC points.

    turn t:   h_t   frozen encoder vector, standardized with train statistics
              x_t = W h_t + e_role + e_justice + e_floor + e_next + e_seg + e_pos + U u_t
                    floor = who is at the lectern, next = who answers, seg = which
                    advocate segment, u_t = [log words, "?" marks, interrupted]
              c_1..T = TransformerEncoder(x)

    justice j, three sources of evidence, concatenated into one small MLP:
      content   z_j   = attention pool of c_t, query q + e_id(j), bonus b on j's own turns
                        (the v0 readout, unchanged)
      counters  m_t   = softplus(A c_t) in R^K: K learned non-negative "weights" per turn.
                phi_j = log(1 + sum of m_t) over six groups of turns:
                        j's turns aimed at petitioner / at respondent, the other justices'
                        turns aimed at each, and each side's own advocate turns.
                        With m_t = 1 this is the question count; the network learns what
                        to count instead ("neural question counting").
      wide      the hand-built count features of features.py (the logistic-regression
                baseline's inputs), also given a direct linear path to the logit
    g_j = MLP([z_j, phi_j, wide_j, e_id(j)])
    bloc layer: s_j = mean over colleagues k of A[j, k] . u_k, with u_k = U g_k a message
                from colleague k and A[j, k] a learned justice-to-justice affinity
    logit_j = (1 + gain_j) v . g_j + lean_j + s_j + w . wide_j

Metadata fusion (optional; the main model does not use it). The case profile from
the Supreme Court Database (issue area, lower-court direction, parties, court of origin,
...) enters in up to three ways, each switched on separately:
    hist   voting-history rates (meta_features.py) appended to the wide features;
    meta   m = sum of one learned embedding per categorical field, concatenated to the
           input of g_j for every justice;
    ideal  an ideal-point term: logit_j += theta_j . a(m), where theta_j is a small
           learned vector per justice (0 for an unknown justice) and
           a(m) = P m + sign * Q m, sign = +1 if the lower-court ruling was conservative
           (the petitioner asks for the liberal outcome), -1 if liberal, 0 if unknown.
           It is the item-response ("ideal point") model of political science, with the
           case side learned from metadata instead of from the votes themselves.
    prior  logit_j += prior_j, the log-odds from a logistic regression on the history
           rates alone (fitted outside the network, cross-fitted on the training cases).
           The network then learns what the case profile cannot explain. Fed as ordinary
           inputs, the history rates were under-used: per-justice parameters fit the same
           pattern on the training cases more easily.

Identity: an unknown justice (new on the Court, or hidden by identity dropout) has a zero
identity vector, in the input and in the readout. "Unknown" then simply means "no
identity information", and identity dropout trains the model for it. In v0 the
unknown-justice embedding was never trained, because every training justice is known.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from scotus.features.features import FLOORS, N_SEG, WIDE_FEATURES

MAX_TURNS = 1024
ROLES = ["<pad>", "justice", "petitioner", "respondent", "amicus", "advocate_unknown",
         "unknown"]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
N_SCAL = 3                      # per-turn scalars: log words, "?" marks, interrupted
                                # (+ the stance score when --stance is given)


# ---------------------------------------------------------------- data

@dataclass
class Case:
    case_id: str
    year: int
    rows: np.ndarray        # [T] rows into the embedding matrix, argument order
    pos: np.ndarray         # [T] position in the case (kept when turns are dropped)
    role: np.ndarray        # [T] ROLES id
    jid: np.ndarray         # [T] justice id of the speaker (0 = not a known justice)
    floor: np.ndarray       # [T] FLOORS id
    nxt: np.ndarray         # [T] FLOORS id of the next advocate
    seg: np.ndarray         # [T] advocate segment
    scal: np.ndarray        # [T, N_SCAL]
    target: np.ndarray      # [T] question-count target: 0 petitioner, 1 respondent, -1 none
    names: list             # [K] voting justices
    vjid: np.ndarray        # [K] justice ids of the voters (0 = unknown)
    labels: np.ndarray      # [K] 1 = pro-petitioner vote
    own: np.ndarray         # [K, T] bool: turn t was spoken by voter k
    wide: np.ndarray        # [K, n_wide] standardized count features
    meta: np.ndarray = None  # [F] categorical case-profile codes (0 = unknown), or None
    prior: np.ndarray = None  # [K] metadata-only log-odds of a pro-petitioner vote, or None
    lc_sign: float = 0.0     # +1 lower court conservative, -1 liberal, 0 unknown


def build_cases(case_ids, turns, feats, votes, vote_feats, justice_vocab, wide_stats,
                wide_cols=WIDE_FEATURES, meta=None, prior=None):
    """One Case per case id. `turns` needs case_id, speaker_id, role; `feats` is
    turn_feats.parquet (same index), optionally with a `stance` column (scaled turn-level
    stance score, 0 where there is none), which becomes a fourth per-turn scalar.
    Wide features `wide_cols` are standardized with `wide_stats`.
    `meta`: optional dict case_id -> (codes [F], lc_sign), the encoded case profile.
    `prior`: optional column name in `vote_feats` holding the metadata-only log-odds."""
    t = turns[turns.case_id.isin(case_ids)].join(feats)
    t = t[t.pos < MAX_TURNS].sort_values(["case_id", "pos"])           # M3: first 1,024
    is_j = (t.role == "justice").to_numpy()
    jid = np.where(is_j, t.speaker_id.map(justice_vocab).fillna(0).to_numpy(), 0)
    scal = [np.log1p(t.n_words.to_numpy()) / 4.0, np.minimum(t.n_q.to_numpy(), 3) / 3.0,
            t.cut.to_numpy().astype(np.float32)]
    if "stance" in t:
        scal.append(t.stance.to_numpy())
    scal = np.stack(scal, axis=1).astype(np.float32)
    arr = dict(rows=t.index.to_numpy(), pos=t.pos.to_numpy(),
               role=t.role.map(ROLE_ID).to_numpy(), jid=jid.astype(np.int64),
               floor=t.floor.to_numpy().astype(np.int64),
               nxt=t.nxt.to_numpy().astype(np.int64),
               seg=t.seg.to_numpy().astype(np.int64), scal=scal,
               target=t.target.to_numpy().astype(np.int64))
    speaker = t.speaker_id.to_numpy()
    bounds = np.flatnonzero(np.r_[True, t.case_id.to_numpy()[1:] != t.case_id.to_numpy()[:-1],
                                  True])
    span = {t.case_id.iloc[a]: (a, b) for a, b in zip(bounds[:-1], bounds[1:])}

    mean, std = wide_stats
    v = votes.merge(vote_feats, on=["case_id", "justice"], how="left")
    wide_all = ((v[list(wide_cols)].to_numpy(np.float32) - mean) / std).astype(np.float32)
    out = []
    for case_id, idx in v.groupby("case_id", sort=False).indices.items():
        if case_id not in span:
            continue
        a, b = span[case_id]
        g = v.iloc[idx]
        names = g.justice.tolist()
        out.append(Case(
            case_id=case_id, year=int(g.year.iloc[0]),
            **{k: x[a:b] for k, x in arr.items()},
            names=names,
            vjid=np.array([justice_vocab.get(j, 0) for j in names], dtype=np.int64),
            labels=g.vote.to_numpy().astype(np.float32),
            own=np.stack([speaker[a:b] == j for j in names]),
            wide=wide_all[idx],
            **({} if meta is None else dict(zip(("meta", "lc_sign"), meta[case_id]))),
            **({} if prior is None else {"prior": g[prior].to_numpy(np.float32)}),
        ))
    return out


def collate(batch, device, turn_drop=0.0, id_drop=0.0, rng=None):
    """Pad a list of Cases into tensors. With rng: training-time turn and identity dropout."""
    kept = []
    for c in batch:
        keep = np.arange(len(c.rows))
        if rng is not None and turn_drop > 0:
            m = rng.random(len(keep)) >= turn_drop
            keep = keep[m] if m.any() else keep[:1]
        kept.append(keep)
    B, T, K = len(batch), max(len(k) for k in kept), max(len(c.vjid) for c in batch)
    i64 = lambda *s: np.zeros(s, dtype=np.int64)
    rows, pos, role, jid, floor, nxt, seg = (i64(B, T) for _ in range(7))
    rows -= 1                                           # -1 = padding row (all zeros)
    target = i64(B, T) - 1
    scal = np.zeros((B, T, batch[0].scal.shape[1]), dtype=np.float32)
    vjid = i64(B, K)
    y = np.zeros((B, K), dtype=np.float32)
    vmask = np.zeros((B, K), dtype=bool)
    own = np.zeros((B, K, T), dtype=np.float32)
    wide = np.zeros((B, K, batch[0].wide.shape[1]), dtype=np.float32)
    turn_vote = i64(B, T) - 1                           # vote of the turn's speaker, if any
    for b, (c, keep) in enumerate(zip(batch, kept)):
        t, k = len(keep), len(c.vjid)
        rows[b, :t], pos[b, :t], role[b, :t] = c.rows[keep], c.pos[keep], c.role[keep]
        floor[b, :t], nxt[b, :t], seg[b, :t] = c.floor[keep], c.nxt[keep], c.seg[keep]
        scal[b, :t], target[b, :t] = c.scal[keep], c.target[keep]
        cj, cv = c.jid[keep], c.vjid
        if rng is not None and id_drop > 0:             # hide some justices' identities
            ids = np.unique(np.r_[cj, cv]); ids = ids[ids > 0]
            hidden = ids[rng.random(len(ids)) < id_drop]
            cj, cv = np.where(np.isin(cj, hidden), 0, cj), np.where(np.isin(cv, hidden), 0, cv)
        jid[b, :t], vjid[b, :k] = cj, cv
        y[b, :k], vmask[b, :k] = c.labels, True
        o = c.own[:, keep]
        own[b, :k, :t], wide[b, :k] = o, c.wide
        spoke = o.any(0)
        turn_vote[b, :t] = np.where(spoke, c.labels[o.argmax(0)].astype(np.int64), -1)
    to = lambda a: torch.from_numpy(a).to(device)
    out = dict(rows=to(rows), pos=to(pos), role=to(role), jid=to(jid), floor=to(floor),
               nxt=to(nxt), seg=to(seg), scal=to(scal), target=to(target), vjid=to(vjid),
               y=to(y), vmask=to(vmask), own=to(own), wide=to(wide),
               turn_vote=to(turn_vote))
    if batch[0].meta is not None:
        out["meta"] = to(np.stack([c.meta for c in batch]).astype(np.int64))
        out["lc_sign"] = to(np.array([c.lc_sign for c in batch], dtype=np.float32))
    if batch[0].prior is not None:
        prior = np.zeros((B, K), dtype=np.float32)
        for b, c in enumerate(batch):
            prior[b, :len(c.prior)] = c.prior
        out["prior"] = to(prior)
    return out


# ---------------------------------------------------------------- model

class TurnModelV1(nn.Module):
    def __init__(self, E, n_justices, d=256, layers=2, heads=4, ff=512, dropout=0.3,
                 in_drop=0.0, counters=8, d_id=32, hidden=128, struct=True, wide=True,
                 content=True, identity=True, stance_aux=False, text=True, bottleneck=0,
                 bench_layers=0, affinity=0, aff_rank=16, aff_mode="pair",
                 n_wide=len(WIDE_FEATURES), n_scal=N_SCAL, aff_unknown=False, mf_iters=1,
                 case_aux=False, meta_sizes=None, d_meta=32, meta_concat=True, ideal=0,
                 prior=0.0, prior_learn=False, center_text=False):
        """E: [n_turns + 1, emb_dim] standardized turn vectors on the device, last row 0."""
        super().__init__()
        self.E, self.d = E, d
        self.struct, self.use_wide, self.content, self.identity = struct, wide, content, identity
        self.k, self.text = counters, text
        if text and bottleneck:                                    # W = low-rank: 768 -> r -> d
            self.proj = nn.Sequential(nn.Linear(E.shape[1], bottleneck, bias=False),
                                      nn.Linear(bottleneck, d))
        elif text:
            self.proj = nn.Linear(E.shape[1], d)                   # W
        self.in_drop = nn.Dropout(in_drop)
        # center_text: subtract the case's mean turn vector from every turn vector. The
        # mean is mostly what the case is about; what is left is how a turn differs from
        # the rest of its own argument. A diagnostic for the encoder comparison.
        self.center_text = center_text
        self.role = nn.Embedding(len(ROLES), d, padding_idx=0)
        self.justice = nn.Embedding(n_justices + 1, d, padding_idx=0)   # 0 = unknown: zero
        nn.init.zeros_(self.justice.weight)                        # starts as "a justice"
        self.pos = nn.Embedding(MAX_TURNS, d)
        if struct:
            self.floor = nn.Embedding(len(FLOORS), d)
            self.nxt = nn.Embedding(len(FLOORS), d)
            self.seg = nn.Embedding(N_SEG, d)
            self.scal = nn.Linear(n_scal, d)
        self.drop = nn.Dropout(dropout)
        if layers:
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            self.encoder = None                                    # ablation: no context
        self.norm = nn.LayerNorm(d)

        n_in = 0
        if content:                                                # the v0 readout
            self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
            self.query_id = nn.Embedding(n_justices + 1, d, padding_idx=0)
            self.own_bias = nn.Parameter(torch.tensor(1.0))
            n_in += d
        if counters:
            self.count = nn.Linear(d, counters)
            self.phi_norm = nn.BatchNorm1d(6 * counters)
            n_in += 6 * counters
        if wide:
            self.wide_lin = nn.Linear(n_wide, 1)
            n_in += n_wide
        if identity:
            self.head_id = nn.Embedding(n_justices + 1, d_id, padding_idx=0)
            n_in += d_id
        # Case vector (optional). The error analysis shows the hard part is which
        # side wins the case, and that the text is what helps there. So one pooled vector
        # for the whole argument goes to every justice's head, and an auxiliary loss asks
        # it to predict the case winner.
        self.case_head = nn.Linear(d, 1) if case_aux else None
        if case_aux:
            self.case_query = nn.Parameter(torch.randn(d) / math.sqrt(d))
            n_in += d
        # Case profile (metadata fusion): one embedding per categorical field, summed.
        # Code 0 = unknown stays a zero vector, so a case without metadata is the v1 model.
        self.meta_emb, self.meta_concat, self.ideal = None, meta_concat, ideal
        if meta_sizes:
            self.meta_emb = nn.ModuleList([nn.Embedding(n, d_meta, padding_idx=0)
                                           for n in meta_sizes])
            self.meta_drop = nn.Dropout(dropout)
            if meta_concat:
                n_in += d_meta
            if ideal:
                # theta_j starts at 0: training starts from the model without this term
                self.ip_justice = nn.Embedding(n_justices + 1, ideal, padding_idx=0)
                nn.init.zeros_(self.ip_justice.weight)
                self.ip_case = nn.Linear(d_meta, ideal, bias=False)
                self.ip_signed = nn.Linear(d_meta, ideal)
        self.pre = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Dropout(dropout))
        # Metadata prior as an offset on the logit: weight `prior`, fixed unless prior_learn.
        self.use_prior = prior != 0
        if self.use_prior and prior_learn:
            self.prior_w = nn.Parameter(torch.tensor(float(prior)))
        elif self.use_prior:
            self.register_buffer("prior_w", torch.tensor(float(prior)))
        if bench_layers:
            # Votes are not independent: justices vote in blocs. One self-attention layer
            # over the (up to 9) justice vectors lets each prediction use WHO else is
            # hostile to which side, e.g. for a silent justice. Without it a justice only
            # sees the other justices as one pooled total.
            layer = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout,
                                               batch_first=True, norm_first=True)
            self.bench = nn.TransformerEncoder(layer, bench_layers, enable_nested_tensor=False)
            self.bench_norm = nn.LayerNorm(hidden)
        else:
            self.bench = None
        self.aff, self.aff_rank, self.aff_mode, self.n_j = affinity, aff_rank, aff_mode, n_justices
        if affinity:
            # Justice affinity (the "bloc" layer). Justice k sends a small message u_k
            # computed from k's evidence; justice j adds the messages of the colleagues,
            # each weighted by a learned affinity A[j, k], which can be negative: an
            # ally's hostility to a side counts with j, an opponent's against. This is
            # how a silent justice (Thomas) can be predicted from WHO presses which side.
            # An unknown justice (id 0) sends and receives nothing here and falls back on
            # the pooled evidence.
            #   "pair":    A[j, k] is a free table entry per ordered pair of justices,
            #              starting at 0, so training starts from the model without blocs
            #              (the neural version of the qc_bloc baseline).
            #   "belief":  the same table, but the first message channel is the colleague's
            #              own predicted vote, tanh(logit_k / 2) in (-1, 1), instead of a
            #              learned feature: logit_j = base_j + sum_k A[j, k] E[vote_k].
            #              This is one mean-field step of a fully connected pairwise model
            #              over the nine votes; mf_iters > 1 repeats it with the updated
            #              logits. Further channels (affinity > 1) stay learned messages.
            #   "lowrank": A[j, k] = a_j . b_k, each justice a point in a small space.
            # aff_unknown: also learn the entries with an unknown justice (id 0). They are
            # trained by identity dropout and mean "a typical justice": a newcomer then
            # still learns from which NAMED colleagues lean where, instead of getting no
            # bloc term at all.
            self.aff_unknown, self.mf_iters = aff_unknown, mf_iters
            n_learned = affinity - (aff_mode == "belief")
            self.aff_msg = nn.Linear(hidden, n_learned) if n_learned else None
            if aff_mode in ("pair", "belief"):
                self.aff_pair = nn.Embedding((n_justices + 1) ** 2, affinity)
                nn.init.zeros_(self.aff_pair.weight)
            else:
                self.aff_recv = nn.Embedding(n_justices + 1, affinity * aff_rank, padding_idx=0)
                self.aff_send = nn.Embedding(n_justices + 1, affinity * aff_rank, padding_idx=0)
        n_out = hidden + (affinity if affinity and aff_mode == "lowrank" else 0)
        self.out = nn.Linear(n_out, 1)
        if identity:
            # Per-justice gain and lean on the logit, starting at 0: some justices'
            # questions say more about their vote than others' do.
            self.gain = nn.Embedding(n_justices + 1, 1, padding_idx=0)
            self.lean = nn.Embedding(n_justices + 1, 1, padding_idx=0)
            nn.init.zeros_(self.gain.weight); nn.init.zeros_(self.lean.weight)
        self.stance = nn.Linear(d, 1) if stance_aux else None

    def forward(self, b, count_scale=1.0):
        pad = b["rows"] < 0
        x = self.role(b["role"]) + self.pos(b["pos"])
        if self.text:
            h = self.E[b["rows"]].float()                           # -1 -> the zero row
            if self.center_text:
                keep = (~pad).float()[..., None]
                h = (h - (h * keep).sum(1, keepdim=True)
                     / keep.sum(1, keepdim=True).clamp(min=1)) * keep
            x = x + self.proj(self.in_drop(h))
        if self.identity:
            x = x + self.justice(b["jid"])
        if self.struct:
            x = (x + self.floor(b["floor"]) + self.nxt(b["nxt"]) + self.seg(b["seg"])
                 + self.scal(b["scal"]))
        x = self.drop(x)
        if self.encoder is not None:
            x = self.encoder(x, src_key_padding_mask=pad)
        c = self.norm(x)                                            # [B, T, d]
        live = (~pad).float()
        out, parts = {}, []

        if self.content:
            query = self.q + (self.query_id(b["vjid"]) if self.identity else 0)
            if query.dim() == 1:
                query = query.expand(b["vjid"].shape + (self.d,))
            scores = torch.einsum("bkd,btd->bkt", query, c) / math.sqrt(self.d)
            scores = scores + self.own_bias * b["own"]
            attn = scores.masked_fill(pad[:, None, :], float("-inf")).softmax(-1)
            parts.append(torch.einsum("bkt,btd->bkd", attn, c))
            out["attn"] = attn

        if self.k:
            m = F.softplus(self.count(c)) * live[..., None]         # [B, T, K] >= 0
            is_j = (b["role"] == 1).float()
            groups = []
            for side in (0, 1):                                     # 0 petitioner, 1 respondent
                aimed = (b["target"] == side).float()
                own = torch.einsum("bkt,btc->bkc", b["own"] * aimed[:, None, :], m)
                bench = torch.einsum("bt,btc->bc", is_j * aimed, m)[:, None, :]
                adv = torch.einsum("bt,btc->bc", (b["role"] == 2 + side).float(), m)
                groups += [own, bench - own, adv[:, None, :].expand_as(own)]
            phi = torch.log1p(torch.cat(groups, -1).clamp(min=0) * count_scale)
            normed = torch.zeros_like(phi)                          # normalize real votes only
            normed[b["vmask"]] = self.phi_norm(phi[b["vmask"]])
            parts.append(normed)
            out["m"] = m

        if self.use_wide:
            parts.append(b["wide"])
        if self.identity:
            parts.append(self.head_id(b["vjid"]))
        if self.case_head is not None:
            w = (c @ self.case_query / math.sqrt(self.d)).masked_fill(pad, float("-inf"))
            case_vec = torch.einsum("bt,btd->bd", w.softmax(-1), c)
            out["case"] = self.case_head(case_vec).squeeze(-1)      # [B] logit: petitioner wins
            parts.append(case_vec[:, None, :].expand(-1, b["vjid"].shape[1], -1))
        mv = None
        if self.meta_emb is not None:
            mv = self.meta_drop(sum(e(b["meta"][:, i]) for i, e in enumerate(self.meta_emb)))
            if self.meta_concat:
                parts.append(mv[:, None, :].expand(-1, b["vjid"].shape[1], -1))
        g = self.pre(torch.cat(parts, -1))                          # [B, K, hidden]
        if self.bench is not None:
            g = self.bench_norm(self.bench(g, src_key_padding_mask=~b["vmask"]))
        A = None
        if self.aff:
            B, K = b["vjid"].shape
            vid = b["vjid"]
            vm = b["vmask"].float()
            who = vm if self.aff_unknown else vm * (vid > 0).float()
            pair = (who[:, :, None] * who[:, None, :]
                    * (1 - torch.eye(K, device=g.device)))[..., None]
            others = (vm.sum(1) - 1).clamp(min=1)[:, None, None]
            if self.aff_mode in ("pair", "belief"):
                A = self.aff_pair(vid[:, :, None] * (self.n_j + 1) + vid[:, None, :])
            else:
                shape = (B, K, self.aff, self.aff_rank)
                A = torch.einsum("bjhr,bkhr->bjkh", self.aff_recv(vid).view(shape),
                                 self.aff_send(vid).view(shape)) / math.sqrt(self.aff_rank)
                s = torch.einsum("bjkh,bkh->bjh", A * pair, self.aff_msg(g)) / others
                g = torch.cat([g, s], -1)
            out["affinity"] = A
        base = self.out(g).squeeze(-1)
        if self.identity:
            base = base * (1 + self.gain(b["vjid"]).squeeze(-1)) + self.lean(b["vjid"]).squeeze(-1)
        if self.use_wide:
            base = base + self.wide_lin(b["wide"]).squeeze(-1)
        if mv is not None and self.ideal:
            a = self.ip_case(mv) + b["lc_sign"][:, None] * self.ip_signed(mv)   # [B, R]
            ideal = torch.einsum("bkr,br->bk", self.ip_justice(b["vjid"]), a)
            base, out["ideal"] = base + ideal, ideal
        if self.use_prior:
            base = base + self.prior_w * b["prior"]
        logit = base                                                # the model without blocs
        if self.aff and self.aff_mode == "pair":
            s = torch.einsum("bjkh,bkh->bjh", A * pair, self.aff_msg(g)) / others
            logit = base + s.sum(-1)
        elif self.aff and self.aff_mode == "belief":
            learned = self.aff_msg(g) if self.aff_msg is not None else None
            for _ in range(self.mf_iters):
                u = torch.tanh(logit / 2)[..., None]                # colleague's expected vote
                if learned is not None:
                    u = torch.cat([u, learned], -1)
                logit = base + (torch.einsum("bjkh,bkh->bjh", A * pair, u) / others).sum(-1)
        out["base"], out["logit"] = base, logit
        if self.stance is not None:
            out["stance"] = self.stance(c).squeeze(-1)              # [B, T]
        return out


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
