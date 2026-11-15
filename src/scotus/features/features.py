"""
Structural (non-text) features per turn and per vote, for model v1 and the count baselines.

Why: the milestone model saw only the turn vector, the speaker and the position, and
could not count. A logistic regression on simple counts of who was questioned how much
predicts votes better than the plain question count; these features make that signal
available to the network.

Two tables, both label-free (they use the transcript only, so they are computed for every
split, test included, without ever touching a test label):

  data/turn_feats.parquet   one row per turn, same row order as data/turns.parquet
      floor    who is at the lectern (FLOORS): for a justice turn this is the advocate
               being questioned, for an advocate turn it is their own side
      nxt      FLOORS code of the next advocate turn in the same transcript
      seg      how many times the advocate at the lectern has changed so far (0..7)
      n_words, n_q ("?" marks), cut (turn ends with "--": the speaker was interrupted)
      target   the question-count baseline's rule: 0 petitioner, 1 respondent, -1 nobody
      pos      turn position within the case (argument order)

  data/vote_feats.parquet   one row per (case, justice) vote: WIDE_FEATURES
      counts of the justice's turns / words / "?" aimed at each side, the same for the
      whole bench and for the other justices, the advocates' words and turns, and
      whether an amicus argued before or after the respondent

Run from the repository root:
    python -m scotus.features.features
    python -m scotus.features.features --data data/ext      # terms 2020-25
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scotus.baselines import baselines as bl

FLOORS = ["none", "petitioner", "respondent", "amicus_before_resp", "amicus_after_resp",
          "advocate_unknown"]
N_SEG = 8
ADVOCATE_ROLES = ("petitioner", "respondent", "amicus", "advocate_unknown")


def turn_features(turns, cases):
    """Per-turn structural features, returned in the row order of `turns`."""
    o = bl.ordered_turns(turns[turns.case_id.isin(cases.case_id)], cases)
    floor = np.zeros(len(o), dtype=np.int8)
    seg = np.zeros(len(o), dtype=np.int8)
    cur, cur_seg, last_adv, seen_resp, prev_convo = 0, 0, None, False, None
    for i, (convo, role, spk) in enumerate(zip(o.convo_id, o.role, o.speaker_id)):
        if convo != prev_convo:                       # new transcript: reset the lectern
            cur, cur_seg, last_adv, seen_resp, prev_convo = 0, 0, None, False, convo
        if role in ADVOCATE_ROLES:
            if role == "petitioner":
                cur = 1
            elif role == "respondent":
                cur, seen_resp = 2, True
            elif role == "amicus":
                # The amicus's side is not in the data. An amicus who argues before the
                # respondent usually supports the petitioner and one who argues after
                # usually supports the respondent, so the two get separate codes and the
                # network learns what each means.
                cur = 4 if seen_resp else 3
            else:
                cur = 5
            if last_adv is not None and spk != last_adv:
                cur_seg = min(cur_seg + 1, N_SEG - 1)
            last_adv = spk
        floor[i], seg[i] = cur, cur_seg               # <INAUDIBLE> turns keep the lectern

    nxt = np.zeros(len(o), dtype=np.int8)             # scan backwards for the next advocate
    cur, prev_convo = 0, None
    convo_arr, role_arr = o.convo_id.to_numpy(), o.role.to_numpy()
    for i in range(len(o) - 1, -1, -1):
        if convo_arr[i] != prev_convo:
            cur, prev_convo = 0, convo_arr[i]
        nxt[i] = cur
        if role_arr[i] in ADVOCATE_ROLES:
            cur = floor[i]

    text = o.text
    out = pd.DataFrame({
        "floor": floor, "nxt": nxt, "seg": seg,
        "n_words": text.str.split().str.len().astype(np.int32).to_numpy(),
        "n_q": text.str.count(r"\?").astype(np.int16).to_numpy(),
        "cut": text.str.rstrip().str.endswith("--").to_numpy(),
        "target": bl.question_targets(o).astype(np.int8).to_numpy(),
        "pos": o.groupby("case_id", sort=False).cumcount().astype(np.int32).to_numpy(),
    }, index=o.index)
    return out.reindex(turns.index)                   # back to turns.parquet row order


def _side_sums(df, keys, prefix):
    """Sum turns / words / '?' by target side. Columns like o_n_P, o_w_R."""
    g = df.groupby(keys + ["target"]).agg(n=("n_words", "size"), w=("n_words", "sum"),
                                          q=("n_q", "sum")).unstack("target").fillna(0)
    g.columns = [f"{prefix}_{m}_{'PR'[t]}" for m, t in g.columns]
    return g.reset_index()


def vote_features(votes, turns, feats):
    """Wide count features, one row per vote in the order of `votes`."""
    t = turns[["case_id", "speaker_id", "role"]].join(feats)
    j = t[(t.role == "justice") & (t.target >= 0)]
    own = _side_sums(j, ["case_id", "speaker_id"], "o").rename(columns={"speaker_id": "justice"})
    bench = _side_sums(j, ["case_id"], "b")
    adv = (t[t.role.isin(["petitioner", "respondent"])].groupby(["case_id", "role"])
           .agg(w=("n_words", "sum"), n=("n_words", "size")).unstack("role").fillna(0))
    adv.columns = [f"a_{m}_{r[0].upper()}" for m, r in adv.columns]
    spoke = (t[t.role == "justice"].groupby(["case_id", "speaker_id"]).size()
             .rename("turns").reset_index().rename(columns={"speaker_id": "justice"}))

    v = (votes[["case_id", "justice"]]
         .merge(own, on=["case_id", "justice"], how="left")
         .merge(bench, on="case_id", how="left")
         .merge(adv.reset_index(), on="case_id", how="left")
         .merge(spoke, on=["case_id", "justice"], how="left").fillna(0))
    for m in "nwq":                                   # the other justices = bench - own
        for s in "PR":
            v[f"x_{m}_{s}"] = v[f"b_{m}_{s}"] - v[f"o_{m}_{s}"]
    for p in "obx":
        for m in "nwq":
            P, R = v[f"{p}_{m}_P"], v[f"{p}_{m}_R"]
            v[f"{p}_{m}_diff"] = np.log1p(R) - np.log1p(P)           # > 0: more at respondent
            v[f"{p}_{m}_share"] = (R + 0.5) / (P + R + 1.0) - 0.5
    v["a_w_diff"] = np.log1p(v.a_w_R) - np.log1p(v.a_w_P)
    v["a_n_diff"] = np.log1p(v.a_n_R) - np.log1p(v.a_n_P)
    v["silent"] = (v.turns == 0).astype(float)

    # Amicus slots. Since 1990 the petitioner wins 78% of cases with an
    # amicus arguing before the respondent (usually the Solicitor General supporting the
    # petitioner), 43% with an amicus arguing after, and 64% with none.
    slot = {3: "early", 4: "late"}
    am = t[t.floor.isin([3, 4])].assign(slot=lambda d: d.floor.map(slot))
    has = (am[am.role == "amicus"].groupby(["case_id", "slot"]).size().unstack("slot")
           .reindex(columns=["early", "late"]).notna().astype(float).add_prefix("am_"))
    aj = am[am.role == "justice"]
    b_am = (aj.groupby(["case_id", "slot"]).agg(n=("n_words", "size"), w=("n_words", "sum"))
            .unstack("slot"))
    b_am.columns = [f"b_am_{m}_{k}" for m, k in b_am.columns]
    o_am = (aj.groupby(["case_id", "speaker_id", "slot"]).size().unstack("slot")
            .add_prefix("o_am_n_").reset_index().rename(columns={"speaker_id": "justice"}))
    v = (v.merge(has.reset_index(), on="case_id", how="left")
         .merge(b_am.reset_index(), on="case_id", how="left")
         .merge(o_am, on=["case_id", "justice"], how="left"))
    for col in AMICUS_FEATURES:
        if col not in v:
            v[col] = 0.0
    v[AMICUS_FEATURES] = v[AMICUS_FEATURES].fillna(0)
    counts = [c for c in AMICUS_FEATURES if not c.startswith("am_")]
    v[counts] = np.sqrt(v[counts])
    # Raw sizes, square-root scale. Checked on the dev fold: sizes on a log scale
    # add nothing to the ratios above (dev AUC 0.683), raw or square-root sizes do
    # (0.699), because they let the model use the plain difference in counts, which is
    # Epstein et al.'s original signal. Square root keeps the long tail in check.
    for p in "ob":
        for m in "nwq":
            for s in "PR":
                v[f"{p}_{m}_{s}_sqrt"] = np.sqrt(v[f"{p}_{m}_{s}"])
    return v[["case_id", "justice", *WIDE_FEATURES]]


STANCE_FEATURES = ["st_o_sum", "st_o_mean", "st_b_sum", "st_b_mean"]


def stance_features(votes, turns, feats, stance):
    """Vote-level summaries of the turn-level stance scores (stance_*.py).

    `stance` is a Series aligned with `turns`: the centred "hostile" logit of each justice
    turn (NaN where there is none). Each score is signed so that + is evidence for the
    petitioner: hostility to the respondent counts +, hostility to the petitioner -.
    For the justice (o) and for the whole bench (b): the sum and the mean of the signed
    scores. With all scores equal this would be the question count again; the scores add
    HOW the questions were asked.
    """
    t = turns[["case_id", "speaker_id", "role"]].assign(target=feats.target, s=stance)
    t = t[(t.role == "justice") & (t.target >= 0) & t.s.notna()]
    t = t.assign(c=np.where(t.target == 1, t.s, -t.s))
    own = (t.groupby(["case_id", "speaker_id"]).c.agg(st_o_sum="sum", st_o_mean="mean")
           .reset_index().rename(columns={"speaker_id": "justice"}))
    bench = t.groupby("case_id").c.agg(st_b_sum="sum", st_b_mean="mean").reset_index()
    v = (votes[["case_id", "justice"]].merge(own, on=["case_id", "justice"], how="left")
         .merge(bench, on="case_id", how="left").fillna(0.0))
    return v[["case_id", "justice", *STANCE_FEATURES]]


AMICUS_FEATURES = (["am_early", "am_late"]
                   + [f"b_am_{m}_{k}" for m in "nw" for k in ("early", "late")]
                   + ["o_am_n_early", "o_am_n_late"])
WIDE_FEATURES = ([f"{p}_{m}_{k}" for p in "obx" for m in "nwq" for k in ("diff", "share")]
                 + ["a_w_diff", "a_n_diff", "silent"]
                 + [f"{p}_{m}_{s}_sqrt" for p in "ob" for m in "nwq" for s in "PR"]
                 + AMICUS_FEATURES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    cases = pd.read_parquet(data / "cases.parquet")
    votes = pd.read_parquet(data / "votes.parquet")
    turns = pd.read_parquet(data / "turns.parquet")
    feats = turn_features(turns, cases)
    feats.to_parquet(data / "turn_feats.parquet")
    vf = vote_features(votes, turns, feats)
    vf.to_parquet(data / "vote_feats.parquet", index=False)
    j = turns.role == "justice"
    print(f"turn_feats: {len(feats):,} rows; vote_feats: {len(vf):,} rows, "
          f"{len(WIDE_FEATURES)} wide features")
    print("justice turns by lectern:",
          {FLOORS[k]: int(n) for k, n in feats.floor[j].value_counts().sort_index().items()})
    print("justice turns where next advocate != lectern: "
          f"{(feats.floor[j] != feats.nxt[j]).mean():.1%}")


if __name__ == "__main__":
    main()
