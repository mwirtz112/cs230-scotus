"""
The clock of the arguments, and a test of whether timing helps predict the votes.

The transcripts record when every turn starts and stops (ConvoKit `start_times`,
`stop_times`, from Oyez). The models ignore that. This script

  --build   reads the times from the ConvoKit corpora already on disk (no download) into
            data/turn_times.parquet, one row per turn in the row order of turns.parquet:
            start, stop, dur (seconds), ok (duration and pace are plausible);
  default   builds twelve timing features per vote (TIMING_FEATURES) and tests them in the
            two count regressions (qc_plus, qc_bloc) on the dev and validation folds
            -> results/timing.md.

The features and the pass mark were fixed before any result was seen.
Label-free inputs; the test split is never scored (eval.load_labels locks it). The
2020-25 terms (data/ext) are not read.

    python -m scotus.analysis.timing --build
    python -m scotus.analysis.timing
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scotus.baselines import baselines as bl
from scotus.evaluation import eval as ev
from scotus.baselines import qc_plus as Q
from scotus.features.features import WIDE_FEATURES
from scotus.data.prepare_data import ID_PATTERN

KEY = ["case_id", "justice"]
PACE_LO, PACE_HI = 0.3, 10.0          # words per second outside this range: time not used
MIN_WORDS = 20                        # timed words needed on each side for a pace difference
PARTS = ["s_diff", "s_share", "pace_diff", "cut_diff", "cut_share", "ans_diff"]
TIMING_FEATURES = [f"{p}_{k}" for p in ("to", "tb") for k in PARTS]     # own, bench
OWN_TIMING = [f for f in TIMING_FEATURES if f.startswith("to_")]
N_BOOT, SEED = 1000, 0


def build_times(data, root):
    """Start / stop of every turn, aligned with turns.parquet. Returns the table."""
    turns = pd.read_parquet(data / "turns.parquet", columns=["convo_id", "year", "turn_idx", "text"])
    rows = []
    for y in sorted(turns.year.unique()):
        convos = {}
        with open(Path(root) / f"supreme-{y}" / "utterances.jsonl") as f:
            for line in f:
                u = json.loads(line)
                convos.setdefault(u["conversation_id"], []).append(u)
        for cid, utts in convos.items():                  # the order prepare_data.py uses
            if all(ID_PATTERN.search(u["id"]) for u in utts):
                utts = sorted(utts, key=lambda u: tuple(map(int, ID_PATTERN.search(u["id"]).groups())))
            for i, u in enumerate(utts):
                st, sp = u["meta"].get("start_times") or [], u["meta"].get("stop_times") or []
                rows.append((cid, i, st[0] if st else np.nan, sp[-1] if sp else np.nan,
                             len((u["text"] or "").split())))
    src = pd.DataFrame(rows, columns=["convo_id", "turn_idx", "start", "stop", "n_src"])
    assert not src.duplicated(["convo_id", "turn_idx"]).any()
    t = turns.merge(src, on=["convo_id", "turn_idx"], how="left", validate="one_to_one")
    assert len(t) == len(turns) and (t.convo_id.to_numpy() == turns.convo_id.to_numpy()).all()
    n_words = turns.text.str.split().str.len().to_numpy()
    print(f"{len(t):,} turns; no source row {t.start.isna().mean():.3%}; word count differs "
          f"from the source {(t.n_src.to_numpy() != n_words).mean():.3%}")
    dur = (t.stop - t.start).to_numpy(float)
    pace = n_words / np.where(dur > 0, dur, np.nan)
    ok = (dur > 0) & (pace >= PACE_LO) & (pace <= PACE_HI)
    out = pd.DataFrame({"start": t.start.to_numpy(float), "stop": t.stop.to_numpy(float),
                        "dur": dur, "ok": ok}, index=turns.index)
    out.to_parquet(data / "turn_times.parquet")
    print(f"wrote {data / 'turn_times.parquet'}: usable time on {ok.mean():.1%} of the turns")
    return out


def _sides(g, keys, prefix):
    a = g.groupby(keys + ["target"]).agg(s=("sec", "sum"), w=("tw", "sum"), cut=("cutin", "sum"),
                                         ans=("ans", "mean")).unstack("target")
    a.columns = [f"{prefix}_{m}_{'PR'[t]}" for m, t in a.columns]
    return a.reset_index()


def timing_features(votes, turns, feats, times, cases):
    """TIMING_FEATURES, one row per vote in the order of `votes`."""
    t = (turns[["case_id", "convo_id", "turn_idx", "speaker_id", "role"]]
         .join(feats[["target", "n_words", "cut"]]).join(times[["dur", "ok"]]))
    o = bl.ordered_turns(t[t.case_id.isin(cases.case_id)], cases)
    adv = o.role.isin(["petitioner", "respondent"])
    g = o.groupby("convo_id", sort=False)
    prev_adv = g.role.shift(1).isin(["petitioner", "respondent"])
    o = o.assign(sec=np.where(o.ok, o.dur, 0.0), tw=np.where(o.ok, o.n_words, 0),
                 cutin=(prev_adv & g.cut.shift(1).fillna(False).astype(bool)).astype(float),
                 ans=np.where(prev_adv & g.ok.shift(1).fillna(False).astype(bool), g.dur.shift(1), np.nan))
    del adv
    j = o[(o.role == "justice") & (o.target >= 0)]
    own = _sides(j, ["case_id", "speaker_id"], "to").rename(columns={"speaker_id": "justice"})
    bench = _sides(j, ["case_id"], "tb")
    v = (votes[KEY].merge(own, on=KEY, how="left").merge(bench, on="case_id", how="left"))
    for p in ("to", "tb"):
        for m in ("s", "w", "cut", "ans"):
            for s in "PR":
                if f"{p}_{m}_{s}" not in v:
                    v[f"{p}_{m}_{s}"] = np.nan
        sP, sR = v[f"{p}_s_P"].fillna(0), v[f"{p}_s_R"].fillna(0)
        wP, wR = v[f"{p}_w_P"].fillna(0), v[f"{p}_w_R"].fillna(0)
        cP, cR = v[f"{p}_cut_P"].fillna(0), v[f"{p}_cut_R"].fillna(0)
        aP, aR = v[f"{p}_ans_P"], v[f"{p}_ans_R"]
        v[f"{p}_s_diff"] = np.log1p(sR) - np.log1p(sP)             # > 0: more at respondent
        v[f"{p}_s_share"] = (sR + 0.5) / (sP + sR + 1.0) - 0.5
        both = (wP >= MIN_WORDS) & (wR >= MIN_WORDS) & (sP > 0) & (sR > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v[f"{p}_pace_diff"] = np.where(both, np.log(wR / sR) - np.log(wP / sP), 0.0)
            v[f"{p}_ans_diff"] = np.where(aP.notna() & aR.notna(), np.log(aR) - np.log(aP), 0.0)
        v[f"{p}_ans_diff"] = v[f"{p}_ans_diff"].clip(-3, 3)
        v[f"{p}_cut_diff"] = np.log1p(cR) - np.log1p(cP)
        v[f"{p}_cut_share"] = (cR + 0.5) / (cP + cR + 1.0) - 0.5
    return v[KEY + TIMING_FEATURES].fillna(0.0)


def plus(train, held, feats, C=1.0):
    z = Q._logit(train, feats, C)
    out = []
    for df in held:
        p = 1 / (1 + np.exp(-z(df)))
        out.append(df[["case_id", "year", "justice", "vote"]].assign(p=p, pred=(p >= 0.5).astype(int)))
    return out


def paired(a, b, rng):
    """b - a on the same votes: AUC and macro-F1 with 95% intervals (bootstrap over cases)."""
    d = a.merge(b[KEY + ["p", "pred"]], on=KEY, suffixes=("_a", "_b"))
    code, cases = pd.factorize(d.case_id)
    diff = lambda w: (roc_auc_score(d.vote, d.p_b, sample_weight=w) - roc_auc_score(d.vote, d.p_a, sample_weight=w),
                      f1_score(d.vote, d.pred_b, average="macro", sample_weight=w)
                      - f1_score(d.vote, d.pred_a, average="macro", sample_weight=w))
    draws = np.array([diff(rng.multinomial(len(cases), np.full(len(cases), 1 / len(cases)))[code])
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    pt = diff(None)
    return pt, [f"{pt[i]:+.3f} [{lo[i]:+.3f}, {hi[i]:+.3f}]" for i in (0, 1)], (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--build", action="store_true", help="(re)build data/turn_times.parquet")
    ap.add_argument("--corpora", default=os.path.expanduser("~/.convokit/saved-corpora"))
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()
    data = Path(args.data)
    if args.build or not (data / "turn_times.parquet").exists():
        build_times(data, args.corpora)
        if args.build:
            return
    cases = pd.read_parquet(data / "cases.parquet")
    votes = pd.read_parquet(data / "votes.parquet")
    turns = pd.read_parquet(data / "turns.parquet")
    feats = pd.read_parquet(data / "turn_feats.parquet")
    times = pd.read_parquet(data / "turn_times.parquet")
    assert len(times) == len(turns) == len(feats)
    tf = timing_features(votes, turns, feats, times, cases)
    tf.to_parquet(data / "vote_timing.parquet", index=False)
    vf = pd.read_parquet(data / "vote_feats.parquet").merge(tf, on=KEY)

    md = ["# Timing test: does the clock of the argument help? (notes §21.1)", "",
          "Start and stop times of every turn from the ConvoKit corpora (Oyez timing). "
          f"Twelve features per vote: {', '.join(TIMING_FEATURES)} (`to_` the justice, `tb_` the "
          "bench; `s` seconds of speech, `pace` words per second, `cut` turns that follow an "
          "advocate cut off with \"--\", `ans` seconds an advocate speaks before the justice comes "
          "in; respondent against petitioner). The features and the pass mark were fixed before "
          "any result. The test split is not scored.", ""]

    # 1. the clock, label-free
    q = turns[["year", "role"]].join(feats[["n_words"]]).join(times)
    q = q.assign(decade=(q.year // 10) * 10, pace=q.n_words / q.dur.where(q.dur > 0))
    rows = []
    for dec, g in q.groupby("decade"):
        jt = g[g.role == "justice"]
        rows.append({"Decade": f"{dec}s", "Turns": f"{len(g):,}", "Usable time %": f"{100 * g.ok.mean():.1f}",
                     "Duration ≤ 0 %": f"{100 * (g.dur <= 0).mean():.1f}",
                     "Median pace, words/s": f"{g.pace[g.ok].median():.2f}",
                     "Median justice turn, s": f"{jt.dur[jt.ok].median():.1f}",
                     "Advocate turns cut off %": f"{100 * feats.cut[g.index][g.role.isin(['petitioner', 'respondent'])].mean():.1f}"})
    md += ["## 1. The clock by decade", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # 2. the test
    lab = {s: ev.load_labels(s, args.data) for s in ["train", "val"]}                 # test locked
    tr = lab["train"][0].merge(vf, on=KEY)
    va = lab["val"][0].merge(vf, on=KEY)
    end, lo, hi = Q.DEV
    dtr, dte = tr[tr.year <= end], tr[tr.year.between(lo, hi)]
    dv = lab["train"][0]; dv = dv[dv.year.between(lo, hi)].reset_index(drop=True)
    dc = lab["train"][1]; dc = dc[dc.case_id.isin(dv.case_id)].reset_index(drop=True)
    folds = {"dev": (dtr, dte, dv, dc), "val": (tr, va, *lab["val"])}
    WT = WIDE_FEATURES + TIMING_FEATURES
    P = {}
    for fold, (a, b, _, _) in folds.items():
        P["qc_plus", fold] = plus(a, [b], WIDE_FEATURES, args.C)[0]
        P["qc_plus + timing", fold] = plus(a, [b], WT, args.C)[0]
        P["timing only", fold] = plus(a, [b], TIMING_FEATURES, args.C)[0]
        P["qc_bloc", fold] = Q.fit_predict_bloc(a, [b], args.C)[0]
        P["qc_bloc + timing", fold] = Q.fit_predict_bloc(a, [b], args.C, pool_feats=WT,
                                                         own_feats=Q.OWN + OWN_TIMING)[0]
    rows = []
    for m in ["timing only", "qc_plus", "qc_plus + timing", "qc_bloc", "qc_bloc + timing"]:
        row = {"Model": m}
        for fold, (_, _, v, c) in folds.items():
            r = ev.score(P[m, fold], v, c)
            row[f"{fold} AUC"], row[f"{fold} macro-F1"] = ev.fmt_ci(*r["auc"]), ev.fmt_ci(*r["macro_f1"])
        p17 = P[m, "val"][P[m, "val"].year == 2017]
        row["2017 AUC"] = f"{roc_auc_score(p17.vote, p17.p):.3f}"
        rows.append(row)
    md += ["## 2. The test (class-balanced logistic regressions, fitted on 1955+)", "",
           "dev: fit ≤ 2011, score 2012–15. val: fit ≤ 2015, score 2016–17. [95% bootstrap "
           "interval over cases].", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    rng = np.random.default_rng(SEED)
    rows, verdict = [], {}
    for base in ["qc_plus", "qc_bloc"]:
        a_all = pd.concat([P[base, "dev"], P[base, "val"]], ignore_index=True)
        b_all = pd.concat([P[base + " + timing", "dev"], P[base + " + timing", "val"]], ignore_index=True)
        sets = {"dev 2012–15": (P[base, "dev"], P[base + " + timing", "dev"]),
                "val 2016–17": (P[base, "val"], P[base + " + timing", "val"]),
                "2017 alone": (P[base, "val"][P[base, "val"].year == 2017],
                               P[base + " + timing", "val"][P[base + " + timing", "val"].year == 2017]),
                "dev + val pooled": (a_all, b_all)}
        for name, (a, b) in sets.items():
            pt, txt, (l, h) = paired(a, b, rng)
            rows.append({"Comparison": f"{base} + timing − {base}", "Scored on": name,
                         "AUC": txt[0], "Macro-F1": txt[1]})
            verdict[base, name] = (pt[0], l[0], h[0])
    md += ["## 3. With timing minus without (paired)", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    d, s17, pool = verdict["qc_bloc", "dev 2012–15"], verdict["qc_bloc", "2017 alone"], verdict["qc_bloc", "dev + val pooled"]
    passed = d[0] > 0 and s17[0] > 0
    clear = pool[1] > 0
    md += ["## 4. Pass mark", "",
           "Fixed in advance: AUC of qc_bloc + timing above qc_bloc on the dev fold and on 2017 "
           "alone; clear only if the pooled interval excludes zero.", "",
           f"- dev {d[0]:+.3f}, 2017 alone {s17[0]:+.3f}, pooled {pool[0]:+.3f} [{pool[1]:+.3f}, {pool[2]:+.3f}].",
           f"- **{'PASSES' if passed else 'DOES NOT PASS'}"
           f"{' and the gain is clear' if passed and clear else ' but the gain is not clear' if passed else ''}.**", ""]

    # 5. what the regression does with the features
    X = tr[WT].to_numpy(float)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=args.C, max_iter=5000, class_weight="balanced").fit(
        (X - X.mean(0)) / (X.std(0) + 1e-6), tr.vote)
    coef = pd.Series(clf.coef_[0], index=WT)
    one = []
    for f in TIMING_FEATURES:
        x = tr[f].to_numpy(float)
        one.append({"Feature": f, "Weight in qc_plus + timing (standardized; + = pro-petitioner)": f"{coef[f]:+.3f}",
                    "AUC alone, train terms": f"{max(roc_auc_score(tr.vote, x), 1 - roc_auc_score(tr.vote, x)):.3f}",
                    "Zero in % of votes": f"{100 * (x == 0).mean():.0f}"})
    md += ["## 5. The timing features in the regression (fit ≤ 2015)", "",
           pd.DataFrame(one).to_markdown(index=False), "",
           "Largest count weights for scale: "
           + ", ".join(f"{k} {x:+.2f}" for k, x in coef[WIDE_FEATURES].reindex(
               coef[WIDE_FEATURES].abs().sort_values(ascending=False).index).head(4).items()) + ".", ""]
    Path("results/timing.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nWrote results/timing.md")


if __name__ == "__main__":
    main()
