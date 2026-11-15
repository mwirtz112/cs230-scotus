"""
The audio trial, step 2: does a justice's pitch, relative to that justice's own normal,
help predict the vote? The features, the test and the pass mark were fixed before any
recording was processed.

Input: data/audio/turn_pitch.parquet (audio_pitch.py), the pitch of every turn of the
argument recordings of the terms 2010-17. No term from 2018 on is read; labels come from
eval.load_labels for train and val only.

    python -m scotus.analysis.audio_test           -> results/audio_pitch.md
    python -m scotus.analysis.audio_test --dry     the same code on shuffled votes (nothing written)

Features per vote (R = at the respondent, P = at the petitioner; z = the turn's value
standardized within the justice's own turns in the same recording):
    po_diff, pb_diff   pitch (median F0 of the turn, semitones): mean z at R - at P,
                       for the justice / for the whole bench            <- primary
    lo_diff, lb_diff   loudness (mean intensity, dB)                    <- secondary
    ro_diff, rb_diff   pitch range inside the turn (sd, semitones)      <- secondary
Test: walk-forward over T = 2012..2017. A class-balanced logistic regression on the
out-of-sample logit of qc_bloc plus the features, fitted on the terms 2010..T-1, predicts
term T; compared with the same regression on the qc_bloc logit alone. Pass mark
(PASS_GAIN): a pooled AUC gain of at least +0.010 that is also positive on 2012-15 and
on 2017 alone.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from scotus.evaluation import eval as ev
from scotus.baselines import qc_plus as Q

KEY = ["case_id", "justice"]
FIRST, START, LAST = 2010, 2012, 2017
MEASURES = {"p": ("st_med", "pitch"), "l": ("int_db", "loudness"), "r": ("st_sd", "pitch range")}
PRIMARY = ["po_diff", "pb_diff"]
SECONDARY = ["lo_diff", "lb_diff", "ro_diff", "rb_diff"]
PASS_GAIN, N_BOOT, SEED = 0.010, 1000, 0
MIN_TURNS_Z, MIN_TURNS_SIDE = 4, 2


def justice_turns():
    """Justice turns of 2010-17 aimed at a side, with the audio measures and their z."""
    t = pd.read_parquet("data/turns.parquet", columns=["case_id", "convo_id", "year", "turn_idx",
                                                       "speaker_id", "role"])
    t = t.join(pd.read_parquet("data/turn_feats.parquet", columns=["target"]))
    t = t[t.year.between(FIRST, LAST)]
    p = pd.read_parquet("data/audio/turn_pitch.parquet")
    p = p.assign(dur=p.stop - p.start)
    rec = p.groupby("convo_id").agg(audio_s=("audio_s", "first"), last=("stop", "max"))
    lined = (rec["last"] <= rec.audio_s + 2) & (rec.audio_s - rec["last"] < 600)
    m = t.merge(p[p.convo_id.isin(lined.index[lined])], on=["convo_id", "turn_idx"], how="left")
    j = m[(m.role == "justice")].copy()
    for k, (col, _) in MEASURES.items():
        g = j.groupby(["convo_id", "speaker_id"])[col]
        n, mu, sd = g.transform("count"), g.transform("mean"), g.transform("std")
        j[f"z{k}"] = ((j[col] - mu) / sd).where((n >= MIN_TURNS_Z) & (sd > 0))
    info = {"transcripts": t.convo_id.nunique(), "recordings": int(p.convo_id.nunique()),
            "lined": int(lined.sum()), "not_lined": sorted(lined.index[~lined]),
            "justice_turns": len(j), "with_pitch": float(j.st_med.notna().mean()),
            "gap": (rec.audio_s - rec["last"]).describe()}
    return j, info


def side_diff(j, z, keys):
    d = j[j[z].notna() & (j.target >= 0) & (j.dur > 0)]
    d = d.assign(wz=d[z] * d.dur)
    a = d.groupby(keys + ["target"]).agg(n=(z, "size"), sw=("dur", "sum"), swz=("wz", "sum")).unstack("target")
    for s in (0, 1):
        for c in ("n", "sw", "swz"):
            if (c, s) not in a:
                a[(c, s)] = np.nan
    ok = (a[("n", 0)] >= MIN_TURNS_SIDE) & (a[("n", 1)] >= MIN_TURNS_SIDE)
    diff = a[("swz", 1)] / a[("sw", 1)] - a[("swz", 0)] / a[("sw", 0)]
    return diff.where(ok, 0.0).fillna(0.0), ok


def features(j, votes):
    v = votes[KEY].copy()
    for k in MEASURES:
        own, ok_o = side_diff(j, f"z{k}", ["case_id", "speaker_id"])
        ben, ok_b = side_diff(j, f"z{k}", ["case_id"])
        o = pd.DataFrame({f"{k}o_diff": own, f"{k}o_ok": ok_o}).reset_index().rename(columns={"speaker_id": "justice"})
        b = pd.DataFrame({f"{k}b_diff": ben, f"{k}b_ok": ok_b}).reset_index()
        v = v.merge(o, on=KEY, how="left").merge(b, on="case_id", how="left")
    for c in v.columns:
        if c.endswith("_ok"):
            v[c] = v[c].fillna(False).astype(bool)
        elif c.endswith("_diff"):
            v[c] = v[c].fillna(0.0)
    return v


def stack(df, feats):
    """Walk-forward predictions for START..LAST from a regression on feats, fitted on FIRST..T-1."""
    out = []
    for t in range(START, LAST + 1):
        tr, te = df[df.year.between(FIRST, t - 1)], df[df.year == t]
        X = tr[feats].to_numpy(float)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit((X - mu) / sd, tr.vote)
        p = clf.predict_proba((te[feats].to_numpy(float) - mu) / sd)[:, 1]
        out.append(te[["case_id", "year", "justice", "vote"]].assign(p=p, pred=(p >= 0.5).astype(int)))
    return pd.concat(out, ignore_index=True), pd.Series(clf.coef_[0], index=feats)


def paired(a, b, rng):
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
    ap.add_argument("--dry", action="store_true", help="shuffle the votes: runs the code, shows nothing real")
    ap.add_argument("--partial", action="store_true", help="allow fewer than 85%% of the recordings")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    j, info = justice_turns()
    share = info["lined"] / info["transcripts"]
    if share < 0.85 and not (args.dry or args.partial):
        raise SystemExit(f"Only {info['lined']} of {info['transcripts']} recordings are on disk and lined "
                         "up (under 85%): the trial's stop rule. Nothing was scored.")

    lab = pd.concat([ev.load_labels(s)[0] for s in ("train", "val")], ignore_index=True)    # test locked
    assert lab.year.max() <= LAST
    if args.dry:
        lab = lab.assign(vote=rng.permutation(lab.vote.to_numpy()))
    vf = pd.read_parquet("data/vote_feats.parquet")
    full = lab.merge(vf, on=KEY)
    z = []
    for t in range(FIRST, LAST + 1):                       # qc_bloc, each term from earlier terms only
        (pr,) = Q.fit_predict_bloc(full[full.year < t], [full[full.year == t]])
        z.append(pr.assign(zq=np.log(pr.p.clip(1e-6, 1 - 1e-6) / (1 - pr.p.clip(1e-6, 1 - 1e-6)))))
    z = pd.concat(z, ignore_index=True)
    df = z.merge(features(j, z), on=KEY)

    base, _ = stack(df, ["zq"])
    prim, w_prim = stack(df, ["zq"] + PRIMARY)
    seco, w_seco = stack(df, ["zq"] + PRIMARY + SECONDARY)
    only, _ = stack(df, PRIMARY)
    raw = z[z.year >= START]
    blocks = {"2012–15": (2012, 2015), "2016–17": (2016, 2017), "2017 alone": (2017, 2017), "2012–17": (START, LAST)}
    sel = lambda d, b: d[d.year.between(*blocks[b])]
    auc = lambda d: roc_auc_score(d.vote, d.p)
    f1 = lambda d: f1_score(d.vote, d.pred, average="macro")

    md = ["# Audio trial: pitch relative to each justice's own normal (notes §21.2)", "",
          ("**DRY RUN ON SHUFFLED VOTES. Not a result.**" if args.dry else
           "Features, test and pass mark were fixed before any recording was processed. "
           "Terms 2010–2017 only; the test terms were not read."), "",
          "## 1. The recordings", "",
          f"- {info['transcripts']} transcripts of 2010–17 in our tables; {info['recordings']} recordings "
          f"processed; {info['lined']} lined up with their turns ({100 * share:.1f}%)"
          + (f"; not lined up: {', '.join(info['not_lined'][:12])}" if info["not_lined"] else "") + ".",
          f"- Audio length minus the end of the last turn: median {info['gap']['50%']:.1f} s, "
          f"largest {info['gap']['max']:.1f} s.",
          f"- {info['justice_turns']:,} justice turns; pitch measured on {100 * info['with_pitch']:.1f}% "
          "(at least 20 voiced frames)."]
    med = j.groupby("speaker_id").f0_med.median().dropna()
    md += [f"- Median pitch by justice: {', '.join(f'{k[3:]} {x:.0f} Hz' for k, x in med.sort_values().items())}.",
           f"- Stop rule (a justice's median outside 70–300 Hz): "
           f"{'TRIGGERED' if ((med < 70) | (med > 300)).any() else 'not triggered'}.", ""]
    cov = {c: 100 * df[df.year >= START][c].mean() for c in df.columns if c.endswith("_ok")}
    md += ["Votes of 2012–17 with a defined difference: "
           + ", ".join(f"`{c[:-3]}_diff` {x:.0f}%" for c, x in cov.items()) + ".", ""]

    rows = []
    for name, d in [("qc_bloc as it is (not refitted)", raw), ("regression on the qc_bloc logit", base),
                    ("+ pitch (2 features, primary)", prim), ("+ pitch, loudness, range (6 features)", seco),
                    ("pitch only (2 features)", only)]:
        rows.append({"Model": name, **{f"AUC {b}": f"{auc(sel(d, b)):.3f}" for b in blocks},
                     "Macro-F1 2012–17": f"{f1(d):.3f}"})
    md += ["## 2. Walk-forward 2012–17 (each term from the terms 2010 to T − 1)", "",
           pd.DataFrame(rows).to_markdown(index=False), ""]
    rows, res = [], {}
    for name, d in [("+ pitch (primary)", prim), ("+ pitch, loudness, range", seco)]:
        for b in blocks:
            pt, txt, (lo, hi) = paired(sel(base, b), sel(d, b), rng)
            rows.append({"Minus the regression on qc_bloc alone": name, "Scored on": b, "AUC": txt[0], "Macro-F1": txt[1]})
            res[name, b] = (pt[0], lo[0], hi[0])
    md += ["## 3. Gain over the regression on qc_bloc alone (paired, 95% interval over cases)", "",
           pd.DataFrame(rows).to_markdown(index=False), ""]
    g, d15, s17 = res["+ pitch (primary)", "2012–17"], res["+ pitch (primary)", "2012–15"], res["+ pitch (primary)", "2017 alone"]
    passed = g[0] >= PASS_GAIN and d15[0] > 0 and s17[0] > 0
    md += ["## 4. Pass mark", "",
           f"Fixed in advance: pooled AUC gain of at least +{PASS_GAIN:.3f} with the two pitch features, and a "
           "positive gain on 2012–15 and on 2017 alone; clear if the pooled interval excludes zero.", "",
           f"- Pooled {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}]; 2012–15 {d15[0]:+.3f}; 2017 alone {s17[0]:+.3f}.",
           f"- **{'PASSES' if passed else 'DOES NOT PASS'}"
           f"{(' and the gain is clear' if g[1] > 0 else ' but the gain is not clear') if passed else ''}.**", ""]

    tr = df[df.year.between(FIRST, 2015)]
    rows = []
    for k, (_, word) in MEASURES.items():
        own = tr[tr[f"{k}o_ok"]]
        agree = ((own[f"{k}o_diff"] > 0) == (own.vote == 1)).to_numpy(float)
        code, cs = pd.factorize(own.case_id)
        dr = [np.average(agree, weights=rng.multinomial(len(cs), np.full(len(cs), 1 / len(cs)))[code]) for _ in range(N_BOOT)]
        lo, hi = np.percentile(dr, [2.5, 97.5])
        ben = tr[tr[f"{k}b_ok"]].drop_duplicates("case_id")
        rows.append({"Measure": word, "Justice–case pairs": f"{len(own):,}",
                     "Higher toward the side voted against %": f"{100 * agree.mean():.1f} [{100 * lo:.1f}, {100 * hi:.1f}]",
                     "AUC of the justice's difference for the vote": f"{roc_auc_score(own.vote, own[f'{k}o_diff']):.3f}",
                     "AUC of the bench's difference for the vote": f"{roc_auc_score(tr[tr[f'{k}b_ok']].vote, tr[tr[f'{k}b_ok']][f'{k}b_diff']):.3f}",
                     "Cases with a bench difference": f"{len(ben):,}"})
    md += ["## 5. The mechanism, training terms 2010–15 only", "",
           "The published direction: a justice's pitch is higher toward the side the justice votes against, "
           "so the share should be above 50% and the AUC above 0.5.", "",
           pd.DataFrame(rows).to_markdown(index=False), "",
           "Weights in the last regression (fitted on 2010–16, standardized inputs; + = pro-petitioner): "
           + ", ".join(f"{k} {x:+.3f}" for k, x in w_seco.items()) + ".",
           "With the two pitch features only: " + ", ".join(f"{k} {x:+.3f}" for k, x in w_prim.items()) + ".", ""]
    text = "\n".join(md)
    print(text)
    if not args.dry:
        Path("results/audio_pitch.md").write_text(text)
        print("\nWrote results/audio_pitch.md")


if __name__ == "__main__":
    main()
