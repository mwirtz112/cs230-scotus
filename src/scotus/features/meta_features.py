"""
Case metadata and voting-history features from the SCDB join (prepare_scdb.py).

    python -m scotus.features.meta_features      -> data/meta_cat.parquet, data/meta_hist.parquet

Two kinds of feature, both knowable before the decision:

1. Case profile (data/meta_cat.parquet, one row per case): the SCDB pre-decision fields,
   cleaned. CAT_FIELDS are categorical codes for embeddings or trees.
     - "No merits" issue codes (90150-90200: writ dismissed as improvidently granted,
       want of jurisdiction, ...) describe how the Court disposed of the case, not what
       the case was about (petitioner win rate 1% for 90150). Those cases get issue and
       issue area "unknown". None occurs in validation or test.
     - lawType / lawSupp are dropped altogether (they carry the same disposition codes).
     - Lower-court direction: conservative / liberal; "unspecifiable" and missing are one
       "unknown" value.

2. Voting history (data/meta_hist.parquet, one row per case x justice, per protocol): what
   the justice, the bench and the Court did in EARLIER decisions, from every orally argued
   SCDB case since 1946 (not only the cases that have a transcript).
   A decision counts only if it was handed down before the "as of" date of the case:
     rolling  as of = the day of the case's first oral argument (walk-forward, as in
              Katz et al. 2017);
     frozen   as of = min(argument day, the end of the fold's training period): nothing
              that happened after the training cutoff is used. Train rows are identical
              under both protocols; only held-out rows differ.
   Rates are exponentially down-weighted with age (HALF_LIFE_YEARS) and shrunk towards
   their parent rate with PSEUDO pseudo-counts (justice x area x direction -> justice x
   direction -> justice -> all votes).

The history table (data/scdb_votes.parquet) stops at the end of the validation period
unless prepare_scdb.py was run with --final, so no test-period outcome can enter.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HALF_LIFE_YEARS = 5.0
PSEUDO = 10.0
NO_MERITS = (90150, 90200)          # issue codes that restate the disposition

# categorical case fields offered to the models (0 = unknown everywhere)
CAT_FIELDS = ["issue_area", "lc_dir", "lc_disp", "lc_disagree", "cert_reason", "jurisdiction",
              "pet", "resp", "pet_group", "resp_group", "source", "origin", "source_group",
              "admin", "three_judge", "arg_month"]
# the three fields SCDB codes from the Court's opinion rather than from the record
OPINION_CODED = ["lc_dir", "lc_disagree", "cert_reason"]

HIST_FEATURES = ["h_exp", "h_pet", "h_lib", "h_dis", "h_pet_lc", "h_pet_area", "h_pet_area_lc",
                 "h_ideo", "b_pet", "b_dis", "b_pet_lc", "b_pet_area_lc", "b_ideo",
                 "d_pet_lc", "d_pet_area_lc", "d_ideo",
                 "c_base", "c_lc", "c_area_lc", "c_source", "c_pet", "c_resp"]

FOLD_CUTOFF = {"val": 2015, "dev": 2011, "dev2": 2007, "nogap": 2013, "newj_gap": 2008,
               "newj_nogap": 2009,      # last training term of each fold in train_v1.FOLDS
               # walk-forward steps (walk_forward.py): "wf2018" = what was known when
               # the 2018 term opened. A step is built only if the history table reaches it.
               **{f"wf{t}": t - 1 for t in range(2012, 2026)}}


def party_group(code):
    """SCDB party codes: 1-26 and 28 state/local government bodies and officials, 27 the
    United States, 100-299 private parties, 300+ federal agencies and officials."""
    c = code.astype("float")
    g = np.select([c.isna(), c == 27, c == 28, c < 100, c < 300], [0, 1, 2, 3, 4], 5)
    return pd.Series(g, index=code.index)


def court_group(code):
    """caseSource / caseOrigin: 21-32 federal courts of appeals, 300+ state courts, other
    low codes special federal courts, 41-299 federal district courts."""
    c = code.astype("float")
    g = np.select([c.isna(), c.between(21, 32), c >= 300, c < 21], [0, 1, 2, 3], 4)
    return pd.Series(g, index=code.index)


def case_profile(sc):
    """Cleaned categorical fields, one row per case; 0 = unknown / not applicable."""
    no_merits = sc.issue.between(*NO_MERITS).fillna(False)
    code = lambda s: s.astype("float").fillna(0).astype(int)
    out = pd.DataFrame({
        "case_id": sc.case_id,
        "issue_area": code(sc.issueArea.where(~no_merits)),
        "issue": code(sc.issue.where(~no_merits)),
        "lc_dir": code(sc.lcDispositionDirection.where(sc.lcDispositionDirection.isin([1, 2]))),
        "lc_disp": code(sc.lcDisposition),
        "lc_disagree": code(sc.lcDisagreement + 1),
        "cert_reason": code(sc.certReason),
        "jurisdiction": code(sc.jurisdiction),
        "pet": code(sc.petitioner), "resp": code(sc.respondent),
        "pet_group": party_group(sc.petitioner), "resp_group": party_group(sc.respondent),
        "source": code(sc.caseSource), "origin": code(sc.caseOrigin),
        "source_group": court_group(sc.caseSource),
        "admin": code(sc.adminAction),
        "three_judge": code(sc.threeJudgeFdc + 1),
        "arg_month": sc.date_argument.dt.month.fillna(0).astype(int),
        "matched": sc.matched.to_numpy(),
    })
    return out


class AsOf:
    """Weighted sums over a key's past decisions, strictly before a query date.

    hist_keys / query_keys: DataFrames with the same columns. Decisions are weighted by
    2 ** (age / half_life); the common factor cancels in a rate, and `n` is rescaled to
    the query date, so `n` is an effective count.
    """

    def __init__(self, hist_keys, dates, values, half_life_years=HALF_LIFE_YEARS):
        self.cols = list(hist_keys.columns)
        self.h = half_life_years * 365.25
        day = (dates - pd.Timestamp("1946-01-01")).dt.days.to_numpy()
        key = pd.MultiIndex.from_frame(hist_keys.astype("int64"))
        self.codes, self.uniques = pd.factorize(key)
        order = np.lexsort((day, self.codes))
        self.codes, self.day = self.codes[order], day[order]
        self.comb = self.codes.astype(np.int64) * 100_000 + self.day
        w = 2.0 ** (self.day / self.h)
        self.cum = {}
        for name, v in values.items():
            v = np.asarray(v, dtype=float)[order]
            ok = ~np.isnan(v)
            self.cum[name] = (np.concatenate([[0], np.cumsum(np.where(ok, w * v, 0))]),
                              np.concatenate([[0], np.cumsum(np.where(ok, w, 0))]))
        self.start = np.searchsorted(self.codes, np.arange(len(self.uniques)))

    def query(self, query_keys, dates):
        day = (dates - pd.Timestamp("1946-01-01")).dt.days.to_numpy()
        key = pd.MultiIndex.from_frame(query_keys[self.cols].astype("int64"))
        code = self.uniques.get_indexer(key)                   # -1: key never seen
        known = code >= 0
        c = np.where(known, code, 0)
        pos = np.searchsorted(self.comb, c.astype(np.int64) * 100_000 + day, side="left")
        lo = self.start[c]
        scale = 2.0 ** (-day / self.h)
        out = {}
        for name, (cs, cn) in self.cum.items():
            s = np.where(known, cs[pos] - cs[lo], 0.0) * scale
            n = np.where(known, cn[pos] - cn[lo], 0.0) * scale
            out[name] = (s, n)
        return out


def shrink(stat, parent, m=PSEUDO):
    s, n = stat
    return (s + m * parent) / (n + m)


def history(votes, prof, asof, hist):
    """History features for every row of `votes` (case_id, justice), as of `asof` (a
    Series indexed like votes: one date per row)."""
    v = votes[["case_id", "justice"]].merge(prof, on="case_id", how="left")
    jcode = {j: i for i, j in enumerate(sorted(set(hist.justice) | set(v.justice)))}
    hk = pd.DataFrame({"j": hist.justice.map(jcode), "area": hist.issue_area,
                       "lc": hist.lc_dir, "all": 0})
    qk = pd.DataFrame({"j": v.justice.map(jcode), "area": v.issue_area, "lc": v.lc_dir,
                       "all": 0})
    vals = {"pet": hist.pet_vote, "lib": hist.lib_vote, "dis": hist.dissent}
    t = pd.Series(asof.to_numpy())

    def q(cols):
        return AsOf(hk[cols], hist.date_decision, vals).query(qk, t)

    g = q(["all"])
    glob_pet = shrink(g["pet"], 0.6, 1.0)
    glob_lib = shrink(g["lib"], 0.5, 1.0)
    J, JL, JA, JAL = q(["j"]), q(["j", "lc"]), q(["j", "area"]), q(["j", "area", "lc"])
    out = pd.DataFrame({"case_id": v.case_id, "justice": v.justice})
    out["h_exp"] = np.log1p(J["pet"][1])
    out["h_pet"] = shrink(J["pet"], glob_pet)
    out["h_lib"] = shrink(J["lib"], glob_lib)
    out["h_dis"] = shrink(J["dis"], shrink(g["dis"], 0.2, 1.0))     # how often j dissents
    lc_known = (v.lc_dir > 0).to_numpy()
    area_known = (v.issue_area > 0).to_numpy()
    out["h_pet_lc"] = np.where(lc_known, shrink(JL["pet"], out.h_pet), out.h_pet)
    out["h_pet_area"] = np.where(area_known, shrink(JA["pet"], out.h_pet), out.h_pet)
    out["h_pet_area_lc"] = np.where(lc_known & area_known, shrink(JAL["pet"], out.h_pet_lc),
                                    out.h_pet_lc)
    # expected vote from the justice's liberal rate in this area: a liberal vote is a
    # vote to reverse a conservative lower-court ruling and to affirm a liberal one
    lib_area = np.where(area_known, shrink(JA["lib"], out.h_lib), out.h_lib)
    out["h_ideo"] = np.select([(v.lc_dir == 1).to_numpy(), (v.lc_dir == 2).to_numpy()],
                              [lib_area, 1 - lib_area], out.h_pet)
    # the bench: mean over the justices voting in the case, and the justice's distance
    for name in ["pet", "dis", "pet_lc", "pet_area_lc", "ideo"]:
        bench = out.groupby("case_id")[f"h_{name}"].transform("mean")
        out[f"b_{name}"] = bench
        if name not in ("pet", "dis"):
            out[f"d_{name}"] = out[f"h_{name}"] - bench

    # case level: how the Court treated similar cases, from case outcomes (one row a case)
    hc = hist.drop_duplicates("scdb_case_id")
    ck = pd.DataFrame({"area": hc.issue_area, "lc": hc.lc_dir, "source": hc.source,
                       "pet": hc.pet, "resp": hc.resp, "all": 0})
    qc = pd.DataFrame({"area": v.issue_area, "lc": v.lc_dir, "source": v.source,
                       "pet": v["pet"], "resp": v.resp, "all": 0})

    def qcase(cols):
        return AsOf(ck[cols], hc.date_decision, {"win": hc.pet_win}).query(qc, t)["win"]

    out["c_base"] = shrink(qcase(["all"]), 0.6, 1.0)
    out["c_lc"] = shrink(qcase(["lc"]), out.c_base)
    out["c_area_lc"] = shrink(qcase(["area", "lc"]), out.c_lc)
    out["c_source"] = shrink(qcase(["source"]), out.c_base)
    out["c_pet"] = shrink(qcase(["pet"]), out.c_base)
    out["c_resp"] = shrink(qcase(["resp"]), out.c_base)
    return out


def load_history(data):
    """SCDB votes with the same cleaned keys as the case profile."""
    h = pd.read_parquet(data / "scdb_votes.parquet")
    no_merits = h.issue.between(*NO_MERITS).fillna(False)
    code = lambda s: s.astype("float").fillna(0).astype(int)
    return pd.DataFrame({
        "scdb_case_id": h.scdb_case_id, "date_decision": h.date_decision,
        "justice": h.justice, "pet_vote": h.pet_vote, "lib_vote": h.lib_vote,
        "dissent": h.dissent, "pet_win": h.pet_win,
        "issue_area": code(h.issueArea.where(~no_merits)),
        "lc_dir": code(h.lcDispositionDirection.where(h.lcDispositionDirection.isin([1, 2]))),
        "source": code(h.caseSource), "pet": code(h.petitioner), "resp": code(h.respondent)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ext", default="data/ext",
                    help="the 2020-25 tables (prepare_oyez.py + prepare_scdb.py), if present")
    args = ap.parse_args()
    data, ext = Path(args.data), Path(args.ext)

    # who sat on which case; the vote column is not read
    sc = pd.read_parquet(data / "scdb_cases.parquet")
    votes = pd.read_parquet(data / "votes.parquet", columns=["case_id", "year", "justice"])
    if (ext / "scdb_cases.parquet").exists():
        sc = pd.concat([sc, pd.read_parquet(ext / "scdb_cases.parquet")], ignore_index=True)
        votes = pd.concat([votes, pd.read_parquet(ext / "votes.parquet",
                                                  columns=["case_id", "year", "justice"])],
                          ignore_index=True)
    prof = case_profile(sc)
    prof.to_parquet(data / "meta_cat.parquet", index=False)

    hist = load_history(data)
    last = hist.date_decision.max()
    asof = votes.case_id.map(sc.set_index("case_id")["asof"])
    # rows whose own period is after the history table ends would get stale features that
    # look "rolling": only build them when the table covers them (prepare_scdb.py --final)
    covered = asof <= last
    frames = [history(votes[covered], prof, asof[covered], hist).assign(protocol="rolling")]
    not_reached = []
    for fold, last_term in FOLD_CUTOFF.items():
        cap = pd.Timestamp(f"{last_term + 1}-10-01")       # the next term's first day
        if last < pd.Timestamp(f"{last_term + 1}-06-01"):
            # the history table stops before this cutoff (it ends with the validation
            # period unless prepare_scdb.py --final was run): "frozen at the cutoff" would
            # silently mean "frozen earlier", so the fold is not built
            not_reached.append(fold)
            continue
        held = votes.year > last_term                      # test rows too: frozen is safe
        f = history(votes[held], prof, asof[held].clip(upper=cap), hist)
        frames.append(f.assign(protocol=f"frozen_{fold}"))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(data / "meta_hist.parquet", index=False)
    print(f"profiles: {len(prof):,} cases; history rows: {len(out):,}; "
          f"history table ends {last.date()}; rolling rows not built: {int((~covered).sum()):,}; "
          f"folds not built (history table ends before them): {', '.join(not_reached) or 'none'}")
    print(out[out.protocol == "rolling"][HIST_FEATURES].describe().T.round(3).to_string())


def load(data="data", fold="val", protocol="frozen"):
    """(profile per case, history per case x justice) for one fold.

    protocol "frozen": train rows as of their argument day, held-out rows as of the fold's
    training cutoff. "rolling": every row as of its argument day.
    """
    data = Path(data)
    prof = pd.read_parquet(data / "meta_cat.parquet")
    h = pd.read_parquet(data / "meta_hist.parquet")
    roll = h[h.protocol == "rolling"].drop(columns="protocol")
    if protocol == "rolling":
        return prof, roll
    held = h[h.protocol == f"frozen_{fold}"].drop(columns="protocol")
    keep = ~roll.case_id.isin(held.case_id)
    return prof, pd.concat([roll[keep], held], ignore_index=True)


if __name__ == "__main__":
    main()
