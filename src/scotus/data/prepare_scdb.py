"""
Join the cases to the Supreme Court Database (SCDB) and keep only what may be used as
input.

Source: http://scdb.wustl.edu/data.php, modern
database, release SCDB_2026_01 (terms 1946-2025), "docket" unit of analysis:
    data/scdb/raw/SCDB_2026_01_caseCentered_Docket.csv.zip      one row per docket
    data/scdb/raw/SCDB_2026_01_justiceCentered_Docket.csv.zip   one row per docket x justice

Run in isolated mode, because it reads downloaded files (it imports only pandas/numpy):
    python -I -m scotus.data.prepare_scdb
    python -I -m scotus.data.prepare_scdb --final     # only for the final test run, see below

Outputs
    data/scdb_cases.parquet   one row per case in data/cases.parquet: the SCDB match, the
                              argument date and the PRE-DECISION fields only (PRE_DECISION
                              below). No outcome field of the case itself is copied.
    data/scdb_votes.parquet   one row per (SCDB case, justice) for every orally argued SCDB
                              case since 1946: how the justice voted. It is the source for
                              "what did this justice do in EARLIER cases" features
                              (meta_features.py), never for the case being predicted.
    results/scdb_match.md     match rates, label agreement with ConvoKit, leakage audit.
    data/ext/scdb_cases.parquet   the same profile for the 2020-25 cases of prepare_oyez.py,
                              for the walk-forward test. Only their ids are read: who sat on which case, never how anyone voted.

Matching. Our case ids are "<term>_<docket>" (Oyez); SCDB has term and docket.
  1. exact (term, docket) after normalizing punctuation and "ORIG" spellings;
  2. if that fails and the docket is globally unique in form ("10-1491", "128 ORIG"), the
     same docket in a term up to two later or earlier: cases held over for reargument are
     filed by Oyez under the first argument's term and by SCDB under the decision term.
     Old-style dockets ("71") are reused every term, so they get no fallback.
  When one (term, docket) has several SCDB decisions, take the orally argued one, then the
  latest decision.

Test lock. Votes and outcomes dated after HISTORY_END (the end of the validation period)
are dropped from scdb_votes.parquet unless --final is given, so that no feature can be
built from a test-period outcome before the final run. The label audit in the report
reads train and validation cases only.

What is left out on purpose (outcome-derived, would leak the label): decisionType,
caseDisposition, caseDispositionUnusual, partyWinning, declarationUncon,
precedentAlteration, decisionDirection, decisionDirectionDissent, authorityDecision1/2,
voteUnclear, majOpinWriter, majOpinAssigner, splitVote, majVotes, minVotes, dateDecision,
and every per-justice vote field of the case itself.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/scdb/raw")
CASE_FILE = RAW / "SCDB_2026_01_caseCentered_Docket.csv.zip"
VOTE_FILE = RAW / "SCDB_2026_01_justiceCentered_Docket.csv.zip"
HISTORY_END = "2018-09-30"      # last day whose decisions may feed features before --final

# Fields known before the decision. lcDispositionDirection is partly derived by SCDB from
# the Supreme Court's own outcome; see the leakage audit in results/scdb_match.md.
PRE_DECISION = [
    "petitioner", "petitionerState", "respondent", "respondentState", "jurisdiction",
    "adminAction", "adminActionState", "threeJudgeFdc", "caseOrigin", "caseOriginState",
    "caseSource", "caseSourceState", "lcDisagreement", "certReason", "lcDisposition",
    "lcDispositionDirection", "issue", "issueArea", "lawType", "lawSupp",
]


def norm_docket(s):
    """'13-553' -> '13-553'; '128 ORIG', '137, Orig.', 'No. 12, Original', '22O141',
    '128-orig' -> '128orig'; '1 MISC', '1-MISC' -> '1misc'."""
    s = str(s).strip().lower()
    # eight SCDB dockets are written with an en dash ("21\u2013707"), which arrives as
    # three bytes when the file is read as latin-1
    s = re.sub("\u00e2\u0080\u0093|[\u2012-\u2015]", "-", s)
    m = re.fullmatch(r"22o(\d+)", s)
    if m:
        return f"{m.group(1)}orig"
    if "orig" in s:
        d = re.findall(r"\d+", s)
        return f"{d[0]}orig" if d else "orig"
    if "misc" in s:
        d = re.findall(r"\d+", s)
        return f"{d[0]}misc" if d else "misc"
    return re.sub(r"[\s,.]+", "", s)


def read(path):
    return pd.read_csv(path, compression="zip", encoding="latin-1", low_memory=False)


def load_scdb():
    """The two SCDB tables with parsed dates and the join keys (term + normalized docket)."""
    sc = read(CASE_FILE)
    jc = read(VOTE_FILE)
    for df in (sc, jc):
        df["date_decision"] = pd.to_datetime(df.dateDecision, format="%m/%d/%Y")
        df["date_argument"] = pd.to_datetime(df.dateArgument, format="%m/%d/%Y")
        df["date_rearg"] = pd.to_datetime(df.dateRearg, format="%m/%d/%Y")
        df["argued"] = df.date_argument.notna()
    sc["dkey"] = sc.docket.map(norm_docket)
    sc["key"] = sc.term.astype(str) + "_" + sc.dkey
    return sc, jc


def pick(cands):
    """One SCDB docket row among several for the same key: argued, then latest decision."""
    c = cands.sort_values(["argued", "date_decision"], ascending=[False, False])
    return c.iloc[0]


def match_cases(cases, sc):
    by_key = {k: g for k, g in sc.groupby("key")}
    by_docket = {k: g for k, g in sc.groupby("dkey")}
    rows = []
    for cid, term, dk in zip(cases.case_id, cases.term, cases.dkey):
        kind, row, n = "none", None, 0
        g = by_key.get(f"{term}_{dk}")
        if g is not None:
            kind, row, n = "exact", pick(g), g.caseId.nunique()
        elif ("-" in dk or dk.endswith("orig")) and dk in by_docket:
            g = by_docket[dk]
            g = g[(g.term - term).abs() <= 2]
            if len(g):
                # nearest term, later terms first (held over for reargument)
                g = g.assign(gap=(g.term - term).abs(), later=(g.term >= term))
                best = g.sort_values(["gap", "later"], ascending=[True, False]).iloc[0]
                g = g[g.term == best.term]
                kind, row, n = "shifted", pick(g), g.caseId.nunique()
        rec = {"case_id": cid, "match": kind, "n_candidates": n}
        if row is not None:
            rec.update(scdb_case_id=row.caseId, scdb_docket_id=row.docketId,
                       scdb_term=int(row.term), case_name=row.caseName,
                       date_argument=row.date_argument, date_rearg=row.date_rearg,
                       **{f: row[f] for f in PRE_DECISION})
        rows.append(rec)
    return pd.DataFrame(rows)


def justice_map(votes, matched, jc):
    """Our justice id -> SCDB justice code, from who sits on the same matched cases."""
    ours = votes.merge(matched[["case_id", "scdb_docket_id"]], on="case_id")
    theirs = jc[["docketId", "justice", "justiceName"]].rename(
        columns={"docketId": "scdb_docket_id", "justice": "code"})
    pairs = ours.merge(theirs, on="scdb_docket_id")
    co = pairs.groupby(["justice", "code", "justiceName"]).size().rename("n").reset_index()
    co = co.join(ours.groupby("justice").size().rename("total"), on="justice")
    # colleagues sit on the same cases, so co-occurrence alone cannot tell them apart:
    # the surname must agree ("j__john_m_harlan2" ~ "JHarlan2"), then take the most cases
    last = co.justice.str.replace(r"^j__|_jr$", "", regex=True).str.split("_").str[-1]
    co = co[[n.lower().endswith(l) for n, l in zip(co.justiceName, last)]]
    best = co.sort_values("n", ascending=False).drop_duplicates("justice")
    best["share"] = best.n / best.total
    assert set(best.justice) == set(votes.justice), set(votes.justice) - set(best.justice)
    assert best.code.is_unique, "two of our justices map to one SCDB justice"
    assert (best.share > 0.9).all(), best[best.share <= 0.9]
    return best


def profile(cases, sc):
    """The SCDB match, argument date and pre-decision fields of each case (case_id, year,
    split). No outcome field is copied."""
    c = cases.assign(term=cases.case_id.str.split("_").str[0].astype(int),
                     dkey=cases.case_id.str.split("_", n=1).str[1].map(norm_docket))
    out = cases[["case_id", "year", "split"]].merge(match_cases(c, sc), on="case_id")
    out["matched"] = out.match != "none"
    # prediction date: first oral argument; unmatched or undated -> start of the term
    term_start = pd.to_datetime(out.year.astype(str) + "-10-01")
    out["asof"] = out.date_argument.fillna(term_start)
    for f in PRE_DECISION:
        out[f] = out[f].astype("Int64")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ext", default="data/ext",
                    help="the 2020-25 tables of prepare_oyez.py, if the folder exists: their "
                         "cases get a profile too (<ext>/scdb_cases.parquet) and their "
                         "justices a place in the id map. Their vote and winner columns "
                         "are not read.")
    ap.add_argument("--final", action="store_true",
                    help="keep SCDB outcomes after the validation period (final test run only)")
    args = ap.parse_args()
    data, ext = Path(args.data), Path(args.ext)

    cases = pd.read_parquet(data / "cases.parquet")
    votes = pd.read_parquet(data / "votes.parquet")

    sc, jc = load_scdb()

    # ---- 1. match -----------------------------------------------------------------------
    out = profile(cases, sc)
    out.to_parquet(data / "scdb_cases.parquet", index=False)
    sitting, matched_all, out_ext = votes[["case_id", "justice"]], out[out.matched], None
    if (ext / "cases.parquet").exists():
        out_ext = profile(pd.read_parquet(ext / "cases.parquet",
                                          columns=["case_id", "year", "split"]), sc)
        out_ext.to_parquet(ext / "scdb_cases.parquet", index=False)
        sitting = pd.concat([sitting, pd.read_parquet(ext / "votes.parquet",
                                                      columns=["case_id", "justice"])])
        matched_all = pd.concat([matched_all, out_ext[out_ext.matched]])

    # ---- 2. justices and the vote history table -----------------------------------------
    jmap = justice_map(sitting, matched_all, jc)
    code_to_id = dict(zip(jmap.code, jmap.justice))
    hv = jc[jc.argued].drop_duplicates(["caseId", "justice"]).copy()   # first docket of a case
    hv["jid"] = hv.justice.map(code_to_id).fillna("scdb__" + hv.justiceName)
    win = hv.partyWinning.where(hv.partyWinning.isin([0, 1]))
    hv["pet_win"] = win
    hv["pet_vote"] = np.where(hv.majority == 2, win, np.where(hv.majority == 1, 1 - win, np.nan))
    hv["lib_vote"] = hv.direction.map({1: 0.0, 2: 1.0})
    hv["dissent"] = hv.majority.map({1: 1.0, 2: 0.0})
    hist = hv[["caseId", "term", "date_decision", "date_argument", "jid", "pet_vote", "lib_vote",
               "dissent", "pet_win", "issueArea", "issue", "lcDispositionDirection", "petitioner",
               "respondent", "caseSource", "certReason", "jurisdiction"]].rename(
        columns={"caseId": "scdb_case_id", "jid": "justice"})
    if not args.final:
        hist = hist[hist.date_decision <= HISTORY_END]
    hist = hist.sort_values(["date_decision", "scdb_case_id"]).reset_index(drop=True)
    hist.to_parquet(data / "scdb_votes.parquet", index=False)

    # ---- 3. report (train and validation labels only; test labels are never read) -------
    open_ = out[out.split != "test"]
    md = ["# SCDB join (prepare_scdb.py)", "",
          "Source: Supreme Court Database, modern release SCDB_2026_01 (terms 1946-2025), "
          f"docket files; {sc.caseId.nunique():,} cases, {len(sc):,} dockets.", "",
          "## Match rate", "",
          out.groupby("split").agg(cases=("matched", "size"), matched=("matched", "sum"),
                                   exact=("match", lambda s: (s == "exact").sum()),
                                   shifted=("match", lambda s: (s == "shifted").sum()),
                                   has_argument_date=("date_argument", lambda s: s.notna().sum())
                                   ).to_markdown(), "",
          f"Several SCDB decisions for the matched docket: {(out.n_candidates > 1).sum()} cases "
          "(the argued one, then the latest decision, is used).", ""]
    if out_ext is not None:
        md += [f"Extension tables ({ext}, terms {out_ext.year.min()}-{out_ext.year.max()}): "
               f"{int(out_ext.matched.sum())} of {len(out_ext)} cases matched "
               f"({int((out_ext.match == 'exact').sum())} exact), "
               f"{int(out_ext.date_argument.notna().sum())} with an argument date. Profile "
               "only; no outcome of these cases is read here.", ""]

    # label agreement: ConvoKit win_side / votes vs SCDB partyWinning / majority
    lab = open_[open_.matched].merge(cases[["case_id", "win_side"]], on="case_id").merge(
        sc[["docketId", "partyWinning"]], left_on="scdb_docket_id", right_on="docketId")
    lab = lab[lab.partyWinning.isin([0, 1])]
    agree_case = (lab.win_side == lab.partyWinning)
    v = votes[votes.split != "test"].merge(out[["case_id", "scdb_docket_id"]], on="case_id")
    v["code"] = v.justice.map(dict(zip(jmap.justice, jmap.code)))
    v = v.merge(jc[["docketId", "justice", "majority", "partyWinning"]].rename(
        columns={"docketId": "scdb_docket_id", "justice": "code"}),
        on=["scdb_docket_id", "code"], how="left")
    ok = v.partyWinning.isin([0, 1]) & v.majority.isin([1, 2])
    sc_vote = np.where(v.majority == 2, v.partyWinning, 1 - v.partyWinning)
    agree_vote = (v.vote == sc_vote)[ok]
    md += ["## Label agreement with ConvoKit (train and validation only)", "",
           f"- Case winner: {agree_case.mean():.4f} of {len(lab):,} matched cases with a clear "
           "SCDB winner.",
           f"- Justice votes: {agree_vote.mean():.4f} of {int(ok.sum()):,} votes found in SCDB "
           f"({(~ok).sum():,} of our votes have no usable SCDB vote).", "",
           "By split:", "",
           pd.DataFrame({"case winner": agree_case.groupby(lab.split).mean(),
                         "votes": agree_vote.groupby(v.split[ok]).mean()}).round(4).to_markdown(),
           ""]

    # justices
    md += ["## Justice id map (by co-occurrence on matched cases)", "",
           jmap[["justice", "justiceName", "total", "share"]].sort_values("justice")
           .round(3).to_markdown(index=False), ""]

    # leakage audit on train: petitioner win rate by value of each pre-decision field
    tr = out[(out.split == "train") & out.matched].merge(cases[["case_id", "win_side"]],
                                                         on="case_id")
    base = tr.win_side.mean()
    md += ["## Leakage audit (train cases only)", "",
           f"Petitioner win rate overall: {base:.3f}. A pre-decision field should not have a "
           "value that almost fixes the winner. Largest deviations among values with at "
           "least 40 train cases:", ""]
    rows = []
    for f in PRE_DECISION:
        g = tr.groupby(tr[f].astype("float").fillna(-1)).win_side.agg(["mean", "size"])
        g = g[g["size"] >= 40]
        if len(g):
            lo, hi = g["mean"].idxmin(), g["mean"].idxmax()
            rows.append({"field": f, "values (>=40)": len(g),
                         "lowest win rate": f"{g['mean'][lo]:.3f} (value {lo:g}, n {g['size'][lo]})",
                         "highest win rate": f"{g['mean'][hi]:.3f} (value {hi:g}, n {g['size'][hi]})"})
    md += [pd.DataFrame(rows).to_markdown(index=False), "", "(value -1 = missing)", ""]
    d = tr.assign(lc=tr.lcDispositionDirection.astype("float").fillna(-1).map(
        {1: "conservative", 2: "liberal", 3: "unspecifiable", -1: "missing"}),
        decade=tr.year // 10 * 10)
    md += ["Lower-court direction x decade (petitioner win rate, n):", "",
           d.pivot_table(index="decade", columns="lc", values="win_side",
                         aggfunc=lambda s: f"{s.mean():.2f} ({len(s)})").fillna("").to_markdown(),
           ""]
    # is the lower-court direction recoverable from pre-decision facts alone?
    crim = tr[tr.issueArea == 1]
    gov_pet = crim.petitioner.isin([27, 28])          # United States, a State
    gov_resp = crim.respondent.isin([27, 28])
    md += ["Is the lower-court direction a pre-decision fact? In criminal-procedure cases "
           "it should follow from who lost below, whatever the Supreme Court then did:", "",
           f"- government (United States or a State) is the petitioner, so the defendant won "
           f"below: lower court coded liberal in "
           f"{(crim[gov_pet].lcDispositionDirection == 2).mean():.3f} of {int(gov_pet.sum())} cases;",
           f"- government is the respondent, so it won below: lower court coded conservative "
           f"in {(crim[gov_resp].lcDispositionDirection == 1).mean():.3f} of "
           f"{int(gov_resp.sum())} cases.", ""]
    Path("results/scdb_match.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"history rows: {len(hist):,}  (decisions through {hist.date_decision.max().date()})")


if __name__ == "__main__":
    main()
