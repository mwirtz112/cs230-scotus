"""
Turn, case and vote tables for the terms after 2019, from the Oyez JSON downloaded by
scrape_oyez.py, in exactly the format prepare_data.py builds from ConvoKit.

    python -I -m scotus.data.prepare_oyez --check 2017
        parser check: rebuild the downloaded sample of a term that ConvoKit also has and
        compare turn by turn with data/turns.parquet (and the votes with data/votes.parquet)
    python -I -m scotus.data.prepare_oyez --terms 2020 2021 2022 2023 2024 2025
        -> data/ext/turns.parquet, cases.parquet, votes.parquet, results/oyez_extension.md

(-I: isolated mode, because it reads downloaded files. The only project module it imports
is prepare_scdb.)

How a transcript becomes turns (the ConvoKit conventions, reproduced):
  * one row per speaker turn, in (section, turn) order; turn_idx counts every turn of
    the transcript, empty ones included (they are then dropped);
  * text = the turn's text blocks joined by newlines;
  * a justice is "j__<oyez identifier>", an advocate "<oyez identifier>", a turn without
    a speaker record "<INAUDIBLE>" (role "unknown");
  * convo_id = the Oyez id of the argument audio; case_id = "<term>_<docket>".
Advocate side, from Oyez's one-line description of whom the advocate argued for:
    "amicus" / "amici" -> 2; else "petitioner" / "appellant" -> 1;
    else "respondent" / "appellee" -> 0; else unknown.
  (Checked on the 12,710 described advocates of the corpus up to 2017: 99.8% the same
   side; the rest are "amici" in the plural. "plaintiff" / "defendant" (original
   jurisdiction) and "applicant" (argued emergency applications) are also read as
   petitioner / respondent; counsel appointed to defend "the judgment below" is amicus.)
  A description that only names a party ("for California, et al.") takes the side of the
  case's first or second party when the name is part of exactly one of them. An advocate who speaks but
  is not on Oyez's list gets a side only when the order of argument leaves no doubt.

Labels. ConvoKit took the votes from the Supreme Court Database; so does this script
(prepare_scdb.py's join on term + docket). vote = 1 if the justice voted for the
petitioner: in the majority of a petitioner win, or dissenting from a respondent win.
Cases without a clear winner, without justice turns or without usable votes are dropped,
as in prepare_data.py.

LOCK. The Oyez case files also say how each case was decided. This script never reads
those fields (decisions, conclusion, decided_by, written_opinion, opinion_announcement).
The new cases get split = "ext" and live in data/ext/, apart from the main tables. They
are test terms of the walk-forward evaluation: only walk_forward.py --final reads their
labels.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scotus.data import prepare_scdb as P

RAW = Path("data/oyez/raw")
SIDE_ROLE = {1: "petitioner", 0: "respondent", 2: "amicus"}
OUTCOME_FIELDS = ("decisions", "conclusion", "decided_by", "written_opinion",
                  "opinion_announcement")                    # never read


def side_of(description):
    d = (description or "").lower()
    if "amicus" in d or "amici" in d or "judgment below" in d:   # court-appointed counsel
        return 2
    if re.search(r"petitioner|appellant|plaintiff|applicant", d):
        return 1
    if re.search(r"respondent|appellee|defendant", d):
        return 0
    return 3


def norm_party(text):
    t = re.sub(r"<[^>]+>", " ", (text or "").lower())
    t = re.sub(r"\bet al\b\.?", " ", t)
    t = re.sub(r"^\s*(for|on behalf of)\s+", "", t.strip())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"^the ", "", t.strip())).strip()


def party_side(description, case):
    """Side for a description that only names a party ("for California, et al."): the
    side of the case's first or second party if the name is part of exactly one of them
    ("california" is in "state of california", not in "state of texas")."""
    generic = {"state", "of", "the", "commonwealth", "city", "county", "and"}
    d = set(norm_party(description).split()) - generic
    hits = [label for party, label in (("first_party", "first_party_label"),
                                       ("second_party", "second_party_label"))
            if d and d <= set(norm_party(case.get(party)).split())]
    return side_of(case.get(hits[0])) if len(hits) == 1 else 3


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def load_case(path):
    case = json.loads(Path(path).read_text())
    for f in OUTCOME_FIELDS:                                 # make misuse impossible
        case.pop(f, None)
    return case


MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"])}


def argument_day(audio):
    """Date in an audio title such as 'Oral Reargument - October 03, 2017' (or None)."""
    m = re.search(r"([A-Za-z]+) (\d{1,2}), (\d{4})", audio.get("title") or "")
    if not m or m.group(1).lower() not in MONTHS:
        return None
    return pd.Timestamp(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)),
                        tz="UTC").timestamp()


def is_justice(speaker, day, bench):
    """A justice sitting on the day of the argument. Oyez gives each justice role a start
    and an end date (0 = still serving), so a later justice who argues as counsel stays
    an advocate, and a justice who joined before a reargument counts as one. Without a
    date in the audio title, fall back on the bench listed for the case."""
    roles = [r for r in speaker.get("roles") or []
             if isinstance(r, dict) and r.get("type") == "scotus_justice"]
    if not roles:
        return False
    if day is None:
        return not bench or speaker.get("identifier") in bench
    return any((r.get("date_start") or 0) <= day + 86400 and
               (not r.get("date_end") or day <= r["date_end"] + 86400) for r in roles)


def infer_sides(rows):
    """Sides for advocates Oyez does not list or describes without a side, from the fixed
    order of argument (ConvoKit marks such advocates "inferred" too). Only two safe cases:
    the first advocate to speak is the petitioner's; a single remaining unknown advocate,
    when nobody else in the transcript speaks for the respondent, is the respondent's."""
    adv = [r for r in rows if r["speaker_type"] == "A"]
    order = list(dict.fromkeys(r["speaker_id"] for r in adv))
    side = {}
    for r in adv:
        side.setdefault(r["speaker_id"], r["side"])
    unknown = [a for a in order if side[a] not in (0, 1, 2)]
    fixed = {}
    if unknown and order[0] == unknown[0] and 1 not in side.values():
        fixed[unknown[0]] = 1
    rest = [a for a in unknown if a not in fixed]
    taken = set(side.values()) | set(fixed.values())
    if len(rest) == 1 and 0 not in taken and 1 in taken:
        fixed[rest[0]] = 0
    for r in adv:
        if r["speaker_id"] in fixed:
            r["side"] = fixed[r["speaker_id"]]
            r["role"] = SIDE_ROLE[r["side"]]
    return len(fixed)


def case_rows(case, term):
    """(turn rows, transcript ids, advocate table) for one case."""
    case_id = f"{term}_{case['docket_number']}"
    sides, adv = {}, []
    for a in case.get("advocates") or []:
        who = (a or {}).get("advocate") or {}
        if who.get("identifier"):
            side = side_of(a.get("advocate_description"))
            if side == 3:
                side = party_side(a.get("advocate_description"), case)
            sides[who["identifier"]] = side
            adv.append({"case_id": case_id, "advocate": who["identifier"],
                        "description": a.get("advocate_description"),
                        "side": sides[who["identifier"]]})
    bench = {m.get("identifier") for h in case.get("heard_by") or [] if h
             for m in h.get("members") or [] if m}
    rows, convos, n_inferred = [], [], 0
    for audio in case.get("oral_argument_audio") or []:
        if not audio or audio.get("unavailable"):
            continue
        path = RAW / f"media_{safe(audio['id'])}.json"
        if not path.exists():
            continue
        transcript = (json.loads(path.read_text()) or {}).get("transcript") or {}
        day = argument_day(audio)
        i = -1
        n_before = len(rows)
        for section in transcript.get("sections") or []:
            for turn in section.get("turns") or []:
                i += 1
                text = "\n".join((b.get("text") or "") for b in turn.get("text_blocks") or [])
                text = text.strip()
                if not text:
                    continue
                sp = turn.get("speaker")
                if not sp or not sp.get("identifier"):
                    who, kind, side, role = "<INAUDIBLE>", None, np.nan, "unknown"
                elif is_justice(sp, day, bench):
                    who, kind, side, role = "j__" + sp["identifier"], "J", np.nan, "justice"
                else:
                    who, kind = sp["identifier"], "A"
                    side = sides.get(sp["identifier"], np.nan)
                    role = SIDE_ROLE.get(side, "advocate_unknown")
                rows.append({"case_id": case_id, "convo_id": str(audio["id"]), "year": term,
                             "turn_idx": i, "speaker_id": who, "speaker_type": kind,
                             "side": side, "role": role, "text": text})
        if len(rows) > n_before:
            convos.append(str(audio["id"]))
            n_inferred += infer_sides(rows[n_before:])
    return rows, convos, adv, n_inferred


def load_term(term):
    """All downloaded cases of a term -> turns, {case_id: [convo ids]}, advocates."""
    listing = json.loads((RAW / f"term_{term}.json").read_text())
    turns, convos, advocates, n_files = [], {}, [], 0
    for c in listing:
        path = RAW / f"case_{term}_{safe(c.get('docket_number'))}.json"
        if not path.exists():
            continue
        n_files += 1
        rows, ids, adv, _ = case_rows(load_case(path), term)
        if ids:
            turns += rows
            convos[f"{term}_{c['docket_number']}"] = ids
        advocates += adv
    return pd.DataFrame(turns), convos, pd.DataFrame(advocates), len(listing), n_files


def scdb_labels(case_ids, justice_ids):
    """SCDB winner and votes for our case ids. Returns (cases table, votes table)."""
    sc, jc = P.load_scdb()
    cases = pd.DataFrame({"case_id": case_ids})
    cases["term"] = cases.case_id.str.split("_").str[0].astype(int)
    cases["dkey"] = cases.case_id.str.split("_", n=1).str[1].map(P.norm_docket)
    m = P.match_cases(cases, sc)
    m = m[m.match != "none"].merge(sc[["docketId", "partyWinning"]],
                                   left_on="scdb_docket_id", right_on="docketId")
    surname = {j.replace("j__", "").removesuffix("_jr").split("_")[-1]: j for j in justice_ids}
    v = m[["case_id", "scdb_docket_id"]].merge(
        jc[["docketId", "justiceName", "majority", "partyWinning"]],
        left_on="scdb_docket_id", right_on="docketId")
    v = v[v.partyWinning.isin([0, 1]) & v.majority.isin([1, 2])].copy()
    v["justice"] = [next((j for s, j in surname.items() if n.lower().endswith(s)), None)
                    for n in v.justiceName]
    unmapped = sorted(set(v.justiceName[v.justice.isna()]))
    assert not unmapped, f"SCDB justices without an Oyez id: {unmapped}"
    v["vote"] = np.where(v.majority == 2, v.partyWinning, 1 - v.partyWinning).astype(int)
    return m, v[["case_id", "justice", "vote"]].drop_duplicates(["case_id", "justice"])


def build(terms, known_justices):
    """Tables for the given terms, with prepare_data.py's filters."""
    frames, convos, advs, listed, files = [], {}, [], {}, {}
    for t in terms:
        turns, c, adv, n_list, n_files = load_term(t)
        frames.append(turns); convos.update(c); advs.append(adv)
        listed[t], files[t] = n_list, n_files
    turns = pd.concat(frames, ignore_index=True)
    justices = set(turns.speaker_id[turns.role == "justice"]) | set(known_justices)
    matched, votes = scdb_labels(sorted(convos), justices)
    win = matched.set_index("case_id").partyWinning
    n_votes = votes.groupby("case_id").size()
    has_j = set(turns.case_id[turns.role == "justice"])
    cases, dropped = [], defaultdict(Counter)
    for case_id, ids in convos.items():
        term = int(case_id.split("_")[0])
        if case_id not in win.index:
            dropped[term]["not in SCDB"] += 1
        elif win[case_id] not in (0, 1):
            dropped[term]["unclear winner"] += 1
        elif case_id not in has_j:
            dropped[term]["no justice turns"] += 1
        elif n_votes.get(case_id, 0) == 0:
            dropped[term]["no usable votes"] += 1
        else:
            cases.append({"case_id": case_id, "year": term, "split": "ext",
                          "win_side": int(win[case_id]), "n_transcripts": len(ids),
                          "convo_ids": ids, "n_votes": int(n_votes[case_id])})
    cases = pd.DataFrame(cases)
    turns = turns[turns.case_id.isin(cases.case_id)].reset_index(drop=True)
    votes = votes[votes.case_id.isin(cases.case_id)].merge(
        cases[["case_id", "year", "split"]], on="case_id")[
        ["case_id", "year", "split", "justice", "vote"]].reset_index(drop=True)
    return turns, cases, votes, pd.concat(advs, ignore_index=True), dropped, listed, files


def check(term, data):
    """Rebuild a term ConvoKit has from the Oyez files and compare."""
    ours = pd.read_parquet(data / "turns.parquet")
    votes_ck = pd.read_parquet(data / "votes.parquet")
    turns, cases, votes, adv, dropped, listed, files = build([term], set(votes_ck.justice))
    ours = ours[ours.convo_id.isin(set(turns.convo_id))]
    both = sorted(set(ours.convo_id))
    print(f"term {term}: {files[term]} case files, {turns.convo_id.nunique()} transcripts "
          f"rebuilt, {len(both)} of them are in data/turns.parquet")
    cols = ["turn_idx", "speaker_id", "speaker_type", "role"]
    same_n = same_meta = same_side = same_text = n_rows = text_rows = 0
    examples = []
    for cid in both:
        a = ours[ours.convo_id == cid].reset_index(drop=True)
        b = turns[turns.convo_id == cid].reset_index(drop=True)
        if len(a) != len(b):
            examples.append(f"  {cid}: {len(a)} turns in ConvoKit, {len(b)} rebuilt")
            continue
        same_n += 1
        n_rows += len(a)
        meta = (a[cols].fillna("-") == b[cols].fillna("-")).all(axis=1)
        side = (a.side.fillna(-1).to_numpy() == b.side.fillna(-1).to_numpy())
        text = (a.text.to_numpy() == b.text.to_numpy())
        same_meta += int(meta.sum()); same_side += int(side.sum()); same_text += int(text.sum())
        for i in np.flatnonzero(~(meta.to_numpy() & side))[:2]:
            examples.append(f"  {cid} turn {i}: ConvoKit {a.loc[i, cols + ['side']].tolist()} "
                            f"vs rebuilt {b.loc[i, cols + ['side']].tolist()}")
        for i in np.flatnonzero(~text)[:1]:
            examples.append(f"  {cid} turn {i} text: {a.text[i][:70]!r} vs {b.text[i][:70]!r}")
    print(f"  transcripts with the same number of turns: {same_n}/{len(both)}")
    print(f"  turns compared: {n_rows:,}; same index/speaker/type/role: {same_meta:,}; "
          f"same side: {same_side:,}; same text: {same_text:,}")
    v = votes.merge(votes_ck, on=["case_id", "justice"], suffixes=("", "_ck"))
    print(f"  votes: {len(votes):,} rebuilt for {votes.case_id.nunique()} cases, "
          f"{len(v):,} also in data/votes.parquet, same vote: {(v.vote == v.vote_ck).sum():,}")
    kept_ck = set(pd.read_parquet(data / "cases.parquet").case_id)
    print(f"  cases kept by both filters: {len(set(cases.case_id) & kept_ck)} of "
          f"{len(cases)} rebuilt; dropped here: {dict(dropped[term])}")
    print("\n".join(examples[:12]) if examples else "  no differences")


def term_stats(turns, cases, votes):
    """Input statistics per term (no outcomes): how long and how busy the arguments are."""
    jt = turns[turns.role == "justice"]
    spoke = jt.groupby("case_id").speaker_id.apply(set)
    votes = votes.assign(speaks=[j in spoke.get(c, set())
                                 for c, j in zip(votes.case_id, votes.justice)])
    words = turns.text.str.split().str.len()
    rows = []
    for term, c in cases.groupby("year"):
        m = turns.case_id.isin(c.case_id)
        t, w = turns[m], words[m]
        v = votes[votes.case_id.isin(c.case_id)]
        a = t[t.speaker_type == "A"]
        thomas = v[v.justice == "j__clarence_thomas"]
        rows.append({
            "term": term, "cases": len(c), "transcripts": int(c.n_transcripts.sum()),
            "votes": len(v), "turns": len(t),
            "median turns / case": int(t.groupby("case_id").size().median()),
            "median words / case": int(w.groupby(t.case_id).sum().median()),
            "justice turns %": round(100 * (t.role == "justice").mean(), 1),
            "median words / justice turn": int(w[t.role == "justice"].median()),
            "unknown speaker %": round(100 * (t.role == "unknown").mean(), 2),
            "advocate side unknown %": round(100 * (a.role == "advocate_unknown").mean(), 2),
            "voting justices who speak %": round(100 * v.speaks.mean(), 1),
            "Thomas speaks, % of his cases": (round(100 * thomas.speaks.mean(), 1)
                                              if len(thomas) else ""),
        })
    return pd.DataFrame(rows)


def report(turns, cases, votes, adv, dropped, listed, files, data):
    stats = term_stats(turns, cases, votes)
    stats.insert(1, "listed by Oyez", [listed[t] for t in stats.term])
    stats.insert(2, "argued, downloaded", [files[t] for t in stats.term])
    stats.insert(4, "dropped", [json.dumps(dict(dropped[t])) for t in stats.term])
    old_cases = pd.read_parquet(data / "cases.parquet")
    old_cases = old_cases[old_cases.year >= 2012]
    old_turns = pd.read_parquet(data / "turns.parquet")
    old_turns = old_turns[old_turns.case_id.isin(old_cases.case_id)]
    old_votes = pd.read_parquet(data / "votes.parquet", columns=["case_id", "justice"])
    before = term_stats(old_turns, old_cases, old_votes[old_votes.case_id.isin(old_cases.case_id)])
    bench = (votes.groupby("year").justice.apply(lambda s: ", ".join(
        sorted(j.replace("j__", "").split("_")[-1] if not j.endswith("_jr")
               else j.replace("j__", "").split("_")[-2] for j in set(s)))))
    unknown = adv[adv.side == 3].description.fillna("(none)").value_counts()
    md = ["# Oyez extension: terms after 2019 (prepare_oyez.py)", "",
          "Transcripts from api.oyez.org, votes from SCDB release 2026_01. Same table format "
          "and filters as the ConvoKit data. Outcomes are not summarized here: these cases "
          "are locked like the test set until their split is decided.", "",
          stats.to_markdown(index=False), "",
          "## For comparison: the same input statistics for 2012-2019 (ConvoKit tables)", "",
          "The 2020 term was argued by telephone, each justice questioning in turn; since "
          "the 2021 term every advocate's time is followed by a round of questions in order "
          "of seniority.", "",
          before.to_markdown(index=False), "",
          "## Justices with votes, by term", "", bench.to_frame("bench").to_markdown(), "",
          "## Advocate descriptions that give no side", "",
          (unknown.head(15).to_frame("advocates").to_markdown() if len(unknown) else "none"), ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--terms", nargs="+", type=int)
    ap.add_argument("--check", type=int, help="compare a term with the ConvoKit tables")
    args = ap.parse_args()
    data = Path(args.data)
    if args.check:
        check(args.check, data)
    if args.terms:
        known = set(pd.read_parquet(data / "votes.parquet").justice)
        turns, cases, votes, adv, dropped, listed, files = build(args.terms, known)
        out = data / "ext"
        out.mkdir(exist_ok=True)
        turns.to_parquet(out / "turns.parquet", index=False)
        cases.to_parquet(out / "cases.parquet", index=False)
        votes.to_parquet(out / "votes.parquet", index=False)
        adv.to_parquet(out / "advocates.parquet", index=False)
        md = report(turns, cases, votes, adv, dropped, listed, files, data)
        Path("results/oyez_extension.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
