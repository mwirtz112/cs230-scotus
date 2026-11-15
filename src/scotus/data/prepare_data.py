"""
Build the turn, case and vote tables from the ConvoKit Supreme Court corpus (terms
1955-2019): speaker roles, filters and the chronological splits.

Run from the repository root:
    python -m scotus.data.prepare_data                     # all years, 1955-2019
    python -m scotus.data.prepare_data --years 2014 2019   # quick subset for a first run

Outputs (in ./data):
    turns.parquet     one row per utterance, in argument order
    cases.parquet     one row per kept case, with split and labels
    votes.parquet     one row per (case, justice) with a usable vote
    stats.md          split-level counts for the Dataset section
    split_table.tex   the same counts as a LaTeX table
    coverage.csv      share of turns with an identified speaker, by decade
    drift.pdf         petitioner win rate by term, bench changes marked
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

TRAIN_END, VAL_END = 2015, 2017          # train <= 2015, val 2016-17, test 2018+ (corpus ends 2019)
SIDE_ROLE = {1: "petitioner", 0: "respondent", 2: "amicus"}


def split_of(year):
    if year <= TRAIN_END:
        return "train"
    if year <= VAL_END:
        return "val"
    return "test"


def role_of(utt_meta):
    """Map ConvoKit speaker_type/side to the four roles in the proposal."""
    st = utt_meta.get("speaker_type")
    if st == "J":
        return "justice"
    if st == "A":
        return SIDE_ROLE.get(utt_meta.get("side"), "advocate_unknown")
    return "unknown"


ID_PATTERN = re.compile(r"__(\d+)_(\d+)$")   # <transcript>__<section>_<turn>


def ordered_utterances(convo):
    """Turns in argument order.

    ConvoKit's get_chronological_utterance_list() sorts on utterance timestamps,
    which this corpus leaves unset (the time sits in meta["timestamp"] and restarts
    each section). Utterance ids encode the order instead: <transcript>__<section>_<turn>.
    """
    utts = list(convo.iter_utterances())
    if all(ID_PATTERN.search(u.id) for u in utts):
        return sorted(utts, key=lambda u: tuple(map(int, ID_PATTERN.search(u.id).groups())))
    return utts                                  # fall back to file order


def process_conversation(convo, year):
    """Return (turn rows, conversation record) for one transcript."""
    turns = []
    for i, utt in enumerate(ordered_utterances(convo)):
        text = (utt.text or "").strip()
        if not text:
            continue
        turns.append({
            "case_id": utt.meta.get("case_id", convo.meta.get("case_id")),
            "convo_id": convo.id,
            "year": year,
            "turn_idx": i,
            "speaker_id": utt.speaker.id,
            "speaker_type": utt.meta.get("speaker_type"),
            "side": utt.meta.get("side"),
            "role": role_of(utt.meta),
            "text": text,
        })
    rec = {
        "case_id": convo.meta.get("case_id"),
        "convo_id": convo.id,
        "year": year,
        "win_side": convo.meta.get("win_side"),
        "votes_side": convo.meta.get("votes_side") or {},
        "n_turns": len(turns),
    }
    return turns, rec


def load_years(years):
    from convokit import Corpus, download   # imported here so tests can skip it
    all_turns, convos = [], []
    for y in years:
        try:
            corpus = Corpus(filename=download(f"supreme-{y}"))
        except Exception as e:                # a missing year should not kill the run
            print(f"  {y}: skipped ({e})")
            continue
        n = 0
        for convo in corpus.iter_conversations():
            t, rec = process_conversation(convo, y)
            all_turns.extend(t)
            convos.append(rec)
            n += 1
        print(f"  {y}: {n} transcripts")
        del corpus
    return pd.DataFrame(all_turns), convos


def build_cases(convos, turns):
    """Merge transcripts into cases and apply the proposal's filters."""
    by_case = defaultdict(list)
    for rec in convos:
        by_case[rec["case_id"]].append(rec)

    justice_turns = turns[turns.role == "justice"].groupby("case_id").size()

    cases, votes, dropped = [], [], defaultdict(int)
    for case_id, recs in by_case.items():
        win_sides = {r["win_side"] for r in recs}
        win = recs[0]["win_side"]
        if len(win_sides) > 1 or win not in (0, 1):
            dropped["unclear winner"] += 1
            continue
        if justice_turns.get(case_id, 0) == 0:
            dropped["no justice turns"] += 1
            continue
        vs = {}
        for r in recs:                                  # union over transcripts
            vs.update({j: v for j, v in r["votes_side"].items() if v in (0, 1)})
        if not vs:
            dropped["no usable votes"] += 1
            continue
        year = recs[0]["year"]
        cases.append({
            "case_id": case_id, "year": year, "split": split_of(year),
            "win_side": win, "n_transcripts": len(recs),
            "convo_ids": [r["convo_id"] for r in recs],
            "n_votes": len(vs),
        })
        for j, v in vs.items():
            votes.append({"case_id": case_id, "year": year,
                          "split": split_of(year), "justice": j, "vote": v})
    return pd.DataFrame(cases), pd.DataFrame(votes), dict(dropped)


def check_vote_keys(votes, turns):
    """Share of voting justices who also speak in that case's transcript."""
    spoke = turns[turns.role == "justice"].groupby("case_id").speaker_id.apply(set)
    hits = [j in spoke.get(c, set()) for c, j in zip(votes.case_id, votes.justice)]
    return sum(hits) / max(len(hits), 1)


def split_stats(cases, votes, turns):
    kept = turns[turns.case_id.isin(cases.case_id)]
    rows = []
    for s in ["train", "val", "test"]:
        c, v = cases[cases.split == s], votes[votes.split == s]
        t = kept[kept.case_id.isin(c.case_id)]
        if c.empty:
            continue
        rows.append({
            "split": s,
            "years": f"{c.year.min()}-{c.year.max()}",
            "cases": len(c),
            "votes": len(v),
            "turns": len(t),
            "petitioner win %": round(100 * c.win_side.mean(), 1),
            "pro-petitioner vote %": round(100 * v.vote.mean(), 1),
            "median turns/case": int(t.groupby("case_id").size().median()),
        })
    return pd.DataFrame(rows)


def coverage_by_decade(turns):
    t = turns.assign(decade=(turns.year // 10) * 10)
    g = t.groupby("decade")
    return pd.DataFrame({
        "turns": g.size(),
        "justice %": (100 * g.role.apply(lambda r: (r == "justice").mean())).round(1),
        "unknown speaker %": (100 * g.role.apply(lambda r: (r == "unknown").mean())).round(1),
    })


def to_latex(stats):
    lines = [r"\begin{table}[h]", r"  \centering", r"  \begin{tabular}{lcrrrc}",
             r"    \toprule",
             r"    Split & Terms & Cases & Votes & Turns & Pro-petitioner votes \\",
             r"    \midrule"]
    for _, r in stats.iterrows():
        lines.append(f"    {r['split'].capitalize()} & {r['years']} & {r['cases']:,} & "
                     f"{r['votes']:,} & {r['turns']:,} & {r['pro-petitioner vote %']}\\% \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
              r"  \caption{Dataset splits after filtering.}",
              r"  \label{tab:data}", r"\end{table}"]
    return "\n".join(lines)


def drift_plot(cases, votes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rate = cases.groupby("year").win_side.mean()
    bench = votes.groupby("year").justice.apply(frozenset)
    changes = [y for prev, y in zip(bench.index[:-1], bench.index[1:])
               if bench[y] != bench[prev]]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for y in changes:
        ax.axvline(y, color="0.85", lw=0.8, zorder=0)
    ax.plot(rate.index, rate.values, lw=1.5)
    ax.axvspan(TRAIN_END + 0.5, VAL_END + 0.5, alpha=0.08)
    ax.axvspan(VAL_END + 0.5, rate.index.max() + 0.5, alpha=0.16)
    ax.set_xlabel("Term")
    ax.set_ylabel("Petitioner win rate")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[1955, 2019])
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print("Downloading and reading corpus by year...")
    turns, convos = load_years(range(args.years[0], args.years[1] + 1))

    cases, votes, dropped = build_cases(convos, turns)
    turns = turns[turns.case_id.isin(cases.case_id)]

    turns.to_parquet(out / "turns.parquet", index=False)
    cases.to_parquet(out / "cases.parquet", index=False)
    votes.to_parquet(out / "votes.parquet", index=False)

    stats = split_stats(cases, votes, turns)
    cov = coverage_by_decade(turns)
    match = check_vote_keys(votes, turns)

    report = [
        "## Split statistics", stats.to_markdown(index=False), "",
        "## Cases dropped", json.dumps(dropped, indent=2), "",
        f"Cases with more than one transcript: {(cases.n_transcripts > 1).sum()}", "",
        f"Voting justices who also speak in the transcript: {100 * match:.1f}%",
        "(should be high for recent terms; a low value means the vote keys and "
        "speaker ids don't line up and need a mapping)", "",
        "## Speaker coverage by decade", cov.to_markdown(), "",
        "## Role counts", turns.role.value_counts().to_markdown(),
    ]
    (out / "stats.md").write_text("\n".join(report))
    (out / "split_table.tex").write_text(to_latex(stats))
    cov.to_csv(out / "coverage.csv")
    drift_plot(cases, votes, out / "drift.pdf")
    print("\n".join(report))
    print(f"\nWrote {out}/turns.parquet, cases.parquet, votes.parquet, "
          "stats.md, split_table.tex, coverage.csv, drift.pdf")


if __name__ == "__main__":
    main()
