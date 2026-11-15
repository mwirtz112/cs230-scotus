"""
The three simplest baselines (the "floor"), scored with eval.py.

    always_petitioner   p = 1 for every vote                       (base rate only)
    justice_rate        p = justice's pro-petitioner rate in train  (who the justice is)
    question_count      p = share of the justice's questions aimed  (how much each side
                            at the respondent (Epstein et al. 2010)   gets questioned)

Run from the repository root:
    python -m scotus.baselines.baselines   # train + val -> data/preds/, results/baselines.{md,tex}
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from scotus.evaluation import eval as ev

SPLITS = ["train", "val"]       # test stays locked until the final report


# ---------------------------------------------------------------- always petitioner

def always_petitioner(votes):
    return votes.assign(p=1.0, pred=1)


# ---------------------------------------------------------------- justice rate

def justice_rate(votes, train_votes):
    """Lookup table of each justice's train pro-petitioner rate. No transcript used.

    Justices with no train votes (Gorsuch, Kavanaugh) get the overall train rate.
    Cutoff = overall train pro-petitioner vote rate: predict petitioner if
    the justice leans petitioner at least as much as the average justice.
    """
    rate = train_votes.groupby("justice").vote.mean()
    overall = train_votes.vote.mean()
    p = votes.justice.map(rate).fillna(overall)
    return votes.assign(p=p, pred=(p >= overall).astype(int)), overall


# ---------------------------------------------------------------- question count

def ordered_turns(turns, cases):
    """Turns in argument order: transcripts in case.convo_ids order, then turn_idx."""
    pos = {(c, cv): i for c, cvs in zip(cases.case_id, cases.convo_ids)
           for i, cv in enumerate(cvs)}
    t = turns.assign(convo_pos=[pos[(c, cv)] for c, cv in zip(turns.case_id, turns.convo_id)])
    return t.sort_values(["case_id", "convo_pos", "turn_idx"])


def question_targets(turns):
    """Which side each justice turn is aimed at: 0 petitioner, 1 respondent, -1 nobody.

    `turns` must be in argument order (see ordered_turns). Rules:
      - petitioner / respondent advocate turn: sets the side being questioned
      - amicus or unknown-side advocate: side unknown -> following questions credit nobody
        (decision A; unknown-side advocates follow the same logic)
      - <INAUDIBLE> / unknown speaker: ignored, side unchanged
      - new transcript (e.g. reargument): side resets
    Non-justice turns get -1. Returned Series shares the index of `turns`.
    """
    out, side, prev_convo = [], None, None
    for convo_id, role in zip(turns.convo_id, turns.role):
        if convo_id != prev_convo:
            side, prev_convo = None, convo_id
        if role == "petitioner":
            side = 0
        elif role == "respondent":
            side = 1
        elif role in ("amicus", "advocate_unknown"):
            side = None
        out.append(side if role == "justice" and side is not None else -1)
    return pd.Series(out, index=turns.index)


def count_questions(turns):
    """{(case_id, justice): [turns aimed at petitioner, turns aimed at respondent]}.

    Each justice TURN counts once, not each "?" (decision E).
    """
    target = question_targets(turns)
    credited = turns[target >= 0]
    counts = defaultdict(lambda: [0, 0])
    for case_id, speaker, side in zip(credited.case_id, credited.speaker_id,
                                      target[credited.index]):
        counts[(case_id, speaker)][side] += 1
    return counts


def question_count(votes, counts):
    """A justice votes AGAINST the side they questioned more.

    p = share of the justice's credited turns aimed at the respondent (more grilling of
    the respondent -> more likely a pro-petitioner vote). Cutoff 0.5 is the rule itself,
    not tuned. If the justice is silent or tied, use the whole bench's counts (decision B);
    if the bench also ties, p = 0.5, which counts as petitioner (p >= 0.5).
    """
    bench = defaultdict(lambda: [0, 0])
    for (case_id, _), (to_pet, to_resp) in counts.items():
        bench[case_id][0] += to_pet
        bench[case_id][1] += to_resp

    ps, sources = [], []
    for case_id, justice in zip(votes.case_id, votes.justice):
        to_pet, to_resp = counts.get((case_id, justice), (0, 0))
        if to_pet != to_resp:
            ps.append(to_resp / (to_pet + to_resp)); sources.append("justice")
            continue
        b_pet, b_resp = bench[case_id]
        if b_pet != b_resp:
            ps.append(b_resp / (b_pet + b_resp)); sources.append("bench")
        else:
            ps.append(0.5); sources.append("default")
    p = pd.Series(ps, index=votes.index)
    out = votes.assign(p=p, pred=(p >= 0.5).astype(int), source=sources)
    return out


# ---------------------------------------------------------------- run + report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-boot", type=int, default=ev.N_BOOT)
    args = ap.parse_args()
    data = Path(args.data)
    (data / "preds").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    cases = pd.read_parquet(data / "cases.parquet")
    turns = pd.read_parquet(data / "turns.parquet",
                            columns=["case_id", "convo_id", "turn_idx", "speaker_id", "role"])
    train_votes, _ = ev.load_labels("train", args.data)

    print("Counting questions...")
    keep = cases[cases.split.isin(SPLITS)]
    counts = count_questions(ordered_turns(turns[turns.case_id.isin(keep.case_id)], keep))

    rows, per_term, notes = [], [], []
    for split in SPLITS:
        votes, split_cases = ev.load_labels(split, args.data)
        jr, cutoff = justice_rate(votes, train_votes)
        qc = question_count(votes, counts)
        src = qc.source.value_counts(normalize=True).mul(100).round(1).to_dict()
        notes.append(f"- {split}: question-count source of p: {src} (% of votes)")
        for name, preds in [("always_petitioner", always_petitioner(votes)),
                            ("justice_rate", jr), ("question_count", qc)]:
            preds.to_parquet(data / "preds" / f"{name}_{split}.parquet", index=False)
            n_boot = args.n_boot if split == "val" else 200   # train CIs are tiny anyway
            res = ev.score(preds, votes, split_cases, n_boot=n_boot)
            rows.append({"method": name, "split": split,
                         **{k: ev.fmt_ci(*res[k]) for k in ev.METRICS}})
            if split == "val":
                per_term.append(ev.score_by_term(preds, votes, split_cases)
                                .assign(method=name))
            print(f"  {split:5s} {name:18s} " +
                  "  ".join(f"{k}={res[k][0]:.3f}" for k in ev.METRICS))
    notes.append(f"- justice_rate cutoff (train pro-petitioner vote rate): {cutoff:.3f}")

    table = pd.DataFrame(rows)
    terms = pd.concat(per_term)[["method", "term", "cases", "votes", "petitioner win %",
                                 *ev.METRICS]].round(3)
    md = ["# Baselines", "",
          "Point estimate [95% bootstrap CI over cases]. Test is locked.", "",
          table.to_markdown(index=False), "",
          "## Val, per term", "", terms.to_markdown(index=False), "",
          "## Notes", *notes, ""]
    Path("results/baselines.md").write_text("\n".join(md))
    val = table[table.split == "val"]
    tex = [r"\begin{table}[h]", r"  \centering", r"  \small",
           r"  \begin{tabular}{lcccc}", r"    \toprule",
           r"    Method & Macro-F1 & AUC & Vote acc. & Case acc. \\", r"    \midrule"]
    for _, r in val.iterrows():
        tex.append("    " + " & ".join([r.method.replace("_", " ").capitalize(),
                                        *(r[k] for k in ev.METRICS)]) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}",
            r"  \caption{Baselines on validation (2016--17). 95\% bootstrap CIs over cases.}",
            r"  \label{tab:baselines}", r"\end{table}"]
    Path("results/baselines.tex").write_text("\n".join(tex))
    print("\n" + "\n".join(md))
    print("Wrote results/baselines.md, results/baselines.tex, data/preds/*.parquet")


if __name__ == "__main__":
    main()
