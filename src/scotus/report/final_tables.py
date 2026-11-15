"""Tables and figure files of the final report, from saved results. Runs no model.

    python -m scotus.report.final_tables     -> final_report/tables/test_main.tex
                                  final_report/figures/*.pdf  (copies of the figures used)

The main table is read from results/walk_forward_final.md, the report that the test run
wrote (terms 2018-2025; checked independently in results/walk_forward_final_check.md).
The LaTeX is written here so that no number in the report is typed by hand. (The
report's own sources, in final_report/, are kept out of this repository.)
"""
import shutil
from pathlib import Path

import pandas as pd

SRC = "results/walk_forward_final.md"
OUT = Path("final_report")
# (method and fit in the test report) -> (label in the paper, what the method uses)
ROWS = [
    (("Always petitioner", "–"), "Always petitioner", "--"),
    (("Justice vote rate", "each term"), "Justice vote rate", "V"),
    (("Flat ModernBERT, first 8k tokens (frozen)", "each term"), "Flat language model", "T"),
    (("Question count", "–"), "Question count", "T"),
    (("Case profile only, Katz-style forest (katz_rf)", "each term"), "Case facts (forest)", "F, V"),
    (("Question features, LR (qc_plus)", "each term"), "Question statistics", "T"),
    (("+ named colleagues, LR (qc_bloc)", "each term"), r"\quad + colleagues", "T"),
    (("+ case profile history, LR (qc_bloc_meta)", "each term"), r"\quad + voting record", "T, V"),
    (("Model v1", "each term"), "My model", "T"),
    (("Model v1 + qc_bloc_meta (mean of logits)", "each term"), "Average of the two rows above", "T, V"),
]
OURS = "My model"
# A small informal study by the author: seven lawyers and law students were each given
# the transcripts of five cases drawn at random from the 2025 term (renamed) and predicted
# each winner; scores 2, 2, 1, 1, 1, 1 and 4 of 5. Not the 481 test cases, so the row is marked.
HUMAN = {"readers": 7, "cases": 5, "correct": 2 + 2 + 1 + 1 + 1 + 1 + 4}
# figures of the report: the test terms one by one (main text) and the look inside the
# network (appendix); drawn by make_figures.py and inside_v1.py
FIGURES = ["fig_walk_final", "fig_inside_paths", "fig_inside_blocs"]


def read_table(path, heading):
    """The markdown table under `heading`, as strings."""
    lines = Path(path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(heading))
    rows = []
    for l in lines[start + 1:]:
        if l.startswith("## "):
            break
        if l.startswith("|") and not set(l) <= set("|:- "):
            rows.append([c.strip() for c in l.strip().strip("|").split("|")])
    return pd.DataFrame(rows[1:], columns=rows[0])


def pct(x):
    return "--" if x in ("–", "", "nan") else f"{100 * float(x):.1f}"


def main():
    heading = "## All scored terms, 2018–2025"
    title = next(l for l in Path(SRC).read_text().splitlines() if l.startswith(heading))
    assert "481 cases, 4,298 votes" in title, title            # the caption states these
    t = read_table(SRC, heading).set_index(["Method", "Fit"])
    body = []
    for key, label, uses in ROWS:
        r = t.loc[key]
        f1, ci = r["Macro-F1"].split(" ", 1)
        cells = [label, uses, rf"{f1} {{\scriptsize {ci}}}", r["AUC"].split(" ")[0], pct(r["Vote acc"]),
                 pct(r["Case acc"]), pct(r["Case acc, base rate restored"])]
        if label == OURS:
            cells = [rf"\textbf{{{c}}}" if i != 2 else rf"\textbf{{{f1}}} {{\scriptsize {ci}}}"
                     for i, c in enumerate(cells)]
        body.append("    " + " & ".join(cells) + r" \\")
    assert HUMAN["correct"] == 12
    human = 100 * HUMAN["correct"] / (HUMAN["readers"] * HUMAN["cases"])
    body += [r"    \midrule",
             rf"    Human readers$^{{\ddagger}}$ & T & -- & -- & -- & {human:.1f} & -- \\"]
    tex = [
        "% Written by final_tables.py from results/walk_forward_final.md. Do not edit by hand.",
        r"\begin{table}[t]",
        r"  \caption{Test terms 2018--2025 (481 cases, 4,298 votes); every method is refit before",
        r"  each term. Uses: T transcript, V the justices' past votes, F facts about the case.",
        r"  Macro-F1 with a 95\% bootstrap interval over cases; votes and cases in per cent called",
        r"  correctly; Cases$^{\dagger}$ with the petitioner's usual advantage put back (see text).",
        r"  $^{\ddagger}$Seven lawyers and law students on five cases of 2025 only (see text).}",
        r"  \label{tab:test}",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llccccc}",
        r"    \toprule",
        r"    Method & Uses & Macro-F1 & AUC & Votes & Cases & Cases$^{\dagger}$ \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables" / "test_main.tex").write_text("\n".join(tex))
    for fig in FIGURES:
        shutil.copyfile(f"results/figures/{fig}.pdf", OUT / "figures" / f"{fig}.pdf")
    print("\n".join(tex))
    print("Wrote final_report/tables/test_main.tex; copied to final_report/figures/:", ", ".join(FIGURES))


if __name__ == "__main__":
    main()
