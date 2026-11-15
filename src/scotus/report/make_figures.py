"""
Figures for the report, built from saved predictions and run logs.

    python -m scotus.report.make_figures            -> results/figures/*.pdf (+ the numbers as .csv)

    fig_results_val   every method on validation: macro-F1 and AUC with 95% CIs
    fig_parts         what each component buys: case-level vs within-case AUC
    fig_era           the question signal by decade (why v1 trains on 1980+)
    fig_curves        learning curves of the main model over seeds
    fig_affinity      whose questioning predicts whose vote (weights of the qc_bloc
                      regression, the linear version of v1's bloc layer)
    fig_walk          the walk-forward run, term by term (--run rehearsal | final; drawn
                      from the csv that walk_forward.py writes)

Only train / dev / val predictions are read; nothing here touches the test split. The one
exception is `--only walk --run final`, which draws the test run's own table after
walk_forward.py --final has been made for the final report.
Style: one accent hue for the model and grey for context, thin marks, hairline grid,
direct labels where they help. Colours are the first slots of a palette checked for
colour-blind separation (blue, orange).
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scotus.evaluation import eval as ev
from scotus.evaluation import summarize_v1 as S

MAIN = "v1_main"                       # the main v1 configuration
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8d8c87"
OUT = Path("results/figures")

plt.rcParams.update({
    "font.size": 8, "font.family": "sans-serif", "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def load(name, fold):
    """Seed ensemble of a v1 configuration, or a baseline's single prediction file."""
    paths = sorted(glob.glob(f"data/preds/{name}_{fold}_s*.parquet"))
    if paths:
        return S.ensemble(paths)
    stem = f"{name}_train" if (name == "question_count" and fold == "dev") else f"{name}_{fold}"
    return pd.read_parquet(f"data/preds/{stem}.parquet")


def dot(ax, x, y, color, size=5.5, **kw):
    ax.plot(x, y, "o", ms=size, color=color, mec="white", mew=1.2, zorder=3, **kw)


# ---------------------------------------------------------------- results with CIs

LADDER = [  # (prediction file stem, label, kind)
    ("always_petitioner", "Always petitioner", "base"),
    ("justice_rate", "Justice vote rate", "base"),
    ("question_count", "Question count", "base"),
    ("embedding_probe", "Embedding probe", "base"),
    ("flat_mb", "Flat ModernBERT, first 8k tokens", "base"),
    ("flat_mb_all", "Flat ModernBERT, whole transcript", "base"),
    ("flat_mb_ft", "Flat ModernBERT, tuned (lr 5e-6)", "base"),          # top 4 layers, first 8k tokens
    ("flat_mb_ft_lr1e5", "Flat ModernBERT, tuned (lr 1e-5)", "base"),
    ("katz_rf", "Case profile only (RF)", "base"),
    ("qc_plus", "Question features (LR)", "base"),
    ("qc_bloc", "+ named colleagues (LR)", "base"),
    ("v0_r1_reg", "Model v0 (milestone)", "old"),
    ("d_notext", "Model v1, no text", "abl"),
    ("c_base_1980", "Model v1, no bloc layer", "abl"),
    (MAIN, "Model v1", "main"),
    ("v1_fusion", "Model v1 + case profile", "abl"),
]


def fig_results_val():
    v, c = S.labels("val")
    rows = []
    for stem, label, kind in LADDER:
        try:
            res = ev.score(load(stem, "val"), v, c)
        except FileNotFoundError:
            continue
        rows.append({"method": label, "kind": kind,
                     **{f"{m}{s}": res[m][i] for m in ("macro_f1", "auc")
                        for i, s in enumerate(("", "_lo", "_hi"))},
                     "vote_acc": res["vote_acc"][0], "case_acc": res["case_acc"][0]})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "fig_results_val.csv", index=False)

    colors = {"base": GREY, "old": GREY, "abl": "#86b6ef", "main": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 0.55 + 0.15 * len(df)), sharey=True)  # 2.8 at 15 rows
    y = np.arange(len(df))[::-1]
    for ax, m, title in [(axes[0], "macro_f1", "Macro-F1 (primary)"), (axes[1], "auc", "ROC-AUC")]:
        for yi, (_, r) in zip(y, df.iterrows()):
            col = colors[r.kind]
            ax.plot([r[f"{m}_lo"], r[f"{m}_hi"]], [yi, yi], color=col, lw=1.6,
                    solid_capstyle="round", zorder=2)
            dot(ax, r[m], yi, col)
        i = int(np.flatnonzero(df.kind == "main")[0])               # label the main model
        ax.annotate(f"{df[m][i]:.3f}", (df[f"{m}_hi"][i], y[i]), xytext=(3, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=7.5)
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(df) - 0.4)
    axes[0].set_yticks(y, df.method)
    fig.supxlabel("validation 2016–17 (123 cases); lines are 95% bootstrap intervals over cases",
                  fontsize=7.5, color=INK2, x=0.6)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_results_val.pdf")
    plt.close(fig)
    return df


# ---------------------------------------------------------------- case vs within-case

def parts(preds, v, c):
    """Case-level AUC (mean p of a case vs the winner) and within-case AUC (split cases:
    are petitioner voters ranked above respondent voters?)."""
    d = v.merge(preds[["case_id", "justice", "p"]], on=["case_id", "justice"])
    mean_p = d.groupby("case_id").p.mean()
    win = c.set_index("case_id").win_side
    case_auc = roc_auc_score(win.loc[mean_p.index], mean_p)
    num = den = 0
    for _, g in d.groupby("case_id"):
        a, b = g.p[g.vote == 1].to_numpy(), g.p[g.vote == 0].to_numpy()
        num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        den += len(a) * len(b)
    return case_auc, num / den


def fig_parts():
    models = [("katz_rf", "Case profile only (RF)"), ("qc_bloc", "Counts + colleagues (LR)"),
              ("d_notext", "v1, no text"), ("c_base_1980", "v1, no bloc layer"), (MAIN, "v1"),
              ("v1_fusion", "v1 + case profile")]
    i_main = [m[0] for m in models].index(MAIN)
    rows = []
    for fold in ("dev", "val"):
        v, c = S.labels(fold)
        for stem, label in models:
            ca, wi = parts(load(stem, fold), v, c)
            rows.append({"model": label, "fold": fold, "case": ca, "within": wi})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "fig_parts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, col, title in [(axes[0], "case", "Which side wins"),
                           (axes[1], "within", "Who dissents")]:
        for fold, color in (("dev", ORANGE), ("val", BLUE)):
            x = df[df.fold == fold].set_index("model")[col].loc[[m[1] for m in models]]
            dot(ax, x.to_numpy(), y, color, label={"dev": "dev 2012–15", "val": "val 2016–17"}[fold])
            ax.annotate(f"{x.iloc[i_main]:.2f}", (x.iloc[i_main], y[i_main]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.set_title(title)
        ax.set_xlabel({"case": "case-level AUC", "within": "within-case AUC (split cases)"}[col])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.1)
        ax.set_ylim(-0.9, len(models) - 0.5)
    axes[0].set_yticks(y, [m[1] for m in models])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=7, handletextpad=0.2,
               columnspacing=1.2, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_parts.pdf")
    plt.close(fig)
    return df


# ---------------------------------------------------------------- signal by decade

def fig_era():
    votes = ev.load_labels("train")[0]
    rows = []
    for stem, label in [("question_count", "Question count"),
                        ("qc_plus", "Question features (LR, in-sample)")]:
        d = votes.merge(pd.read_parquet(f"data/preds/{stem}_train.parquet")
                        [["case_id", "justice", "p"]], on=["case_id", "justice"])
        d["decade"] = d.year // 10 * 10
        for dec, g in d.groupby("decade"):
            rows.append({"method": label, "decade": dec, "auc": roc_auc_score(g.vote, g.p),
                         "votes": len(g)})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "fig_era.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for (label, g), color in zip(df.groupby("method", sort=False), (GREY, BLUE)):
        ax.plot(g.decade, g.auc, color=color, lw=1.8, solid_capstyle="round", label=label)
        dot(ax, g.decade, g.auc, color, size=4.5)
    ax.axhline(0.5, color=INK2, lw=0.6)
    ax.annotate("chance", (2010, 0.5), xytext=(0, 3), textcoords="offset points",
                ha="right", fontsize=7, color=INK2)
    ax.axvline(1980, color=GRID, lw=0.8, zorder=0)
    ax.annotate("v1 trains\nfrom 1980", (1980, 0.585), xytext=(4, 0),
                textcoords="offset points", fontsize=7, color=INK2, va="top")
    ax.set_xticks(sorted(df.decade.unique()), [f"{d}s" for d in sorted(df.decade.unique())])
    ax.set_ylabel("ROC-AUC on training votes")
    ax.set_ylim(0.47, 0.74)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=7, ncol=1,
               handlelength=1.4, bbox_to_anchor=(0.16, 1.0))
    fig.savefig(OUT / "fig_era.pdf")
    plt.close(fig)
    return df


# ---------------------------------------------------------------- learning curves

def fig_curves(name=MAIN):
    runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"results/v1_runs/{name}_*_s*.json"))]
    if not runs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0))
    hist = {f: np.array([[h["ev_auc"] for h in r["history"]] for r in runs if r["fold"] == f])
            for f in ("dev", "val")}
    loss = np.array([[h["train_loss"] for h in r["history"]] for r in runs if r["fold"] == "val"])
    ep = np.arange(1, loss.shape[1] + 1)
    axes[0].fill_between(ep, loss.min(0), loss.max(0), color=BLUE, alpha=0.12, lw=0)
    axes[0].plot(ep, loss.mean(0), color=BLUE, lw=1.8)
    axes[0].set_title("Training loss (balanced BCE)")
    for fold, color, label in (("val", BLUE, "val 2016–17"), ("dev", ORANGE, "dev 2012–15")):
        a = hist[fold]
        axes[1].fill_between(ep, a.min(0), a.max(0), color=color, alpha=0.12, lw=0)
        axes[1].plot(ep, a.mean(0), color=color, lw=1.8, label=label)
        axes[1].annotate(label, (ep[-1], a.mean(0)[-1]), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=7, color=INK)
    tr = np.array([[h.get("train_auc", np.nan) for h in r["history"]] for r in runs
                   if r["fold"] == "val"])
    m = ~np.isnan(tr).all(0)
    axes[1].plot(ep[m], np.nanmean(tr, 0)[m], color=GREY, lw=1.2, ls=(0, (1, 1.5)))
    dot(axes[1], ep[m], np.nanmean(tr, 0)[m], GREY, size=4)
    axes[1].annotate("train subset", (ep[m][0], np.nanmean(tr, 0)[m][0]), xytext=(0, -9),
                     textcoords="offset points", ha="center", fontsize=7, color=INK2)
    axes[1].set_title("ROC-AUC of held-out terms")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xticks(ep)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xlim(ep[0] - 0.2, ep[-1] + 1.9)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_curves.pdf")
    plt.close(fig)
    return len(runs)


# ---------------------------------------------------------------- bloc affinities

def short(j):
    """j__ruth_bader_ginsburg -> Ginsburg."""
    parts = [p for p in j.replace("j__", "").split("_") if p not in ("jr", "ii", "iii")]
    return parts[-1].capitalize()


def fig_affinity():
    """Whose questioning predicts whose vote: stage-2 weights of the qc_bloc regression
    (fit on training terms only), for the justices who sat in 2010-15. Rows and columns are
    ordered by the leading component of the matrix, not by any outside ideology score."""
    from scotus.baselines import qc_plus as Q
    vf = pd.read_parquet("data/vote_feats.parquet")
    tr = ev.load_labels("train")[0].merge(vf, on=["case_id", "justice"])
    _, coef = Q.fit_predict_bloc(tr, [tr.iloc[:50]], return_coef=True)
    bench = tr[tr.year >= 2010].justice.value_counts()
    names = [j for j in bench.index if j in coef.index]
    M = coef.loc[names, names].to_numpy(dtype=float, copy=True)
    np.fill_diagonal(M, np.nan)
    sym = np.nan_to_num((M + M.T) / 2)
    order = np.argsort(np.linalg.eigh(sym)[1][:, -1])              # leading eigenvector
    A = pd.DataFrame(M, index=names, columns=names).iloc[order, order]
    A.rename(index=short, columns=short).round(3).to_csv(OUT / "fig_affinity.csv")

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", BLUE])
    cmap.set_bad("#ffffff")
    lim = np.nanmax(np.abs(A.values))
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    im = ax.imshow(A.values, cmap=cmap, vmin=-lim, vmax=lim)
    labels = [short(j) for j in A.index]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)                  # 2px surface gaps
    ax.tick_params(which="both", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xlabel("colleague whose questioning is observed")
    ax.set_ylabel("justice whose vote is predicted")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7)
    cb.set_label("weight (+: votes with, −: against)", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_affinity.pdf")
    plt.close(fig)
    return A.rename(index=short, columns=short)


# ---------------------------------------------------------------- walk-forward, per term

def fig_walk(run="rehearsal"):
    """Per-term macro-F1 and AUC of a walk-forward run (walk_forward.py): model v1 refit
    before each term, the same model fitted once, and the strongest baseline. Drawn from
    the run's csv, so this function reads no labels (for run "final" it shows the test)."""
    src = Path(f"results/walk_forward_{run}_terms.csv")
    if not src.exists():
        return None
    df = pd.read_csv(src)
    terms = sorted(df.term.unique())
    series = [("qc_bloc", "each", GREY, "Count regression with blocs (qc_bloc), refit each term"),
              ("v1", "once", ORANGE, f"Model v1, fitted once (terms ≤ {terms[0] - 1})"),
              ("v1", "each", BLUE, "Model v1, refit each term (band: 95% CI)")]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharex=True)
    for ax, (m, title) in zip(axes, [("macro_f1", "Macro-F1"), ("auc", "ROC-AUC")]):
        for name, fit, color, label in series:
            g = df[(df.method == name) & (df.fit == fit)].sort_values("term")
            if g.empty:
                continue
            if g[f"{m}_lo"].notna().all():
                ax.fill_between(g.term, g[f"{m}_lo"], g[f"{m}_hi"], color=color, alpha=0.12, lw=0)
            ax.plot(g.term, g[m], color=color, lw=1.8, solid_capstyle="round", label=label)
            dot(ax, g.term, g[m], color, size=4.5)
        if 2020 in terms and terms[0] < 2020:
            ax.axvline(2019.5, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=0)
            ax.annotate("new argument\nformat", (2019.5, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -2), textcoords="offset points", va="top", fontsize=6.5,
                        color=INK2)
        ax.set_title(title)
        ax.set_xticks(terms, [str(t) if len(terms) <= 6 else f"’{t % 100:02d}" for t in terms])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_xlabel("Term predicted")
    axes[1].set_xlabel("Term predicted")
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", fontsize=7, ncol=1,
               handlelength=1.4, bbox_to_anchor=(0.06, 1.0))
    fig.savefig(OUT / f"fig_walk_{run}.pdf")
    fig.savefig(OUT / f"fig_walk_{run}.png", dpi=200)
    plt.close(fig)
    keep = df[df.method.isin(["qc_bloc", "v1"])]
    return keep.pivot_table(index="term", columns=["method", "fit"], values="auc").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--run", default="rehearsal",
                    help="walk-forward run to draw for fig_walk (rehearsal / final)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {"results_val": fig_results_val, "parts": fig_parts, "era": fig_era,
            "curves": fig_curves, "affinity": fig_affinity,
            "walk": lambda: fig_walk(args.run)}
    for name, fn in figs.items():
        if args.only and name not in args.only:
            continue
        out = fn()
        if name == "walk" and out is None:
            print(f"fig_walk: no results/walk_forward_{args.run}_terms.csv yet, skipped")
            continue
        print(f"fig_{name}: written")
        if isinstance(out, pd.DataFrame):
            print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
