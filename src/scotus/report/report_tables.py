"""
Results tables on the validation and dev folds, for every method.

    python -m scotus.report.report_tables     -> results/v1_main.md, results/v1_main.tex,
                                   results/v1_ablations.md, results/v1_fusion.md,
                                   results/v1_encoders.md

One row per method, all scored by eval.py on the same votes:
    val  = validation 2016-17 (models fit on <= 2015)
    dev  = 2012-15 (models refit on <= 2011); only for methods that have a dev fold
Each method is shown at cutoff 0.5 and, where a dev fold exists, at the "train-rate"
cutoff: the cutoff at which the share of predicted petitioner votes on dev equals the
training vote rate (59.9%), applied unchanged to val.
95% confidence intervals: bootstrap over cases. The test split is not read.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scotus.evaluation import eval as ev
from scotus.evaluation import summarize_v1 as S

MAIN = "v1_main"
FUSION_RUN = "v1_fusion"                        # v1 + metadata prior (--prior 1), 5 seeds
BLEND = f"blend:{MAIN}+qc_bloc"                 # "blend:a+b+..." = mean of the models' logits
BLEND_META = f"blend:{MAIN}+qc_bloc_meta"
METHODS = [  # (label, prediction stem, is a v1 configuration)
    ("Always petitioner", "always_petitioner", False),
    ("Justice vote rate", "justice_rate", False),
    ("Question count", "question_count", False),
    ("Embedding probe", "embedding_probe", False),
    ("Flat ModernBERT, first 8k tokens, frozen (flat_mb)", "flat_mb", False),
    ("Flat ModernBERT, whole transcript, frozen (flat_mb_all)", "flat_mb_all", False),
    ("Flat ModernBERT, top 4 layers tuned, lr 5e-6 (flat_mb_ft)", "flat_mb_ft", False),
    ("Flat ModernBERT, top 4 layers tuned, lr 1e-5 (flat_mb_ft_lr1e5)", "flat_mb_ft_lr1e5", False),
    ("Case profile only, Katz-style forest (katz_rf)", "katz_rf", False),
    ("Question features, LR (qc_plus)", "qc_plus", False),
    ("+ named colleagues, LR (qc_bloc)", "qc_bloc", False),
    ("+ case profile history, LR (qc_bloc_meta)", "qc_bloc_meta", False),
    ("Model v0 + regularization (milestone)", "v0_r1_reg", False),
    ("Model v1, no text", "d_notext", True),
    ("Model v1, no bloc layer", "c_base_1980", True),
    ("Model v1 (5 seeds)", MAIN, True),
    ("Model v1 + metadata prior (fusion, 5 seeds)", FUSION_RUN, True),
    ("Model v1 + qc_bloc (mean of logits)", BLEND, True),
    ("Model v1 + qc_bloc_meta (mean of logits)", BLEND_META, True),
]


def preds(stem, fold):
    if stem.startswith("blend:"):
        parts = [preds(n, fold) for n in stem[6:].split("+")]
        if fold == "dev":                       # baseline dev files also hold other terms
            keys = S.labels("dev")[0][["case_id", "justice"]]
            parts = [keys.merge(p, on=["case_id", "justice"]) for p in parts]
        return S.blend_many(parts)
    paths = sorted(glob.glob(f"data/preds/{stem}_{fold}_s*.parquet"))
    if paths:
        return S.ensemble(paths)
    name = f"{stem}_train" if (fold == "dev" and stem == "question_count") else f"{stem}_{fold}"
    path = Path(f"data/preds/{name}.parquet")
    return pd.read_parquet(path) if path.exists() else None


def main():
    lab = {f: S.labels(f) for f in ("dev", "val")}
    rows, terms = [], []
    for label, stem, _ in METHODS:
        val = preds(stem, "val")
        if val is None:
            continue
        dev = preds(stem, "dev")
        r = ev.score(val, *lab["val"])
        row = {"Method": label, "Val macro-F1": ev.fmt_ci(*r["macro_f1"]),
               "Val AUC": ev.fmt_ci(*r["auc"]), "Val vote acc": f"{r['vote_acc'][0]:.3f}",
               "Val case acc": f"{r['case_acc'][0]:.3f}"}
        if dev is not None:
            dev = lab["dev"][0][["case_id", "justice"]].merge(dev, on=["case_id", "justice"])
            d = ev.score(dev, *lab["dev"], n_boot=0)
            cut = S.rate_cutoff(dev)
            rr = ev.score(val.assign(pred=(val.p >= cut).astype(int)), *lab["val"])
            row.update({"Dev macro-F1": f"{d['macro_f1'][0]:.3f}", "Dev AUC": f"{d['auc'][0]:.3f}",
                        "Cutoff": f"{cut:.3f}", "Val macro-F1 @rate": ev.fmt_ci(*rr["macro_f1"]),
                        "Val vote acc @rate": f"{rr['vote_acc'][0]:.3f}",
                        "Val case acc @rate": f"{rr['case_acc'][0]:.3f}"})
        rows.append(row)
        if stem in (MAIN, FUSION_RUN, "qc_bloc", "katz_rf", BLEND, BLEND_META):
            for fold in ("dev", "val"):
                p = preds(stem, fold)
                p = lab[fold][0][["case_id", "justice"]].merge(p, on=["case_id", "justice"])
                terms.append(ev.score_by_term(p, *lab[fold]).assign(method=label))
    df = pd.DataFrame(rows).fillna("")
    main_cols = ["Method", "Val macro-F1", "Val AUC", "Val vote acc", "Val case acc",
                 "Dev macro-F1", "Dev AUC"]
    rate_cols = ["Method", "Cutoff", "Val macro-F1 @rate", "Val vote acc @rate",
                 "Val case acc @rate"]
    per_term = pd.concat(terms)[["method", "term", "cases", "votes", "petitioner win %",
                                 *ev.METRICS]].round(3)
    md = ["# Main results (validation and dev; test untouched)", "",
          "Point estimate [95% bootstrap CI over cases]. Cutoff 0.5 unless stated. Dev columns "
          "are blank for methods without a dev-fold run.", "",
          df[main_cols].to_markdown(index=False), "",
          "## Train-rate cutoff (set on dev, applied to val)", "",
          df[df.Cutoff != ""][rate_cols].to_markdown(index=False), "",
          "## Per term", "", per_term.to_markdown(index=False), ""]
    Path("results/v1_main.md").write_text("\n".join(md))

    short = lambda s: s.replace("0.", ".", 1) if s.startswith("0.") else s
    tex = [r"\begin{table}[h]", r"  \centering", r"  \small",
           r"  \begin{tabular}{lcccc}", r"    \toprule",
           r"    Method & Macro-F1 & AUC & Vote acc. & Case acc. \\", r"    \midrule"]
    for _, r in df.iterrows():
        f1, lo_hi = r["Val macro-F1"].split(" ", 1) if " " in r["Val macro-F1"] else (r["Val macro-F1"], "")
        ci = lo_hi.replace("0.", ".").replace("[", r"{\scriptsize[").replace("]", "]}") if lo_hi else ""
        tex.append(f"    {r.Method.replace('_', chr(92) + '_')} & {f1} {ci} & "
                   f"{r['Val AUC'].split(' ')[0]} & {r['Val vote acc']} & {r['Val case acc']} \\\\")
    tex += [r"    \bottomrule", r"  \end{tabular}",
            r"  \caption{Validation (2016--17) results at cutoff 0.5, with 95\% bootstrap CIs "
            r"over cases for macro-F1.}", r"  \label{tab:main}", r"\end{table}"]
    Path("results/v1_main.tex").write_text("\n".join(tex))
    print("\n".join(md))
    ablations(lab)
    fusion(lab)
    encoders(lab)


ABLATIONS = [  # (label, run name); every row is the ensemble of seeds 0-2
    ("Full model v1", MAIN),
    ("- bloc layer", "c_base_1980"),
    ("- text (turn embeddings)", "d_notext"),
    ("- structural turn inputs", "h_nostruct"),
    ("- learned counters", "h_nocount"),
    ("- wide count features", "h_nowide"),
    ("- attention content readout", "h_nocontent"),
    ("- turn transformer (0 layers)", "h_l0"),
    ("- justice identity (removes the bloc layer too)", "h_noid"),
]


def ablations(lab):
    """Each component removed in turn, same recipe, seeds 0-2 -> results/v1_ablations.md."""
    from scotus.report.make_figures import parts
    rows = []
    for label, name in ABLATIONS:
        row = {"Variant": label}
        for fold in ("dev", "val"):
            paths = [f"data/preds/{name}_{fold}_s{s}.parquet" for s in (0, 1, 2)]
            if not all(Path(p).exists() for p in paths):
                break
            p = S.ensemble(paths)
            sc = S.scores(p, lab[fold][0])
            case_auc, within = parts(p, *lab[fold])
            row.update({f"{fold} AUC": sc["auc"], f"{fold} F1": sc["f1"],
                        f"{fold} case-level": case_auc, f"{fold} within-case": within})
            if fold == "val":
                row.update({"val 2017 AUC": sc["auc17"], "val 2017 F1": sc["f117"]})
        else:
            rows.append(row)
    df = pd.DataFrame(rows).round(3)
    full = df.iloc[0]
    delta = df.copy()
    for col in df.columns[1:]:
        delta[col] = [f"{v:.3f}" if i == 0 else f"{v:.3f} ({v - full[col]:+.3f})"
                      for i, v in enumerate(df[col])]
    md = ["# Model v1 ablations", "",
          "One component removed at a time; same recipe (train 1980+, 6 epochs); every row is "
          "the mean-probability ensemble of seeds 0-2; cutoff 0.5. In brackets: change from "
          "the full model. case-level = AUC of a case's mean probability against the winner; "
          "within-case = AUC of ranking the justices inside non-unanimous cases. "
          "Differences below about 0.01 are within seed noise. Test untouched.", "",
          delta.to_markdown(index=False), ""]
    Path("results/v1_ablations.md").write_text("\n".join(md))
    print("\n".join(md))


FUSION = [  # (label, run name / baseline stem / "blend:a+b", train_v1.py flags added to v1)
    ("Model v1 (transcript only)", MAIN, ""),
    ("Case profile only: Katz-style forest (no transcript)", "katz_rf", ""),
    ("+ history rates as inputs", "m_hist", "--hist"),
    ("+ history rates as inputs, rolling history", "f_hist_roll",
     "--hist --meta-protocol rolling"),
    ("+ profile embeddings", "f_meta", "--meta"),
    ("+ profile embeddings + ideal-point term", "f_ideal", "--meta --ideal 4"),
    ("+ ideal-point term only", "f_ideal_only", "--meta --ideal 4 --no-meta-concat"),
    ("+ history + profile embeddings + ideal point", "f_all", "--hist --meta --ideal 4"),
    ("+ metadata prior on the logit", "m_prior", "--prior 1"),
    ("+ metadata prior, weight 0.5", "m_prior05", "--prior 0.5"),
    ("+ metadata prior, learned weight", "m_prior_learn", "--prior 1 --prior-learn"),
    ("+ metadata prior + history rates as inputs", "m_prior_hist", "--prior 1 --hist"),
    ("+ metadata prior, rolling history", "m_prior_roll",
     "--prior 1 --meta-protocol rolling"),
    ("+ prior = qc_bloc regression (counts + colleagues, no metadata)", "m_resid_qc",
     "--prior 1 --prior-source qc_bloc"),
    ("+ prior = qc_bloc_meta regression (counts + colleagues + history)", "m_resid_qcm",
     "--prior 1 --prior-source qc_bloc_meta"),
    ("+ prior = qc_bloc_meta; network without wide features or bloc layer",
     "m_resid_qcm_lean", "--affinity 0 --no-wide --prior 1 --prior-source qc_bloc_meta"),
    ("v1 without the bloc layer", "c_base_1980", "--affinity 0"),
    ("v1 without the bloc layer + history rates", "m_hist_nobloc", "--affinity 0 --hist"),
    ("v1 without justice identity or bloc layer", "h_noid", "--affinity 0 --no-identity"),
    ("v1 without identity or bloc layer + history rates", "m_hist_noid",
     "--affinity 0 --no-identity --hist"),
    ("v1 without identity or bloc layer + metadata prior", "m_prior_noid",
     "--affinity 0 --no-identity --prior 1"),
    ("Blend: v1 + qc_bloc", f"blend:{MAIN}+qc_bloc", ""),
    ("Blend: v1 + metadata regression (meta_lr)", f"blend:{MAIN}+meta_lr", ""),
    ("Blend: v1 + qc_bloc_meta", f"blend:{MAIN}+qc_bloc_meta", ""),
    ("Blend: v1 + qc_bloc + meta_lr", f"blend:{MAIN}+qc_bloc+meta_lr", ""),
]


def fusion_preds(name, fold, lab):
    """Seeds 0-2 ensemble of a run, a baseline's single file, or an equal-weight blend."""
    if name.startswith("blend:"):
        parts = [fusion_preds(n, fold, lab) for n in name[6:].split("+")]
        return None if any(p is None for p in parts) else S.blend_many(parts)
    paths = [f"data/preds/{name}_{fold}_s{s}.parquet" for s in (0, 1, 2)]
    if all(Path(p).exists() for p in paths):
        return S.ensemble(paths)
    if Path(f"data/preds/{name}_{fold}.parquet").exists():
        return lab[fold][0][["case_id", "justice"]].merge(
            pd.read_parquet(f"data/preds/{name}_{fold}.parquet"), on=["case_id", "justice"])
    return None


def paired_auc_ci(a, b, lab, n=1000, seed=0):
    """95% CI of AUC(b) - AUC(a), resampling cases (same resamples for both)."""
    d = lab[0].merge(a[["case_id", "justice", "p"]], on=["case_id", "justice"]).merge(
        b[["case_id", "justice", "p"]], on=["case_id", "justice"], suffixes=("_a", "_b"))
    groups = [g for _, g in d.groupby("case_id")]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(roc_auc_score(s.vote, s.p_b) - roc_auc_score(s.vote, s.p_a))
    return np.percentile(diffs, [2.5, 97.5])


ENCODERS = [  # (label, run name / "blend:a+b", train_v1.py flags that differ from v1)
    ("Model v1, Legal-BERT turn vectors", MAIN, ""),
    ("Model v1 without text (for scale)", "d_notext", "--no-text"),
    ("Model v1, BGE turn vectors", "e_bge", "--emb embeddings/turns_bge.npy"),
    ("Model v1, both encoders concatenated", "e_both",
     "--emb embeddings/turns_last.npy embeddings/turns_bge.npy"),
    ("BGE, low-rank text projection (rank 64)", "e_bge_bn64",
     "--emb embeddings/turns_bge.npy --bottleneck 64"),
    ("BGE, dropout 0.3 on the turn vectors", "e_bge_drop",
     "--emb embeddings/turns_bge.npy --in-drop 0.3"),
    ("BGE, 4 epochs instead of 6", "e_bge_e4", "--emb embeddings/turns_bge.npy --epochs 4"),
    ("Legal-BERT, case-centred turn vectors", "e_lb_center", "--center-text"),
    ("BGE, case-centred turn vectors", "e_bge_center",
     "--emb embeddings/turns_bge.npy --center-text"),
    ("Blend: v1 (Legal-BERT) + v1 (BGE)", f"blend:{MAIN}+e_bge", ""),
]


def fusion(lab):
    """Metadata fusion variants against transcript-only v1 -> results/v1_fusion.md."""
    variant_table(
        lab, FUSION, "results/v1_fusion.md",
        "# Metadata fusion: model v1 with the SCDB case profile",
        "Voting history of held-out cases is frozen at the training cutoff unless the row "
        "says rolling.")


def encoders(lab):
    """Frozen-encoder comparison -> results/v1_encoders.md."""
    variant_table(
        lab, ENCODERS, "results/v1_encoders.md",
        "# Encoder comparison: Legal-BERT vs BGE turn vectors",
        "Legal-BERT: nlpaueb/legal-bert-base-uncased, mean-pooled last layer. BGE: "
        "BAAI/bge-base-en-v1.5, [CLS] vector, unit length. Both frozen, 768-d, same "
        "chunking of long turns.")


def variant_table(lab, variants, path, title, note):
    """Variants of v1 against v1 itself, on dev and val.

    Every network row is the ensemble of seeds 0-2 (v1 too, so the comparison is like for
    like); the CI is a paired case bootstrap of the AUC difference to v1."""
    from scotus.report.make_figures import parts
    rows = []
    for label, name, flags in variants:
        row = {"Variant": label, "Flags": flags}
        for fold in ("dev", "val"):
            p = fusion_preds(name, fold, lab)
            if p is None:
                break
            sc = S.scores(p, lab[fold][0])
            case_auc, within = parts(p, *lab[fold])
            row.update({f"{fold} AUC": f"{sc['auc']:.3f}", f"{fold} F1": f"{sc['f1']:.3f}",
                        f"{fold} case-level": f"{case_auc:.3f}",
                        f"{fold} within-case": f"{within:.3f}"})
            if fold == "val":
                row.update({"val 2017 AUC": f"{sc['auc17']:.3f}",
                            "val 2017 F1": f"{sc['f117']:.3f}"})
            if name != MAIN:
                lo, hi = paired_auc_ci(fusion_preds(MAIN, fold, lab), p, lab[fold])
                row[f"{fold} AUC vs v1"] = f"[{lo:+.3f}, {hi:+.3f}]"
        else:
            rows.append(row)
    df = pd.DataFrame(rows).fillna("")
    md = [title, "",
          "Same recipe as v1 (train 1980+, 6 epochs); network rows are the mean-probability "
          "ensemble of seeds 0-2; cutoff 0.5. 'AUC vs v1': 95% CI of the AUC difference to "
          "v1, paired bootstrap over cases. case-level = AUC of a case's mean probability "
          "against the winner; within-case = AUC of ranking the justices inside "
          f"non-unanimous cases. {note} Test untouched.", "",
          df.to_markdown(index=False), ""]
    Path(path).write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
