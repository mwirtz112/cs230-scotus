# Reading the Room: Predicting Supreme Court Votes from Oral Arguments

CS230 (Deep Learning), Stanford, Autumn 2026 — Matthew Wirtz

Given the transcript of a Supreme Court oral argument, the model predicts how each justice
will vote: for the petitioner or for the respondent. A frozen Legal-BERT turns each turn of
the argument into a vector. A small transformer, trained here, reads those vectors and
gathers evidence for each justice (a summary of the argument, learned counts of the
questioning, hand-built counts, and who the justice is). A voting-bloc layer then lets each
justice's prediction depend on how the colleagues lean. The case goes to the side with the
most predicted votes.

## Results

The test covers the eight terms 2018–2025 (481 cases, 4,298 votes) and is walk-forward:
before each term, every method is refit on all earlier terms and then predicts that term.
It was run once, after the model, its training recipe and the baselines had been fixed
([`results/walk_forward_final.md`](results/walk_forward_final.md); every score is
recomputed independently in [`results/walk_forward_final_check.md`](results/walk_forward_final_check.md)).

| Method | Uses | Macro-F1 [95% CI] | AUC | Votes right | Cases right |
|---|---|---|---|---|---|
| Always petitioner | – | 0.381 [0.369, 0.393] | 0.500 | 61.5% | 67.6% |
| Justice vote rate | V | 0.508 [0.497, 0.520] | 0.527 | 56.9% | 67.6% |
| Flat language model (frozen ModernBERT) | T | 0.553 [0.528, 0.577] | 0.569 | 58.0% | 59.9% |
| Question count | T | 0.603 [0.582, 0.624] | 0.656 | 60.5% | 54.9% |
| Case facts (Katz-style random forest) | F, V | 0.623 [0.600, 0.647] | 0.673 | 65.9% | 63.8% |
| Question statistics (regression) | T | 0.644 [0.623, 0.667] | 0.729 | 64.7% | 58.6% |
| &nbsp;&nbsp;+ colleagues | T | 0.676 [0.654, 0.699] | 0.755 | 68.1% | 63.4% |
| &nbsp;&nbsp;+ voting record | T, V | 0.689 [0.667, 0.713] | 0.766 | 69.5% | 64.4% |
| **Model v1** | **T** | **0.702 [0.677, 0.726]** | **0.781** | **70.8%** | **66.3%** |
| Average of the two rows above | T, V | 0.723 [0.700, 0.746] | 0.800 | 72.9% | 67.6% |

Uses: T the transcript, V the justices' past votes, F facts about the case (Supreme Court
Database). Macro-F1 is the main score; the interval is a 95% bootstrap over cases.

- The model calls 70.8% of the votes right, against 61.5% for always choosing the
  petitioner, and has the highest macro-F1 and AUC of the methods rebuilt from prior papers.
- It does not name the winner of a case more often than always choosing the petitioner
  (66.3% against 67.6%). Training weights both sides equally, so the model calls the
  respondent too often; a rule fixed before the test that puts the petitioner's usual
  advantage back raises this to 71.1% ([`results/case_call_final.md`](results/case_call_final.md)).
- The voting-bloc layer learns the Court's two camps by itself: its weights order the
  justices of 2010–15 the way their actual votes do (correlation 0.98;
  [`results/v1_inside.md`](results/v1_inside.md)).

Validation results (terms 2016–17) and the ablations are in
[`results/v1_main.md`](results/v1_main.md) and [`results/v1_ablations.md`](results/v1_ablations.md).

## Repository layout

```
src/scotus/
  data/        prepare_data.py     ConvoKit corpus -> turn, case and vote tables (1955-2019)
               scrape_oyez.py      transcripts of 2020-2025 from the Oyez API
               prepare_oyez.py     the same tables for 2020-2025
               prepare_scdb.py     join to the Supreme Court Database (pre-decision fields only)
  features/    encode_turns.py     one frozen Legal-BERT vector per turn
               features.py         structural features per turn, counts per vote
               meta_features.py    case profile and past-only voting history
               flat_encode.py      one ModernBERT vector per transcript (flat baseline)
  models/      model_v1.py         the network
               train_v1.py         training on the dev and validation folds
               probe.py            a linear probe on the turn vectors
  baselines/   baselines.py        always petitioner, justice vote rate, question count
               qc_plus.py          regressions on question counts, alone and with colleagues
               meta_baselines.py   case facts: a Katz-style random forest; + voting record
               flat_baseline.py    flat language model (frozen ModernBERT)
               flat_finetune.py    the same with its top 4 layers trained
  evaluation/  eval.py             shared scores and bootstrap intervals
               walk_forward.py     the test (2018-2025) and its rehearsal (2012-2017)
               summarize_v1.py     seed ensembles and comparison tables
               check_*.py          checks on the splits, the model, the fine-tuning runs and
                                   the test run
  analysis/    inside_v1.py        which parts of the trained network carry the prediction
               case_call.py        how well each method names the winner of a case
               timing.py           does the timing of the argument help? (no)
               audio_pitch.py      the pitch of every turn in the argument recordings
               audio_test.py       does the justices' vocal pitch help? (no)
  report/      make_figures.py, report_tables.py, final_tables.py    figures and tables
results/       every table and figure, plus one record per trained network
data/          a small part of the data (the scripts rebuild the rest):
                 cases.parquet, votes.parquet   cases and vote labels, 1955-2019
                 ext/                           the 2020-2025 terms built from Oyez
                 preds/                         every saved prediction, test run included
```

Every script starts with a header that says what it does, what it reads and writes, and how
to run it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run every script from the repository root as a module, for example
`python -m scotus.evaluation.walk_forward`. Scripts that read downloaded files run in
isolated mode: `python -I -m ...`. Outputs go to `results/` and to `data/`, `embeddings/`
and `checkpoints/`; of those last three, only the small part of `data/` listed above is in
the repository.

## Data

- **ConvoKit Supreme Court corpus**, terms 1955–2019: transcripts from Oyez, votes from the
  Supreme Court Database (`scotus.data.prepare_data` downloads it).
- **Oyez API**, terms 2020–2025, which ConvoKit does not cover (`scotus.data.scrape_oyez`,
  one request every 10 seconds as Oyez's robots.txt asks). Oyez content is CC BY-NC 4.0.
- **Supreme Court Database**, release `SCDB_2026_01`: download the two "docket" CSV zips
  from <http://scdb.wustl.edu/data.php> into `data/scdb/raw/`
  (`SCDB_2026_01_caseCentered_Docket.csv.zip`, `SCDB_2026_01_justiceCentered_Docket.csv.zip`).
  Only fields known before the decision are used; see the leakage audit in
  [`results/scdb_match.md`](results/scdb_match.md).

Splits are chronological: training up to 2015 (the network is trained on the terms from
1980 onward), validation 2016–17, test 2018–2025. The test labels are locked: `eval` and
`walk_forward` refuse the test terms without `--final`.

**Included in this repository** (31 MB of `data/`): the cases and vote labels of
1955–2019, the 2020–2025 tables built from Oyez (`data/ext/`; their transcript text is
from the Oyez Project, CC BY-NC 4.0), and every saved prediction (`data/preds/`; the test
run is in `data/preds/wf_final/`). With these, the test scores can be rechecked without
training anything:

```bash
python -m scotus.evaluation.check_walk_final
python -m scotus.analysis.case_call --final
```

## Pipeline

| Step | Command | Writes |
|---|---|---|
| 1 | `python -m scotus.data.prepare_data` | `data/{turns,cases,votes}.parquet` |
| 2 | `python -I -m scotus.data.scrape_oyez --terms 2020 2021 2022 2023 2024 2025`, then `python -I -m scotus.data.prepare_oyez --terms 2020 2021 2022 2023 2024 2025` | `data/oyez/raw/`, `data/ext/` |
| 3 | `python -m scotus.evaluation.check_splits` | `data/split_checks.md` |
| 4 | `python -m scotus.features.encode_turns`, and again with `--data data/ext --out embeddings/ext` | `embeddings/` |
| 5 | `python -m scotus.features.features`, and again with `--data data/ext` | `data/{turn,vote}_feats.parquet` |
| 6 | `python -I -m scotus.data.prepare_scdb`, then `python -m scotus.features.meta_features` | `data/scdb_*.parquet`, `data/meta_*.parquet` |
| 7 | `python -m scotus.baselines.baselines`, `... .qc_plus`, `... .meta_baselines`; `python -I -m scotus.features.flat_encode` (`--rest` for the whole transcript), then `python -m scotus.baselines.flat_baseline` | `data/preds/`, `results/` |
| 8 | `python -m scotus.models.train_v1 --name v1_main --epochs 6 --train-start 1980 --affinity 4 --seeds 0 1 2 3 4 --save-model --eval-hide-ids` | `results/v1_runs/`, `checkpoints/` |
| 9 | `python -m scotus.evaluation.summarize_v1 --ci v1_main`, `python -m scotus.report.report_tables` | `results/v1_*.md` |
| 10 | `python -m scotus.evaluation.walk_forward` (rehearsal on 2012–17; no test label is read) | `results/walk_forward_rehearsal.md` |
| 11 | The test, once: `python -I -m scotus.data.prepare_scdb --final`, `python -m scotus.features.meta_features`, `python -m scotus.evaluation.walk_forward --final` | `results/walk_forward_final.md` |
| 12 | `python -m scotus.evaluation.check_walk_final`, `python -m scotus.analysis.case_call --final`, `python -m scotus.analysis.inside_v1`, `python -m scotus.report.make_figures` | `results/` |

Step 11 has been made. Running it again would replace the test predictions with new ones.

## Side experiments

These were tested and are reported as results that did not help:

- **Timing** of the turns: speaking time, pace and interruptions add +0.002 AUC to the
  regression with colleagues ([`results/timing.md`](results/timing.md)).
- **Vocal pitch**, from 514 argument recordings of 2010–17 (7.8 GB from Oyez; needs
  `ffmpeg`). The published effect replicates: a justice's pitch is higher toward the side
  the justice votes against in 55.1% of justice–case pairs. But it adds only +0.003 AUC
  ([`results/audio_pitch.md`](results/audio_pitch.md)).
- **Fine-tuning** the top four layers of the flat ModernBERT: no gain
  (`results/flat_runs/`).
- **A general sentence encoder** (`BAAI/bge-base-en-v1.5`) in place of Legal-BERT: worse
  ([`results/v1_encoders.md`](results/v1_encoders.md)).
- **Case facts inside the network**: no clear gain ([`results/v1_fusion.md`](results/v1_fusion.md)).

Code that is no longer here (the milestone model v0 and a stance-label experiment) is in
the git history.
