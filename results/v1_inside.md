# Inside the trained network (analysis C)

`inside_v1.py`, written on Nov 14 after the test run; design fixed first in docs/writeup_notes.md §23.1. The saved fits of the main recipe (`v1_main`, five seeds a fold) are run on their own held-out terms. Nothing is fitted; `model_v1.py` and `train_v1.py` are unchanged. The fits of the test run were not saved, so the test terms play no part in M1–M4.

- **val fold**: trained on terms 1980–2015, analysed on 2016–17 (123 cases, 1,049 votes). The loaded networks reproduce their saved predictions: largest difference in a probability 3.6e-07.
- **dev fold**: trained on terms 1980–2011, analysed on 2012–15 (256 cases, 2,238 votes). The loaded networks reproduce their saved predictions: largest difference in a probability 2.7e-07.

## M1. Which path carries the prediction (knock-outs at inference)

One thing at a time is replaced by its average over the held-out votes, in the trained network; nothing is retrained. Mean probability over the five seeds, cutoff 0.5. Change = knocked-out minus intact, 95% interval from a paired bootstrap over cases (1,000 draws). Case-level AUC = a case's mean probability against the winner; within-case AUC = ranking the justices inside split cases. A knock-out shows what the trained network relies on; the retrained ablations (`results/v1_ablations.md`) show what it needs.

**val fold, terms 2016–17**

| Knocked out                                                                         | AUC   | Macro-F1   | Case-level AUC   | Within-case AUC   | Pred. petitioner %   | AUC change              | Macro-F1 change         |
|:------------------------------------------------------------------------------------|:------|:-----------|:-----------------|:------------------|:---------------------|:------------------------|:------------------------|
| nothing (the intact network)                                                        | 0.821 | 0.748      | 0.800            | 0.905             | 60.5                 |                         |                         |
| content readout: the attention readout's content vector → its average               | 0.794 | 0.722      | 0.723            | 0.908             | 62.6                 | -0.027 [-0.058, +0.002] | -0.026 [-0.059, +0.008] |
| learned counters: the learned counters → their average                              | 0.823 | 0.743      | 0.812            | 0.888             | 62.2                 | +0.002 [-0.008, +0.011] | -0.005 [-0.019, +0.011] |
| count features: the hand-built count features (MLP and direct path) → their average | 0.793 | 0.700      | 0.806            | 0.878             | 60.9                 | -0.028 [-0.049, -0.009] | -0.048 [-0.080, -0.016] |
| turn text: every turn's Legal-BERT vector → the training mean                       | 0.778 | 0.722      | 0.709            | 0.899             | 66.5                 | -0.043 [-0.089, +0.004] | -0.026 [-0.073, +0.023] |
| identity: every justice identity hidden (the bloc term goes with it)                | 0.778 | 0.703      | 0.790            | 0.831             | 60.2                 | -0.043 [-0.064, -0.023] | -0.044 [-0.075, -0.018] |
| bloc term: the bloc term removed from the logit                                     | 0.786 | 0.715      | 0.798            | 0.849             | 60.2                 | -0.035 [-0.054, -0.019] | -0.033 [-0.056, -0.012] |
| uniform attention: the readout's attention made uniform over the turns              | 0.819 | 0.746      | 0.788            | 0.908             | 62.5                 | -0.002 [-0.011, +0.006] | -0.002 [-0.015, +0.010] |
| no own-turn bonus: the bonus on the justice's own turns set to 0                    | 0.817 | 0.740      | 0.799            | 0.896             | 60.5                 | -0.004 [-0.007, -0.001] | -0.008 [-0.017, +0.000] |
| no identity in the query: the identity removed from the readout's query only        | 0.822 | 0.745      | 0.790            | 0.909             | 62.6                 | +0.001 [-0.008, +0.009] | -0.003 [-0.017, +0.010] |

**dev fold, terms 2012–15**

| Knocked out                                                                         | AUC   | Macro-F1   | Case-level AUC   | Within-case AUC   | Pred. petitioner %   | AUC change              | Macro-F1 change         |
|:------------------------------------------------------------------------------------|:------|:-----------|:-----------------|:------------------|:---------------------|:------------------------|:------------------------|
| nothing (the intact network)                                                        | 0.744 | 0.667      | 0.708            | 0.801             | 48.2                 |                         |                         |
| content readout: the attention readout's content vector → its average               | 0.739 | 0.659      | 0.696            | 0.804             | 47.7                 | -0.005 [-0.034, +0.023] | -0.008 [-0.036, +0.021] |
| learned counters: the learned counters → their average                              | 0.732 | 0.646      | 0.706            | 0.776             | 45.7                 | -0.012 [-0.023, -0.002] | -0.022 [-0.036, -0.007] |
| count features: the hand-built count features (MLP and direct path) → their average | 0.704 | 0.624      | 0.684            | 0.770             | 45.2                 | -0.040 [-0.055, -0.025] | -0.043 [-0.062, -0.025] |
| turn text: every turn's Legal-BERT vector → the training mean                       | 0.696 | 0.635      | 0.675            | 0.779             | 59.7                 | -0.049 [-0.082, -0.019] | -0.032 [-0.068, +0.003] |
| identity: every justice identity hidden (the bloc term goes with it)                | 0.709 | 0.629      | 0.707            | 0.752             | 46.3                 | -0.035 [-0.050, -0.021] | -0.038 [-0.057, -0.020] |
| bloc term: the bloc term removed from the logit                                     | 0.715 | 0.628      | 0.706            | 0.763             | 44.9                 | -0.029 [-0.041, -0.018] | -0.039 [-0.055, -0.023] |
| uniform attention: the readout's attention made uniform over the turns              | 0.739 | 0.670      | 0.709            | 0.807             | 51.4                 | -0.005 [-0.014, +0.004] | +0.003 [-0.013, +0.018] |
| no own-turn bonus: the bonus on the justice's own turns set to 0                    | 0.739 | 0.660      | 0.707            | 0.793             | 48.1                 | -0.005 [-0.008, -0.003] | -0.007 [-0.014, -0.001] |
| no identity in the query: the identity removed from the readout's query only        | 0.743 | 0.667      | 0.709            | 0.805             | 52.0                 | -0.001 [-0.010, +0.007] | -0.001 [-0.015, +0.013] |

## M2. What the readout per justice looks at

Attention of a justice's readout over the turns of the case, summed by who speaks, next to that group's share of the turns (what a uniform readout would give). Averages over the held-out votes and the five seeds.

**val fold, justices who speak in the case (900 votes)**

| Who speaks              | Share of the turns %   | Attention %   | Attention %, no own-turn bonus   | Attention %, no identity in the query   |
|:------------------------|:-----------------------|:--------------|:---------------------------------|:----------------------------------------|
| The justice's own turns | 7.1                    | 17.0          | 7.0                              | 17.8                                    |
| The other justices      | 44.7                   | 39.1          | 44.1                             | 40.4                                    |
| Petitioner's advocate   | 22.6                   | 21.1          | 23.5                             | 19.5                                    |
| Respondent's advocate   | 20.8                   | 16.9          | 19.0                             | 18.1                                    |
| Anyone else             | 4.8                    | 5.8           | 6.5                              | 4.2                                     |

**val fold, silent justices (149 votes)**

| Who speaks              | Share of the turns %   | Attention %   | Attention %, no own-turn bonus   | Attention %, no identity in the query   |
|:------------------------|:-----------------------|:--------------|:---------------------------------|:----------------------------------------|
| The justice's own turns | 0.0                    | 0.0           | 0.0                              | 0.0                                     |
| The other justices      | 51.7                   | 47.3          | 47.3                             | 52.5                                    |
| Petitioner's advocate   | 23.1                   | 26.3          | 26.3                             | 22.5                                    |
| Respondent's advocate   | 20.3                   | 21.7          | 21.7                             | 20.1                                    |
| Anyone else             | 4.9                    | 4.7           | 4.7                              | 4.9                                     |

- val: a case has 231 turns on average; the readout spreads its attention over the equivalent of 150 turns. A justice's own turns get 2.41 times their share of the turns.
- val: taking the identity out of the query moves 32.3% of the attention (total-variation distance); setting the own-turn bonus to 0 moves 10.0% (justices who speak).

**val fold: how far the weights moved from their start** (mean of five seeds; the starting weights are rebuilt from the seed; relative change = size of the change divided by the size at the start)

| Part                                                            | Size at the start   | Size after training   | Relative change   | Cosine with the start   |
|:----------------------------------------------------------------|:--------------------|:----------------------|:------------------|:------------------------|
| Shared query q                                                  | 0.954               | 0.951                 | 0.144             | 0.990                   |
| Identity in the query, justices with training votes             | 71.720              | 70.356                | 0.021             | 1.000                   |
| Identity in the query, speakers without a training vote (check) | 50.665              | 49.706                | 0.019             | 1.000                   |
| Identity vector in the head, justices with training votes       | 25.264              | 24.753                | 0.022             | 1.000                   |
| Text projection W                                               | 9.230               | 10.385                | 0.545             | 0.875                   |
| The two transformer layers                                      | 49.747              | 49.171                | 0.143             | 0.990                   |
| Counter weights                                                 | 1.636               | 1.848                 | 0.437             | 0.923                   |
| Head MLP, first layer                                           | 6.536               | 6.680                 | 0.289             | 0.959                   |
| Speaker embeddings (start: 0)                                   | 0.000               | 1.286                 |                   |                         |
| Pair table of the bloc layer (start: 0)                         | 0.000               | 6.049                 |                   |                         |
| Gain and lean per justice (start: 0)                            | 0.000               | 0.565                 |                   |                         |
| Own-turn bonus (one number)                                     | 1.000               | 1.069                 |                   |                         |

**dev fold, justices who speak in the case (1,943 votes)**

| Who speaks              | Share of the turns %   | Attention %   | Attention %, no own-turn bonus   | Attention %, no identity in the query   |
|:------------------------|:-----------------------|:--------------|:---------------------------------|:----------------------------------------|
| The justice's own turns | 6.7                    | 16.5          | 6.7                              | 17.1                                    |
| The other justices      | 44.8                   | 40.4          | 45.3                             | 40.8                                    |
| Petitioner's advocate   | 20.9                   | 20.5          | 22.7                             | 18.0                                    |
| Respondent's advocate   | 21.1                   | 15.7          | 17.6                             | 18.3                                    |
| Anyone else             | 6.5                    | 6.9           | 7.7                              | 5.8                                     |

**dev fold, silent justices (295 votes)**

| Who speaks              | Share of the turns %   | Attention %   | Attention %, no own-turn bonus   | Attention %, no identity in the query   |
|:------------------------|:-----------------------|:--------------|:---------------------------------|:----------------------------------------|
| The justice's own turns | 0.0                    | 0.0           | 0.0                              | 0.0                                     |
| The other justices      | 51.4                   | 48.0          | 48.0                             | 52.5                                    |
| Petitioner's advocate   | 21.0                   | 23.1          | 23.1                             | 20.4                                    |
| Respondent's advocate   | 21.0                   | 21.7          | 21.7                             | 20.5                                    |
| Anyone else             | 6.6                    | 7.2           | 7.2                              | 6.6                                     |

- dev: a case has 227 turns on average; the readout spreads its attention over the equivalent of 138 turns. A justice's own turns get 2.47 times their share of the turns.
- dev: taking the identity out of the query moves 36.2% of the attention (total-variation distance); setting the own-turn bonus to 0 moves 9.8% (justices who speak).

**dev fold: how far the weights moved from their start** (mean of five seeds; the starting weights are rebuilt from the seed; relative change = size of the change divided by the size at the start)

| Part                                                            | Size at the start   | Size after training   | Relative change   | Cosine with the start   |
|:----------------------------------------------------------------|:--------------------|:----------------------|:------------------|:------------------------|
| Shared query q                                                  | 0.954               | 0.952                 | 0.128             | 0.992                   |
| Identity in the query, justices with training votes             | 71.720              | 70.462                | 0.020             | 1.000                   |
| Identity in the query, speakers without a training vote (check) | 50.665              | 49.778                | 0.018             | 1.000                   |
| Identity vector in the head, justices with training votes       | 25.264              | 24.791                | 0.020             | 1.000                   |
| Text projection W                                               | 9.230               | 10.329                | 0.531             | 0.880                   |
| The two transformer layers                                      | 49.747              | 49.205                | 0.136             | 0.991                   |
| Counter weights                                                 | 1.636               | 1.838                 | 0.429             | 0.924                   |
| Head MLP, first layer                                           | 6.536               | 6.676                 | 0.282             | 0.961                   |
| Speaker embeddings (start: 0)                                   | 0.000               | 1.299                 |                   |                         |
| Pair table of the bloc layer (start: 0)                         | 0.000               | 5.861                 |                   |                         |
| Gain and lean per justice (start: 0)                            | 0.000               | 0.559                 |                   |                         |
| Own-turn bonus (one number)                                     | 1.000               | 1.073                 |                   |                         |

## M3. What the learned counters count

Weight of a turn = sum of its eight counters, scaled so that the mean justice turn is 1 (a plain count gives every turn 1). Mean over the five seeds; the last column is the range of the five seeds' own means.

**val fold**

| Justice turns by   | Group                         | Turns   | Share %   | Mean weight   | Seeds, min–max   |
|:-------------------|:------------------------------|:--------|:----------|:--------------|:-----------------|
| Length             | 1–5 words                     | 2890    | 19.7      | 0.52          | 0.40–0.79        |
| Length             | 6–20 words                    | 4839    | 33.0      | 0.70          | 0.61–0.86        |
| Length             | 21–50 words                   | 3783    | 25.8      | 1.14          | 1.05–1.25        |
| Length             | 51+ words                     | 3143    | 21.4      | 1.73          | 1.32–2.05        |
| Question mark      | none                          | 8081    | 55.1      | 0.93          | 0.88–0.97        |
| Question mark      | at least one                  | 6574    | 44.9      | 1.09          | 1.03–1.15        |
| Cut off            | no                            | 12358   | 84.3      | 1.05          | 1.03–1.09        |
| Cut off            | yes                           | 2297    | 15.7      | 0.71          | 0.54–0.84        |
| Aimed at           | the petitioner                | 6802    | 46.4      | 1.13          | 0.79–1.51        |
| Aimed at           | the respondent                | 6379    | 43.5      | 0.80          | 0.47–1.10        |
| Aimed at           | nobody (e.g. after an amicus) | 1474    | 10.1      | 1.24          | 0.96–1.51        |

- Spread over justice turns: coefficient of variation 0.76; the heaviest fifth of the turns carries 45% of the weight (20% under a plain count). Rank correlation of weight and length: 0.72. An advocate's turn weighs 1.53 on the same scale.
- Does the learned weighting sharpen the question count? Feature: log(1 + weight of the justice's turns aimed at the respondent) − the same for the petitioner; 896 votes of justices who question at least one side. AUC for the vote: plain count 0.723, learned weights 0.761 (difference +0.038 [+0.021, +0.056]; single seeds 0.746–0.773). For reference, added after the first results: weight = the turn's number of words, 0.739 (learned − words: +0.022 [+0.004, +0.040]). The 40 single counters (8 × 5 seeds): 0.653 to 0.778, median 0.747; the best one is a choice made after the fact.

**dev fold**

| Justice turns by   | Group                         | Turns   | Share %   | Mean weight   | Seeds, min–max   |
|:-------------------|:------------------------------|:--------|:----------|:--------------|:-----------------|
| Length             | 1–5 words                     | 4615    | 15.4      | 0.53          | 0.26–0.78        |
| Length             | 6–20 words                    | 9790    | 32.7      | 0.72          | 0.56–0.84        |
| Length             | 21–50 words                   | 8703    | 29.1      | 1.11          | 1.06–1.17        |
| Length             | 51+ words                     | 6786    | 22.7      | 1.59          | 1.30–1.93        |
| Question mark      | none                          | 15982   | 53.5      | 0.95          | 0.91–1.00        |
| Question mark      | at least one                  | 13912   | 46.5      | 1.05          | 1.00–1.10        |
| Cut off            | no                            | 26398   | 88.3      | 1.03          | 1.02–1.05        |
| Cut off            | yes                           | 3496    | 11.7      | 0.75          | 0.64–0.87        |
| Aimed at           | the petitioner                | 12956   | 43.3      | 1.17          | 0.88–1.62        |
| Aimed at           | the respondent                | 13231   | 44.3      | 0.77          | 0.41–1.02        |
| Aimed at           | nobody (e.g. after an amicus) | 3707    | 12.4      | 1.25          | 0.93–1.60        |

- Spread over justice turns: coefficient of variation 0.69; the heaviest fifth of the turns carries 43% of the weight (20% under a plain count). Rank correlation of weight and length: 0.65. An advocate's turn weighs 1.35 on the same scale.
- Does the learned weighting sharpen the question count? Feature: log(1 + weight of the justice's turns aimed at the respondent) − the same for the petitioner; 1,909 votes of justices who question at least one side. AUC for the vote: plain count 0.640, learned weights 0.670 (difference +0.030 [+0.018, +0.042]; single seeds 0.654–0.673). For reference, added after the first results: weight = the turn's number of words, 0.656 (learned − words: +0.014 [+0.001, +0.027]). The 40 single counters (8 × 5 seeds): 0.612 to 0.680, median 0.661; the best one is a choice made after the fact.

## M4. What the bloc layer learned

Effective weight W[j, k]: how far justice j's logit moves when colleague k's own evidence moves by one logit toward the petitioner, through the bloc layer, on a full bench (the table's entry, times the slope of the messages on the sender's own logit, divided by 8 colleagues). Rows: the justice whose vote is predicted; columns: the colleague. Mean of five seeds; order = leading eigenvector of the table, no outside ideology score.

**val fold: the network's table**

|           |   Roberts |   Scalia |    Alito |   Thomas |   Kennedy |   Breyer |   Ginsburg |   Sotomayor |    Kagan |
|:----------|----------:|---------:|---------:|---------:|----------:|---------:|-----------:|------------:|---------:|
| Roberts   |  +nan     |   +0.185 |   +0.074 |   +0.030 |    -0.118 |   +0.079 |     +0.100 |      -0.032 |   -0.052 |
| Scalia    |    +0.171 | +nan     |   +0.019 |   +0.041 |    -0.085 |   -0.016 |     +0.035 |      -0.069 |   -0.140 |
| Alito     |    +0.182 |   +0.168 | +nan     |   +0.013 |    -0.059 |   +0.052 |     +0.086 |      -0.073 |   -0.126 |
| Thomas    |    +0.173 |   +0.261 |   +0.115 | +nan     |    -0.052 |   -0.035 |     +0.044 |      -0.126 |   -0.161 |
| Kennedy   |    +0.047 |   +0.153 |   +0.102 |   +0.017 |  +nan     |   +0.068 |     +0.164 |      -0.016 |   -0.021 |
| Breyer    |    -0.105 |   -0.053 |   -0.055 |   +0.025 |    -0.113 | +nan     |     +0.255 |      -0.038 |   +0.057 |
| Ginsburg  |    -0.147 |   -0.090 |   -0.156 |   -0.020 |    -0.124 |   +0.081 |   +nan     |      +0.062 |   +0.125 |
| Sotomayor |    -0.112 |   +0.019 |   -0.131 |   +0.031 |    -0.122 |   +0.123 |     +0.123 |    +nan     |   +0.055 |
| Kagan     |    -0.072 |   +0.047 |   -0.064 |   +0.020 |    -0.144 |   +0.097 |     +0.117 |      -0.010 | +nan     |

**For comparison: correlation of the same justices' actual votes, terms 2010–15**

|           |   Roberts |   Scalia |   Alito |   Thomas |   Kennedy |   Breyer |   Ginsburg |   Sotomayor |   Kagan |
|:----------|----------:|---------:|--------:|---------:|----------:|---------:|-----------:|------------:|--------:|
| Roberts   |      1.00 |     0.71 |    0.73 |     0.63 |      0.65 |     0.45 |       0.31 |        0.36 |    0.39 |
| Scalia    |      0.71 |     1.00 |    0.65 |     0.75 |      0.51 |     0.20 |       0.23 |        0.23 |    0.34 |
| Alito     |      0.73 |     0.65 |    1.00 |     0.69 |      0.59 |     0.34 |       0.21 |        0.22 |    0.28 |
| Thomas    |      0.63 |     0.75 |    0.69 |     1.00 |      0.49 |     0.25 |       0.16 |        0.20 |    0.24 |
| Kennedy   |      0.65 |     0.51 |    0.59 |     0.49 |      1.00 |     0.55 |       0.44 |        0.48 |    0.55 |
| Breyer    |      0.45 |     0.20 |    0.34 |     0.25 |      0.55 |     1.00 |       0.71 |        0.69 |    0.75 |
| Ginsburg  |      0.31 |     0.23 |    0.21 |     0.16 |      0.44 |     0.71 |       1.00 |        0.74 |    0.80 |
| Sotomayor |      0.36 |     0.23 |    0.22 |     0.20 |      0.48 |     0.69 |       0.74 |        1.00 |    0.77 |
| Kagan     |      0.39 |     0.34 |    0.28 |     0.24 |      0.55 |     0.75 |       0.80 |        0.77 |    1.00 |

- Order from the table alone: Roberts, Scalia, Alito, Thomas, Kennedy, Breyer, Ginsburg, Sotomayor, Kagan.
- Position on that axis (farthest justice = 1), and for comparison the position on the axis that divides the Court in the actual votes of 2010–15 (the second principal axis of the vote correlations; the first is general agreement): Roberts -1.00 / -0.69; Scalia -0.99 / -0.94; Alito -0.92 / -0.90; Thomas -0.80 / -1.00; Kennedy -0.25 / -0.15; Breyer +0.54 / +0.78; Ginsburg +0.66 / +0.99; Sotomayor +0.71 / +0.91; Kagan +0.83 / +0.85. Correlation of the two line-ups: 0.98.
- Rank correlation over the 72 ordered pairs: with the correlation of the actual votes in 2010–15, 0.53 (2012–15 only: 0.52); with the weights of the `qc_bloc` regression, 0.91.
- A colleague's average weight over the eight others (the column mean): Roberts +0.017, Alito -0.012, Scalia +0.086, Thomas +0.020, Kennedy -0.102, Breyer +0.056, Ginsburg +0.116, Sotomayor -0.038, Kagan -0.033. With that average removed (added after the first results), the rank correlation with the actual votes is 0.77.
- Mean weight for pairs whose votes agree more than the median pair: +0.063; for the others: -0.039.
- Two seeds' tables correlate at 0.97 on average. The bloc term has a standard deviation of 0.53 logits over the votes (the bloc-free logit: 1.12); pair constants plus colleagues' own logits weighted by W explain 67% of its variance.

**dev fold: the network's table**

|           |    Alito |   Roberts |   Scalia |   Thomas |   Kennedy |   Sotomayor |   Breyer |   Ginsburg |    Kagan |
|:----------|---------:|----------:|---------:|---------:|----------:|------------:|---------:|-----------:|---------:|
| Alito     | +nan     |    +0.166 |   +0.154 |   +0.027 |    +0.013 |      -0.059 |   +0.048 |     +0.087 |   -0.080 |
| Roberts   |   +0.062 |  +nan     |   +0.203 |   +0.031 |    -0.057 |      -0.021 |   +0.056 |     +0.114 |   -0.075 |
| Scalia    |   +0.048 |    +0.180 | +nan     |   +0.041 |    -0.086 |      -0.043 |   -0.008 |     +0.035 |   -0.092 |
| Thomas    |   +0.130 |    +0.180 |   +0.244 | +nan     |    -0.041 |      -0.075 |   -0.038 |     +0.047 |   -0.111 |
| Kennedy   |   +0.093 |    +0.094 |   +0.163 |   +0.005 |  +nan     |      +0.000 |   +0.070 |     +0.203 |   -0.008 |
| Sotomayor |   -0.138 |    -0.058 |   +0.058 |   +0.026 |    +0.015 |    +nan     |   +0.115 |     +0.141 |   +0.072 |
| Breyer    |   -0.130 |    -0.120 |   -0.058 |   +0.009 |    -0.099 |      -0.019 | +nan     |     +0.273 |   +0.037 |
| Ginsburg  |   -0.191 |    -0.150 |   -0.101 |   -0.032 |    -0.088 |      +0.070 |   +0.066 |   +nan     |   +0.103 |
| Kagan     |   -0.129 |    +0.016 |   +0.047 |   +0.007 |    -0.062 |      +0.017 |   +0.057 |     +0.108 | +nan     |

**For comparison: correlation of the same justices' actual votes, terms 2010–15**

|           |   Alito |   Roberts |   Scalia |   Thomas |   Kennedy |   Sotomayor |   Breyer |   Ginsburg |   Kagan |
|:----------|--------:|----------:|---------:|---------:|----------:|------------:|---------:|-----------:|--------:|
| Alito     |    1.00 |      0.73 |     0.65 |     0.69 |      0.59 |        0.22 |     0.34 |       0.21 |    0.28 |
| Roberts   |    0.73 |      1.00 |     0.71 |     0.63 |      0.65 |        0.36 |     0.45 |       0.31 |    0.39 |
| Scalia    |    0.65 |      0.71 |     1.00 |     0.75 |      0.51 |        0.23 |     0.20 |       0.23 |    0.34 |
| Thomas    |    0.69 |      0.63 |     0.75 |     1.00 |      0.49 |        0.20 |     0.25 |       0.16 |    0.24 |
| Kennedy   |    0.59 |      0.65 |     0.51 |     0.49 |      1.00 |        0.48 |     0.55 |       0.44 |    0.55 |
| Sotomayor |    0.22 |      0.36 |     0.23 |     0.20 |      0.48 |        1.00 |     0.69 |       0.74 |    0.77 |
| Breyer    |    0.34 |      0.45 |     0.20 |     0.25 |      0.55 |        0.69 |     1.00 |       0.71 |    0.75 |
| Ginsburg  |    0.21 |      0.31 |     0.23 |     0.16 |      0.44 |        0.74 |     0.71 |       1.00 |    0.80 |
| Kagan     |    0.28 |      0.39 |     0.34 |     0.24 |      0.55 |        0.77 |     0.75 |       0.80 |    1.00 |

- Order from the table alone: Alito, Roberts, Scalia, Thomas, Kennedy, Sotomayor, Breyer, Ginsburg, Kagan.
- Position on that axis (farthest justice = 1), and for comparison the position on the axis that divides the Court in the actual votes of 2010–15 (the second principal axis of the vote correlations; the first is general agreement): Alito -1.00 / -0.90; Roberts -0.98 / -0.69; Scalia -0.97 / -0.94; Thomas -0.79 / -1.00; Kennedy -0.18 / -0.15; Sotomayor +0.60 / +0.91; Breyer +0.63 / +0.78; Ginsburg +0.69 / +0.99; Kagan +0.70 / +0.85. Correlation of the two line-ups: 0.99.
- Rank correlation over the 72 ordered pairs: with the correlation of the actual votes in 2010–15, 0.56 (2012–15 only: 0.55); with the weights of the `qc_bloc` regression, 0.88.
- A colleague's average weight over the eight others (the column mean): Roberts +0.038, Alito -0.032, Scalia +0.089, Thomas +0.014, Kennedy -0.051, Breyer +0.046, Ginsburg +0.126, Sotomayor -0.016, Kagan -0.019. With that average removed (added after the first results), the rank correlation with the actual votes is 0.75.
- Mean weight for pairs whose votes agree more than the median pair: +0.072; for the others: -0.029.
- Two seeds' tables correlate at 0.93 on average. The bloc term has a standard deviation of 0.56 logits over the votes (the bloc-free logit: 1.13); pair constants plus colleagues' own logits weighted by W explain 63% of its variance.

## M5. Untrained identity rows (a defect of the tested recipe)

ConvoKit marks a speaker as a justice in every case, also in a case argued as an advocate before joining the Court; `prepare_data.py` takes the role from that mark. 138 cases (all in the train split) have a speaker tagged as a justice before that justice's first vote. In 90 of them the speaker has ten turns or more, an advocate's part; since 1980 that is every one of the 52 cases (Roberts 39, Kagan 6, Alito 6, Kavanaugh 1). In the others the speaker has a few turns: more likely a sitting justice's turns filed under the wrong name in the source (in 1964–67 the Chief Justice's calling of the case is filed under Burger). In an argued case that side's advocate turns are missing and are counted as turns of the bench.

| Later justice   |   Cases |   With 10+ turns |   Median turns a case |   First term |   Last term |
|:----------------|--------:|-----------------:|----------------------:|-------------:|------------:|
| Roberts         |      39 |               39 |                  45   |         1988 |        2002 |
| Marshall        |      21 |               18 |                  59   |         1957 |        1966 |
| Kagan           |       6 |                6 |                  57.5 |         2008 |        2009 |
| Alito           |       6 |                6 |                  25.5 |         1981 |        1984 |
| Fortas          |       8 |                5 |                  29.5 |         1955 |        1964 |
| Ginsburg        |       7 |                5 |                  21   |         1971 |        1978 |
| Goldberg        |       7 |                4 |                  21   |         1955 |        1959 |
| Stevens         |      14 |                3 |                   2   |         1958 |        1973 |
| Powell          |      10 |                1 |                   5.5 |         1966 |        1969 |
| Stewart         |       6 |                1 |                   3   |         1956 |        1957 |
| Burger          |       6 |                1 |                   1   |         1964 |        1967 |
| Scalia          |       1 |                1 |                  14   |         1974 |        1974 |
| Kavanaugh       |       1 |                1 |                  77   |         1997 |        1997 |
| White           |       6 |                0 |                   1   |         1957 |        1960 |
| Rehnquist       |       2 |                0 |                   1.5 |         1968 |        1969 |
| Blackmun        |       1 |                0 |                   1   |         1967 |        1967 |
| Oconnor         |       1 |                0 |                   1   |         1976 |        1976 |

`train_v1.py` gives an identity row to every speaker seen with the justice role in a training case. Kavanaugh therefore had rows at every step of the test run, without a training vote before the 2019 step: his query and head rows were the untrained random start; his gain, lean and pair-table entries are exactly 0 (the script checks it), so those parts fell back as designed. Affected: his 61 votes of the 2018 term in the primary row, and all 472 of his votes in the rows fitted once.

**The check on held-out data.** Each held-out vote of a known justice, with only that justice changed. Mean probability over five seeds, as in the test run.

**val fold, terms 2016–17**

| How the justice is identified       | Votes   | AUC   | Macro-F1   | Vote acc   | Pred. petitioner %   |
|:------------------------------------|:--------|:------|:-----------|:-----------|:---------------------|
| as trained                          | 978     | 0.821 | 0.747      | 0.764      | 59.8                 |
| unknown justice (designed fallback) | 978     | 0.776 | 0.703      | 0.723      | 59.8                 |
| untrained rows (as in the test run) | 978     | 0.776 | 0.705      | 0.727      | 61.2                 |

- Untrained rows − designed fallback: AUC +0.000 [-0.007, +0.007], macro-F1 +0.002 [-0.013, +0.015]. Mean absolute change of a probability 0.019; 3.1% of the calls change.
- Designed fallback − as trained: AUC -0.046 [-0.068, -0.024], macro-F1 -0.044 [-0.074, -0.016]. Untrained rows − as trained: AUC -0.046 [-0.069, -0.025], macro-F1 -0.042 [-0.075, -0.013].
- Size of the rows (vector norms, mean of five seeds): readout query, untrained 15.7 against trained 15.7 (the shared query: 1.0); head, 5.6 against 5.5; speaker embedding, 0.12 against 0.27. Learned own-turn bonus: 1.07 (start: 1).

**dev fold, terms 2012–15**

| How the justice is identified       | Votes   | AUC   | Macro-F1   | Vote acc   | Pred. petitioner %   |
|:------------------------------------|:--------|:------|:-----------|:-----------|:---------------------|
| as trained                          | 2238    | 0.744 | 0.667      | 0.670      | 48.2                 |
| unknown justice (designed fallback) | 2238    | 0.710 | 0.629      | 0.631      | 46.3                 |
| untrained rows (as in the test run) | 2238    | 0.709 | 0.639      | 0.644      | 50.3                 |

- Untrained rows − designed fallback: AUC -0.001 [-0.006, +0.004], macro-F1 +0.010 [-0.000, +0.020]. Mean absolute change of a probability 0.024; 4.1% of the calls change.
- Designed fallback − as trained: AUC -0.035 [-0.049, -0.021], macro-F1 -0.038 [-0.057, -0.020]. Untrained rows − as trained: AUC -0.036 [-0.050, -0.021], macro-F1 -0.028 [-0.049, -0.007].
- Size of the rows (vector norms, mean of five seeds): readout query, untrained 15.7 against trained 15.7 (the shared query: 1.0); head, 5.6 against 5.5; speaker embedding, 0.13 against 0.27. Learned own-turn bonus: 1.07 (start: 1).

**From the saved predictions of the test run (descriptive; small samples).** Newcomers' first terms, models refit before each term:

| Term   | Votes of           | Handled as                    | Votes   | Model v1 AUC   | Model v1 acc %   | qc_bloc AUC   | qc_bloc acc %   | qc_plus AUC   | qc_plus acc %   |
|:-------|:-------------------|:------------------------------|:--------|:---------------|:-----------------|:--------------|:----------------|:--------------|:----------------|
| 2018   | Kavanaugh          | untrained identity rows       | 61      | 0.680          | 65.6             | 0.748         | 73.8            | 0.748         | 73.8            |
| 2018   | the other justices |                               | 552     | 0.816          | 74.3             | 0.796         | 71.9            | 0.762         | 66.3            |
| 2020   | Barrett            | unknown justice (as designed) | 47      | 0.744          | 63.8             | 0.631         | 57.4            | 0.631         | 57.4            |
| 2020   | the other justices |                               | 455     | 0.750          | 65.5             | 0.730         | 67.0            | 0.744         | 67.9            |
| 2022   | Jackson            | unknown justice (as designed) | 58      | 0.762          | 67.2             | 0.755         | 69.0            | 0.755         | 69.0            |
| 2022   | the other justices |                               | 472     | 0.692          | 66.3             | 0.671         | 61.7            | 0.648         | 58.1            |

The fit made once before 2018, terms 2018–2025, by justice:

| Votes of                                                 | Votes   | v1 once AUC   | v1 once acc %   | v1 each term AUC   | v1 each term acc %   | qc_plus each term AUC   | qc_plus each term acc %   |
|:---------------------------------------------------------|:--------|:--------------|:----------------|:-------------------|:---------------------|:------------------------|:--------------------------|
| Kavanaugh (untrained identity rows in the fit made once) | 472     | 0.698         | 66.3            | 0.746              | 69.5                 | 0.705                   | 67.2                      |
| Barrett, Jackson (unknown justices, as designed)         | 581     | 0.752         | 63.2            | 0.773              | 69.4                 | 0.745                   | 61.1                      |
| Gorsuch (71 training votes before 2018)                  | 479     | 0.763         | 71.0            | 0.778              | 72.0                 | 0.743                   | 68.7                      |
| The justices who sat before 2016                         | 2766    | 0.781         | 72.1            | 0.788              | 71.0                 | 0.727                   | 64.3                      |
