# Timing test: does the clock of the argument help? (notes §21.1)

Start and stop times of every turn from the ConvoKit corpora (Oyez timing). Twelve features per vote: to_s_diff, to_s_share, to_pace_diff, to_cut_diff, to_cut_share, to_ans_diff, tb_s_diff, tb_s_share, tb_pace_diff, tb_cut_diff, tb_cut_share, tb_ans_diff (`to_` the justice, `tb_` the bench; `s` seconds of speech, `pace` words per second, `cut` turns that follow an advocate cut off with "--", `ans` seconds an advocate speaks before the justice comes in; respondent against petitioner). The features and the pass mark were fixed before any result. The test split is not scored.

## 1. The clock by decade

| Decade   |   Turns |   Usable time % |   Duration ≤ 0 % |   Median pace, words/s |   Median justice turn, s |   Advocate turns cut off % |
|:---------|--------:|----------------:|-----------------:|-----------------------:|-------------------------:|---------------------------:|
| 1950s    | 195,269 |            93.8 |              0   |                   2.92 |                      4   |                       31.6 |
| 1960s    | 348,590 |            93.9 |              0   |                   2.82 |                      3.9 |                       30.6 |
| 1970s    | 307,148 |            98.1 |              0   |                   2.81 |                      4.8 |                       25.4 |
| 1980s    | 274,550 |            96.8 |              2.3 |                   2.81 |                      5.3 |                       31.5 |
| 1990s    | 215,450 |            96.8 |              2.2 |                   2.97 |                      6.2 |                       38.5 |
| 2000s    | 188,212 |            97.5 |              0.6 |                   2.94 |                      6.8 |                       41.7 |
| 2010s    | 148,826 |            97.5 |              0.3 |                   3.07 |                      7.5 |                       44.1 |

## 2. The test (class-balanced logistic regressions, fitted on 1955+)

dev: fit ≤ 2011, score 2012–15. val: fit ≤ 2015, score 2016–17. [95% bootstrap interval over cases].

| Model            | dev AUC              | dev macro-F1         | val AUC              | val macro-F1         |   2017 AUC |
|:-----------------|:---------------------|:---------------------|:---------------------|:---------------------|-----------:|
| timing only      | 0.671 [0.640, 0.704] | 0.616 [0.587, 0.644] | 0.726 [0.680, 0.769] | 0.635 [0.594, 0.672] |      0.745 |
| qc_plus          | 0.703 [0.668, 0.740] | 0.639 [0.606, 0.670] | 0.780 [0.728, 0.827] | 0.675 [0.629, 0.720] |      0.764 |
| qc_plus + timing | 0.703 [0.668, 0.739] | 0.639 [0.606, 0.670] | 0.780 [0.730, 0.826] | 0.675 [0.632, 0.716] |      0.767 |
| qc_bloc          | 0.726 [0.689, 0.761] | 0.663 [0.629, 0.693] | 0.818 [0.777, 0.862] | 0.735 [0.690, 0.774] |      0.841 |
| qc_bloc + timing | 0.727 [0.689, 0.763] | 0.663 [0.630, 0.692] | 0.822 [0.780, 0.864] | 0.731 [0.690, 0.771] |      0.847 |

## 3. With timing minus without (paired)

| Comparison                 | Scored on        | AUC                     | Macro-F1                |
|:---------------------------|:-----------------|:------------------------|:------------------------|
| qc_plus + timing − qc_plus | dev 2012–15      | +0.000 [-0.003, +0.004] | +0.000 [-0.007, +0.008] |
| qc_plus + timing − qc_plus | val 2016–17      | +0.000 [-0.004, +0.006] | +0.000 [-0.016, +0.016] |
| qc_plus + timing − qc_plus | 2017 alone       | +0.003 [-0.003, +0.009] | +0.000 [-0.021, +0.021] |
| qc_plus + timing − qc_plus | dev + val pooled | +0.000 [-0.003, +0.003] | +0.000 [-0.007, +0.008] |
| qc_bloc + timing − qc_bloc | dev 2012–15      | +0.002 [-0.003, +0.006] | -0.000 [-0.010, +0.009] |
| qc_bloc + timing − qc_bloc | val 2016–17      | +0.004 [-0.001, +0.009] | -0.004 [-0.015, +0.008] |
| qc_bloc + timing − qc_bloc | 2017 alone       | +0.006 [-0.001, +0.013] | +0.011 [-0.004, +0.028] |
| qc_bloc + timing − qc_bloc | dev + val pooled | +0.002 [-0.001, +0.006] | -0.002 [-0.008, +0.005] |

## 4. Pass mark

Fixed in advance: AUC of qc_bloc + timing above qc_bloc on the dev fold and on 2017 alone; clear only if the pooled interval excludes zero.

- dev +0.002, 2017 alone +0.006, pooled +0.002 [-0.001, +0.006].
- **PASSES but the gain is not clear.**

## 5. The timing features in the regression (fit ≤ 2015)

| Feature      |   Weight in qc_plus + timing (standardized; + = pro-petitioner) |   AUC alone, train terms |   Zero in % of votes |
|:-------------|----------------------------------------------------------------:|-------------------------:|---------------------:|
| to_s_diff    |                                                          -0.353 |                    0.596 |                   34 |
| to_s_share   |                                                           0.098 |                    0.595 |                   34 |
| to_pace_diff |                                                          -0     |                    0.509 |                   63 |
| to_cut_diff  |                                                           0.137 |                    0.59  |                   48 |
| to_cut_share |                                                           0.016 |                    0.59  |                   48 |
| to_ans_diff  |                                                           0.024 |                    0.516 |                   58 |
| tb_s_diff    |                                                           0.686 |                    0.575 |                    9 |
| tb_s_share   |                                                          -0.509 |                    0.575 |                    9 |
| tb_pace_diff |                                                          -0.059 |                    0.501 |                   26 |
| tb_cut_diff  |                                                          -0.228 |                    0.557 |                   16 |
| tb_cut_share |                                                           0.177 |                    0.556 |                   16 |
| tb_ans_diff  |                                                          -0.02  |                    0.57  |                   23 |

Largest count weights for scale: a_n_diff +0.82, b_n_share +0.76, b_w_diff -0.69, o_w_R_sqrt +0.69.
