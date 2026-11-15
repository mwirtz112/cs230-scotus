# The test run, checked (terms 2018–2025)

Reads the saved predictions of the test run (`data/preds/wf_final`) and the labels; fits nothing. `check_walk_final.py`, written on Nov 14 after the run.

## 1. Independent recomputation

481 cases, 4,298 votes, 24 method × fit tables, all predictions finite in 24 of them. Per-term macro-F1, AUC, vote accuracy and case accuracy, rebuilt without the harness's scoring code and compared with `results/walk_forward_final_terms.csv`: **768 cells, largest difference 5.0e-05, 0 beyond rounding (the file is rounded to 4 decimals).**

Recomputed over all test terms and the two blocks (compare with `results/walk_forward_final.md`; 'each' = refit before each term, 'once' = the fit made before 2018):

| Method                                         | Fit   |   Macro-F1 2018–2025 |   AUC 2018–2025 |   Vote acc |   Case acc |   Pred. petitioner % |   Macro-F1 2018–19 |   AUC 2018–19 |   Macro-F1 2020–25 |   AUC 2020–25 |
|:-----------------------------------------------|:------|---------------------:|----------------:|-----------:|-----------:|---------------------:|-------------------:|--------------:|-------------------:|--------------:|
| Always petitioner                              | each  |                0.381 |           0.5   |      0.615 |      0.676 |                100   |              0.364 |         0.5   |              0.387 |         0.5   |
| Justice vote rate                              | each  |                0.508 |           0.527 |      0.569 |      0.676 |                 73.6 |              0.504 |         0.516 |              0.509 |         0.526 |
| Justice vote rate                              | once  |                0.474 |           0.513 |      0.592 |      0.676 |                 85.9 |              0.488 |         0.516 |              0.462 |         0.512 |
| Question count                                 | each  |                0.603 |           0.656 |      0.605 |      0.549 |                 45.9 |              0.631 |         0.655 |              0.593 |         0.663 |
| Flat ModernBERT, first 8k tokens (frozen)      | each  |                0.553 |           0.569 |      0.58  |      0.599 |                 63   |              0.542 |         0.566 |              0.552 |         0.567 |
| Flat ModernBERT, first 8k tokens (frozen)      | once  |                0.536 |           0.551 |      0.558 |      0.555 |                 60.4 |              0.549 |         0.574 |              0.531 |         0.541 |
| Flat ModernBERT, whole transcript (frozen)     | each  |                0.539 |           0.575 |      0.583 |      0.632 |                 69.3 |              0.55  |         0.588 |              0.527 |         0.565 |
| Flat ModernBERT, whole transcript (frozen)     | once  |                0.526 |           0.544 |      0.566 |      0.593 |                 67.7 |              0.546 |         0.587 |              0.509 |         0.522 |
| Case profile only, Katz-style forest (katz_rf) | each  |                0.623 |           0.673 |      0.659 |      0.638 |                 69.3 |              0.585 |         0.641 |              0.636 |         0.683 |
| Case profile only, Katz-style forest (katz_rf) | once  |                0.609 |           0.652 |      0.64  |      0.642 |                 66.6 |              0.593 |         0.643 |              0.615 |         0.658 |
| Question features, LR (qc_plus)                | each  |                0.644 |           0.729 |      0.647 |      0.586 |                 47.9 |              0.661 |         0.729 |              0.638 |         0.737 |
| Question features, LR (qc_plus)                | once  |                0.634 |           0.72  |      0.637 |      0.563 |                 46.3 |              0.662 |         0.729 |              0.624 |         0.726 |
| + named colleagues, LR (qc_bloc)               | each  |                0.676 |           0.755 |      0.681 |      0.634 |                 50.7 |              0.686 |         0.749 |              0.673 |         0.763 |
| + named colleagues, LR (qc_bloc)               | once  |                0.609 |           0.659 |      0.614 |      0.561 |                 49.3 |              0.689 |         0.749 |              0.582 |         0.634 |
| + case profile history, LR (qc_bloc_meta)      | each  |                0.689 |           0.766 |      0.695 |      0.644 |                 53.2 |              0.688 |         0.753 |              0.689 |         0.775 |
| + case profile history, LR (qc_bloc_meta)      | once  |                0.609 |           0.667 |      0.616 |      0.588 |                 51.1 |              0.687 |         0.754 |              0.583 |         0.641 |
| Model v1                                       | each  |                0.702 |           0.781 |      0.708 |      0.663 |                 52.4 |              0.712 |         0.794 |              0.698 |         0.783 |
| Model v1                                       | once  |                0.695 |           0.767 |      0.701 |      0.655 |                 52.9 |              0.706 |         0.784 |              0.691 |         0.77  |
| Model v1 + metadata prior                      | each  |                0.706 |           0.785 |      0.711 |      0.651 |                 52.5 |              0.729 |         0.791 |              0.698 |         0.789 |
| Model v1 + metadata prior                      | once  |                0.688 |           0.764 |      0.695 |      0.647 |                 53   |              0.713 |         0.786 |              0.68  |         0.764 |
| Model v1 + qc_bloc (mean of logits)            | each  |                0.716 |           0.791 |      0.721 |      0.655 |                 51.8 |              0.718 |         0.792 |              0.716 |         0.798 |
| Model v1 + qc_bloc (mean of logits)            | once  |                0.663 |           0.739 |      0.668 |      0.62  |                 50.9 |              0.706 |         0.786 |              0.648 |         0.73  |
| Model v1 + qc_bloc_meta (mean of logits)       | each  |                0.723 |           0.8   |      0.729 |      0.676 |                 53.4 |              0.72  |         0.796 |              0.724 |         0.808 |
| Model v1 + qc_bloc_meta (mean of logits)       | once  |                0.677 |           0.748 |      0.683 |      0.636 |                 51.8 |              0.713 |         0.79  |              0.665 |         0.739 |

## 2. Differences the test report does not list (computed after the run)

First method minus second, both refit before each term; 95% interval from a bootstrap over cases, paired (`walk_forward.paired`, 1,000 draws, seed 0). These rows were not fixed before the test run: they describe the result.

| Comparison                                                                           | Terms     |   Cases | AUC                     | Macro-F1                | Note                                                                    |
|:-------------------------------------------------------------------------------------|:----------|--------:|:------------------------|:------------------------|:------------------------------------------------------------------------|
| Model v1 − + named colleagues, LR (qc_bloc)                                          | 2018–2025 |     481 | +0.026 [+0.007, +0.045] | +0.025 [+0.006, +0.046] | in the test report: +0.026 [+0.007, +0.045] and +0.025 [+0.006, +0.046] |
| Model v1 − + named colleagues, LR (qc_bloc)                                          | 2018–19   |     124 | +0.045 [+0.012, +0.083] | +0.027 [-0.010, +0.064] |                                                                         |
| Model v1 − + named colleagues, LR (qc_bloc)                                          | 2020–25   |     357 | +0.020 [-0.005, +0.044] | +0.025 [+0.001, +0.048] |                                                                         |
| Model v1 − + case profile history, LR (qc_bloc_meta)                                 | 2018–2025 |     481 | +0.015 [-0.006, +0.035] | +0.013 [-0.008, +0.037] |                                                                         |
| Model v1 − + case profile history, LR (qc_bloc_meta)                                 | 2018–19   |     124 | +0.041 [+0.005, +0.078] | +0.024 [-0.018, +0.065] |                                                                         |
| Model v1 − + case profile history, LR (qc_bloc_meta)                                 | 2020–25   |     357 | +0.008 [-0.020, +0.033] | +0.009 [-0.018, +0.034] |                                                                         |
| Model v1 − Case profile only, Katz-style forest (katz_rf)                            | 2018–2025 |     481 | +0.108 [+0.073, +0.143] | +0.079 [+0.047, +0.110] |                                                                         |
| Model v1 − Question count                                                            | 2018–2025 |     481 | +0.125 [+0.101, +0.152] | +0.099 [+0.078, +0.122] |                                                                         |
| Model v1 + qc_bloc_meta (mean of logits) − Model v1                                  | 2018–2025 |     481 | +0.019 [+0.011, +0.027] | +0.022 [+0.010, +0.034] |                                                                         |
| Model v1 + qc_bloc_meta (mean of logits) − + case profile history, LR (qc_bloc_meta) | 2018–2025 |     481 | +0.034 [+0.021, +0.047] | +0.035 [+0.017, +0.050] | same inputs on both sides: transcript and case profile                  |
