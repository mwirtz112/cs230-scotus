# Metadata baselines (no text, no deep learning)

SCDB case profile and voting-history rates (meta_features.py). katz_rf = random forest, 1,000 trees, on the justice, the profile and the history rates; katz_rf_rec = the same on facts of record only (no lower-court direction, lower-court dissent or cert reason); meta_lr = logistic regression on the history rates; qc_meta = qc_plus + history rates; qc_bloc_meta = qc_bloc with the history rates in its first stage. Class-balanced, cutoff 0.5. 'frozen': a held-out case sees voting history up to the fold's training cutoff; 'rolling': up to its argument day. Validation rows carry 95% bootstrap CIs over cases. Test untouched.

## Validation 2016-17 (fit on terms <= 2015)

| method       | history   | macro_f1             | auc                  |   vote_acc |   case_acc |   AUC 2017 |   pred petitioner % |
|:-------------|:----------|:---------------------|:---------------------|-----------:|-----------:|-----------:|--------------------:|
| katz_rf      | frozen    | 0.624 [0.577, 0.672] | 0.715 [0.670, 0.765] |      0.686 |      0.691 |      0.72  |                74.9 |
| katz_rf_rec  | frozen    | 0.573 [0.524, 0.624] | 0.657 [0.598, 0.715] |      0.675 |      0.707 |      0.603 |                82.9 |
| meta_lr      | frozen    | 0.622 [0.581, 0.663] | 0.703 [0.660, 0.755] |      0.634 |      0.577 |      0.732 |                52   |
| qc_meta      | frozen    | 0.715 [0.673, 0.752] | 0.812 [0.768, 0.857] |      0.724 |      0.707 |      0.812 |                52.8 |
| qc_bloc_meta | frozen    | 0.724 [0.683, 0.765] | 0.813 [0.771, 0.856] |      0.739 |      0.691 |      0.84  |                57.5 |
| katz_rf      | rolling   | 0.639 [0.595, 0.688] | 0.717 [0.666, 0.771] |      0.684 |      0.667 |      0.715 |                69.8 |
| katz_rf_rec  | rolling   | 0.548 [0.501, 0.597] | 0.651 [0.595, 0.719] |      0.638 |      0.675 |      0.578 |                78.6 |
| meta_lr      | rolling   | 0.625 [0.583, 0.667] | 0.701 [0.655, 0.754] |      0.642 |      0.593 |      0.733 |                55.2 |
| qc_meta      | rolling   | 0.709 [0.667, 0.747] | 0.812 [0.767, 0.858] |      0.72  |      0.699 |      0.813 |                53.7 |
| qc_bloc_meta | rolling   | 0.721 [0.681, 0.760] | 0.812 [0.769, 0.854] |      0.736 |      0.699 |      0.838 |                57.6 |

## Dev 2012-15 (fit on terms <= 2011)

| method       | history   |   macro_f1 |   auc |   vote_acc |   case_acc | AUC 2017   |   pred petitioner % |
|:-------------|:----------|-----------:|------:|-----------:|-----------:|:-----------|--------------------:|
| katz_rf      | frozen    |      0.592 | 0.621 |      0.628 |      0.598 |            |                68.5 |
| katz_rf_rec  | frozen    |      0.535 | 0.566 |      0.605 |      0.594 |            |                77.4 |
| meta_lr      | frozen    |      0.571 | 0.636 |      0.575 |      0.527 |            |                47.5 |
| qc_meta      | frozen    |      0.651 | 0.724 |      0.656 |      0.609 |            |                50.7 |
| qc_bloc_meta | frozen    |      0.667 | 0.736 |      0.673 |      0.648 |            |                52.2 |
| katz_rf      | rolling   |      0.582 | 0.622 |      0.613 |      0.582 |            |                65.8 |
| katz_rf_rec  | rolling   |      0.529 | 0.564 |      0.587 |      0.621 |            |                73.7 |
| meta_lr      | rolling   |      0.573 | 0.631 |      0.578 |      0.523 |            |                49.4 |
| qc_meta      | rolling   |      0.648 | 0.722 |      0.655 |      0.605 |            |                52.6 |
| qc_bloc_meta | rolling   |      0.663 | 0.734 |      0.671 |      0.645 |            |                53.7 |

## katz_rf: largest impurity importances (validation fold, frozen)

|               |   importance |
|:--------------|-------------:|
| h_pet_area_lc |       0.0636 |
| h_ideo        |       0.0607 |
| d_ideo        |       0.0506 |
| d_pet_area_lc |       0.0463 |
| c_area_lc     |       0.0451 |
| d_pet_lc      |       0.0384 |
| c_pet         |       0.0374 |
| c_resp        |       0.0373 |
| h_pet_lc      |       0.0358 |
| c_source      |       0.0354 |
| b_pet_area_lc |       0.0328 |
| b_ideo        |       0.0325 |
