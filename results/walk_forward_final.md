# Walk-forward evaluation, terms 2018–2025

THE TEST RUN (terms 2018–2025), made once for the final report. For each term every method is refit on the cases of earlier terms decided before that term opened (1 October), then predicts the term. 'each term' = that refit; 'once' = the fit made before 2018, applied unchanged to every term (the two agree on the first term by construction). Network rows are the mean-probability ensemble of 5 seeds of the fixed recipe (train from 1980, 6 epochs, no early stopping). Cutoff 0.5 (justice vote rate: the training rate). Point estimate [95% bootstrap CI over cases]. '@rate' columns: the same predictions cut, within each term, where the share of predicted petitioner votes equals the pro-petitioner share of the votes the model was fitted on (59.9–60.1%); this uses the term's predictions but none of its labels. 'Case acc, base rate restored': the case call when a vote counts for the petitioner if p > 1 − π, π being that same training share. The class-balanced loss takes the base rate out of p and this puts it back, with no fitted number and nothing of the predicted term; a secondary column, fixed before the test run (notes §19.7). '–' for the methods without a class-balanced probability.

## All scored terms, 2018–2025 (481 cases, 4,298 votes)

| Method                                         | Fit                 | Macro-F1             | AUC                  |   Vote acc |   Case acc |   Pred. petitioner % |   Macro-F1 @rate |   Case acc @rate | Case acc, base rate restored   |
|:-----------------------------------------------|:--------------------|:---------------------|:---------------------|-----------:|-----------:|---------------------:|-----------------:|-----------------:|:-------------------------------|
| Always petitioner                              | –                   | 0.381 [0.369, 0.393] | 0.500 [0.500, 0.500] |      0.615 |      0.676 |                100   |            0.381 |            0.676 | –                              |
| Justice vote rate                              | each term           | 0.508 [0.497, 0.520] | 0.527 [0.511, 0.543] |      0.569 |      0.676 |                 73.6 |            0.514 |            0.676 | –                              |
| Justice vote rate                              | once (terms ≤ 2017) | 0.474 [0.461, 0.487] | 0.513 [0.502, 0.525] |      0.592 |      0.676 |                 85.9 |            0.485 |            0.676 | –                              |
| Question count                                 | –                   | 0.603 [0.582, 0.624] | 0.656 [0.633, 0.680] |      0.605 |      0.549 |                 45.9 |            0.624 |            0.657 | –                              |
| Flat ModernBERT, first 8k tokens (frozen)      | each term           | 0.553 [0.528, 0.577] | 0.569 [0.537, 0.604] |      0.58  |      0.599 |                 63   |            0.551 |            0.582 | 0.653                          |
| Flat ModernBERT, first 8k tokens (frozen)      | once (terms ≤ 2017) | 0.536 [0.512, 0.560] | 0.551 [0.521, 0.583] |      0.558 |      0.555 |                 60.4 |            0.536 |            0.551 | 0.609                          |
| Flat ModernBERT, whole transcript (frozen)     | each term           | 0.539 [0.515, 0.564] | 0.575 [0.544, 0.607] |      0.583 |      0.632 |                 69.3 |            0.55  |            0.586 | 0.680                          |
| Flat ModernBERT, whole transcript (frozen)     | once (terms ≤ 2017) | 0.526 [0.501, 0.552] | 0.544 [0.512, 0.579] |      0.566 |      0.593 |                 67.7 |            0.525 |            0.557 | 0.672                          |
| Case profile only, Katz-style forest (katz_rf) | each term           | 0.623 [0.600, 0.647] | 0.673 [0.645, 0.701] |      0.659 |      0.638 |                 69.3 |            0.623 |            0.624 | 0.686                          |
| Case profile only, Katz-style forest (katz_rf) | once (terms ≤ 2017) | 0.609 [0.586, 0.632] | 0.652 [0.625, 0.679] |      0.64  |      0.642 |                 66.6 |            0.603 |            0.576 | 0.682                          |
| Question features, LR (qc_plus)                | each term           | 0.644 [0.623, 0.667] | 0.729 [0.704, 0.756] |      0.647 |      0.586 |                 47.9 |            0.667 |            0.699 | 0.709                          |
| Question features, LR (qc_plus)                | once (terms ≤ 2017) | 0.634 [0.612, 0.658] | 0.720 [0.696, 0.748] |      0.637 |      0.563 |                 46.3 |            0.663 |            0.684 | 0.707                          |
| + named colleagues, LR (qc_bloc)               | each term           | 0.676 [0.654, 0.699] | 0.755 [0.730, 0.780] |      0.681 |      0.634 |                 50.7 |            0.685 |            0.678 | 0.701                          |
| + named colleagues, LR (qc_bloc)               | once (terms ≤ 2017) | 0.609 [0.591, 0.630] | 0.659 [0.638, 0.681] |      0.614 |      0.561 |                 49.3 |            0.623 |            0.655 | 0.661                          |
| + case profile history, LR (qc_bloc_meta)      | each term           | 0.689 [0.667, 0.713] | 0.766 [0.742, 0.792] |      0.695 |      0.644 |                 53.2 |            0.695 |            0.669 | 0.699                          |
| + case profile history, LR (qc_bloc_meta)      | once (terms ≤ 2017) | 0.609 [0.589, 0.630] | 0.667 [0.644, 0.689] |      0.616 |      0.588 |                 51.1 |            0.621 |            0.651 | 0.674                          |
| Model v1                                       | each term           | 0.702 [0.677, 0.726] | 0.781 [0.755, 0.807] |      0.708 |      0.663 |                 52.4 |            0.712 |            0.701 | 0.711                          |
| Model v1                                       | once (terms ≤ 2017) | 0.695 [0.672, 0.718] | 0.767 [0.739, 0.793] |      0.701 |      0.655 |                 52.9 |            0.703 |            0.696 | 0.717                          |
| Model v1 + metadata prior                      | each term           | 0.706 [0.682, 0.730] | 0.785 [0.760, 0.811] |      0.711 |      0.651 |                 52.5 |            0.722 |            0.711 | 0.709                          |
| Model v1 + metadata prior                      | once (terms ≤ 2017) | 0.688 [0.665, 0.712] | 0.764 [0.738, 0.791] |      0.695 |      0.647 |                 53   |            0.702 |            0.69  | 0.692                          |
| Model v1 + qc_bloc (mean of logits)            | each term           | 0.716 [0.693, 0.740] | 0.791 [0.766, 0.816] |      0.721 |      0.655 |                 51.8 |            0.722 |            0.699 | 0.717                          |
| Model v1 + qc_bloc (mean of logits)            | once (terms ≤ 2017) | 0.663 [0.642, 0.686] | 0.739 [0.716, 0.764] |      0.668 |      0.62  |                 50.9 |            0.676 |            0.69  | 0.721                          |
| Model v1 + qc_bloc_meta (mean of logits)       | each term           | 0.723 [0.700, 0.746] | 0.800 [0.775, 0.824] |      0.729 |      0.676 |                 53.4 |            0.73  |            0.705 | 0.730                          |
| Model v1 + qc_bloc_meta (mean of logits)       | once (terms ≤ 2017) | 0.677 [0.656, 0.700] | 0.748 [0.725, 0.773] |      0.683 |      0.636 |                 51.8 |            0.686 |            0.69  | 0.726                          |

## 2018–19, the test set of the approved design (124 cases, 1,105 votes)

| Method                                         | Fit                 | Macro-F1             | AUC                  |   Vote acc |   Case acc |   Pred. petitioner % |   Macro-F1 @rate |   Case acc @rate | Case acc, base rate restored   |
|:-----------------------------------------------|:--------------------|:---------------------|:---------------------|-----------:|-----------:|---------------------:|-----------------:|-----------------:|:-------------------------------|
| Always petitioner                              | –                   | 0.364 [0.340, 0.388] | 0.500 [0.500, 0.500] |      0.572 |      0.597 |                100   |            0.364 |            0.597 | –                              |
| Justice vote rate                              | each term           | 0.504 [0.481, 0.526] | 0.516 [0.491, 0.542] |      0.548 |      0.597 |                 72.7 |            0.502 |            0.597 | –                              |
| Justice vote rate                              | once (terms ≤ 2017) | 0.488 [0.467, 0.510] | 0.516 [0.496, 0.537] |      0.55  |      0.597 |                 77.6 |            0.494 |            0.597 | –                              |
| Question count                                 | –                   | 0.631 [0.592, 0.666] | 0.655 [0.612, 0.695] |      0.635 |      0.613 |                 53.7 |            0.635 |            0.613 | –                              |
| Flat ModernBERT, first 8k tokens (frozen)      | each term           | 0.542 [0.492, 0.587] | 0.566 [0.505, 0.624] |      0.547 |      0.532 |                 53.3 |            0.553 |            0.548 | 0.556                          |
| Flat ModernBERT, first 8k tokens (frozen)      | once (terms ≤ 2017) | 0.549 [0.500, 0.595] | 0.574 [0.510, 0.634] |      0.559 |      0.589 |                 58.2 |            0.55  |            0.573 | 0.589                          |
| Flat ModernBERT, whole transcript (frozen)     | each term           | 0.550 [0.503, 0.596] | 0.588 [0.530, 0.645] |      0.558 |      0.548 |                 56.3 |            0.563 |            0.597 | 0.613                          |
| Flat ModernBERT, whole transcript (frozen)     | once (terms ≤ 2017) | 0.546 [0.498, 0.591] | 0.587 [0.527, 0.641] |      0.551 |      0.548 |                 53.2 |            0.564 |            0.613 | 0.621                          |
| Case profile only, Katz-style forest (katz_rf) | each term           | 0.585 [0.541, 0.632] | 0.641 [0.588, 0.703] |      0.623 |      0.565 |                 72.9 |            0.594 |            0.565 | 0.597                          |
| Case profile only, Katz-style forest (katz_rf) | once (terms ≤ 2017) | 0.593 [0.549, 0.642] | 0.643 [0.589, 0.701] |      0.626 |      0.581 |                 71.4 |            0.594 |            0.573 | 0.597                          |
| Question features, LR (qc_plus)                | each term           | 0.661 [0.620, 0.702] | 0.729 [0.678, 0.774] |      0.666 |      0.605 |                 54.8 |            0.663 |            0.653 | 0.685                          |
| Question features, LR (qc_plus)                | once (terms ≤ 2017) | 0.662 [0.621, 0.702] | 0.729 [0.679, 0.774] |      0.667 |      0.597 |                 54.5 |            0.663 |            0.645 | 0.685                          |
| + named colleagues, LR (qc_bloc)               | each term           | 0.686 [0.642, 0.728] | 0.749 [0.699, 0.796] |      0.691 |      0.645 |                 56.4 |            0.684 |            0.645 | 0.645                          |
| + named colleagues, LR (qc_bloc)               | once (terms ≤ 2017) | 0.689 [0.645, 0.732] | 0.749 [0.700, 0.796] |      0.694 |      0.629 |                 56.1 |            0.684 |            0.669 | 0.637                          |
| + case profile history, LR (qc_bloc_meta)      | each term           | 0.688 [0.646, 0.731] | 0.753 [0.705, 0.801] |      0.694 |      0.629 |                 56.8 |            0.689 |            0.629 | 0.653                          |
| + case profile history, LR (qc_bloc_meta)      | once (terms ≤ 2017) | 0.687 [0.643, 0.731] | 0.754 [0.706, 0.802] |      0.693 |      0.653 |                 56.9 |            0.686 |            0.653 | 0.629                          |
| Model v1                                       | each term           | 0.712 [0.663, 0.757] | 0.794 [0.743, 0.840] |      0.72  |      0.661 |                 59.5 |            0.712 |            0.653 | 0.710                          |
| Model v1                                       | once (terms ≤ 2017) | 0.706 [0.659, 0.750] | 0.784 [0.731, 0.833] |      0.717 |      0.669 |                 62   |            0.704 |            0.645 | 0.718                          |
| Model v1 + metadata prior                      | each term           | 0.729 [0.681, 0.774] | 0.791 [0.741, 0.839] |      0.736 |      0.669 |                 58.6 |            0.73  |            0.677 | 0.694                          |
| Model v1 + metadata prior                      | once (terms ≤ 2017) | 0.713 [0.663, 0.759] | 0.786 [0.737, 0.834] |      0.722 |      0.677 |                 61.1 |            0.721 |            0.694 | 0.694                          |
| Model v1 + qc_bloc (mean of logits)            | each term           | 0.718 [0.674, 0.761] | 0.792 [0.742, 0.838] |      0.724 |      0.637 |                 57.3 |            0.717 |            0.669 | 0.702                          |
| Model v1 + qc_bloc (mean of logits)            | once (terms ≤ 2017) | 0.706 [0.663, 0.751] | 0.786 [0.736, 0.832] |      0.714 |      0.637 |                 59   |            0.706 |            0.645 | 0.702                          |
| Model v1 + qc_bloc_meta (mean of logits)       | each term           | 0.720 [0.675, 0.764] | 0.796 [0.746, 0.842] |      0.727 |      0.661 |                 58.1 |            0.721 |            0.661 | 0.702                          |
| Model v1 + qc_bloc_meta (mean of logits)       | once (terms ≤ 2017) | 0.713 [0.667, 0.757] | 0.790 [0.739, 0.836] |      0.721 |      0.653 |                 59.7 |            0.716 |            0.645 | 0.685                          |

## 2020–25, the new argument format (357 cases, 3,193 votes)

| Method                                         | Fit                 | Macro-F1             | AUC                  |   Vote acc |   Case acc |   Pred. petitioner % |   Macro-F1 @rate |   Case acc @rate | Case acc, base rate restored   |
|:-----------------------------------------------|:--------------------|:---------------------|:---------------------|-----------:|-----------:|---------------------:|-----------------:|-----------------:|:-------------------------------|
| Always petitioner                              | –                   | 0.387 [0.374, 0.399] | 0.500 [0.500, 0.500] |      0.63  |      0.703 |                100   |            0.387 |            0.703 | –                              |
| Justice vote rate                              | each term           | 0.509 [0.496, 0.522] | 0.526 [0.507, 0.546] |      0.576 |      0.703 |                 73.9 |            0.518 |            0.703 | –                              |
| Justice vote rate                              | once (terms ≤ 2017) | 0.462 [0.446, 0.478] | 0.512 [0.499, 0.525] |      0.607 |      0.703 |                 88.8 |            0.474 |            0.703 | –                              |
| Question count                                 | –                   | 0.593 [0.567, 0.617] | 0.663 [0.632, 0.692] |      0.595 |      0.527 |                 43.3 |            0.62  |            0.672 | –                              |
| Flat ModernBERT, first 8k tokens (frozen)      | each term           | 0.552 [0.521, 0.581] | 0.567 [0.525, 0.605] |      0.591 |      0.622 |                 66.3 |            0.549 |            0.594 | 0.686                          |
| Flat ModernBERT, first 8k tokens (frozen)      | once (terms ≤ 2017) | 0.531 [0.499, 0.558] | 0.541 [0.504, 0.575] |      0.558 |      0.543 |                 61.1 |            0.531 |            0.543 | 0.616                          |
| Flat ModernBERT, whole transcript (frozen)     | each term           | 0.527 [0.498, 0.558] | 0.565 [0.527, 0.602] |      0.591 |      0.661 |                 73.8 |            0.545 |            0.583 | 0.703                          |
| Flat ModernBERT, whole transcript (frozen)     | once (terms ≤ 2017) | 0.509 [0.476, 0.537] | 0.522 [0.483, 0.560] |      0.572 |      0.608 |                 72.7 |            0.51  |            0.538 | 0.689                          |
| Case profile only, Katz-style forest (katz_rf) | each term           | 0.636 [0.609, 0.663] | 0.683 [0.649, 0.716] |      0.671 |      0.664 |                 68   |            0.633 |            0.644 | 0.717                          |
| Case profile only, Katz-style forest (katz_rf) | once (terms ≤ 2017) | 0.615 [0.588, 0.641] | 0.658 [0.627, 0.688] |      0.645 |      0.664 |                 64.9 |            0.606 |            0.577 | 0.711                          |
| Question features, LR (qc_plus)                | each term           | 0.638 [0.610, 0.661] | 0.737 [0.706, 0.769] |      0.64  |      0.58  |                 45.5 |            0.668 |            0.714 | 0.717                          |
| Question features, LR (qc_plus)                | once (terms ≤ 2017) | 0.624 [0.596, 0.649] | 0.726 [0.694, 0.759] |      0.626 |      0.552 |                 43.4 |            0.662 |            0.697 | 0.714                          |
| + named colleagues, LR (qc_bloc)               | each term           | 0.673 [0.648, 0.696] | 0.763 [0.734, 0.791] |      0.678 |      0.63  |                 48.7 |            0.685 |            0.689 | 0.720                          |
| + named colleagues, LR (qc_bloc)               | once (terms ≤ 2017) | 0.582 [0.562, 0.602] | 0.634 [0.612, 0.657] |      0.586 |      0.538 |                 46.9 |            0.602 |            0.65  | 0.669                          |
| + case profile history, LR (qc_bloc_meta)      | each term           | 0.689 [0.663, 0.712] | 0.775 [0.745, 0.801] |      0.696 |      0.65  |                 52   |            0.697 |            0.683 | 0.714                          |
| + case profile history, LR (qc_bloc_meta)      | once (terms ≤ 2017) | 0.583 [0.561, 0.603] | 0.641 [0.617, 0.662] |      0.589 |      0.566 |                 49.1 |            0.598 |            0.65  | 0.689                          |
| Model v1                                       | each term           | 0.698 [0.668, 0.726] | 0.783 [0.754, 0.815] |      0.703 |      0.664 |                 50   |            0.711 |            0.717 | 0.711                          |
| Model v1                                       | once (terms ≤ 2017) | 0.691 [0.661, 0.718] | 0.770 [0.738, 0.803] |      0.696 |      0.65  |                 49.8 |            0.702 |            0.714 | 0.717                          |
| Model v1 + metadata prior                      | each term           | 0.698 [0.672, 0.726] | 0.789 [0.759, 0.820] |      0.703 |      0.644 |                 50.4 |            0.719 |            0.723 | 0.714                          |
| Model v1 + metadata prior                      | once (terms ≤ 2017) | 0.680 [0.651, 0.707] | 0.764 [0.731, 0.793] |      0.685 |      0.636 |                 50.2 |            0.695 |            0.689 | 0.692                          |
| Model v1 + qc_bloc (mean of logits)            | each term           | 0.716 [0.688, 0.741] | 0.798 [0.767, 0.828] |      0.72  |      0.661 |                 49.9 |            0.723 |            0.709 | 0.723                          |
| Model v1 + qc_bloc (mean of logits)            | once (terms ≤ 2017) | 0.648 [0.623, 0.672] | 0.730 [0.702, 0.757] |      0.653 |      0.613 |                 48   |            0.665 |            0.706 | 0.728                          |
| Model v1 + qc_bloc_meta (mean of logits)       | each term           | 0.724 [0.696, 0.750] | 0.808 [0.779, 0.836] |      0.73  |      0.681 |                 51.7 |            0.733 |            0.72  | 0.739                          |
| Model v1 + qc_bloc_meta (mean of logits)       | once (terms ≤ 2017) | 0.665 [0.640, 0.689] | 0.739 [0.711, 0.766] |      0.67  |      0.63  |                 49.1 |            0.675 |            0.706 | 0.739                          |

## Per term: AUC

|   Term |   Cases |   qc_bloc, each term |   qc_bloc, once |   katz_rf, each term |   katz_rf, once |   v1, each term |   v1, once |   v1+qc_bloc, each term |   v1+qc_bloc, once |
|-------:|--------:|---------------------:|----------------:|---------------------:|----------------:|----------------:|-----------:|------------------------:|-------------------:|
|   2018 |      69 |                0.794 |           0.794 |                0.637 |           0.637 |           0.808 |      0.808 |                   0.818 |              0.818 |
|   2019 |      55 |                0.691 |           0.692 |                0.648 |           0.65  |           0.775 |      0.754 |                   0.754 |              0.74  |
|   2020 |      57 |                0.723 |           0.611 |                0.683 |           0.661 |           0.75  |      0.775 |                   0.784 |              0.733 |
|   2021 |      62 |                0.764 |           0.642 |                0.724 |           0.677 |           0.779 |      0.757 |                   0.789 |              0.72  |
|   2022 |      59 |                0.681 |           0.597 |                0.618 |           0.594 |           0.7   |      0.694 |                   0.705 |              0.66  |
|   2023 |      61 |                0.798 |           0.674 |                0.657 |           0.66  |           0.814 |      0.787 |                   0.825 |              0.756 |
|   2024 |      60 |                0.763 |           0.632 |                0.735 |           0.696 |           0.81  |      0.798 |                   0.816 |              0.734 |
|   2025 |      58 |                0.835 |           0.646 |                0.69  |           0.665 |           0.858 |      0.815 |                   0.867 |              0.771 |

## Per term: Macro-F1

|   Term |   Cases |   qc_bloc, each term |   qc_bloc, once |   katz_rf, each term |   katz_rf, once |   v1, each term |   v1, once |   v1+qc_bloc, each term |   v1+qc_bloc, once |
|-------:|--------:|---------------------:|----------------:|---------------------:|----------------:|----------------:|-----------:|------------------------:|-------------------:|
|   2018 |      69 |                0.717 |           0.717 |                0.593 |           0.593 |           0.728 |      0.728 |                   0.743 |              0.743 |
|   2019 |      55 |                0.646 |           0.652 |                0.573 |           0.592 |           0.692 |      0.676 |                   0.687 |              0.66  |
|   2020 |      57 |                0.635 |           0.567 |                0.641 |           0.634 |           0.647 |      0.702 |                   0.702 |              0.659 |
|   2021 |      62 |                0.697 |           0.58  |                0.659 |           0.617 |           0.65  |      0.653 |                   0.677 |              0.621 |
|   2022 |      59 |                0.623 |           0.562 |                0.57  |           0.556 |           0.663 |      0.648 |                   0.648 |              0.612 |
|   2023 |      61 |                0.7   |           0.611 |                0.651 |           0.621 |           0.711 |      0.688 |                   0.729 |              0.661 |
|   2024 |      60 |                0.648 |           0.573 |                0.647 |           0.647 |           0.749 |      0.743 |                   0.753 |              0.663 |
|   2025 |      58 |                0.711 |           0.58  |                0.646 |           0.613 |           0.752 |      0.693 |                   0.772 |              0.656 |

## Differences (paired bootstrap over cases, 95% CI)

| Comparison                                                                                                    | Terms     | AUC                     | Macro-F1                |
|:--------------------------------------------------------------------------------------------------------------|:----------|:------------------------|:------------------------|
| Justice vote rate: each term − once                                                                           | 2019–2025 | +0.015 [-0.006, +0.036] | +0.043 [+0.030, +0.057] |
| Flat ModernBERT, first 8k tokens (frozen): each term − once                                                   | 2019–2025 | +0.021 [-0.005, +0.046] | +0.019 [-0.004, +0.043] |
| Flat ModernBERT, whole transcript (frozen): each term − once                                                  | 2019–2025 | +0.037 [+0.010, +0.065] | +0.015 [-0.009, +0.040] |
| Case profile only, Katz-style forest (katz_rf): each term − once                                              | 2019–2025 | +0.023 [+0.003, +0.042] | +0.016 [-0.006, +0.037] |
| Question features, LR (qc_plus): each term − once                                                             | 2019–2025 | +0.009 [+0.005, +0.013] | +0.011 [+0.005, +0.017] |
| + named colleagues, LR (qc_bloc): each term − once                                                            | 2019–2025 | +0.112 [+0.092, +0.132] | +0.078 [+0.062, +0.094] |
| + case profile history, LR (qc_bloc_meta): each term − once                                                   | 2019–2025 | +0.116 [+0.097, +0.137] | +0.092 [+0.074, +0.111] |
| Model v1: each term − once                                                                                    | 2019–2025 | +0.017 [+0.006, +0.028] | +0.008 [-0.007, +0.022] |
| Model v1 + metadata prior: each term − once                                                                   | 2019–2025 | +0.025 [+0.013, +0.037] | +0.020 [+0.007, +0.035] |
| Model v1 + qc_bloc (mean of logits): each term − once                                                         | 2019–2025 | +0.062 [+0.048, +0.076] | +0.062 [+0.047, +0.077] |
| Model v1 + qc_bloc_meta (mean of logits): each term − once                                                    | 2019–2025 | +0.062 [+0.048, +0.077] | +0.054 [+0.038, +0.069] |
| Flat ModernBERT, whole transcript (frozen) − Flat ModernBERT, first 8k tokens (frozen) (both refit each term) | 2018–2025 | +0.005 [-0.035, +0.042] | -0.014 [-0.043, +0.015] |
| Model v1 − Flat ModernBERT, whole transcript (frozen) (both refit each term)                                  | 2018–2025 | +0.206 [+0.168, +0.247] | +0.163 [+0.131, +0.197] |
| Model v1 − + named colleagues, LR (qc_bloc) (both refit each term)                                            | 2018–2025 | +0.026 [+0.007, +0.045] | +0.025 [+0.006, +0.046] |
| Model v1 + qc_bloc (mean of logits) − + named colleagues, LR (qc_bloc) (both refit each term)                 | 2018–2025 | +0.036 [+0.024, +0.049] | +0.040 [+0.025, +0.055] |
| Model v1 + qc_bloc (mean of logits) − Model v1 (both refit each term)                                         | 2018–2025 | +0.010 [+0.003, +0.017] | +0.014 [+0.004, +0.025] |
| Model v1 + metadata prior − Model v1 (both refit each term)                                                   | 2018–2025 | +0.004 [-0.005, +0.014] | +0.004 [-0.008, +0.016] |
| Model v1 + qc_bloc_meta (mean of logits) − Model v1 + qc_bloc (mean of logits) (both refit each term)         | 2018–2025 | +0.009 [+0.005, +0.014] | +0.007 [-0.001, +0.016] |

## The steps

|   Term predicted |   Fitted on (cases, all terms) |   of which since 1980 (network) | Held over, not yet decided   |   Cases predicted |
|-----------------:|-------------------------------:|--------------------------------:|:-----------------------------|------------------:|
|             2018 |                           6511 |                            3509 | –                            |                69 |
|             2019 |                           6580 |                            3578 | –                            |                55 |
|             2020 |                           6635 |                            3633 | –                            |                57 |
|             2021 |                           6692 |                            3690 | –                            |                62 |
|             2022 |                           6754 |                            3752 | –                            |                59 |
|             2023 |                           6813 |                            3811 | –                            |                61 |
|             2024 |                           6874 |                            3872 | –                            |                60 |
|             2025 |                           6934 |                            3932 | –                            |                58 |

Cases without an SCDB decision day, taken as decided within their own term: 20.
