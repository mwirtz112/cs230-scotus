# Walk-forward evaluation, terms 2012–2017

Rehearsal on train and validation terms. The test terms (2018–25) were not read. For each term every method is refit on the cases of earlier terms decided before that term opened (1 October), then predicts the term. 'each term' = that refit; 'once' = the fit made before 2012, applied unchanged to every term (the two agree on the first term by construction). Network rows are the mean-probability ensemble of 5 seeds of the fixed recipe (train from 1980, 6 epochs, no early stopping). Cutoff 0.5 (justice vote rate: the training rate). Point estimate [95% bootstrap CI over cases]. '@rate' columns: the same predictions cut, within each term, where the share of predicted petitioner votes equals the pro-petitioner share of the votes the model was fitted on (59.8–60.0%); this uses the term's predictions but none of its labels. 'Case acc, base rate restored': the case call when a vote counts for the petitioner if p > 1 − π, π being that same training share. The class-balanced loss takes the base rate out of p and this puts it back, with no fitted number and nothing of the predicted term; a secondary column, fixed before the test run (notes §19.7). '–' for the methods without a class-balanced probability.

## All scored terms, 2012–2017 (379 cases, 3,287 votes)

| Method                                         | Fit                 | Macro-F1             | AUC                  |   Vote acc |   Case acc |   Pred. petitioner % |   Macro-F1 @rate |   Case acc @rate | Case acc, base rate restored   |
|:-----------------------------------------------|:--------------------|:---------------------|:---------------------|-----------:|-----------:|---------------------:|-----------------:|-----------------:|:-------------------------------|
| Always petitioner                              | –                   | 0.386 [0.372, 0.399] | 0.500 [0.500, 0.500] |      0.628 |      0.675 |                100   |            0.386 |            0.675 | –                              |
| Justice vote rate                              | each term           | 0.517 [0.501, 0.532] | 0.513 [0.494, 0.531] |      0.536 |      0.544 |                 57.2 |            0.51  |            0.673 | –                              |
| Justice vote rate                              | once (terms ≤ 2011) | 0.503 [0.489, 0.517] | 0.522 [0.506, 0.537] |      0.509 |      0.412 |                 48.1 |            0.518 |            0.673 | –                              |
| Question count                                 | –                   | 0.609 [0.585, 0.634] | 0.649 [0.621, 0.680] |      0.614 |      0.58  |                 48.4 |            0.616 |            0.681 | –                              |
| Flat ModernBERT, first 8k tokens (frozen)      | each term           | 0.592 [0.560, 0.620] | 0.611 [0.570, 0.648] |      0.614 |      0.612 |                 60.3 |            0.587 |            0.609 | 0.681                          |
| Flat ModernBERT, first 8k tokens (frozen)      | once (terms ≤ 2011) | 0.585 [0.554, 0.613] | 0.613 [0.571, 0.650] |      0.611 |      0.646 |                 61.9 |            0.581 |            0.609 | 0.681                          |
| Flat ModernBERT, whole transcript (frozen)     | each term           | 0.588 [0.557, 0.617] | 0.613 [0.574, 0.652] |      0.611 |      0.623 |                 60.9 |            0.581 |            0.62  | 0.654                          |
| Flat ModernBERT, whole transcript (frozen)     | once (terms ≤ 2011) | 0.573 [0.546, 0.598] | 0.606 [0.569, 0.640] |      0.587 |      0.615 |                 55.3 |            0.57  |            0.623 | 0.665                          |
| Case profile only, Katz-style forest (katz_rf) | each term           | 0.605 [0.580, 0.633] | 0.647 [0.617, 0.680] |      0.643 |      0.609 |                 67.9 |            0.602 |            0.583 | 0.678                          |
| Case profile only, Katz-style forest (katz_rf) | once (terms ≤ 2011) | 0.603 [0.572, 0.630] | 0.641 [0.606, 0.674] |      0.642 |      0.631 |                 68.8 |            0.602 |            0.602 | 0.670                          |
| Question features, LR (qc_plus)                | each term           | 0.650 [0.623, 0.676] | 0.726 [0.696, 0.756] |      0.655 |      0.591 |                 49.6 |            0.659 |            0.665 | 0.694                          |
| Question features, LR (qc_plus)                | once (terms ≤ 2011) | 0.650 [0.624, 0.676] | 0.726 [0.697, 0.756] |      0.655 |      0.588 |                 49   |            0.658 |            0.673 | 0.694                          |
| + named colleagues, LR (qc_bloc)               | each term           | 0.682 [0.655, 0.708] | 0.752 [0.721, 0.780] |      0.692 |      0.649 |                 54.3 |            0.681 |            0.668 | 0.702                          |
| + named colleagues, LR (qc_bloc)               | once (terms ≤ 2011) | 0.679 [0.652, 0.705] | 0.749 [0.719, 0.779] |      0.688 |      0.66  |                 53.1 |            0.679 |            0.686 | 0.699                          |
| + case profile history, LR (qc_bloc_meta)      | each term           | 0.684 [0.659, 0.708] | 0.756 [0.728, 0.783] |      0.694 |      0.66  |                 54.8 |            0.685 |            0.673 | 0.686                          |
| + case profile history, LR (qc_bloc_meta)      | once (terms ≤ 2011) | 0.676 [0.649, 0.701] | 0.758 [0.729, 0.785] |      0.683 |      0.641 |                 52.4 |            0.682 |            0.675 | 0.699                          |
| Model v1                                       | each term           | 0.687 [0.660, 0.715] | 0.767 [0.737, 0.799] |      0.697 |      0.678 |                 54.9 |            0.685 |            0.673 | 0.704                          |
| Model v1                                       | once (terms ≤ 2011) | 0.680 [0.649, 0.710] | 0.767 [0.737, 0.798] |      0.686 |      0.644 |                 50.6 |            0.685 |            0.686 | 0.699                          |
| Model v1 + metadata prior                      | each term           | 0.689 [0.660, 0.718] | 0.769 [0.738, 0.798] |      0.696 |      0.665 |                 52   |            0.69  |            0.683 | 0.683                          |
| Model v1 + metadata prior                      | once (terms ≤ 2011) | 0.683 [0.654, 0.713] | 0.772 [0.741, 0.801] |      0.686 |      0.649 |                 46.4 |            0.69  |            0.691 | 0.681                          |
| Model v1 + qc_bloc (mean of logits)            | each term           | 0.695 [0.669, 0.722] | 0.776 [0.745, 0.807] |      0.705 |      0.673 |                 55.2 |            0.695 |            0.691 | 0.707                          |
| Model v1 + qc_bloc (mean of logits)            | once (terms ≤ 2011) | 0.690 [0.663, 0.717] | 0.776 [0.745, 0.807] |      0.696 |      0.668 |                 51.7 |            0.69  |            0.691 | 0.707                          |
| Model v1 + qc_bloc_meta (mean of logits)       | each term           | 0.701 [0.674, 0.726] | 0.783 [0.754, 0.812] |      0.711 |      0.691 |                 55.2 |            0.702 |            0.699 | 0.712                          |
| Model v1 + qc_bloc_meta (mean of logits)       | once (terms ≤ 2011) | 0.695 [0.668, 0.724] | 0.786 [0.756, 0.815] |      0.701 |      0.683 |                 51   |            0.704 |            0.689 | 0.702                          |

## 2012–15, the dev terms (256 cases, 2,238 votes)

| Method                                         | Fit                     | Macro-F1             | AUC                  |   Vote acc |   Case acc |   Pred. petitioner % |   Macro-F1 @rate |   Case acc @rate | Case acc, base rate restored   |
|:-----------------------------------------------|:------------------------|:---------------------|:---------------------|-----------:|-----------:|---------------------:|-----------------:|-----------------:|:-------------------------------|
| Always petitioner                              | –                       | 0.380 [0.364, 0.397] | 0.500 [0.500, 0.500] |      0.614 |      0.648 |                100   |            0.38  |            0.648 | –                              |
| Justice vote rate                              | each term               | 0.504 [0.486, 0.521] | 0.507 [0.484, 0.529] |      0.511 |      0.453 |                 50.7 |            0.51  |            0.645 | –                              |
| Justice vote rate                              | once (terms ≤ 2011)     | 0.500 [0.484, 0.514] | 0.517 [0.499, 0.534] |      0.502 |      0.352 |                 45.5 |            0.513 |            0.645 | –                              |
| Question count                                 | –                       | 0.601 [0.574, 0.628] | 0.627 [0.589, 0.662] |      0.606 |      0.605 |                 49.5 |            0.606 |            0.672 | –                              |
| Flat ModernBERT, first 8k tokens (frozen)      | each term               | 0.584 [0.545, 0.618] | 0.604 [0.556, 0.649] |      0.603 |      0.594 |                 60.2 |            0.578 |            0.59  | 0.641                          |
| Flat ModernBERT, first 8k tokens (frozen)      | once (terms ≤ 2011)     | 0.581 [0.544, 0.613] | 0.602 [0.555, 0.645] |      0.609 |      0.641 |                 64.4 |            0.572 |            0.598 | 0.648                          |
| Flat ModernBERT, first 8k tokens (frozen)      | dev fold (terms ≤ 2011) | 0.587 [0.552, 0.620] | 0.603 [0.555, 0.648] |      0.61  |      0.617 |                 62   |            0.587 |            0.625 | 0.637                          |
| Flat ModernBERT, whole transcript (frozen)     | each term               | 0.582 [0.545, 0.617] | 0.601 [0.552, 0.648] |      0.601 |      0.613 |                 59.7 |            0.578 |            0.617 | 0.637                          |
| Flat ModernBERT, whole transcript (frozen)     | once (terms ≤ 2011)     | 0.580 [0.545, 0.612] | 0.612 [0.567, 0.655] |      0.596 |      0.633 |                 58   |            0.571 |            0.625 | 0.660                          |
| Flat ModernBERT, whole transcript (frozen)     | dev fold (terms ≤ 2011) | 0.573 [0.540, 0.607] | 0.608 [0.563, 0.652] |      0.595 |      0.625 |                 61.2 |            0.572 |            0.633 | 0.664                          |
| Case profile only, Katz-style forest (katz_rf) | each term               | 0.596 [0.564, 0.630] | 0.622 [0.582, 0.660] |      0.629 |      0.59  |                 67.1 |            0.593 |            0.578 | 0.656                          |
| Case profile only, Katz-style forest (katz_rf) | once (terms ≤ 2011)     | 0.589 [0.553, 0.624] | 0.621 [0.582, 0.661] |      0.627 |      0.617 |                 68.9 |            0.591 |            0.594 | 0.648                          |
| Case profile only, Katz-style forest (katz_rf) | dev fold (terms ≤ 2011) | 0.592 [0.558, 0.627] | 0.621 [0.583, 0.662] |      0.628 |      0.598 |                 68.5 |            0.596 |            0.582 | 0.648                          |
| Question features, LR (qc_plus)                | each term               | 0.638 [0.605, 0.669] | 0.703 [0.668, 0.740] |      0.642 |      0.594 |                 48.8 |            0.642 |            0.652 | 0.672                          |
| Question features, LR (qc_plus)                | once (terms ≤ 2011)     | 0.639 [0.605, 0.670] | 0.703 [0.667, 0.739] |      0.642 |      0.582 |                 48.5 |            0.638 |            0.66  | 0.672                          |
| Question features, LR (qc_plus)                | dev fold (terms ≤ 2011) | 0.639 [0.606, 0.670] | 0.703 [0.668, 0.740] |      0.643 |      0.578 |                 48.4 |            0.639 |            0.664 | 0.676                          |
| + named colleagues, LR (qc_bloc)               | each term               | 0.660 [0.628, 0.689] | 0.725 [0.688, 0.759] |      0.667 |      0.617 |                 52.7 |            0.659 |            0.641 | 0.652                          |
| + named colleagues, LR (qc_bloc)               | once (terms ≤ 2011)     | 0.660 [0.626, 0.690] | 0.725 [0.689, 0.762] |      0.666 |      0.637 |                 51.7 |            0.658 |            0.648 | 0.660                          |
| + named colleagues, LR (qc_bloc)               | dev fold (terms ≤ 2011) | 0.663 [0.629, 0.693] | 0.726 [0.689, 0.761] |      0.669 |      0.641 |                 51.8 |            0.655 |            0.641 | 0.664                          |
| + case profile history, LR (qc_bloc_meta)      | each term               | 0.667 [0.634, 0.696] | 0.734 [0.698, 0.765] |      0.675 |      0.633 |                 53.4 |            0.67  |            0.652 | 0.664                          |
| + case profile history, LR (qc_bloc_meta)      | once (terms ≤ 2011)     | 0.669 [0.636, 0.696] | 0.736 [0.700, 0.768] |      0.675 |      0.641 |                 52.2 |            0.665 |            0.645 | 0.660                          |
| + case profile history, LR (qc_bloc_meta)      | dev fold (terms ≤ 2011) | 0.667 [0.633, 0.695] | 0.736 [0.701, 0.768] |      0.673 |      0.648 |                 52.2 |            0.665 |            0.648 | 0.656                          |
| Model v1                                       | each term               | 0.665 [0.628, 0.697] | 0.744 [0.704, 0.782] |      0.669 |      0.617 |                 50.6 |            0.666 |            0.621 | 0.648                          |
| Model v1                                       | once (terms ≤ 2011)     | 0.662 [0.623, 0.695] | 0.743 [0.705, 0.781] |      0.665 |      0.594 |                 48   |            0.671 |            0.652 | 0.660                          |
| Model v1                                       | dev fold (terms ≤ 2011) | 0.667 [0.629, 0.700] | 0.744 [0.705, 0.782] |      0.67  |      0.629 |                 48.2 |            0.668 |            0.656 | 0.656                          |
| Model v1 + metadata prior                      | each term               | 0.666 [0.628, 0.699] | 0.747 [0.709, 0.783] |      0.669 |      0.617 |                 47.2 |            0.671 |            0.648 | 0.645                          |
| Model v1 + metadata prior                      | once (terms ≤ 2011)     | 0.667 [0.628, 0.699] | 0.749 [0.711, 0.784] |      0.668 |      0.621 |                 44.3 |            0.669 |            0.664 | 0.641                          |
| Model v1 + metadata prior                      | dev fold (terms ≤ 2011) | 0.659 [0.620, 0.692] | 0.752 [0.716, 0.788] |      0.66  |      0.617 |                 42.8 |            0.671 |            0.664 | 0.637                          |
| Model v1 + qc_bloc (mean of logits)            | each term               | 0.668 [0.634, 0.699] | 0.751 [0.713, 0.786] |      0.674 |      0.617 |                 52.1 |            0.675 |            0.656 | 0.660                          |
| Model v1 + qc_bloc (mean of logits)            | once (terms ≤ 2011)     | 0.673 [0.636, 0.703] | 0.751 [0.714, 0.788] |      0.676 |      0.633 |                 49.6 |            0.673 |            0.664 | 0.656                          |
| Model v1 + qc_bloc (mean of logits)            | dev fold (terms ≤ 2011) | 0.671 [0.635, 0.702] | 0.753 [0.715, 0.789] |      0.675 |      0.629 |                 49.6 |            0.673 |            0.664 | 0.656                          |
| Model v1 + qc_bloc_meta (mean of logits)       | each term               | 0.682 [0.647, 0.711] | 0.760 [0.723, 0.795] |      0.688 |      0.652 |                 52.2 |            0.688 |            0.66  | 0.672                          |
| Model v1 + qc_bloc_meta (mean of logits)       | once (terms ≤ 2011)     | 0.680 [0.642, 0.711] | 0.763 [0.726, 0.798] |      0.684 |      0.648 |                 49.6 |            0.687 |            0.664 | 0.656                          |
| Model v1 + qc_bloc_meta (mean of logits)       | dev fold (terms ≤ 2011) | 0.682 [0.645, 0.713] | 0.764 [0.728, 0.800] |      0.686 |      0.664 |                 49.9 |            0.686 |            0.656 | 0.664                          |

## 2016–17, the validation terms (123 cases, 1,049 votes)

| Method                                         | Fit                            | Macro-F1             | AUC                  |   Vote acc |   Case acc |   Pred. petitioner % |   Macro-F1 @rate |   Case acc @rate | Case acc, base rate restored   |
|:-----------------------------------------------|:-------------------------------|:---------------------|:---------------------|-----------:|-----------:|---------------------:|-----------------:|-----------------:|:-------------------------------|
| Always petitioner                              | –                              | 0.397 [0.375, 0.419] | 0.500 [0.500, 0.500] |      0.659 |      0.732 |                100   |            0.397 |            0.732 | –                              |
| Justice vote rate                              | each term                      | 0.526 [0.506, 0.543] | 0.509 [0.481, 0.538] |      0.59  |      0.732 |                 71   |            0.509 |            0.732 | –                              |
| Justice vote rate                              | once (terms ≤ 2011)            | 0.506 [0.479, 0.532] | 0.530 [0.503, 0.559] |      0.524 |      0.537 |                 53.6 |            0.528 |            0.732 | –                              |
| Justice vote rate                              | validation fold (terms ≤ 2015) | 0.521 [0.503, 0.539] | 0.535 [0.514, 0.557] |      0.607 |      0.732 |                 76.5 |            0.509 |            0.732 | –                              |
| Question count                                 | –                              | 0.626 [0.582, 0.664] | 0.700 [0.656, 0.747] |      0.631 |      0.528 |                 45.9 |            0.638 |            0.699 | –                              |
| Flat ModernBERT, first 8k tokens (frozen)      | each term                      | 0.609 [0.556, 0.661] | 0.628 [0.557, 0.694] |      0.636 |      0.65  |                 60.3 |            0.605 |            0.65  | 0.764                          |
| Flat ModernBERT, first 8k tokens (frozen)      | once (terms ≤ 2011)            | 0.595 [0.537, 0.647] | 0.646 [0.577, 0.715] |      0.616 |      0.659 |                 56.6 |            0.598 |            0.634 | 0.748                          |
| Flat ModernBERT, first 8k tokens (frozen)      | validation fold (terms ≤ 2015) | 0.580 [0.524, 0.632] | 0.617 [0.542, 0.688] |      0.596 |      0.585 |                 53.7 |            0.58  |            0.626 | 0.748                          |
| Flat ModernBERT, whole transcript (frozen)     | each term                      | 0.600 [0.542, 0.652] | 0.640 [0.572, 0.706] |      0.634 |      0.642 |                 63.4 |            0.587 |            0.626 | 0.691                          |
| Flat ModernBERT, whole transcript (frozen)     | once (terms ≤ 2011)            | 0.557 [0.501, 0.609] | 0.600 [0.533, 0.666] |      0.567 |      0.577 |                 49.5 |            0.565 |            0.618 | 0.675                          |
| Flat ModernBERT, whole transcript (frozen)     | validation fold (terms ≤ 2015) | 0.607 [0.545, 0.662] | 0.633 [0.562, 0.701] |      0.636 |      0.626 |                 61.1 |            0.584 |            0.618 | 0.675                          |
| Case profile only, Katz-style forest (katz_rf) | each term                      | 0.624 [0.577, 0.671] | 0.705 [0.656, 0.758] |      0.671 |      0.65  |                 69.6 |            0.619 |            0.593 | 0.724                          |
| Case profile only, Katz-style forest (katz_rf) | once (terms ≤ 2011)            | 0.632 [0.580, 0.677] | 0.695 [0.643, 0.749] |      0.675 |      0.659 |                 68.4 |            0.624 |            0.618 | 0.715                          |
| Case profile only, Katz-style forest (katz_rf) | validation fold (terms ≤ 2015) | 0.624 [0.577, 0.672] | 0.715 [0.670, 0.765] |      0.686 |      0.691 |                 74.9 |            0.633 |            0.602 | 0.724                          |
| Question features, LR (qc_plus)                | each term                      | 0.673 [0.629, 0.717] | 0.778 [0.727, 0.826] |      0.683 |      0.585 |                 51.3 |            0.697 |            0.691 | 0.740                          |
| Question features, LR (qc_plus)                | once (terms ≤ 2011)            | 0.675 [0.633, 0.717] | 0.780 [0.730, 0.827] |      0.684 |      0.602 |                 50   |            0.7   |            0.699 | 0.740                          |
| Question features, LR (qc_plus)                | validation fold (terms ≤ 2015) | 0.675 [0.629, 0.720] | 0.780 [0.728, 0.827] |      0.684 |      0.585 |                 50.6 |            0.699 |            0.691 | 0.740                          |
| + named colleagues, LR (qc_bloc)               | each term                      | 0.730 [0.687, 0.772] | 0.813 [0.770, 0.858] |      0.745 |      0.715 |                 57.8 |            0.729 |            0.724 | 0.805                          |
| + named colleagues, LR (qc_bloc)               | once (terms ≤ 2011)            | 0.721 [0.679, 0.760] | 0.804 [0.760, 0.847] |      0.734 |      0.707 |                 56.1 |            0.722 |            0.764 | 0.780                          |
| + named colleagues, LR (qc_bloc)               | validation fold (terms ≤ 2015) | 0.735 [0.690, 0.774] | 0.818 [0.777, 0.862] |      0.748 |      0.715 |                 56.9 |            0.731 |            0.724 | 0.821                          |
| + case profile history, LR (qc_bloc_meta)      | each term                      | 0.720 [0.681, 0.759] | 0.806 [0.763, 0.850] |      0.736 |      0.715 |                 57.8 |            0.717 |            0.715 | 0.732                          |
| + case profile history, LR (qc_bloc_meta)      | once (terms ≤ 2011)            | 0.691 [0.650, 0.729] | 0.809 [0.767, 0.850] |      0.702 |      0.642 |                 53   |            0.718 |            0.74  | 0.780                          |
| + case profile history, LR (qc_bloc_meta)      | validation fold (terms ≤ 2015) | 0.724 [0.683, 0.765] | 0.813 [0.771, 0.856] |      0.739 |      0.691 |                 57.5 |            0.727 |            0.715 | 0.756                          |
| Model v1                                       | each term                      | 0.732 [0.685, 0.779] | 0.812 [0.755, 0.864] |      0.756 |      0.805 |                 64   |            0.725 |            0.78  | 0.821                          |
| Model v1                                       | once (terms ≤ 2011)            | 0.717 [0.664, 0.765] | 0.815 [0.762, 0.862] |      0.73  |      0.748 |                 56.1 |            0.716 |            0.756 | 0.780                          |
| Model v1                                       | validation fold (terms ≤ 2015) | 0.748 [0.704, 0.791] | 0.821 [0.767, 0.873] |      0.765 |      0.805 |                 60.5 |            0.738 |            0.805 | 0.797                          |
| Model v1 + metadata prior                      | each term                      | 0.732 [0.684, 0.776] | 0.813 [0.757, 0.862] |      0.753 |      0.764 |                 62.2 |            0.731 |            0.756 | 0.764                          |
| Model v1 + metadata prior                      | once (terms ≤ 2011)            | 0.715 [0.667, 0.758] | 0.819 [0.767, 0.866] |      0.724 |      0.707 |                 51   |            0.735 |            0.748 | 0.764                          |
| Model v1 + metadata prior                      | validation fold (terms ≤ 2015) | 0.756 [0.712, 0.796] | 0.822 [0.767, 0.872] |      0.771 |      0.772 |                 59.4 |            0.748 |            0.772 | 0.813                          |
| Model v1 + qc_bloc (mean of logits)            | each term                      | 0.750 [0.704, 0.794] | 0.829 [0.777, 0.877] |      0.769 |      0.789 |                 61.7 |            0.74  |            0.764 | 0.805                          |
| Model v1 + qc_bloc (mean of logits)            | once (terms ≤ 2011)            | 0.724 [0.677, 0.769] | 0.828 [0.781, 0.871] |      0.738 |      0.74  |                 56.2 |            0.727 |            0.748 | 0.813                          |
| Model v1 + qc_bloc (mean of logits)            | validation fold (terms ≤ 2015) | 0.747 [0.700, 0.788] | 0.836 [0.789, 0.882] |      0.764 |      0.805 |                 59.8 |            0.744 |            0.78  | 0.797                          |
| Model v1 + qc_bloc_meta (mean of logits)       | each term                      | 0.739 [0.693, 0.781] | 0.830 [0.780, 0.877] |      0.759 |      0.772 |                 61.6 |            0.731 |            0.78  | 0.797                          |
| Model v1 + qc_bloc_meta (mean of logits)       | once (terms ≤ 2011)            | 0.726 [0.680, 0.770] | 0.835 [0.789, 0.878] |      0.737 |      0.756 |                 54.1 |            0.741 |            0.74  | 0.797                          |
| Model v1 + qc_bloc_meta (mean of logits)       | validation fold (terms ≤ 2015) | 0.751 [0.708, 0.793] | 0.837 [0.790, 0.882] |      0.767 |      0.789 |                 59.6 |            0.736 |            0.789 | 0.797                          |

## Per term: AUC

|   Term |   Cases |   qc_bloc, each term |   qc_bloc, once |   katz_rf, each term |   katz_rf, once |   v1, each term |   v1, once |   v1+qc_bloc, each term |   v1+qc_bloc, once |
|-------:|--------:|---------------------:|----------------:|---------------------:|----------------:|----------------:|-----------:|------------------------:|-------------------:|
|   2012 |      73 |                0.745 |           0.745 |                0.609 |           0.609 |           0.761 |      0.761 |                   0.769 |              0.769 |
|   2013 |      66 |                0.68  |           0.677 |                0.631 |           0.642 |           0.705 |      0.697 |                   0.706 |              0.699 |
|   2014 |      53 |                0.785 |           0.781 |                0.654 |           0.661 |           0.777 |      0.781 |                   0.797 |              0.798 |
|   2015 |      64 |                0.69  |           0.692 |                0.581 |           0.561 |           0.729 |      0.725 |                   0.726 |              0.729 |
|   2016 |      62 |                0.79  |           0.794 |                0.7   |           0.663 |           0.838 |      0.833 |                   0.833 |              0.833 |
|   2017 |      61 |                0.832 |           0.813 |                0.693 |           0.711 |           0.806 |      0.8   |                   0.835 |              0.824 |

## Per term: Macro-F1

|   Term |   Cases |   qc_bloc, each term |   qc_bloc, once |   katz_rf, each term |   katz_rf, once |   v1, each term |   v1, once |   v1+qc_bloc, each term |   v1+qc_bloc, once |
|-------:|--------:|---------------------:|----------------:|---------------------:|----------------:|----------------:|-----------:|------------------------:|-------------------:|
|   2012 |      73 |                0.672 |           0.672 |                0.596 |           0.596 |           0.666 |      0.666 |                   0.679 |              0.679 |
|   2013 |      66 |                0.642 |           0.646 |                0.595 |           0.585 |           0.623 |      0.621 |                   0.636 |              0.623 |
|   2014 |      53 |                0.675 |           0.682 |                0.62  |           0.608 |           0.689 |      0.694 |                   0.696 |              0.721 |
|   2015 |      64 |                0.643 |           0.629 |                0.567 |           0.56  |           0.678 |      0.663 |                   0.656 |              0.663 |
|   2016 |      62 |                0.701 |           0.689 |                0.624 |           0.602 |           0.728 |      0.708 |                   0.734 |              0.714 |
|   2017 |      61 |                0.751 |           0.744 |                0.612 |           0.65  |           0.734 |      0.719 |                   0.762 |              0.729 |

## Differences (paired bootstrap over cases, 95% CI)

| Comparison                                                                                                    | Terms     | AUC                     | Macro-F1                |
|:--------------------------------------------------------------------------------------------------------------|:----------|:------------------------|:------------------------|
| Justice vote rate: each term − once                                                                           | 2013–2017 | -0.012 [-0.029, +0.005] | +0.015 [-0.002, +0.030] |
| Flat ModernBERT, first 8k tokens (frozen): each term − once                                                   | 2013–2017 | -0.002 [-0.027, +0.020] | +0.008 [-0.017, +0.031] |
| Flat ModernBERT, whole transcript (frozen): each term − once                                                  | 2013–2017 | +0.009 [-0.014, +0.034] | +0.018 [-0.003, +0.040] |
| Case profile only, Katz-style forest (katz_rf): each term − once                                              | 2013–2017 | +0.008 [-0.009, +0.023] | +0.003 [-0.020, +0.028] |
| Question features, LR (qc_plus): each term − once                                                             | 2013–2017 | -0.000 [-0.002, +0.002] | -0.001 [-0.006, +0.003] |
| + named colleagues, LR (qc_bloc): each term − once                                                            | 2013–2017 | +0.004 [-0.004, +0.011] | +0.004 [-0.008, +0.015] |
| + case profile history, LR (qc_bloc_meta): each term − once                                                   | 2013–2017 | -0.002 [-0.010, +0.005] | +0.011 [-0.000, +0.021] |
| Model v1: each term − once                                                                                    | 2013–2017 | +0.001 [-0.008, +0.009] | +0.009 [-0.006, +0.021] |
| Model v1 + metadata prior: each term − once                                                                   | 2013–2017 | -0.003 [-0.013, +0.007] | +0.007 [-0.010, +0.020] |
| Model v1 + qc_bloc (mean of logits): each term − once                                                         | 2013–2017 | +0.001 [-0.007, +0.008] | +0.006 [-0.006, +0.018] |
| Model v1 + qc_bloc_meta (mean of logits): each term − once                                                    | 2013–2017 | -0.003 [-0.010, +0.004] | +0.008 [-0.003, +0.019] |
| Flat ModernBERT, whole transcript (frozen) − Flat ModernBERT, first 8k tokens (frozen) (both refit each term) | 2012–2017 | +0.003 [-0.032, +0.036] | -0.004 [-0.031, +0.025] |
| Model v1 − Flat ModernBERT, whole transcript (frozen) (both refit each term)                                  | 2012–2017 | +0.154 [+0.107, +0.198] | +0.099 [+0.061, +0.135] |
| Model v1 − + named colleagues, LR (qc_bloc) (both refit each term)                                            | 2012–2017 | +0.016 [-0.007, +0.037] | +0.005 [-0.018, +0.027] |
| Model v1 + qc_bloc (mean of logits) − + named colleagues, LR (qc_bloc) (both refit each term)                 | 2012–2017 | +0.025 [+0.011, +0.038] | +0.012 [-0.003, +0.027] |
| Model v1 + qc_bloc (mean of logits) − Model v1 (both refit each term)                                         | 2012–2017 | +0.009 [+0.000, +0.019] | +0.007 [-0.005, +0.020] |
| Model v1 + metadata prior − Model v1 (both refit each term)                                                   | 2012–2017 | +0.002 [-0.010, +0.013] | +0.002 [-0.012, +0.017] |
| Model v1 + qc_bloc_meta (mean of logits) − Model v1 + qc_bloc (mean of logits) (both refit each term)         | 2012–2017 | +0.007 [+0.001, +0.013] | +0.006 [-0.003, +0.017] |

## The steps

|   Term predicted |   Fitted on (cases, all terms) |   of which since 1980 (network) | Held over, not yet decided   |   Cases predicted |
|-----------------:|-------------------------------:|--------------------------------:|:-----------------------------|------------------:|
|             2012 |                           6131 |                            3129 | 2011_10-1491                 |                73 |
|             2013 |                           6205 |                            3203 | –                            |                66 |
|             2014 |                           6271 |                            3269 | –                            |                53 |
|             2015 |                           6324 |                            3322 | –                            |                64 |
|             2016 |                           6388 |                            3386 | –                            |                62 |
|             2017 |                           6450 |                            3448 | –                            |                61 |

Cases without an SCDB decision day, taken as decided within their own term: 20.

## Harness check: step 2016 against the validation fold

Step 2016 is fitted on exactly the validation fold's training terms (≤ 2015, no case held over), so its predictions for the 2016 term should equal the ones already on disk from the standard runs.

| Method            |   Votes compared |   Largest gap in p |   Mean gap in p |
|:------------------|-----------------:|-------------------:|----------------:|
| always_petitioner |              505 |            0       |         0       |
| justice_rate      |              505 |            0       |         0       |
| question_count    |              505 |            0       |         0       |
| qc_plus           |              505 |            3.3e-16 |         5.9e-18 |
| qc_bloc           |              505 |            3.3e-16 |         5.7e-18 |
| flat_mb           |              505 |            0       |         0       |
| flat_mb_all       |              505 |            0       |         0       |
| katz_rf           |              505 |            2.2e-16 |         2.6e-17 |
| qc_bloc_meta      |              505 |            2.2e-16 |         4.6e-18 |
| v1, seed 0        |              505 |            0.0018  |         0.00028 |
| v1, seed 1        |              505 |            0.0024  |         0.00042 |
| v1, seed 2        |              505 |            0.0015  |         0.00024 |
| v1, seed 3        |              505 |            0.002   |         0.00031 |
| v1, seed 4        |              505 |            0.0019  |         0.00041 |
| v1_fusion, seed 0 |              505 |            0.002   |         0.00029 |
| v1_fusion, seed 1 |              505 |            0.0019  |         0.00027 |
| v1_fusion, seed 2 |              505 |            0.0018  |         0.00025 |
| v1_fusion, seed 3 |              505 |            0.0016  |         0.00029 |
| v1_fusion, seed 4 |              505 |            0.002   |         0.00025 |
