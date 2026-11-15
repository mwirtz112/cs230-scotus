# Main results (validation and dev; test untouched)

Point estimate [95% bootstrap CI over cases]. Cutoff 0.5 unless stated. Dev columns are blank for methods without a dev-fold run.

| Method                                                          | Val macro-F1         | Val AUC              |   Val vote acc |   Val case acc |   Dev macro-F1 |   Dev AUC |
|:----------------------------------------------------------------|:---------------------|:---------------------|---------------:|---------------:|---------------:|----------:|
| Always petitioner                                               | 0.397 [0.375, 0.419] | 0.500 [0.500, 0.500] |          0.659 |          0.732 |                |           |
| Justice vote rate                                               | 0.521 [0.503, 0.539] | 0.535 [0.514, 0.557] |          0.607 |          0.732 |                |           |
| Question count                                                  | 0.626 [0.582, 0.664] | 0.700 [0.656, 0.747] |          0.631 |          0.528 |          0.601 |     0.627 |
| Embedding probe                                                 | 0.595 [0.559, 0.629] | 0.667 [0.631, 0.705] |          0.6   |          0.528 |                |           |
| Flat ModernBERT, first 8k tokens, frozen (flat_mb)              | 0.580 [0.524, 0.632] | 0.617 [0.542, 0.688] |          0.596 |          0.585 |          0.587 |     0.603 |
| Flat ModernBERT, whole transcript, frozen (flat_mb_all)         | 0.607 [0.545, 0.662] | 0.633 [0.562, 0.701] |          0.636 |          0.626 |          0.573 |     0.608 |
| Flat ModernBERT, top 4 layers tuned, lr 5e-6 (flat_mb_ft)       | 0.575 [0.518, 0.629] | 0.624 [0.549, 0.694] |          0.592 |          0.585 |                |           |
| Flat ModernBERT, top 4 layers tuned, lr 1e-5 (flat_mb_ft_lr1e5) | 0.577 [0.521, 0.629] | 0.625 [0.551, 0.697] |          0.595 |          0.569 |                |           |
| Case profile only, Katz-style forest (katz_rf)                  | 0.624 [0.577, 0.672] | 0.715 [0.670, 0.765] |          0.686 |          0.691 |          0.592 |     0.621 |
| Question features, LR (qc_plus)                                 | 0.675 [0.629, 0.720] | 0.780 [0.728, 0.827] |          0.684 |          0.585 |          0.639 |     0.703 |
| + named colleagues, LR (qc_bloc)                                | 0.735 [0.690, 0.774] | 0.818 [0.777, 0.862] |          0.748 |          0.715 |          0.663 |     0.726 |
| + case profile history, LR (qc_bloc_meta)                       | 0.724 [0.683, 0.765] | 0.813 [0.771, 0.856] |          0.739 |          0.691 |          0.667 |     0.736 |
| Model v0 + regularization (milestone)                           | 0.625 [0.572, 0.672] | 0.667 [0.605, 0.727] |          0.673 |          0.732 |                |           |
| Model v1, no text                                               | 0.726 [0.678, 0.775] | 0.803 [0.755, 0.852] |          0.742 |          0.724 |          0.65  |     0.717 |
| Model v1, no bloc layer                                         | 0.712 [0.659, 0.754] | 0.793 [0.734, 0.844] |          0.74  |          0.772 |          0.656 |     0.711 |
| Model v1 (5 seeds)                                              | 0.748 [0.704, 0.791] | 0.821 [0.767, 0.873] |          0.765 |          0.805 |          0.667 |     0.744 |
| Model v1 + metadata prior (fusion, 5 seeds)                     | 0.756 [0.712, 0.796] | 0.822 [0.767, 0.872] |          0.771 |          0.772 |          0.659 |     0.752 |
| Model v1 + qc_bloc (mean of logits)                             | 0.747 [0.700, 0.788] | 0.836 [0.789, 0.882] |          0.764 |          0.805 |          0.671 |     0.753 |
| Model v1 + qc_bloc_meta (mean of logits)                        | 0.751 [0.708, 0.793] | 0.837 [0.790, 0.882] |          0.767 |          0.789 |          0.682 |     0.764 |

## Train-rate cutoff (set on dev, applied to val)

| Method                                                  |   Cutoff | Val macro-F1 @rate   |   Val vote acc @rate |   Val case acc @rate |
|:--------------------------------------------------------|---------:|:---------------------|---------------------:|---------------------:|
| Question count                                          |    0.412 | 0.635 [0.594, 0.672] |                0.654 |                0.659 |
| Flat ModernBERT, first 8k tokens, frozen (flat_mb)      |    0.513 | 0.568 [0.510, 0.621] |                0.581 |                0.569 |
| Flat ModernBERT, whole transcript, frozen (flat_mb_all) |    0.504 | 0.606 [0.546, 0.660] |                0.634 |                0.634 |
| Case profile only, Katz-style forest (katz_rf)          |    0.542 | 0.640 [0.598, 0.683] |                0.668 |                0.634 |
| Question features, LR (qc_plus)                         |    0.447 | 0.708 [0.666, 0.749] |                0.729 |                0.707 |
| + named colleagues, LR (qc_bloc)                        |    0.455 | 0.740 [0.698, 0.783] |                0.763 |                0.764 |
| + case profile history, LR (qc_bloc_meta)               |    0.45  | 0.735 [0.695, 0.776] |                0.76  |                0.732 |
| Model v1, no text                                       |    0.459 | 0.729 [0.683, 0.775] |                0.753 |                0.74  |
| Model v1, no bloc layer                                 |    0.45  | 0.707 [0.654, 0.754] |                0.745 |                0.78  |
| Model v1 (5 seeds)                                      |    0.399 | 0.742 [0.693, 0.789] |                0.777 |                0.797 |
| Model v1 + metadata prior (fusion, 5 seeds)             |    0.352 | 0.722 [0.669, 0.770] |                0.767 |                0.789 |
| Model v1 + qc_bloc (mean of logits)                     |    0.434 | 0.760 [0.713, 0.801] |                0.789 |                0.821 |
| Model v1 + qc_bloc_meta (mean of logits)                |    0.434 | 0.757 [0.706, 0.800] |                0.786 |                0.813 |

## Per term

| method                                         |   term |   cases |   votes |   petitioner win % |   macro_f1 |   auc |   vote_acc |   case_acc |
|:-----------------------------------------------|-------:|--------:|--------:|-------------------:|-----------:|------:|-----------:|-----------:|
| Case profile only, Katz-style forest (katz_rf) |   2012 |      73 |     651 |               67.1 |      0.601 | 0.608 |      0.633 |      0.562 |
| Case profile only, Katz-style forest (katz_rf) |   2013 |      66 |     589 |               68.2 |      0.59  | 0.642 |      0.643 |      0.636 |
| Case profile only, Katz-style forest (katz_rf) |   2014 |      53 |     477 |               67.9 |      0.61  | 0.662 |      0.66  |      0.642 |
| Case profile only, Katz-style forest (katz_rf) |   2015 |      64 |     521 |               56.2 |      0.56  | 0.565 |      0.576 |      0.562 |
| Case profile only, Katz-style forest (katz_rf) |   2016 |      62 |     505 |               75.8 |      0.624 | 0.7   |      0.711 |      0.742 |
| Case profile only, Katz-style forest (katz_rf) |   2017 |      61 |     544 |               70.5 |      0.62  | 0.72  |      0.664 |      0.639 |
| + named colleagues, LR (qc_bloc)               |   2012 |      73 |     651 |               67.1 |      0.675 | 0.745 |      0.682 |      0.603 |
| + named colleagues, LR (qc_bloc)               |   2013 |      66 |     589 |               68.2 |      0.645 | 0.678 |      0.664 |      0.697 |
| + named colleagues, LR (qc_bloc)               |   2014 |      53 |     477 |               67.9 |      0.68  | 0.782 |      0.688 |      0.623 |
| + named colleagues, LR (qc_bloc)               |   2015 |      64 |     521 |               56.2 |      0.641 | 0.691 |      0.641 |      0.641 |
| + named colleagues, LR (qc_bloc)               |   2016 |      62 |     505 |               75.8 |      0.701 | 0.79  |      0.733 |      0.742 |
| + named colleagues, LR (qc_bloc)               |   2017 |      61 |     544 |               70.5 |      0.758 | 0.841 |      0.763 |      0.689 |
| Model v1 (5 seeds)                             |   2012 |      73 |     651 |               67.1 |      0.672 | 0.76  |      0.676 |      0.644 |
| Model v1 (5 seeds)                             |   2013 |      66 |     589 |               68.2 |      0.639 | 0.705 |      0.649 |      0.606 |
| Model v1 (5 seeds)                             |   2014 |      53 |     477 |               67.9 |      0.698 | 0.776 |      0.706 |      0.642 |
| Model v1 (5 seeds)                             |   2015 |      64 |     521 |               56.2 |      0.653 | 0.724 |      0.655 |      0.625 |
| Model v1 (5 seeds)                             |   2016 |      62 |     505 |               75.8 |      0.728 | 0.838 |      0.762 |      0.79  |
| Model v1 (5 seeds)                             |   2017 |      61 |     544 |               70.5 |      0.761 | 0.809 |      0.768 |      0.82  |
| Model v1 + metadata prior (fusion, 5 seeds)    |   2012 |      73 |     651 |               67.1 |      0.69  | 0.783 |      0.691 |      0.63  |
| Model v1 + metadata prior (fusion, 5 seeds)    |   2013 |      66 |     589 |               68.2 |      0.629 | 0.719 |      0.635 |      0.576 |
| Model v1 + metadata prior (fusion, 5 seeds)    |   2014 |      53 |     477 |               67.9 |      0.648 | 0.765 |      0.65  |      0.585 |
| Model v1 + metadata prior (fusion, 5 seeds)    |   2015 |      64 |     521 |               56.2 |      0.652 | 0.731 |      0.656 |      0.672 |
| Model v1 + metadata prior (fusion, 5 seeds)    |   2016 |      62 |     505 |               75.8 |      0.743 | 0.846 |      0.774 |      0.774 |
| Model v1 + metadata prior (fusion, 5 seeds)    |   2017 |      61 |     544 |               70.5 |      0.762 | 0.803 |      0.768 |      0.77  |
| Model v1 + qc_bloc (mean of logits)            |   2012 |      73 |     651 |               67.1 |      0.68  | 0.77  |      0.687 |      0.589 |
| Model v1 + qc_bloc (mean of logits)            |   2013 |      66 |     589 |               68.2 |      0.623 | 0.704 |      0.635 |      0.621 |
| Model v1 + qc_bloc (mean of logits)            |   2014 |      53 |     477 |               67.9 |      0.716 | 0.796 |      0.723 |      0.642 |
| Model v1 + qc_bloc (mean of logits)            |   2015 |      64 |     521 |               56.2 |      0.661 | 0.729 |      0.662 |      0.672 |
| Model v1 + qc_bloc (mean of logits)            |   2016 |      62 |     505 |               75.8 |      0.734 | 0.833 |      0.766 |      0.79  |
| Model v1 + qc_bloc (mean of logits)            |   2017 |      61 |     544 |               70.5 |      0.754 | 0.839 |      0.761 |      0.82  |
| Model v1 + qc_bloc_meta (mean of logits)       |   2012 |      73 |     651 |               67.1 |      0.7   | 0.789 |      0.707 |      0.671 |
| Model v1 + qc_bloc_meta (mean of logits)       |   2013 |      66 |     589 |               68.2 |      0.637 | 0.722 |      0.65  |      0.621 |
| Model v1 + qc_bloc_meta (mean of logits)       |   2014 |      53 |     477 |               67.9 |      0.71  | 0.801 |      0.717 |      0.679 |
| Model v1 + qc_bloc_meta (mean of logits)       |   2015 |      64 |     521 |               56.2 |      0.671 | 0.734 |      0.672 |      0.688 |
| Model v1 + qc_bloc_meta (mean of logits)       |   2016 |      62 |     505 |               75.8 |      0.719 | 0.832 |      0.752 |      0.774 |
| Model v1 + qc_bloc_meta (mean of logits)       |   2017 |      61 |     544 |               70.5 |      0.775 | 0.842 |      0.781 |      0.803 |
