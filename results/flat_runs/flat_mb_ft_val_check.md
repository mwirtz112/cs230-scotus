# Checks of the fine-tuned flat baseline: `flat_mb_ft_val`

Written by check_flat_finetune.py. Run: top 4 layers of ModernBERT-base and the head, 3 epochs, encoder rate 5e-06, head rate 0.0001, 8 cases per update, seed 0; 3,386 training cases, 123 held out; 6.3 hours. It starts from one seed of `flat_mb` (the head trained on the frozen vectors).

## 1. Did it train?

Reference losses on the training cases: 0.5713 for a constant p = 0.5 (knows nothing); the stage-1 head on the frozen vectors 0.4174 with dropout as in training (0.3906 with dropout off). Control: stage 2 repeated with the encoder frozen (same head, schedule, case order and identity dropout).

Training loss by epoch (mean over cases, dropout on):

|   epoch |   run (top layers + head) |   control (encoder frozen) |   run - control |
|--------:|--------------------------:|---------------------------:|----------------:|
|       1 |                    0.4245 |                     0.4187 |          0.0058 |
|       2 |                    0.4035 |                     0.4    |          0.0035 |
|       3 |                    0.3858 |                     0.3924 |         -0.0065 |

**Passed**: the last epoch's training loss is nearer to the frozen baseline's than to the know-nothing level.

## 2. How far did the weights move?

Relative change ||w - w0|| / ||w0|| of the unfrozen encoder weights against the pretrained model:

| part       |   values |   relative change |
|:-----------|---------:|------------------:|
| layer 18   |  5015040 |           0.00079 |
| layer 19   |  5015040 |           0.00073 |
| layer 20   |  5015040 |           0.00071 |
| layer 21   |  5015040 |           0.00074 |
| final_norm |      768 |           5e-05   |

All unfrozen encoder weights: 0.074% (mean absolute change 6.4e-05 against a mean absolute weight of 8.6e-02). The head, against the stage-1 head: 6.2%.

## 3. The held-out predictions (no labels used)

Before fine-tuning the live model reproduces the head on the cached vectors: mean logit difference 0.0131, largest 0.2960, over 1,049 votes.

| model   |   epoch |   logit sd |   variance between justices |   variance between cases |   predicted petitioner % |   correlation with epoch 0 |   sd of the change |
|:--------|--------:|-----------:|----------------------------:|-------------------------:|-------------------------:|---------------------------:|-------------------:|
| run     |       0 |      1.196 |                       0.128 |                    0.627 |                   45.472 |                    nan     |            nan     |
| run     |       1 |      1.27  |                       0.081 |                    0.648 |                   69.495 |                      0.947 |              0.409 |
| run     |       2 |      1.248 |                       0.086 |                    0.634 |                   53.956 |                      0.929 |              0.462 |
| run     |       3 |      1.24  |                       0.066 |                    0.641 |                   54.242 |                      0.915 |              0.504 |
| control |       0 |      1.198 |                       0.129 |                    0.627 |                   45.091 |                    nan     |            nan     |
| control |       1 |      1.182 |                       0.079 |                    0.651 |                   58.532 |                      0.953 |              0.365 |
| control |       2 |      1.157 |                       0.06  |                    0.649 |                   52.622 |                      0.922 |              0.467 |
| control |       3 |      1.162 |                       0.063 |                    0.647 |                   55.291 |                      0.927 |              0.451 |

The encoder's change alone (the run's final head on the fine-tuned encoder against the same head on the frozen vectors): logit correlation 0.990, sd of the difference 0.201 (sd of the logit 1.239), 58 of 1,049 votes (5.5%) change side at cutoff 0.5.

## 4. Scores (val fold, terms 2016-2017: 123 cases, 1,049 votes)

Cutoff 0.5; 95% bootstrap CI over cases. The rows after epochs before the last are a diagnostic, never a way to choose an epoch.

| method                                                     | macro-F1             | AUC                  |   vote acc |   case acc |   predicted petitioner % |   case-level AUC |   within-case AUC |
|:-----------------------------------------------------------|:---------------------|:---------------------|-----------:|-----------:|-------------------------:|-----------------:|------------------:|
| flat_mb_ft (the row: after 3 epochs)                       | 0.575 [0.518, 0.629] | 0.624 [0.549, 0.694] |      0.592 |      0.585 |                       54 |            0.626 |             0.631 |
| its start: the stage-1 head, one seed of flat_mb (epoch 0) | 0.583 [0.527, 0.635] | 0.643 [0.565, 0.711] |      0.588 |      0.553 |                       45 |            0.646 |             0.65  |
| the run after epoch 1 (diagnostic)                         | 0.592 [0.537, 0.648] | 0.629 [0.550, 0.699] |      0.643 |      0.74  |                       69 |            0.632 |             0.653 |
| the run after epoch 2 (diagnostic)                         | 0.582 [0.526, 0.637] | 0.629 [0.553, 0.697] |      0.599 |      0.593 |                       54 |            0.625 |             0.636 |
| control: encoder frozen, same schedule, after 3 epochs     | 0.594 [0.536, 0.650] | 0.628 [0.554, 0.696] |      0.612 |      0.593 |                       55 |            0.621 |             0.632 |
| the run's final head on the frozen vectors                 | 0.590 [0.532, 0.641] | 0.633 [0.556, 0.700] |      0.602 |      0.585 |                       51 |            0.629 |             0.645 |
| flat_mb (5 seeds)                                          | 0.580 [0.524, 0.632] | 0.617 [0.542, 0.688] |      0.596 |      0.585 |                       54 |            0.617 |             0.639 |
| flat_mb_all (5 seeds)                                      | 0.607 [0.545, 0.662] | 0.633 [0.562, 0.701] |      0.636 |      0.626 |                       61 |            0.608 |             0.688 |
| question_count                                             | 0.626 [0.582, 0.664] | 0.700 [0.656, 0.747] |      0.631 |      0.528 |                       46 |            0.71  |             0.787 |
| qc_bloc                                                    | 0.735 [0.690, 0.774] | 0.818 [0.777, 0.862] |      0.748 |      0.715 |                       57 |            0.743 |             0.906 |
| v1_main (5 seeds)                                          | 0.748 [0.704, 0.791] | 0.821 [0.767, 0.873] |      0.765 |      0.805 |                       61 |            0.8   |             0.905 |

Paired differences (the same 1,000 resamples of cases for both models):

| difference                                                                 | AUC                     | macro-F1                | vote acc                | case acc                |
|:---------------------------------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| fine-tuned - its start (all of stage 2)                                    | -0.018 [-0.047, +0.009] | -0.008 [-0.040, +0.026] | +0.004 [-0.030, +0.041] | +0.033 [-0.033, +0.106] |
| fine-tuned - control (unfreezing the encoder, same schedule)               | -0.004 [-0.016, +0.010] | -0.019 [-0.036, -0.000] | -0.020 [-0.039, -0.001] | -0.008 [-0.057, +0.041] |
| fine-tuned - its final head on frozen vectors (the encoder's change alone) | -0.008 [-0.020, +0.003] | -0.015 [-0.034, +0.004] | -0.010 [-0.028, +0.009] | +0.000 [-0.033, +0.033] |
| control - its start (three more epochs of the head alone)                  | -0.015 [-0.039, +0.011] | +0.015 [-0.016, +0.047] | +0.028 [-0.005, +0.063] | +0.041 [-0.024, +0.114] |
| fine-tuned - flat_mb (5 seeds)                                             | +0.007 [-0.019, +0.032] | -0.005 [-0.035, +0.025] | -0.004 [-0.035, +0.027] | +0.000 [-0.065, +0.065] |
| fine-tuned - question_count                                                | -0.076 [-0.160, +0.007] | -0.051 [-0.110, +0.014] | -0.039 [-0.102, +0.026] | +0.057 [-0.065, +0.179] |
| v1_main - fine-tuned                                                       | +0.197 [+0.127, +0.273] | +0.173 [+0.114, +0.235] | +0.173 [+0.114, +0.236] | +0.220 [+0.114, +0.333] |

Per term:

| method                                                     |   term |   cases |   votes |   macro_f1 |   auc |   vote_acc |   case_acc |
|:-----------------------------------------------------------|-------:|--------:|--------:|-----------:|------:|-----------:|-----------:|
| flat_mb_ft (the row: after 3 epochs)                       |   2016 |      62 |     505 |      0.614 | 0.722 |      0.648 |      0.629 |
| flat_mb_ft (the row: after 3 epochs)                       |   2017 |      61 |     544 |      0.534 | 0.541 |      0.54  |      0.541 |
| its start: the stage-1 head, one seed of flat_mb (epoch 0) |   2016 |      62 |     505 |      0.653 | 0.748 |      0.675 |      0.661 |
| its start: the stage-1 head, one seed of flat_mb (epoch 0) |   2017 |      61 |     544 |      0.507 | 0.546 |      0.507 |      0.443 |
| control: encoder frozen, same schedule, after 3 epochs     |   2016 |      62 |     505 |      0.641 | 0.719 |      0.679 |      0.694 |
| control: encoder frozen, same schedule, after 3 epochs     |   2017 |      61 |     544 |      0.545 | 0.546 |      0.55  |      0.492 |
