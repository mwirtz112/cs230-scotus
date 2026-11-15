# Baselines

Point estimate [95% bootstrap CI over cases]. Test is locked.

| method            | split   | macro_f1             | auc                  | vote_acc             | case_acc             |
|:------------------|:--------|:---------------------|:---------------------|:---------------------|:---------------------|
| always_petitioner | train   | 0.375 [0.371, 0.377] | 0.500 [0.500, 0.500] | 0.599 [0.591, 0.605] | 0.641 [0.631, 0.651] |
| justice_rate      | train   | 0.529 [0.524, 0.533] | 0.555 [0.549, 0.560] | 0.532 [0.528, 0.535] | 0.483 [0.473, 0.492] |
| question_count    | train   | 0.561 [0.555, 0.567] | 0.582 [0.575, 0.590] | 0.563 [0.557, 0.570] | 0.564 [0.551, 0.574] |
| always_petitioner | val     | 0.397 [0.375, 0.419] | 0.500 [0.500, 0.500] | 0.659 [0.599, 0.721] | 0.732 [0.659, 0.805] |
| justice_rate      | val     | 0.521 [0.503, 0.539] | 0.535 [0.514, 0.557] | 0.607 [0.574, 0.641] | 0.732 [0.659, 0.805] |
| question_count    | val     | 0.626 [0.582, 0.664] | 0.700 [0.656, 0.747] | 0.631 [0.590, 0.668] | 0.528 [0.439, 0.618] |

## Val, per term

| method            |   term |   cases |   votes |   petitioner win % |   macro_f1 |   auc |   vote_acc |   case_acc |
|:------------------|-------:|--------:|--------:|-------------------:|-----------:|------:|-----------:|-----------:|
| always_petitioner |   2016 |      62 |     505 |               75.8 |      0.419 | 0.5   |      0.721 |      0.758 |
| always_petitioner |   2017 |      61 |     544 |               70.5 |      0.375 | 0.5   |      0.601 |      0.705 |
| justice_rate      |   2016 |      62 |     505 |               75.8 |      0.532 | 0.524 |      0.638 |      0.758 |
| justice_rate      |   2017 |      61 |     544 |               70.5 |      0.509 | 0.538 |      0.579 |      0.705 |
| question_count    |   2016 |      62 |     505 |               75.8 |      0.595 | 0.682 |      0.608 |      0.565 |
| question_count    |   2017 |      61 |     544 |               70.5 |      0.651 | 0.722 |      0.653 |      0.492 |

## Notes
- train: question-count source of p: {'justice': 63.9, 'bench': 27.3, 'default': 8.8} (% of votes)
- val: question-count source of p: {'justice': 81.6, 'bench': 17.9, 'default': 0.5} (% of votes)
- justice_rate cutoff (train pro-petitioner vote rate): 0.599
