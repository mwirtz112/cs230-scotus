# Flat ModernBERT baseline (frozen encoder)

The transcript as one text (one line per turn: a justice by surname, an advocate by side) through ModernBERT-base, one vector per case (flat_encode.py); then a one-hidden-layer head with a small vector and offset per justice (flat_baseline.py). flat_mb = the first 8,192 tokens only, as proposed; flat_mb_all = the whole transcript, read in windows of 8,192 tokens whose vectors are averaged. Trained on terms from 1980, class-balanced, cutoff 0.5, five seeds averaged. Settings were chosen on the dev fold by AUC; validation was scored once with them. Test untouched.

## What the first window of 8,192 tokens covers

| period             |   cases | longer_than_window   |   median_tokens_in_transcript | median_share_kept   |   median_turns_seen |   median_turns | respondent_inside_window   |
|:-------------------|--------:|:---------------------|------------------------------:|:--------------------|--------------------:|---------------:|:---------------------------|
| train 1980-2015    |    3386 | 96.2%                |                       12354.5 | 66%                 |                 150 |            223 | 93.7%                      |
| validation 2016-17 |     123 | 100.0%               |                       13638   | 60%                 |                 138 |            225 | 93.5%                      |

## Dev-fold grid (3 seeds per setting; per method, the top row is the one used)

| method      | pool    |   hidden |   dropout |   epochs |   dev AUC |   dev macro-F1 |   pred petitioner % |
|:------------|:--------|---------:|----------:|---------:|----------:|---------------:|--------------------:|
| flat_mb     | cls     |      256 |       0.2 |       10 |    0.605  |         0.5717 |                51.3 |
| flat_mb     | cls     |      256 |       0.5 |       10 |    0.5987 |         0.569  |                50.9 |
| flat_mb     | cls     |       64 |       0.5 |       10 |    0.5984 |         0.5629 |                57.6 |
| flat_mb     | cls     |       64 |       0.2 |       10 |    0.5981 |         0.5713 |                55.8 |
| flat_mb     | cls     |       64 |       0.2 |       30 |    0.5881 |         0.5766 |                61.5 |
| flat_mb     | cls     |      256 |       0.2 |       30 |    0.5851 |         0.557  |                63.1 |
| flat_mb     | mean    |      256 |       0.2 |       30 |    0.5744 |         0.5491 |                60.7 |
| flat_mb     | mean    |      256 |       0.5 |       10 |    0.5722 |         0.5568 |                59.4 |
| flat_mb     | cls     |      256 |       0.5 |       30 |    0.5714 |         0.5493 |                60.4 |
| flat_mb     | mean    |      256 |       0.2 |       10 |    0.5684 |         0.5464 |                64.3 |
| flat_mb     | cls     |       64 |       0.5 |       30 |    0.5676 |         0.5519 |                56.4 |
| flat_mb     | mean    |      256 |       0.5 |       30 |    0.5659 |         0.5471 |                61.8 |
| flat_mb     | mean    |       64 |       0.2 |       30 |    0.5621 |         0.5483 |                58.7 |
| flat_mb     | mean    |       64 |       0.2 |       10 |    0.5618 |         0.5479 |                67.7 |
| flat_mb     | mean    |       64 |       0.5 |       10 |    0.5575 |         0.5466 |                58.6 |
| flat_mb     | mean    |       64 |       0.5 |       30 |    0.5486 |         0.5392 |                59.1 |
| flat_mb_all | clsall  |       64 |       0.5 |       10 |    0.6008 |         0.5766 |                58.5 |
| flat_mb_all | clsall  |       64 |       0.2 |       10 |    0.5983 |         0.5791 |                58.9 |
| flat_mb_all | clsall  |      256 |       0.2 |       10 |    0.5932 |         0.5645 |                58.8 |
| flat_mb_all | clsall  |      256 |       0.5 |       10 |    0.5928 |         0.557  |                48.8 |
| flat_mb_all | clsall  |       64 |       0.2 |       30 |    0.5727 |         0.5521 |                62.7 |
| flat_mb_all | clsall  |       64 |       0.5 |       30 |    0.5722 |         0.5513 |                63.9 |
| flat_mb_all | clsall  |      256 |       0.2 |       30 |    0.5678 |         0.5423 |                64.4 |
| flat_mb_all | clsall  |      256 |       0.5 |       30 |    0.5486 |         0.5318 |                66.7 |
| flat_mb_all | meanall |       64 |       0.5 |       10 |    0.5449 |         0.5359 |                60.5 |
| flat_mb_all | meanall |      256 |       0.5 |       10 |    0.5435 |         0.5315 |                63.8 |
| flat_mb_all | meanall |      256 |       0.5 |       30 |    0.542  |         0.5396 |                62.6 |
| flat_mb_all | meanall |      256 |       0.2 |       10 |    0.5401 |         0.5194 |                61.5 |
| flat_mb_all | meanall |      256 |       0.2 |       30 |    0.5393 |         0.5258 |                54.9 |
| flat_mb_all | meanall |       64 |       0.5 |       30 |    0.5356 |         0.5213 |                59.8 |
| flat_mb_all | meanall |       64 |       0.2 |       10 |    0.5303 |         0.5089 |                65.8 |
| flat_mb_all | meanall |       64 |       0.2 |       30 |    0.5241 |         0.5156 |                55   |

## Results

Settings used: flat_mb: {'pool': 'cls', 'hidden': 256, 'dropout': 0.2, 'epochs': 10}; flat_mb_all: {'pool': 'clsall', 'hidden': 64, 'dropout': 0.5, 'epochs': 10}

| method      | fold                                  | macro_f1             | auc                  | vote_acc             | case_acc             |   pred petitioner % |
|:------------|:--------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|--------------------:|
| flat_mb     | dev 2012-15 (fit on 1980-2011)        | 0.587 [0.552, 0.620] | 0.603 [0.555, 0.648] | 0.610 [0.575, 0.641] | 0.617 [0.559, 0.676] |                62   |
| flat_mb     | validation 2016-17 (fit on 1980-2015) | 0.580 [0.524, 0.632] | 0.617 [0.542, 0.688] | 0.596 [0.540, 0.651] | 0.585 [0.496, 0.675] |                53.7 |
| flat_mb_all | dev 2012-15 (fit on 1980-2011)        | 0.573 [0.540, 0.607] | 0.608 [0.563, 0.652] | 0.595 [0.563, 0.630] | 0.625 [0.570, 0.684] |                61.2 |
| flat_mb_all | validation 2016-17 (fit on 1980-2015) | 0.607 [0.545, 0.662] | 0.633 [0.562, 0.701] | 0.636 [0.576, 0.689] | 0.626 [0.537, 0.707] |                61.1 |

## Per term

|   term |   cases |   votes |   petitioner win % |   macro_f1 |   auc |   vote_acc |   case_acc | method      |
|-------:|--------:|--------:|-------------------:|-----------:|------:|-----------:|-----------:|:------------|
|   2012 |      73 |     651 |               67.1 |      0.578 | 0.587 |      0.593 |      0.589 | flat_mb     |
|   2013 |      66 |     589 |               68.2 |      0.614 | 0.618 |      0.65  |      0.697 | flat_mb     |
|   2014 |      53 |     477 |               67.9 |      0.548 | 0.568 |      0.581 |      0.491 | flat_mb     |
|   2015 |      64 |     521 |               56.2 |      0.601 | 0.621 |      0.612 |      0.672 | flat_mb     |
|   2016 |      62 |     505 |               75.8 |      0.635 | 0.714 |      0.671 |      0.661 | flat_mb     |
|   2017 |      61 |     544 |               70.5 |      0.522 | 0.528 |      0.526 |      0.508 | flat_mb     |
|   2012 |      73 |     651 |               67.1 |      0.593 | 0.639 |      0.608 |      0.658 | flat_mb_all |
|   2013 |      66 |     589 |               68.2 |      0.551 | 0.574 |      0.594 |      0.667 | flat_mb_all |
|   2014 |      53 |     477 |               67.9 |      0.562 | 0.58  |      0.591 |      0.528 | flat_mb_all |
|   2015 |      64 |     521 |               56.2 |      0.575 | 0.629 |      0.583 |      0.625 | flat_mb_all |
|   2016 |      62 |     505 |               75.8 |      0.637 | 0.681 |      0.693 |      0.694 | flat_mb_all |
|   2017 |      61 |     544 |               70.5 |      0.572 | 0.579 |      0.583 |      0.557 | flat_mb_all |
