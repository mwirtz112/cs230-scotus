# embedding_probe

Embeddings: `embeddings/turns_last.npy`; logistic regression C=1.0; cutoff = train pro-petitioner rate 0.599.

| method          | split   | macro_f1             | auc                  | vote_acc             | case_acc             |   pred petitioner % |
|:----------------|:--------|:---------------------|:---------------------|:---------------------|:---------------------|--------------------:|
| embedding_probe | train   | 0.568 [0.563, 0.573] | 0.655 [0.650, 0.660] | 0.569 [0.564, 0.574] | 0.535 [0.526, 0.545] |                35.4 |
| embedding_probe | val     | 0.595 [0.559, 0.629] | 0.667 [0.631, 0.705] | 0.600 [0.566, 0.632] | 0.528 [0.439, 0.618] |                44.7 |

## Val, per term

|   term |   cases |   votes |   petitioner win % |   macro_f1 |   auc |   vote_acc |   case_acc |
|-------:|--------:|--------:|-------------------:|-----------:|------:|-----------:|-----------:|
|   2016 |      62 |     505 |               75.8 |      0.592 | 0.689 |      0.606 |      0.548 |
|   2017 |      61 |     544 |               70.5 |      0.593 | 0.651 |      0.594 |      0.508 |
