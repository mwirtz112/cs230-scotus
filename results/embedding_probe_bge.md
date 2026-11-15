# embedding_probe_bge

Embeddings: `embeddings/turns_bge.npy`; logistic regression C=1.0; cutoff = train pro-petitioner rate 0.599.

| method              | split   | macro_f1             | auc                  | vote_acc             | case_acc             |   pred petitioner % |
|:--------------------|:--------|:---------------------|:---------------------|:---------------------|:---------------------|--------------------:|
| embedding_probe_bge | train   | 0.569 [0.564, 0.574] | 0.656 [0.651, 0.661] | 0.570 [0.565, 0.575] | 0.537 [0.527, 0.549] |                35.6 |
| embedding_probe_bge | val     | 0.595 [0.559, 0.630] | 0.680 [0.641, 0.719] | 0.599 [0.567, 0.633] | 0.480 [0.390, 0.561] |                43.1 |

## Val, per term

|   term |   cases |   votes |   petitioner win % |   macro_f1 |   auc |   vote_acc |   case_acc |
|-------:|--------:|--------:|-------------------:|-----------:|------:|-----------:|-----------:|
|   2016 |      62 |     505 |               75.8 |      0.565 | 0.679 |      0.574 |      0.387 |
|   2017 |      61 |     544 |               70.5 |      0.621 | 0.686 |      0.621 |      0.574 |
