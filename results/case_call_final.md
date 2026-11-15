# Calling the winner of a case, test terms 2018–2025

Walk-forward predictions of the test run (481 cases, 4,298 votes; every method refit before each term). The test run was made once; this table only reads its saved predictions. π = pro-petitioner share of the votes of all earlier terms (0.599–0.601). Intervals: 95%, bootstrap over cases, paired, 2,000 resamples.

## 1. Against always-petitioner (standard rule, cutoff 0.5)

| Method                                         |   Vote acc 2018–19 |   Case acc 2018–19 |   Vote acc 2020–25 |   Case acc 2020–25 |   Vote acc 2018–2025 |   Case acc 2018–2025 |
|:-----------------------------------------------|-------------------:|-------------------:|-------------------:|-------------------:|---------------------:|---------------------:|
| Always petitioner                              |               57.2 |               59.7 |               63   |               70.3 |                 61.5 |                 67.6 |
| Question count                                 |               63.5 |               61.3 |               59.5 |               52.7 |                 60.5 |                 54.9 |
| Case profile only, Katz-style forest (katz_rf) |               62.3 |               56.5 |               67.1 |               66.4 |                 65.9 |                 63.8 |
| + named colleagues, LR (qc_bloc)               |               69.1 |               64.5 |               67.8 |               63   |                 68.1 |                 63.4 |
| Model v1                                       |               72   |               66.1 |               70.3 |               66.4 |                 70.8 |                 66.3 |
| Model v1 + qc_bloc (mean of logits)            |               72.4 |               63.7 |               72   |               66.1 |                 72.1 |                 65.5 |
| Model v1 + qc_bloc_meta (mean of logits)       |               72.7 |               66.1 |               73   |               68.1 |                 72.9 |                 67.6 |

- Model v1 − always petitioner, vote accuracy, 2018–2025: +0.092 [+0.049, +0.135].
- Model v1 − always petitioner, case accuracy, 2018–19 (124 cases): +0.065 [-0.040, +0.169].
- Model v1 − always petitioner, case accuracy, 2020–25 (357 cases): -0.039 [-0.109, +0.031].
- Model v1 − always petitioner, case accuracy, 2018–2025 (481 cases): -0.012 [-0.075, +0.048].

|   Term |   Cases |   Vote acc, v1 |   Vote acc, always petitioner |   Case acc, v1 |   Case acc, always petitioner |
|-------:|--------:|---------------:|------------------------------:|---------------:|------------------------------:|
|   2018 |      69 |           73.4 |                          56.4 |           62.3 |                          58   |
|   2019 |      55 |           70.3 |                          58.1 |           70.9 |                          61.8 |
|   2020 |      57 |           65.3 |                          64.7 |           64.9 |                          75.4 |
|   2021 |      62 |           65.1 |                          65.3 |           53.2 |                          75.8 |
|   2022 |      59 |           66.4 |                          59.4 |           64.4 |                          61   |
|   2023 |      61 |           73.1 |                          68.4 |           67.2 |                          73.8 |
|   2024 |      60 |           76.5 |                          62.8 |           78.3 |                          70   |
|   2025 |      58 |           75.2 |                          57.4 |           70.7 |                          65.5 |

## 2. Does the model know which side wins?

Cases sorted by the mean probability over the justices and cut in three equal groups: the share of cases the petitioner wins in each group. Always-petitioner gives every case the same answer (67.6% of these cases are won by the petitioner).

| Method                                         |   Case-level AUC |   Lowest third: petitioner wins % |   Middle third: petitioner wins % |   Top third: petitioner wins % |   Calls respondent, % of cases |   Respondent calls that are right % |   Petitioner calls that are right % |
|:-----------------------------------------------|-----------------:|----------------------------------:|----------------------------------:|-------------------------------:|-------------------------------:|------------------------------------:|------------------------------------:|
| Case profile only, Katz-style forest (katz_rf) |            0.548 |                              62.1 |                              67.5 |                           73.1 |                           15.4 |                                37.8 |                                68.6 |
| + named colleagues, LR (qc_bloc)               |            0.692 |                              52.8 |                              67.5 |                           82.5 |                           49.1 |                                45.8 |                                80.4 |
| Model v1                                       |            0.727 |                              46.6 |                              71.9 |                           84.4 |                           44.9 |                                48.6 |                                80.8 |
| Model v1 + qc_bloc (mean of logits)            |            0.731 |                              46   |                              73.1 |                           83.8 |                           45.3 |                                47.7 |                                80.2 |
| Model v1 + qc_bloc_meta (mean of logits)       |            0.737 |                              43.5 |                              73.8 |                           85.6 |                           42.8 |                                50   |                                80.7 |

| Model v1, terms   |   Cases |   Case-level AUC |   Lowest third: petitioner wins % |   Middle third: petitioner wins % |   Top third: petitioner wins % |
|:------------------|--------:|-----------------:|----------------------------------:|----------------------------------:|-------------------------------:|
| 2018–19           |     124 |            0.742 |                              35.7 |                              63.4 |                           80.5 |
| 2020–25           |     357 |            0.741 |                              49.6 |                              72.3 |                           89.1 |
| 2018–2025         |     481 |            0.727 |                              46.6 |                              71.9 |                           84.4 |

## 3. Rules for calling the winner

C1 and C2 have no fitted number: they put the base rate back that the class-balanced loss takes out. A and B are fitted on the earlier scored terms, so they start in 2019. Case accuracy, %.

| Case profile only, Katz-style forest (katz_rf)                       | 2018–19   |   2020–25 | 2018–2025   |   2019–2025 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 56.5      |      66.4 | 63.8        |        65.8 |                 84.5 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 59.7      |      71.7 | 68.6        |        70.4 |                 96.8 |
| C2: mean probability with the class balance undone, above 0.5        | 59.7      |      71.7 | 68.6        |        70.4 |                 96.8 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      72.3 | –           |        70.9 |                 95.4 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      70.9 | –           |        68.4 |                 95.4 |
| Always petitioner                                                    | 59.7      |      70.3 | 67.6        |        69.2 |                100   |

- 2018–2025: C1 − standard +0.048 [+0.017, +0.079]; C1 − always petitioner +0.010 [-0.004, +0.025]; C2 − standard +0.048 [+0.017, +0.079]; C2 − always petitioner +0.010 [-0.004, +0.025].
- 2019–2025: A − always petitioner +0.017 [-0.005, +0.039]; B − always petitioner -0.007 [-0.029, +0.012]; B − standard +0.027 [-0.012, +0.066].

| + named colleagues, LR (qc_bloc)                                     | 2018–19   |   2020–25 | 2018–2025   |   2019–2025 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 64.5      |      63   | 63.4        |        63.3 |                 49   |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 64.5      |      72   | 70.1        |        70.4 |                 76.9 |
| C2: mean probability with the class balance undone, above 0.5        | 65.3      |      71.7 | 70.1        |        69.9 |                 79.9 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      70   | –           |        68.4 |                 78.4 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      72.3 | –           |        71.6 |                 76.7 |
| Always petitioner                                                    | 59.7      |      70.3 | 67.6        |        69.2 |                100   |

- 2018–2025: C1 − standard +0.067 [+0.021, +0.110]; C1 − always petitioner +0.025 [-0.017, +0.069]; C2 − standard +0.067 [+0.017, +0.112]; C2 − always petitioner +0.025 [-0.019, +0.064].
- 2019–2025: A − always petitioner -0.007 [-0.051, +0.034]; B − always petitioner +0.024 [-0.022, +0.068]; B − standard +0.083 [+0.032, +0.133].

| Model v1                                                             | 2018–19   |   2020–25 | 2018–2025   |   2019–2025 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 66.1      |      66.4 | 66.3        |        67   |                 54.1 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 71.0      |      71.1 | 71.1        |        71.1 |                 65   |
| C2: mean probability with the class balance undone, above 0.5        | 71.0      |      70.6 | 70.7        |        70.9 |                 66.7 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      70.6 | –           |        70.6 |                 76.7 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      72   | –           |        72.1 |                 75.7 |
| Always petitioner                                                    | 59.7      |      70.3 | 67.6        |        69.2 |                100   |

- 2018–2025: C1 − standard +0.048 [+0.019, +0.077]; C1 − always petitioner +0.035 [-0.017, +0.087]; C2 − standard +0.044 [+0.012, +0.075]; C2 − always petitioner +0.031 [-0.019, +0.083].
- 2019–2025: A − always petitioner +0.015 [-0.029, +0.058]; B − always petitioner +0.029 [-0.017, +0.075]; B − standard +0.051 [+0.007, +0.097].

| Model v1 + qc_bloc (mean of logits)                                  | 2018–19   |   2020–25 | 2018–2025   |   2019–2025 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 63.7      |      66.1 | 65.5        |        66   |                 53.2 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 70.2      |      72.3 | 71.7        |        72.1 |                 70.9 |
| C2: mean probability with the class balance undone, above 0.5        | 69.4      |      71.7 | 71.1        |        71.6 |                 72.8 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      70.6 | –           |        71.1 |                 74.8 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      72.5 | –           |        72.3 |                 73.1 |
| Always petitioner                                                    | 59.7      |      70.3 | 67.6        |        69.2 |                100   |

- 2018–2025: C1 − standard +0.062 [+0.027, +0.102]; C1 − always petitioner +0.042 [-0.006, +0.089]; C2 − standard +0.056 [+0.017, +0.096]; C2 − always petitioner +0.035 [-0.012, +0.083].
- 2019–2025: A − always petitioner +0.019 [-0.027, +0.066]; B − always petitioner +0.032 [-0.019, +0.085]; B − standard +0.063 [+0.019, +0.107].

| Model v1 + qc_bloc_meta (mean of logits)                             | 2018–19   |   2020–25 | 2018–2025   |   2019–2025 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 66.1      |      68.1 | 67.6        |        68.2 |                 55.8 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 70.2      |      73.9 | 73.0        |        73.5 |                 73.3 |
| C2: mean probability with the class balance undone, above 0.5        | 71.0      |      72.8 | 72.3        |        73.1 |                 74.8 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      72.5 | –           |        72.6 |                 80.6 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      72.5 | –           |        72.3 |                 74.5 |
| Always petitioner                                                    | 59.7      |      70.3 | 67.6        |        69.2 |                100   |

- 2018–2025: C1 − standard +0.054 [+0.017, +0.089]; C1 − always petitioner +0.054 [+0.006, +0.100]; C2 − standard +0.048 [+0.010, +0.087]; C2 − always petitioner +0.048 [+0.000, +0.091].
- 2019–2025: A − always petitioner +0.034 [-0.007, +0.075]; B − always petitioner +0.032 [-0.017, +0.083]; B − standard +0.041 [+0.000, +0.085].

|   Model v1, term |   Cases |   Standard |   C1 |   C2 |   Always petitioner |
|-----------------:|--------:|-----------:|-----:|-----:|--------------------:|
|             2018 |      69 |       62.3 | 71   | 69.6 |                58   |
|             2019 |      55 |       70.9 | 70.9 | 72.7 |                61.8 |
|             2020 |      57 |       64.9 | 78.9 | 80.7 |                75.4 |
|             2021 |      62 |       53.2 | 58.1 | 54.8 |                75.8 |
|             2022 |      59 |       64.4 | 61   | 61   |                61   |
|             2023 |      61 |       67.2 | 73.8 | 73.8 |                73.8 |
|             2024 |      60 |       78.3 | 80   | 78.3 |                70   |
|             2025 |      58 |       70.7 | 75.9 | 75.9 |                65.5 |

Votes of model v1 with the class balance undone (p > 1 − π), 2018–2025: vote accuracy 73.4% (at 0.5: 70.8%), macro-F1 0.717 (at 0.5: 0.702), predicted petitioner 63.1% (at 0.5: 52.4%; true 61.5%).
