# Calling the winner of a case, rehearsal terms 2012–2017

Walk-forward predictions of the rehearsal (379 cases, 3,287 votes; every method refit before each term). The test terms (2018–25) were not read. π = pro-petitioner share of the votes of all earlier terms (0.598–0.600). Intervals: 95%, bootstrap over cases, paired, 2,000 resamples.

## 1. Against always-petitioner (standard rule, cutoff 0.5)

| Method                                         |   Vote acc 2012–15 |   Case acc 2012–15 |   Vote acc 2016–17 |   Case acc 2016–17 |   Vote acc 2012–2017 |   Case acc 2012–2017 |
|:-----------------------------------------------|-------------------:|-------------------:|-------------------:|-------------------:|---------------------:|---------------------:|
| Always petitioner                              |               61.4 |               64.8 |               65.9 |               73.2 |                 62.8 |                 67.5 |
| Question count                                 |               60.6 |               60.5 |               63.1 |               52.8 |                 61.4 |                 58   |
| Case profile only, Katz-style forest (katz_rf) |               62.9 |               59   |               67.1 |               65   |                 64.3 |                 60.9 |
| + named colleagues, LR (qc_bloc)               |               66.7 |               61.7 |               74.5 |               71.5 |                 69.2 |                 64.9 |
| Model v1                                       |               66.9 |               61.7 |               75.6 |               80.5 |                 69.7 |                 67.8 |
| Model v1 + qc_bloc (mean of logits)            |               67.4 |               61.7 |               76.9 |               78.9 |                 70.5 |                 67.3 |
| Model v1 + qc_bloc_meta (mean of logits)       |               68.8 |               65.2 |               75.9 |               77.2 |                 71.1 |                 69.1 |

- Model v1 − always petitioner, vote accuracy, 2012–2017: +0.069 [+0.025, +0.116].
- Model v1 − always petitioner, case accuracy, 2012–15 (256 cases): -0.031 [-0.117, +0.055].
- Model v1 − always petitioner, case accuracy, 2016–17 (123 cases): +0.073 [-0.024, +0.171].
- Model v1 − always petitioner, case accuracy, 2012–2017 (379 cases): +0.003 [-0.063, +0.069].

|   Term |   Cases |   Vote acc, v1 |   Vote acc, always petitioner |   Case acc, v1 |   Case acc, always petitioner |
|-------:|--------:|---------------:|------------------------------:|---------------:|------------------------------:|
|   2012 |      73 |           67   |                          62.1 |           64.4 |                          67.1 |
|   2013 |      66 |           63.8 |                          64.7 |           59.1 |                          68.2 |
|   2014 |      53 |           69.8 |                          64.2 |           60.4 |                          67.9 |
|   2015 |      64 |           67.8 |                          54.3 |           62.5 |                          56.2 |
|   2016 |      62 |           76.2 |                          72.1 |           79   |                          75.8 |
|   2017 |      61 |           75   |                          60.1 |           82   |                          70.5 |

## 2. Does the model know which side wins?

Cases sorted by the mean probability over the justices and cut in three equal groups: the share of cases the petitioner wins in each group. Always-petitioner gives every case the same answer (67.5% of these cases are won by the petitioner).

| Method                                         |   Case-level AUC |   Lowest third: petitioner wins % |   Middle third: petitioner wins % |   Top third: petitioner wins % |   Calls respondent, % of cases |   Respondent calls that are right % |   Petitioner calls that are right % |
|:-----------------------------------------------|-----------------:|----------------------------------:|----------------------------------:|-------------------------------:|-------------------------------:|------------------------------------:|------------------------------------:|
| Case profile only, Katz-style forest (katz_rf) |            0.547 |                              66.9 |                              61.1 |                           74.6 |                           23   |                                35.6 |                                68.5 |
| + named colleagues, LR (qc_bloc)               |            0.694 |                              49.6 |                              72.2 |                           81   |                           44.3 |                                47   |                                79.1 |
| Model v1                                       |            0.74  |                              44.1 |                              71.4 |                           87.3 |                           44.6 |                                50.3 |                                81.9 |
| Model v1 + qc_bloc (mean of logits)            |            0.739 |                              46.5 |                              69.8 |                           86.5 |                           43.5 |                                49.7 |                                80.8 |
| Model v1 + qc_bloc_meta (mean of logits)       |            0.743 |                              47.2 |                              67.5 |                           88.1 |                           43.8 |                                51.8 |                                82.6 |

| Model v1, terms   |   Cases |   Case-level AUC |   Lowest third: petitioner wins % |   Middle third: petitioner wins % |   Top third: petitioner wins % |
|:------------------|--------:|-----------------:|----------------------------------:|----------------------------------:|-------------------------------:|
| 2012–15           |     256 |            0.701 |                              47.7 |                              63.5 |                           83.5 |
| 2016–17           |     123 |            0.81  |                              43.9 |                              82.9 |                           92.7 |
| 2012–2017         |     379 |            0.74  |                              44.1 |                              71.4 |                           87.3 |

## 3. Rules for calling the winner

C1 and C2 have no fitted number: they put the base rate back that the class-balanced loss takes out. A and B are fitted on the earlier scored terms, so they start in 2013. Case accuracy, %.

| Case profile only, Katz-style forest (katz_rf)                       | 2012–15   |   2016–17 | 2012–2017   |   2017 |   2013–2017 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|-------:|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 59.0      |      65   | 60.9        |   55.7 |        62.1 |                 76.1 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 65.6      |      72.4 | 67.8        |   68.9 |        68.3 |                 97.4 |
| C2: mean probability with the class balance undone, above 0.5        | 65.6      |      70.7 | 67.3        |   65.6 |        67.6 |                 97.4 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      70.7 | –           |   65.6 |        66.7 |                 99   |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      73.2 | –           |   70.5 |        67.6 |                100   |
| Always petitioner                                                    | 64.8      |      73.2 | 67.5        |   70.5 |        67.6 |                100   |

- 2012–2017: C1 − standard +0.069 [+0.021, +0.113]; C1 − always petitioner +0.003 [-0.016, +0.021]; C2 − standard +0.063 [+0.016, +0.108]; C2 − always petitioner -0.003 [-0.018, +0.016].
- 2013–2017: A − always petitioner -0.010 [-0.023, +0.000]; B − always petitioner +0.000 [+0.000, +0.000]; B − standard +0.056 [+0.000, +0.111].

| + named colleagues, LR (qc_bloc)                                     | 2012–15   |   2016–17 | 2012–2017   |   2017 |   2013–2017 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|-------:|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 61.7      |      71.5 | 64.9        |   68.9 |        65.4 |                 56.5 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 64.8      |      80.5 | 69.9        |   78.7 |        70.9 |                 76.5 |
| C2: mean probability with the class balance undone, above 0.5        | 66.4      |      76.4 | 69.7        |   73.8 |        69.9 |                 81.4 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      76.4 | –           |   77   |        69   |                 85.6 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      75.6 | –           |   77   |        69.9 |                 88.6 |
| Always petitioner                                                    | 64.8      |      73.2 | 67.5        |   70.5 |        67.6 |                100   |

- 2012–2017: C1 − standard +0.050 [+0.005, +0.098]; C1 − always petitioner +0.024 [-0.021, +0.074]; C2 − standard +0.047 [-0.003, +0.100]; C2 − always petitioner +0.021 [-0.024, +0.063].
- 2013–2017: A − always petitioner +0.013 [-0.029, +0.056]; B − always petitioner +0.023 [-0.013, +0.062]; B − standard +0.046 [-0.020, +0.111].

| Model v1                                                             | 2012–15   |   2016–17 | 2012–2017   |   2017 |   2013–2017 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|-------:|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 61.7      |      80.5 | 67.8        |   82   |        68.6 |                 56.5 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 64.8      |      82.1 | 70.4        |   80.3 |        72.5 |                 69.6 |
| C2: mean probability with the class balance undone, above 0.5        | 66.0      |      80.5 | 70.7        |   77   |        72.5 |                 70.9 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      78.9 | –           |   77   |        71.6 |                 85.6 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      78   | –           |   75.4 |        71.6 |                 87.6 |
| Always petitioner                                                    | 64.8      |      73.2 | 67.5        |   70.5 |        67.6 |                100   |

- 2012–2017: C1 − standard +0.026 [-0.008, +0.063]; C1 − always petitioner +0.029 [-0.029, +0.084]; C2 − standard +0.029 [-0.011, +0.069]; C2 − always petitioner +0.032 [-0.024, +0.087].
- 2013–2017: A − always petitioner +0.039 [+0.000, +0.082]; B − always petitioner +0.039 [+0.000, +0.078]; B − standard +0.029 [-0.036, +0.092].

| Model v1 + qc_bloc (mean of logits)                                  | 2012–15   |   2016–17 | 2012–2017   |   2017 |   2013–2017 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|-------:|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 61.7      |      78.9 | 67.3        |   78.7 |        69   |                 56.9 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 66.0      |      80.5 | 70.7        |   78.7 |        71.6 |                 71.9 |
| C2: mean probability with the class balance undone, above 0.5        | 69.1      |      79.7 | 72.6        |   77   |        74.5 |                 75.5 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      78   | –           |   77   |        72.5 |                 84.6 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      78.9 | –           |   78.7 |        72.2 |                 87.6 |
| Always petitioner                                                    | 64.8      |      73.2 | 67.5        |   70.5 |        67.6 |                100   |

- 2012–2017: C1 − standard +0.034 [-0.005, +0.074]; C1 − always petitioner +0.032 [-0.024, +0.090]; C2 − standard +0.053 [+0.011, +0.098]; C2 − always petitioner +0.050 [+0.000, +0.103].
- 2013–2017: A − always petitioner +0.049 [+0.007, +0.092]; B − always petitioner +0.046 [+0.007, +0.085]; B − standard +0.033 [-0.029, +0.095].

| Model v1 + qc_bloc_meta (mean of logits)                             | 2012–15   |   2016–17 | 2012–2017   |   2017 |   2013–2017 |   Calls petitioner % |
|:---------------------------------------------------------------------|:----------|----------:|:------------|-------:|------------:|---------------------:|
| Standard: majority of the votes called at 0.5                        | 65.2      |      77.2 | 69.1        |   77   |        69.9 |                 57.2 |
| C1: majority of the votes, class balance undone (p > 1 − π)          | 67.2      |      79.7 | 71.2        |   78.7 |        73.2 |                 74.2 |
| C2: mean probability with the class balance undone, above 0.5        | 68.4      |      78.9 | 71.8        |   77   |        73.9 |                 75.5 |
| A: cutoff on the mean probability, best on earlier terms             | –         |      78.9 | –           |   77   |        71.6 |                 83.7 |
| B: logistic fit of the winner on the mean probability, earlier terms | –         |      79.7 | –           |   77   |        72.5 |                 86.6 |
| Always petitioner                                                    | 64.8      |      73.2 | 67.5        |   70.5 |        67.6 |                100   |

- 2012–2017: C1 − standard +0.021 [-0.021, +0.063]; C1 − always petitioner +0.037 [-0.016, +0.090]; C2 − standard +0.026 [-0.016, +0.066]; C2 − always petitioner +0.042 [-0.013, +0.092].
- 2013–2017: A − always petitioner +0.039 [-0.007, +0.085]; B − always petitioner +0.049 [+0.010, +0.088]; B − standard +0.026 [-0.036, +0.092].

|   Model v1, term |   Cases |   Standard |   C1 |   C2 |   Always petitioner |
|-----------------:|--------:|-----------:|-----:|-----:|--------------------:|
|             2012 |      73 |       64.4 | 61.6 | 63   |                67.1 |
|             2013 |      66 |       59.1 | 62.1 | 63.6 |                68.2 |
|             2014 |      53 |       60.4 | 66   | 66   |                67.9 |
|             2015 |      64 |       62.5 | 70.3 | 71.9 |                56.2 |
|             2016 |      62 |       79   | 83.9 | 83.9 |                75.8 |
|             2017 |      61 |       82   | 80.3 | 77   |                70.5 |

Votes of model v1 with the class balance undone (p > 1 − π), 2012–2017: vote accuracy 71.3% (at 0.5: 69.7%), macro-F1 0.686 (at 0.5: 0.687), predicted petitioner 66.3% (at 0.5: 54.9%; true 62.8%).
