# Audio trial: pitch relative to each justice's own normal (notes §21.2)

Features, test and pass mark were fixed before any recording was processed. Terms 2010–2017 only; the test terms were not read.

## 1. The recordings

- 535 transcripts of 2010–17 in our tables; 534 recordings processed; 514 lined up with their turns (96.1%); not lined up: 21353, 21480, 21578, 21609, 21916, 22118, 22575, 22684, 22936, 23218, 23226, 23228.
- Audio length minus the end of the last turn: median 1.4 s, largest 3384.1 s.
- 61,563 justice turns; pitch measured on 91.4% (at least 20 voiced frames).
- Median pitch by justice: clarence_thomas 80 Hz, neil_gorsuch 96 Hz, samuel_a_alito_jr 99 Hz, anthony_m_kennedy 111 Hz, stephen_g_breyer 114 Hz, john_g_roberts_jr 124 Hz, antonin_scalia 129 Hz, ruth_bader_ginsburg 151 Hz, elena_kagan 153 Hz, sonia_sotomayor 157 Hz.
- Stop rule (a justice's median outside 70–300 Hz): not triggered.

Votes of 2012–17 with a defined difference: `po_diff` 53%, `pb_diff` 89%, `lo_diff` 53%, `lb_diff` 89%, `ro_diff` 53%, `rb_diff` 89%.

## 2. Walk-forward 2012–17 (each term from the terms 2010 to T − 1)

| Model                                 |   AUC 2012–15 |   AUC 2016–17 |   AUC 2017 alone |   AUC 2012–17 |   Macro-F1 2012–17 |
|:--------------------------------------|--------------:|--------------:|-----------------:|--------------:|-------------------:|
| qc_bloc as it is (not refitted)       |         0.725 |         0.813 |            0.832 |         0.752 |              0.683 |
| regression on the qc_bloc logit       |         0.725 |         0.813 |            0.832 |         0.751 |              0.683 |
| + pitch (2 features, primary)         |         0.733 |         0.802 |            0.819 |         0.754 |              0.681 |
| + pitch, loudness, range (6 features) |         0.725 |         0.794 |            0.818 |         0.747 |              0.667 |
| pitch only (2 features)               |         0.592 |         0.572 |            0.571 |         0.584 |              0.53  |

## 3. Gain over the regression on qc_bloc alone (paired, 95% interval over cases)

| Minus the regression on qc_bloc alone   | Scored on   | AUC                     | Macro-F1                |
|:----------------------------------------|:------------|:------------------------|:------------------------|
| + pitch (primary)                       | 2012–15     | +0.009 [+0.001, +0.016] | +0.008 [-0.002, +0.018] |
| + pitch (primary)                       | 2016–17     | -0.012 [-0.035, +0.007] | -0.022 [-0.052, +0.004] |
| + pitch (primary)                       | 2017 alone  | -0.013 [-0.033, +0.007] | -0.029 [-0.058, +0.001] |
| + pitch (primary)                       | 2012–17     | +0.003 [-0.005, +0.012] | -0.002 [-0.012, +0.009] |
| + pitch, loudness, range                | 2012–15     | +0.001 [-0.015, +0.016] | -0.010 [-0.027, +0.006] |
| + pitch, loudness, range                | 2016–17     | -0.019 [-0.044, +0.004] | -0.029 [-0.057, -0.003] |
| + pitch, loudness, range                | 2017 alone  | -0.014 [-0.040, +0.014] | -0.024 [-0.057, +0.005] |
| + pitch, loudness, range                | 2012–17     | -0.005 [-0.017, +0.008] | -0.016 [-0.030, -0.002] |

## 4. Pass mark

Fixed in advance: pooled AUC gain of at least +0.010 with the two pitch features, and a positive gain on 2012–15 and on 2017 alone; clear if the pooled interval excludes zero.

- Pooled +0.003 [-0.005, +0.012]; 2012–15 +0.009; 2017 alone -0.013.
- **DOES NOT PASS.**

## 5. The mechanism, training terms 2010–15 only

The published direction: a justice's pitch is higher toward the side the justice votes against, so the share should be above 50% and the AUC above 0.5.

| Measure     |   Justice–case pairs | Higher toward the side voted against %   |   AUC of the justice's difference for the vote |   AUC of the bench's difference for the vote |   Cases with a bench difference |
|:------------|---------------------:|:-----------------------------------------|-----------------------------------------------:|---------------------------------------------:|--------------------------------:|
| pitch       |                1,890 | 55.1 [52.8, 57.2]                        |                                          0.561 |                                        0.582 |                             355 |
| loudness    |                1,890 | 53.9 [51.3, 56.4]                        |                                          0.546 |                                        0.552 |                             355 |
| pitch range |                1,890 | 55.4 [53.1, 57.8]                        |                                          0.572 |                                        0.59  |                             355 |

Weights in the last regression (fitted on 2010–16, standardized inputs; + = pro-petitioner): zq +0.932, po_diff +0.013, pb_diff +0.151, lo_diff -0.006, lb_diff -0.005, ro_diff +0.055, rb_diff +0.161.
With the two pitch features only: zq +0.934, po_diff +0.025, pb_diff +0.213.
