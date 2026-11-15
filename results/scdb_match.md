# SCDB join (prepare_scdb.py)

Source: Supreme Court Database, modern release SCDB_2026_01 (terms 1946-2025), docket files; 9,409 cases, 10,932 dockets.

## Match rate

| split   |   cases |   matched |   exact |   shifted |   has_argument_date |
|:--------|--------:|----------:|--------:|----------:|--------------------:|
| test    |     124 |       124 |     124 |         0 |                 124 |
| train   |    6388 |      6373 |    6339 |        34 |                6368 |
| val     |     123 |       123 |     123 |         0 |                 123 |

Several SCDB decisions for the matched docket: 7 cases (the argued one, then the latest decision, is used).

Extension tables (data/ext, terms 2020-2025): 357 of 357 cases matched (357 exact), 357 with an argument date. Profile only; no outcome of these cases is read here.

## Label agreement with ConvoKit (train and validation only)

- Case winner: 0.9991 of 6,496 matched cases with a clear SCDB winner.
- Justice votes: 0.9991 of 56,862 votes found in SCDB (145 of our votes have no usable SCDB vote).

By split:

| split   |   case winner |   votes |
|:--------|--------------:|--------:|
| train   |        0.9991 |  0.9991 |
| val     |        1      |  1      |

## Justice id map (by co-occurrence on matched cases)

| justice                  | justiceName   |   total |   share |
|:-------------------------|:--------------|--------:|--------:|
| j__abe_fortas            | AFortas       |     364 |   1     |
| j__amy_coney_barrett     | ACBarrett     |     345 |   1     |
| j__anthony_m_kennedy     | AMKennedy     |    2473 |   1     |
| j__antonin_scalia        | AScalia       |    2514 |   1     |
| j__arthur_j_goldberg     | AJGoldberg    |     334 |   0.985 |
| j__brett_m_kavanaugh     | BMKavanaugh   |     472 |   1     |
| j__byron_r_white         | BRWhite       |    3869 |   0.999 |
| j__charles_e_whittaker   | CEWhittaker   |     541 |   0.998 |
| j__clarence_thomas       | CThomas       |    2503 |   1     |
| j__david_h_souter        | DHSouter      |    1540 |   1     |
| j__earl_warren           | EWarren       |    1529 |   1     |
| j__elena_kagan           | EKagan        |     963 |   1     |
| j__felix_frankfurter     | FFrankfurter  |     706 |   1     |
| j__harold_burton         | HHBurton      |     323 |   1     |
| j__harry_a_blackmun      | HABlackmun    |    3047 |   1     |
| j__hugo_l_black          | HLBlack       |    1744 |   1     |
| j__john_g_roberts_jr     | JGRoberts     |    1361 |   1     |
| j__john_m_harlan2        | JHarlan2      |    1732 |   1     |
| j__john_paul_stevens     | JPStevens     |    3554 |   1     |
| j__ketanji_brown_jackson | KBJackson     |     236 |   1     |
| j__lewis_f_powell_jr     | LFPowell      |    1982 |   1     |
| j__neil_gorsuch          | NMGorsuch     |     550 |   1     |
| j__potter_stewart        | PStewart      |    2733 |   1     |
| j__ruth_bader_ginsburg   | RBGinsburg    |    1952 |   0.999 |
| j__samuel_a_alito_jr     | SAAlito       |    1323 |   1     |
| j__sandra_day_oconnor    | SDOConnor     |    2483 |   1     |
| j__sherman_minton        | SMinton       |      93 |   1     |
| j__sonia_sotomayor       | SSotomayor    |    1068 |   1     |
| j__stanley_reed          | SFReed        |     133 |   1     |
| j__stephen_g_breyer      | SGBreyer      |    1972 |   1     |
| j__thurgood_marshall     | TMarshall     |    3006 |   1     |
| j__tom_c_clark           | TCClark       |    1310 |   1     |
| j__warren_e_burger       | WEBurger      |    2233 |   1     |
| j__william_h_rehnquist   | WHRehnquist   |    3719 |   1     |
| j__william_j_brennan_jr  | WJBrennan     |    4194 |   1     |
| j__william_o_douglas     | WODouglas     |    2275 |   1     |

## Leakage audit (train cases only)

Petitioner win rate overall: 0.641. A pre-decision field should not have a value that almost fixes the winner. Largest deviations among values with at least 40 train cases:

| field                  |   values (>=40) | lowest win rate           | highest win rate          |
|:-----------------------|----------------:|:--------------------------|:--------------------------|
| petitioner             |              31 | 0.365 (value 240, n 74)   | 0.765 (value 27, n 574)   |
| petitionerState        |              11 | 0.545 (value 51, n 66)    | 0.778 (value 27, n 45)    |
| respondent             |              34 | 0.327 (value 369, n 49)   | 0.850 (value 198, n 40)   |
| respondentState        |              13 | 0.489 (value 45, n 45)    | 0.787 (value 60, n 61)    |
| jurisdiction           |               2 | 0.636 (value 2, n 1025)   | 0.643 (value 1, n 5330)   |
| adminAction            |               9 | 0.535 (value 69, n 200)   | 0.734 (value 66, n 79)    |
| adminActionState       |               3 | 0.481 (value 6, n 52)     | 0.642 (value -1, n 5840)  |
| threeJudgeFdc          |               2 | 0.641 (value 0, n 5966)   | 0.645 (value 1, n 403)    |
| caseOrigin             |              42 | 0.364 (value -1, n 44)    | 0.812 (value 49, n 48)    |
| caseOriginState        |              11 | 0.535 (value 6, n 142)    | 0.790 (value 51, n 62)    |
| caseSource             |              18 | 0.546 (value 22, n 454)   | 0.704 (value 29, n 805)   |
| caseSourceState        |              11 | 0.535 (value 6, n 142)    | 0.790 (value 51, n 62)    |
| lcDisagreement         |               2 | 0.620 (value 0, n 4891)   | 0.712 (value 1, n 1477)   |
| certReason             |               9 | 0.550 (value 2, n 1110)   | 0.750 (value 13, n 64)    |
| lcDisposition          |              11 | 0.605 (value 2, n 3062)   | 0.780 (value 8, n 109)    |
| lcDispositionDirection |               3 | 0.400 (value 3, n 40)     | 0.649 (value 2, n 2888)   |
| issue                  |              46 | 0.011 (value 90150, n 94) | 0.860 (value 90090, n 43) |
| issueArea              |              11 | 0.533 (value 12, n 227)   | 0.675 (value 4, n 252)    |
| lawType                |               7 | 0.392 (value 5, n 212)    | 0.700 (value -1, n 523)   |
| lawSupp                |              34 | 0.011 (value 508, n 95)   | 0.860 (value 213, n 43)   |

(value -1 = missing)

Lower-court direction x decade (petitioner win rate, n):

|   decade | conservative   | liberal    | missing   | unspecifiable   |
|---------:|:---------------|:-----------|:----------|:----------------|
|     1950 | 0.67 (382)     | 0.44 (165) | 0.00 (1)  | 0.00 (2)        |
|     1960 | 0.75 (832)     | 0.58 (269) | 1.00 (2)  | 0.33 (3)        |
|     1970 | 0.60 (653)     | 0.68 (675) | 0.50 (2)  | 0.00 (1)        |
|     1980 | 0.57 (551)     | 0.64 (793) |           | 0.50 (10)       |
|     1990 | 0.56 (476)     | 0.67 (400) | 0.20 (5)  | 0.17 (6)        |
|     2000 | 0.65 (320)     | 0.72 (404) | 0.14 (7)  | 0.43 (14)       |
|     2010 | 0.61 (213)     | 0.66 (182) | 0.00 (1)  | 0.75 (4)        |

Is the lower-court direction a pre-decision fact? In criminal-procedure cases it should follow from who lost below, whatever the Supreme Court then did:

- government (United States or a State) is the petitioner, so the defendant won below: lower court coded liberal in 0.972 of 508 cases;
- government is the respondent, so it won below: lower court coded conservative in 0.969 of 799 cases.
