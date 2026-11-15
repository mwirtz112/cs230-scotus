# Oyez extension: terms after 2019 (prepare_oyez.py)

Transcripts from api.oyez.org, votes from SCDB release 2026_01. Same table format and filters as the ConvoKit data. Outcomes are not summarized here: these cases are locked like the test set until their split is decided.

|   term |   listed by Oyez |   argued, downloaded |   cases | dropped                |   transcripts |   votes |   turns |   median turns / case |   median words / case |   justice turns % |   median words / justice turn |   unknown speaker % |   advocate side unknown % |   voting justices who speak % |   Thomas speaks, % of his cases |
|-------:|-----------------:|---------------------:|--------:|:-----------------------|--------------:|--------:|--------:|----------------------:|----------------------:|------------------:|------------------------------:|--------------------:|--------------------------:|------------------------------:|--------------------------------:|
|   2020 |               64 |                   58 |      57 | {"unclear winner": 1}  |            57 |     502 |   13682 |                   246 |                 12769 |              57.9 |                            15 |                   0 |                      0.95 |                         100   |                           100   |
|   2021 |               67 |                   63 |      62 | {"no usable votes": 1} |            62 |     556 |   17687 |                   282 |                 13922 |              54.2 |                            14 |                   0 |                      0    |                          95.3 |                            93.5 |
|   2022 |               62 |                   60 |      59 | {}                     |            59 |     530 |   21461 |                   343 |                 15647 |              54.2 |                            14 |                   0 |                      3.7  |                          96.8 |                            98.3 |
|   2023 |               62 |                   61 |      61 | {}                     |            61 |     547 |   22856 |                   371 |                 16004 |              54.4 |                            14 |                   0 |                      0.83 |                          98.5 |                            96.7 |
|   2024 |               62 |                   61 |      60 | {"no usable votes": 1} |            60 |     537 |   21292 |                   330 |                 14136 |              54.2 |                            14 |                   0 |                      0    |                          96.6 |                           100   |
|   2025 |               60 |                   58 |      58 | {}                     |            59 |     521 |   21901 |                   385 |                 14936 |              54.3 |                            13 |                   0 |                      1.57 |                          98.5 |                           100   |

## For comparison: the same input statistics for 2012-2019 (ConvoKit tables)

The 2020 term was argued by telephone, each justice questioning in turn; since the 2021 term every advocate's time is followed by a round of questions in order of seniority.

|   term |   cases |   transcripts |   votes |   turns |   median turns / case |   median words / case |   justice turns % |   median words / justice turn |   unknown speaker % |   advocate side unknown % |   voting justices who speak % |   Thomas speaks, % of his cases |
|-------:|--------:|--------------:|--------:|--------:|----------------------:|----------------------:|------------------:|------------------------------:|--------------------:|--------------------------:|------------------------------:|--------------------------------:|
|   2012 |      73 |            74 |     651 |   16606 |                   217 |                 10497 |              51.2 |                            23 |                0    |                      0    |                          86.3 |                             1.4 |
|   2013 |      66 |            67 |     589 |   14556 |                   218 |                 10542 |              51.3 |                            24 |                2.72 |                      9.78 |                          88.3 |                             0   |
|   2014 |      53 |            55 |     477 |   12182 |                   216 |                 10569 |              51.7 |                            21 |                2.89 |                      1.03 |                          87   |                             0   |
|   2015 |      64 |            65 |     521 |   14739 |                   223 |                 10452 |              51.7 |                            19 |                0.12 |                      0.8  |                          85.6 |                             1.6 |
|   2016 |      62 |            62 |     505 |   13060 |                   206 |                 10373 |              51.3 |                            19 |                0    |                      1.01 |                          85.3 |                             0   |
|   2017 |      61 |            63 |     544 |   15225 |                   242 |                 10669 |              52.2 |                            18 |                0.89 |                      0    |                          86.2 |                             0   |
|   2018 |      69 |            70 |     613 |   16305 |                   227 |                 10448 |              52.1 |                            17 |                0    |                      0    |                          84.7 |                             1.4 |
|   2019 |      55 |            55 |     492 |   12979 |                   232 |                 10804 |              53.3 |                            15 |                0    |                      3.01 |                          87.4 |                            18.2 |

## Justices with votes, by term

|   year | bench                                                                          |
|-------:|:-------------------------------------------------------------------------------|
|   2020 | alito, barrett, breyer, gorsuch, kagan, kavanaugh, roberts, sotomayor, thomas  |
|   2021 | alito, barrett, breyer, gorsuch, kagan, kavanaugh, roberts, sotomayor, thomas  |
|   2022 | alito, barrett, gorsuch, jackson, kagan, kavanaugh, roberts, sotomayor, thomas |
|   2023 | alito, barrett, gorsuch, jackson, kagan, kavanaugh, roberts, sotomayor, thomas |
|   2024 | alito, barrett, gorsuch, jackson, kagan, kavanaugh, roberts, sotomayor, thomas |
|   2025 | alito, barrett, gorsuch, jackson, kagan, kavanaugh, roberts, sotomayor, thomas |

## Advocate descriptions that give no side

| description                           |   advocates |
|:--------------------------------------|------------:|
| for the United States                 |           3 |
| for the federal parties               |           2 |
| for the U.S. House of Representatives |           1 |
| for the United States, et al.         |           1 |
| on behalf of the federal parties      |           1 |
| for Smith & Nephew, Inc., et al.      |           1 |
| for Arkansas et al.                   |           1 |
| for Chad Everet Brackeen, et al.      |           1 |
| for Texas                             |           1 |
| for the tribal parties                |           1 |
| for the SEC, et al.                   |           1 |
| On behalf of the Federal parties      |           1 |
| On behalf of the State parties        |           1 |
| for the Intervenors                   |           1 |
| for the private parties               |           1 |
