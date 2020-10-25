# CSI 4900 - Covid Prediction

**Author**: Quang-Vinh Do



## Goal 🌟
Create covid prediction model to forecast infection/mortality/recovered rates in Canada.


## SEIR Model
- [ ] Reorganize code. Make it easier to do cross validation and apply other optimization methods.
- [ ] Add machine learning layer to SEIR model



## TODO
- [ ] Use more up to date data from https://github.com/ishaberry/Covid19Canada


## Results



## Datasets
- Canada Dataset https://github.com/ishaberry/Covid19Canada 


# References
- YGG Model / SIR Model https://covid19-projections.com/model-details/





## TEMP

(covid-prediction) PS F:\OneDrive\Projects\CSI 4900> python .\yyg-seir-simulator\src\learn_simulation.py -v --country Canada
====================================================
YYG/C19PRO Simulator
Current time: 2020-10-12 03:29:58.202716
====================================================
Loading params file: yyg-seir-simulator\src\..\best_params\latest/subregion\Canada_Ontario.json
best params type: mean
================================
Canada | ALL | Ontario
================================
Parameters:
INITIAL_R_0               : 1.8621476756657092
LOCKDOWN_R_0              : 0.8121886339025861
INFLECTION_DAY            : 2020-03-27
RATE_OF_INFLECTION        : 0.3497453946619491
LOCKDOWN_FATIGUE          : 1.0
DAILY_IMPORTS             : 194.06780757414793
MORTALITY_RATE            : 0.012500000000000002
REOPEN_DATE               : 2020-06-01
REOPEN_SHIFT_DAYS         : 14.24962572681525
REOPEN_R                  : 0.9350336996413784
REOPEN_INFLECTION         : 0.30222943084385256
POST_REOPEN_EQUILIBRIUM_R : 0.9875365749849131
FALL_R_MULTIPLIER         : 1.0010158770538358
--------------------------
Running simulation...
--------------------------
Finding new optimum with error: 182676.20797784862
Finding new optimum with error: 172505.9862695099
Finding new optimum with error: 162631.65797633678
Finding new optimum with error: 153053.15040140558
Finding new optimum with error: 143770.39087135627
Finding new optimum with error: 134783.3067363771
Finding new optimum with error: 126091.82537019548
Finding new optimum with error: 117695.87417007002
Finding new optimum with error: 109595.38055677342
Finding new optimum with error: 101790.27197459071
Finding new optimum with error: 94280.47589130381
Finding new optimum with error: 87065.91979817595
Finding new optimum with error: 80146.53120995169
Finding new optimum with error: 73522.23766484042
Finding new optimum with error: 67192.9667245043
Finding new optimum with error: 61158.64597405293
Finding new optimum with error: 55419.20302202585
Finding new optimum with error: 49974.565500390774
Finding new optimum with error: 44824.66106452471
Finding new optimum with error: 39968.49733751463
Finding new optimum with error: 35406.91042226498
Finding new optimum with error: 31139.9579458519
Finding new optimum with error: 27167.567617870074
Finding new optimum with error: 23489.66717129426
Finding new optimum with error: 20106.184362467386
Finding new optimum with error: 17017.046971088737
Finding new optimum with error: 14222.182800207316
Finding new optimum with error: 11721.519676209089
Finding new optimum with error: 9514.985448806663
Finding new optimum with error: 7602.507991027722
Finding new optimum with error: 5984.015199207749
Finding new optimum with error: 4659.434992975447
Finding new optimum with error: 3628.6953152458873
Finding new optimum with error: 2891.7241322085556
Finding new optimum with error: 2448.4494333164666
Finding new optimum with error: 2298.799231276642
Finding new optimum with error: 2016.1191924175712
Finding new optimum with error: 1914.5752604255154
Finding new optimum with error: 1645.724274469668
Finding new optimum with error: 1598.3281600636583
Finding new optimum with error: 1344.026208398145
Finding new optimum with error: 1340.9819726227126
Finding new optimum with error: 1118.434997818828
Finding new optimum with error: 977.060725471656
Finding new optimum with error: 928.7684526230399
Finding new optimum with error: 922.4822181998111
Finding new optimum with error: 920.3609985400153
{'INITIAL_R_0': 1.3605078936496406, 'LOCKDOWN_R_0': 0.8015019413512363, 'DAILY_IMPORTS': 203.77119795285535, 'MORTALITY_RATE': 0.03125000000000001}
7809.0946934223175
1   - 2020-02-22 - New / total infections: 204 / 204 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
2   - 2020-02-23 - New / total infections: 204 / 408 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
3   - 2020-02-24 - New / total infections: 204 / 611 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
4   - 2020-02-25 - New / total infections: 204 / 815 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
5   - 2020-02-26 - New / total infections: 204 / 1,019 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
6   - 2020-02-27 - New / total infections: 204 / 1,223 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
7   - 2020-02-28 - New / total infections: 204 / 1,426 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
8   - 2020-02-29 - New / total infections: 204 / 1,630 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
9   - 2020-03-01 - New / total infections: 204 / 1,834 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
10  - 2020-03-02 - New / total infections: 481 / 2,315 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
11  - 2020-03-03 - New / total infections: 479 / 2,794 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
12  - 2020-03-04 - New / total infections: 495 / 3,289 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.862 - IFR: 3.13%
13  - 2020-03-05 - New / total infections: 531 / 3,820 - Hospital beds in use: 0 - New / total deaths: 0.00 / 0.0 - Mean R: 1.861 - IFR: 3.13%
14  - 2020-03-06 - New / total infections: 605 / 4,424 - Hospital beds in use: 4 - New / total deaths: 0.00 / 0.0 - Mean R: 1.861 - IFR: 3.13%
15  - 2020-03-07 - New / total infections: 719 / 5,144 - Hospital beds in use: 8 - New / total deaths: 0.00 / 0.0 - Mean R: 1.861 - IFR: 3.13%
16  - 2020-03-08 - New / total infections: 806 / 5,949 - Hospital beds in use: 12 - New / total deaths: 0.00 / 0.0 - Mean R: 1.860 - IFR: 3.13%
17  - 2020-03-09 - New / total infections: 875 / 6,824 - Hospital beds in use: 16 - New / total deaths: 0.01 / 0.0 - Mean R: 1.860 - IFR: 3.13%
18  - 2020-03-10 - New / total infections: 951 / 7,775 - Hospital beds in use: 20 - New / total deaths: 0.03 / 0.0 - Mean R: 1.859 - IFR: 3.13%
19  - 2020-03-11 - New / total infections: 1,038 / 8,813 - Hospital beds in use: 24 - New / total deaths: 0.07 / 0.1 - Mean R: 1.858 - IFR: 3.13%
20  - 2020-03-12 - New / total infections: 1,146 / 9,960 - Hospital beds in use: 28 - New / total deaths: 0.12 / 0.2 - Mean R: 1.856 - IFR: 3.13%
21  - 2020-03-13 - New / total infections: 1,257 / 11,216 - Hospital beds in use: 32 - New / total deaths: 0.17 / 0.4 - Mean R: 1.854 - IFR: 3.13%
22  - 2020-03-14 - New / total infections: 1,365 / 12,581 - Hospital beds in use: 36 - New / total deaths: 0.25 / 0.7 - Mean R: 1.850 - IFR: 3.13%
23  - 2020-03-15 - New / total infections: 1,475 / 14,056 - Hospital beds in use: 46 - New / total deaths: 0.34 / 1.0 - Mean R: 1.846 - IFR: 3.13%
24  - 2020-03-16 - New / total infections: 1,592 / 15,647 - Hospital beds in use: 55 - New / total deaths: 0.45 / 1.4 - Mean R: 1.839 - IFR: 3.13%
25  - 2020-03-17 - New / total infections: 1,718 / 17,365 - Hospital beds in use: 61 - New / total deaths: 0.58 / 2.0 - Mean R: 1.831 - IFR: 3.13%
26  - 2020-03-18 - New / total infections: 1,849 / 19,214 - Hospital beds in use: 68 - New / total deaths: 0.77 / 2.8 - Mean R: 1.819 - IFR: 3.13%
27  - 2020-03-19 - New / total infections: 1,981 / 21,195 - Hospital beds in use: 76 - New / total deaths: 1.03 / 3.8 - Mean R: 1.802 - IFR: 3.13%
28  - 2020-03-20 - New / total infections: 2,111 / 23,306 - Hospital beds in use: 86 - New / total deaths: 1.36 / 5.2 - Mean R: 1.779 - IFR: 3.13%
29  - 2020-03-21 - New / total infections: 2,238 / 25,544 - Hospital beds in use: 98 - New / total deaths: 1.75 / 6.9 - Mean R: 1.749 - IFR: 3.13%
30  - 2020-03-22 - New / total infections: 2,358 / 27,902 - Hospital beds in use: 112 - New / total deaths: 2.17 / 9.1 - Mean R: 1.709 - IFR: 3.13%
31  - 2020-03-23 - New / total infections: 2,467 / 30,369 - Hospital beds in use: 126 - New / total deaths: 2.58 / 11.7 - Mean R: 1.659 - IFR: 3.13%
32  - 2020-03-24 - New / total infections: 2,558 / 32,927 - Hospital beds in use: 143 - New / total deaths: 3.05 / 14.7 - Mean R: 1.596 - IFR: 3.11%
33  - 2020-03-25 - New / total infections: 2,625 / 35,553 - Hospital beds in use: 162 - New / total deaths: 3.59 / 18.3 - Mean R: 1.523 - IFR: 3.09%
34  - 2020-03-26 - New / total infections: 2,666 / 38,219 - Hospital beds in use: 178 - New / total deaths: 4.21 / 22.5 - Mean R: 1.440 - IFR: 3.08%
35  - 2020-03-27 - New / total infections: 2,681 / 40,900 - Hospital beds in use: 195 - New / total deaths: 4.91 / 27.5 - Mean R: 1.352 - IFR: 3.06%
36  - 2020-03-28 - New / total infections: 2,671 / 43,570 - Hospital beds in use: 215 - New / total deaths: 5.70 / 33.2 - Mean R: 1.265 - IFR: 3.05%
37  - 2020-03-29 - New / total infections: 2,641 / 46,212 - Hospital beds in use: 236 - New / total deaths: 6.59 / 39.8 - Mean R: 1.182 - IFR: 3.03%
38  - 2020-03-30 - New / total infections: 2,597 / 48,809 - Hospital beds in use: 258 - New / total deaths: 7.58 / 47.3 - Mean R: 1.109 - IFR: 3.02%
39  - 2020-03-31 - New / total infections: 2,543 / 51,352 - Hospital beds in use: 281 - New / total deaths: 8.70 / 56.0 - Mean R: 1.047 - IFR: 3.00%
40  - 2020-04-01 - New / total infections: 2,482 / 53,833 - Hospital beds in use: 304 - New / total deaths: 9.96 / 66.0 - Mean R: 0.996 - IFR: 2.99%
41  - 2020-04-02 - New / total infections: 2,416 / 56,250 - Hospital beds in use: 329 - New / total deaths: 11.32 / 77.3 - Mean R: 0.957 - IFR: 2.97%
42  - 2020-04-03 - New / total infections: 2,349 / 58,599 - Hospital beds in use: 355 - New / total deaths: 12.80 / 90.1 - Mean R: 0.927 - IFR: 2.96%
43  - 2020-04-04 - New / total infections: 2,281 / 60,880 - Hospital beds in use: 381 - New / total deaths: 14.41 / 104.5 - Mean R: 0.904 - IFR: 2.94%
44  - 2020-04-05 - New / total infections: 2,214 / 63,094 - Hospital beds in use: 408 - New / total deaths: 16.16 / 120.7 - Mean R: 0.888 - IFR: 2.93%
45  - 2020-04-06 - New / total infections: 2,147 / 65,242 - Hospital beds in use: 434 - New / total deaths: 18.06 / 138.7 - Mean R: 0.875 - IFR: 2.91%
46  - 2020-04-07 - New / total infections: 2,082 / 67,324 - Hospital beds in use: 459 - New / total deaths: 20.10 / 158.8 - Mean R: 0.867 - IFR: 2.90%
47  - 2020-04-08 - New / total infections: 2,018 / 69,342 - Hospital beds in use: 483 - New / total deaths: 22.27 / 181.1 - Mean R: 0.860 - IFR: 2.88%
48  - 2020-04-09 - New / total infections: 1,957 / 71,299 - Hospital beds in use: 505 - New / total deaths: 24.57 / 205.7 - Mean R: 0.855 - IFR: 2.87%
49  - 2020-04-10 - New / total infections: 1,897 / 73,195 - Hospital beds in use: 524 - New / total deaths: 27.00 / 232.7 - Mean R: 0.852 - IFR: 2.86%
50  - 2020-04-11 - New / total infections: 1,838 / 75,034 - Hospital beds in use: 539 - New / total deaths: 29.53 / 262.2 - Mean R: 0.850 - IFR: 2.84%
51  - 2020-04-12 - New / total infections: 1,782 / 76,816 - Hospital beds in use: 552 - New / total deaths: 32.14 / 294.3 - Mean R: 0.848 - IFR: 2.83%
52  - 2020-04-13 - New / total infections: 1,728 / 78,544 - Hospital beds in use: 560 - New / total deaths: 34.80 / 329.1 - Mean R: 0.846 - IFR: 2.81%
53  - 2020-04-14 - New / total infections: 1,675 / 80,219 - Hospital beds in use: 565 - New / total deaths: 37.46 / 366.6 - Mean R: 0.845 - IFR: 2.80%
54  - 2020-04-15 - New / total infections: 1,624 / 81,843 - Hospital beds in use: 566 - New / total deaths: 40.11 / 406.7 - Mean R: 0.845 - IFR: 2.78%
55  - 2020-04-16 - New / total infections: 1,575 / 83,418 - Hospital beds in use: 564 - New / total deaths: 42.68 / 449.4 - Mean R: 0.844 - IFR: 2.77%
56  - 2020-04-17 - New / total infections: 1,527 / 84,946 - Hospital beds in use: 559 - New / total deaths: 45.15 / 494.5 - Mean R: 0.843 - IFR: 2.76%
57  - 2020-04-18 - New / total infections: 1,481 / 86,427 - Hospital beds in use: 550 - New / total deaths: 47.49 / 542.0 - Mean R: 0.843 - IFR: 2.74%
58  - 2020-04-19 - New / total infections: 1,437 / 87,864 - Hospital beds in use: 540 - New / total deaths: 49.65 / 591.7 - Mean R: 0.843 - IFR: 2.73%
59  - 2020-04-20 - New / total infections: 1,394 / 89,258 - Hospital beds in use: 528 - New / total deaths: 51.61 / 643.3 - Mean R: 0.842 - IFR: 2.72%
60  - 2020-04-21 - New / total infections: 1,352 / 90,610 - Hospital beds in use: 515 - New / total deaths: 53.35 / 696.6 - Mean R: 0.842 - IFR: 2.70%
61  - 2020-04-22 - New / total infections: 1,311 / 91,921 - Hospital beds in use: 501 - New / total deaths: 54.86 / 751.5 - Mean R: 0.842 - IFR: 2.69%
62  - 2020-04-23 - New / total infections: 1,272 / 93,193 - Hospital beds in use: 487 - New / total deaths: 56.12 / 807.6 - Mean R: 0.841 - IFR: 2.68%
63  - 2020-04-24 - New / total infections: 1,234 / 94,428 - Hospital beds in use: 473 - New / total deaths: 56.99 / 864.6 - Mean R: 0.841 - IFR: 2.66%
64  - 2020-04-25 - New / total infections: 1,198 / 95,625 - Hospital beds in use: 459 - New / total deaths: 57.49 / 922.1 - Mean R: 0.841 - IFR: 2.65%
65  - 2020-04-26 - New / total infections: 1,162 / 96,787 - Hospital beds in use: 445 - New / total deaths: 57.66 / 979.8 - Mean R: 0.841 - IFR: 2.64%
66  - 2020-04-27 - New / total infections: 1,127 / 97,914 - Hospital beds in use: 432 - New / total deaths: 57.54 / 1,037.3 - Mean R: 0.841 - IFR: 2.62%
67  - 2020-04-28 - New / total infections: 1,094 / 99,008 - Hospital beds in use: 419 - New / total deaths: 57.16 / 1,094.5 - Mean R: 0.840 - IFR: 2.61%
68  - 2020-04-29 - New / total infections: 1,061 / 100,069 - Hospital beds in use: 406 - New / total deaths: 56.57 / 1,151.0 - Mean R: 0.840 - IFR: 2.60%
69  - 2020-04-30 - New / total infections: 1,029 / 101,098 - Hospital beds in use: 394 - New / total deaths: 55.79 / 1,206.8 - Mean R: 0.840 - IFR: 2.58%
70  - 2020-05-01 - New / total infections: 998 / 102,096 - Hospital beds in use: 382 - New / total deaths: 54.86 / 1,261.7 - Mean R: 0.840 - IFR: 2.57%
71  - 2020-05-02 - New / total infections: 968 / 103,065 - Hospital beds in use: 370 - New / total deaths: 53.81 / 1,315.5 - Mean R: 0.840 - IFR: 2.56%
72  - 2020-05-03 - New / total infections: 939 / 104,004 - Hospital beds in use: 359 - New / total deaths: 52.66 / 1,368.1 - Mean R: 0.840 - IFR: 2.54%
73  - 2020-05-04 - New / total infections: 911 / 104,915 - Hospital beds in use: 348 - New / total deaths: 51.44 / 1,419.6 - Mean R: 0.839 - IFR: 2.53%
74  - 2020-05-05 - New / total infections: 883 / 105,798 - Hospital beds in use: 337 - New / total deaths: 50.16 / 1,469.7 - Mean R: 0.839 - IFR: 2.52%
75  - 2020-05-06 - New / total infections: 856 / 106,654 - Hospital beds in use: 327 - New / total deaths: 48.84 / 1,518.6 - Mean R: 0.839 - IFR: 2.51%
76  - 2020-05-07 - New / total infections: 830 / 107,484 - Hospital beds in use: 317 - New / total deaths: 47.50 / 1,566.1 - Mean R: 0.839 - IFR: 2.49%
77  - 2020-05-08 - New / total infections: 804 / 108,289 - Hospital beds in use: 308 - New / total deaths: 46.14 / 1,612.2 - Mean R: 0.839 - IFR: 2.48%
78  - 2020-05-09 - New / total infections: 779 / 109,068 - Hospital beds in use: 298 - New / total deaths: 44.78 / 1,657.0 - Mean R: 0.839 - IFR: 2.47%
79  - 2020-05-10 - New / total infections: 755 / 109,823 - Hospital beds in use: 289 - New / total deaths: 43.42 / 1,700.4 - Mean R: 0.839 - IFR: 2.46%
80  - 2020-05-11 - New / total infections: 731 / 110,555 - Hospital beds in use: 281 - New / total deaths: 42.08 / 1,742.5 - Mean R: 0.839 - IFR: 2.44%
81  - 2020-05-12 - New / total infections: 708 / 111,263 - Hospital beds in use: 272 - New / total deaths: 40.75 / 1,783.2 - Mean R: 0.838 - IFR: 2.43%
82  - 2020-05-13 - New / total infections: 685 / 111,948 - Hospital beds in use: 264 - New / total deaths: 39.44 / 1,822.7 - Mean R: 0.838 - IFR: 2.42%
83  - 2020-05-14 - New / total infections: 663 / 112,611 - Hospital beds in use: 256 - New / total deaths: 38.15 / 1,860.8 - Mean R: 0.838 - IFR: 2.41%
84  - 2020-05-15 - New / total infections: 641 / 113,253 - Hospital beds in use: 249 - New / total deaths: 36.89 / 1,897.7 - Mean R: 0.838 - IFR: 2.40%
85  - 2020-05-16 - New / total infections: 620 / 113,873 - Hospital beds in use: 241 - New / total deaths: 35.66 / 1,933.4 - Mean R: 0.838 - IFR: 2.38%
86  - 2020-05-17 - New / total infections: 599 / 114,473 - Hospital beds in use: 234 - New / total deaths: 34.46 / 1,967.8 - Mean R: 0.838 - IFR: 2.37%
87  - 2020-05-18 - New / total infections: 579 / 115,052 - Hospital beds in use: 227 - New / total deaths: 33.29 / 2,001.1 - Mean R: 0.838 - IFR: 2.36%
88  - 2020-05-19 - New / total infections: 559 / 115,611 - Hospital beds in use: 220 - New / total deaths: 32.15 / 2,033.3 - Mean R: 0.838 - IFR: 2.35%
89  - 2020-05-20 - New / total infections: 539 / 116,150 - Hospital beds in use: 213 - New / total deaths: 31.05 / 2,064.3 - Mean R: 0.838 - IFR: 2.34%
90  - 2020-05-21 - New / total infections: 520 / 116,670 - Hospital beds in use: 207 - New / total deaths: 29.98 / 2,094.3 - Mean R: 0.838 - IFR: 2.32%
91  - 2020-05-22 - New / total infections: 501 / 117,171 - Hospital beds in use: 201 - New / total deaths: 28.94 / 2,123.3 - Mean R: 0.837 - IFR: 2.31%
92  - 2020-05-23 - New / total infections: 483 / 117,654 - Hospital beds in use: 195 - New / total deaths: 27.94 / 2,151.2 - Mean R: 0.837 - IFR: 2.30%
93  - 2020-05-24 - New / total infections: 465 / 118,119 - Hospital beds in use: 189 - New / total deaths: 26.97 / 2,178.2 - Mean R: 0.837 - IFR: 2.29%
94  - 2020-05-25 - New / total infections: 447 / 118,565 - Hospital beds in use: 183 - New / total deaths: 26.03 / 2,204.2 - Mean R: 0.837 - IFR: 2.28%
95  - 2020-05-26 - New / total infections: 429 / 118,994 - Hospital beds in use: 177 - New / total deaths: 25.12 / 2,229.3 - Mean R: 0.837 - IFR: 2.27%
96  - 2020-05-27 - New / total infections: 412 / 119,406 - Hospital beds in use: 172 - New / total deaths: 24.24 / 2,253.6 - Mean R: 0.837 - IFR: 2.26%
97  - 2020-05-28 - New / total infections: 396 / 119,802 - Hospital beds in use: 166 - New / total deaths: 23.40 / 2,277.0 - Mean R: 0.837 - IFR: 2.24%
98  - 2020-05-29 - New / total infections: 381 / 120,183 - Hospital beds in use: 161 - New / total deaths: 22.58 / 2,299.5 - Mean R: 0.837 - IFR: 2.23%
99  - 2020-05-30 - New / total infections: 367 / 120,549 - Hospital beds in use: 156 - New / total deaths: 21.78 / 2,321.3 - Mean R: 0.837 - IFR: 2.22%
100 - 2020-05-31 - New / total infections: 353 / 120,902 - Hospital beds in use: 151 - New / total deaths: 21.02 / 2,342.3 - Mean R: 0.837 - IFR: 2.21%
101 - 2020-06-01 - New / total infections: 339 / 121,242 - Hospital beds in use: 146 - New / total deaths: 20.28 / 2,362.6 - Mean R: 0.837 - IFR: 2.20%
102 - 2020-06-02 - New / total infections: 327 / 121,568 - Hospital beds in use: 141 - New / total deaths: 19.56 / 2,382.2 - Mean R: 0.837 - IFR: 2.19%
103 - 2020-06-03 - New / total infections: 315 / 121,883 - Hospital beds in use: 136 - New / total deaths: 18.86 / 2,401.0 - Mean R: 0.837 - IFR: 2.18%
104 - 2020-06-04 - New / total infections: 303 / 122,186 - Hospital beds in use: 132 - New / total deaths: 18.19 / 2,419.2 - Mean R: 0.837 - IFR: 2.17%
105 - 2020-06-05 - New / total infections: 292 / 122,478 - Hospital beds in use: 127 - New / total deaths: 17.54 / 2,436.8 - Mean R: 0.837 - IFR: 2.16%
106 - 2020-06-06 - New / total infections: 282 / 122,760 - Hospital beds in use: 123 - New / total deaths: 16.90 / 2,453.7 - Mean R: 0.837 - IFR: 2.15%
107 - 2020-06-07 - New / total infections: 271 / 123,031 - Hospital beds in use: 119 - New / total deaths: 16.29 / 2,470.0 - Mean R: 0.836 - IFR: 2.14%
108 - 2020-06-08 - New / total infections: 262 / 123,293 - Hospital beds in use: 114 - New / total deaths: 15.70 / 2,485.7 - Mean R: 0.836 - IFR: 2.12%
109 - 2020-06-09 - New / total infections: 252 / 123,545 - Hospital beds in use: 110 - New / total deaths: 15.12 / 2,500.8 - Mean R: 0.836 - IFR: 2.11%
110 - 2020-06-10 - New / total infections: 244 / 123,789 - Hospital beds in use: 106 - New / total deaths: 14.56 / 2,515.3 - Mean R: 0.836 - IFR: 2.10%
111 - 2020-06-11 - New / total infections: 235 / 124,024 - Hospital beds in use: 102 - New / total deaths: 14.02 / 2,529.4 - Mean R: 0.836 - IFR: 2.09%
112 - 2020-06-12 - New / total infections: 227 / 124,251 - Hospital beds in use: 98 - New / total deaths: 13.49 / 2,542.8 - Mean R: 0.837 - IFR: 2.08%
113 - 2020-06-13 - New / total infections: 219 / 124,471 - Hospital beds in use: 95 - New / total deaths: 12.98 / 2,555.8 - Mean R: 0.837 - IFR: 2.07%
114 - 2020-06-14 - New / total infections: 212 / 124,683 - Hospital beds in use: 91 - New / total deaths: 12.48 / 2,568.3 - Mean R: 0.837 - IFR: 2.06%
115 - 2020-06-15 - New / total infections: 205 / 124,888 - Hospital beds in use: 87 - New / total deaths: 11.99 / 2,580.3 - Mean R: 0.837 - IFR: 2.05%
116 - 2020-06-16 - New / total infections: 198 / 125,086 - Hospital beds in use: 84 - New / total deaths: 11.52 / 2,591.8 - Mean R: 0.837 - IFR: 2.04%
117 - 2020-06-17 - New / total infections: 192 / 125,278 - Hospital beds in use: 81 - New / total deaths: 11.07 / 2,602.9 - Mean R: 0.838 - IFR: 2.03%
118 - 2020-06-18 - New / total infections: 186 / 125,465 - Hospital beds in use: 78 - New / total deaths: 10.63 / 2,613.5 - Mean R: 0.838 - IFR: 2.02%
119 - 2020-06-19 - New / total infections: 181 / 125,645 - Hospital beds in use: 75 - New / total deaths: 10.20 / 2,623.7 - Mean R: 0.839 - IFR: 2.01%
120 - 2020-06-20 - New / total infections: 175 / 125,820 - Hospital beds in use: 72 - New / total deaths: 9.79 / 2,633.5 - Mean R: 0.840 - IFR: 2.00%
121 - 2020-06-21 - New / total infections: 170 / 125,991 - Hospital beds in use: 69 - New / total deaths: 9.39 / 2,642.9 - Mean R: 0.841 - IFR: 1.99%
122 - 2020-06-22 - New / total infections: 166 / 126,156 - Hospital beds in use: 67 - New / total deaths: 9.00 / 2,651.9 - Mean R: 0.842 - IFR: 1.98%
123 - 2020-06-23 - New / total infections: 162 / 126,318 - Hospital beds in use: 64 - New / total deaths: 8.63 / 2,660.5 - Mean R: 0.844 - IFR: 1.97%
124 - 2020-06-24 - New / total infections: 158 / 126,476 - Hospital beds in use: 62 - New / total deaths: 8.27 / 2,668.8 - Mean R: 0.847 - IFR: 1.96%
125 - 2020-06-25 - New / total infections: 154 / 126,630 - Hospital beds in use: 60 - New / total deaths: 7.93 / 2,676.7 - Mean R: 0.850 - IFR: 1.95%
126 - 2020-06-26 - New / total infections: 151 / 126,782 - Hospital beds in use: 58 - New / total deaths: 7.60 / 2,684.3 - Mean R: 0.854 - IFR: 1.94%
127 - 2020-06-27 - New / total infections: 149 / 126,930 - Hospital beds in use: 55 - New / total deaths: 7.29 / 2,691.6 - Mean R: 0.859 - IFR: 1.93%
128 - 2020-06-28 - New / total infections: 147 / 127,077 - Hospital beds in use: 54 - New / total deaths: 6.98 / 2,698.6 - Mean R: 0.864 - IFR: 1.92%
129 - 2020-06-29 - New / total infections: 145 / 127,223 - Hospital beds in use: 52 - New / total deaths: 6.69 / 2,705.3 - Mean R: 0.870 - IFR: 1.91%
130 - 2020-06-30 - New / total infections: 144 / 127,366 - Hospital beds in use: 50 - New / total deaths: 6.42 / 2,711.7 - Mean R: 0.876 - IFR: 1.90%
131 - 2020-07-01 - New / total infections: 143 / 127,509 - Hospital beds in use: 48 - New / total deaths: 6.15 / 2,717.8 - Mean R: 0.882 - IFR: 1.89%
132 - 2020-07-02 - New / total infections: 142 / 127,651 - Hospital beds in use: 47 - New / total deaths: 5.90 / 2,723.7 - Mean R: 0.888 - IFR: 1.88%
133 - 2020-07-03 - New / total infections: 141 / 127,793 - Hospital beds in use: 45 - New / total deaths: 5.66 / 2,729.4 - Mean R: 0.893 - IFR: 1.87%
134 - 2020-07-04 - New / total infections: 141 / 127,934 - Hospital beds in use: 44 - New / total deaths: 5.42 / 2,734.8 - Mean R: 0.898 - IFR: 1.86%
135 - 2020-07-05 - New / total infections: 141 / 128,074 - Hospital beds in use: 42 - New / total deaths: 5.20 / 2,740.0 - Mean R: 0.902 - IFR: 1.86%
136 - 2020-07-06 - New / total infections: 141 / 128,215 - Hospital beds in use: 41 - New / total deaths: 4.99 / 2,745.0 - Mean R: 0.905 - IFR: 1.85%
137 - 2020-07-07 - New / total infections: 140 / 128,355 - Hospital beds in use: 40 - New / total deaths: 4.79 / 2,749.8 - Mean R: 0.907 - IFR: 1.84%
138 - 2020-07-08 - New / total infections: 140 / 128,496 - Hospital beds in use: 38 - New / total deaths: 4.60 / 2,754.4 - Mean R: 0.909 - IFR: 1.83%
139 - 2020-07-09 - New / total infections: 140 / 128,636 - Hospital beds in use: 37 - New / total deaths: 4.42 / 2,758.8 - Mean R: 0.911 - IFR: 1.82%
140 - 2020-07-10 - New / total infections: 140 / 128,776 - Hospital beds in use: 36 - New / total deaths: 4.25 / 2,763.1 - Mean R: 0.912 - IFR: 1.81%
141 - 2020-07-11 - New / total infections: 140 / 128,917 - Hospital beds in use: 35 - New / total deaths: 4.08 / 2,767.2 - Mean R: 0.913 - IFR: 1.80%
142 - 2020-07-12 - New / total infections: 140 / 129,057 - Hospital beds in use: 35 - New / total deaths: 3.93 / 2,771.1 - Mean R: 0.914 - IFR: 1.76%
143 - 2020-07-13 - New / total infections: 140 / 129,197 - Hospital beds in use: 34 - New / total deaths: 3.77 / 2,774.9 - Mean R: 0.914 - IFR: 1.71%
144 - 2020-07-14 - New / total infections: 140 / 129,337 - Hospital beds in use: 33 - New / total deaths: 3.62 / 2,778.5 - Mean R: 0.915 - IFR: 1.67%
145 - 2020-07-15 - New / total infections: 140 / 129,478 - Hospital beds in use: 33 - New / total deaths: 3.46 / 2,781.9 - Mean R: 0.915 - IFR: 1.63%
146 - 2020-07-16 - New / total infections: 140 / 129,618 - Hospital beds in use: 32 - New / total deaths: 3.32 / 2,785.3 - Mean R: 0.915 - IFR: 1.59%
147 - 2020-07-17 - New / total infections: 140 / 129,759 - Hospital beds in use: 32 - New / total deaths: 3.17 / 2,788.4 - Mean R: 0.915 - IFR: 1.55%
148 - 2020-07-18 - New / total infections: 141 / 129,899 - Hospital beds in use: 31 - New / total deaths: 3.03 / 2,791.5 - Mean R: 0.915 - IFR: 1.51%
149 - 2020-07-19 - New / total infections: 141 / 130,040 - Hospital beds in use: 31 - New / total deaths: 2.90 / 2,794.4 - Mean R: 0.916 - IFR: 1.47%
150 - 2020-07-20 - New / total infections: 141 / 130,181 - Hospital beds in use: 31 - New / total deaths: 2.78 / 2,797.2 - Mean R: 0.916 - IFR: 1.43%
151 - 2020-07-21 - New / total infections: 141 / 130,321 - Hospital beds in use: 31 - New / total deaths: 2.66 / 2,799.8 - Mean R: 0.916 - IFR: 1.40%
152 - 2020-07-22 - New / total infections: 141 / 130,462 - Hospital beds in use: 31 - New / total deaths: 2.55 / 2,802.4 - Mean R: 0.916 - IFR: 1.36%
153 - 2020-07-23 - New / total infections: 141 / 130,603 - Hospital beds in use: 31 - New / total deaths: 2.44 / 2,804.8 - Mean R: 0.916 - IFR: 1.33%
154 - 2020-07-24 - New / total infections: 141 / 130,744 - Hospital beds in use: 31 - New / total deaths: 2.35 / 2,807.2 - Mean R: 0.916 - IFR: 1.30%
155 - 2020-07-25 - New / total infections: 141 / 130,885 - Hospital beds in use: 30 - New / total deaths: 2.25 / 2,809.4 - Mean R: 0.916 - IFR: 1.26%
156 - 2020-07-26 - New / total infections: 141 / 131,026 - Hospital beds in use: 30 - New / total deaths: 2.17 / 2,811.6 - Mean R: 0.916 - IFR: 1.23%
157 - 2020-07-27 - New / total infections: 141 / 131,167 - Hospital beds in use: 30 - New / total deaths: 2.09 / 2,813.7 - Mean R: 0.916 - IFR: 1.20%
158 - 2020-07-28 - New / total infections: 141 / 131,308 - Hospital beds in use: 30 - New / total deaths: 2.01 / 2,815.7 - Mean R: 0.916 - IFR: 1.17%
159 - 2020-07-29 - New / total infections: 141 / 131,449 - Hospital beds in use: 30 - New / total deaths: 1.94 / 2,817.6 - Mean R: 0.916 - IFR: 1.14%
160 - 2020-07-30 - New / total infections: 141 / 131,590 - Hospital beds in use: 30 - New / total deaths: 1.88 / 2,819.5 - Mean R: 0.916 - IFR: 1.11%
161 - 2020-07-31 - New / total infections: 141 / 131,731 - Hospital beds in use: 30 - New / total deaths: 1.81 / 2,821.3 - Mean R: 0.916 - IFR: 1.09%
162 - 2020-08-01 - New / total infections: 141 / 131,873 - Hospital beds in use: 30 - New / total deaths: 1.76 / 2,823.1 - Mean R: 0.916 - IFR: 1.06%
163 - 2020-08-02 - New / total infections: 141 / 132,014 - Hospital beds in use: 30 - New / total deaths: 1.70 / 2,824.8 - Mean R: 0.916 - IFR: 1.03%
164 - 2020-08-03 - New / total infections: 141 / 132,155 - Hospital beds in use: 30 - New / total deaths: 1.65 / 2,826.4 - Mean R: 0.916 - IFR: 1.01%
165 - 2020-08-04 - New / total infections: 141 / 132,297 - Hospital beds in use: 30 - New / total deaths: 1.60 / 2,828.0 - Mean R: 0.916 - IFR: 0.98%
166 - 2020-08-05 - New / total infections: 141 / 132,438 - Hospital beds in use: 30 - New / total deaths: 1.55 / 2,829.6 - Mean R: 0.916 - IFR: 0.96%
167 - 2020-08-06 - New / total infections: 142 / 132,580 - Hospital beds in use: 30 - New / total deaths: 1.51 / 2,831.1 - Mean R: 0.916 - IFR: 0.94%
168 - 2020-08-07 - New / total infections: 142 / 132,721 - Hospital beds in use: 30 - New / total deaths: 1.47 / 2,832.5 - Mean R: 0.915 - IFR: 0.94%
169 - 2020-08-08 - New / total infections: 142 / 132,863 - Hospital beds in use: 30 - New / total deaths: 1.43 / 2,834.0 - Mean R: 0.915 - IFR: 0.94%
170 - 2020-08-09 - New / total infections: 142 / 133,005 - Hospital beds in use: 30 - New / total deaths: 1.40 / 2,835.4 - Mean R: 0.915 - IFR: 0.94%
171 - 2020-08-10 - New / total infections: 142 / 133,146 - Hospital beds in use: 30 - New / total deaths: 1.38 / 2,836.7 - Mean R: 0.915 - IFR: 0.94%
172 - 2020-08-11 - New / total infections: 142 / 133,288 - Hospital beds in use: 30 - New / total deaths: 1.36 / 2,838.1 - Mean R: 0.915 - IFR: 0.94%
173 - 2020-08-12 - New / total infections: 142 / 133,430 - Hospital beds in use: 31 - New / total deaths: 1.34 / 2,839.4 - Mean R: 0.915 - IFR: 0.94%
174 - 2020-08-13 - New / total infections: 142 / 133,572 - Hospital beds in use: 31 - New / total deaths: 1.32 / 2,840.8 - Mean R: 0.915 - IFR: 0.94%
175 - 2020-08-14 - New / total infections: 142 / 133,714 - Hospital beds in use: 31 - New / total deaths: 1.31 / 2,842.1 - Mean R: 0.915 - IFR: 0.94%
176 - 2020-08-15 - New / total infections: 142 / 133,855 - Hospital beds in use: 31 - New / total deaths: 1.30 / 2,843.4 - Mean R: 0.915 - IFR: 0.94%
177 - 2020-08-16 - New / total infections: 142 / 133,997 - Hospital beds in use: 31 - New / total deaths: 1.29 / 2,844.7 - Mean R: 0.915 - IFR: 0.94%
178 - 2020-08-17 - New / total infections: 142 / 134,139 - Hospital beds in use: 31 - New / total deaths: 1.29 / 2,846.0 - Mean R: 0.915 - IFR: 0.94%
179 - 2020-08-18 - New / total infections: 142 / 134,281 - Hospital beds in use: 31 - New / total deaths: 1.28 / 2,847.2 - Mean R: 0.915 - IFR: 0.94%
180 - 2020-08-19 - New / total infections: 142 / 134,423 - Hospital beds in use: 31 - New / total deaths: 1.28 / 2,848.5 - Mean R: 0.915 - IFR: 0.94%
181 - 2020-08-20 - New / total infections: 142 / 134,565 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,849.8 - Mean R: 0.915 - IFR: 0.94%
182 - 2020-08-21 - New / total infections: 142 / 134,707 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,851.1 - Mean R: 0.915 - IFR: 0.94%
183 - 2020-08-22 - New / total infections: 142 / 134,850 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,852.3 - Mean R: 0.915 - IFR: 0.94%
184 - 2020-08-23 - New / total infections: 142 / 134,992 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,853.6 - Mean R: 0.915 - IFR: 0.94%
185 - 2020-08-24 - New / total infections: 142 / 135,134 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,854.9 - Mean R: 0.916 - IFR: 0.94%
186 - 2020-08-25 - New / total infections: 142 / 135,277 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,856.1 - Mean R: 0.917 - IFR: 0.94%
187 - 2020-08-26 - New / total infections: 143 / 135,419 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,857.4 - Mean R: 0.918 - IFR: 0.94%
188 - 2020-08-27 - New / total infections: 143 / 135,562 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,858.6 - Mean R: 0.919 - IFR: 0.94%
189 - 2020-08-28 - New / total infections: 143 / 135,705 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,859.9 - Mean R: 0.920 - IFR: 0.94%
190 - 2020-08-29 - New / total infections: 143 / 135,848 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,861.2 - Mean R: 0.921 - IFR: 0.94%
191 - 2020-08-30 - New / total infections: 144 / 135,992 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,862.4 - Mean R: 0.922 - IFR: 0.94%
192 - 2020-08-31 - New / total infections: 144 / 136,136 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,863.7 - Mean R: 0.923 - IFR: 0.94%
193 - 2020-09-01 - New / total infections: 144 / 136,280 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,864.9 - Mean R: 0.924 - IFR: 0.94%
194 - 2020-09-02 - New / total infections: 145 / 136,425 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,866.2 - Mean R: 0.925 - IFR: 0.94%
195 - 2020-09-03 - New / total infections: 145 / 136,569 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,867.5 - Mean R: 0.925 - IFR: 0.94%
196 - 2020-09-04 - New / total infections: 145 / 136,715 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,868.7 - Mean R: 0.926 - IFR: 0.94%
197 - 2020-09-05 - New / total infections: 146 / 136,860 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,870.0 - Mean R: 0.927 - IFR: 0.94%
198 - 2020-09-06 - New / total infections: 146 / 137,007 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,871.2 - Mean R: 0.928 - IFR: 0.94%
199 - 2020-09-07 - New / total infections: 147 / 137,153 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,872.5 - Mean R: 0.929 - IFR: 0.94%
200 - 2020-09-08 - New / total infections: 147 / 137,300 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,873.8 - Mean R: 0.930 - IFR: 0.94%
201 - 2020-09-09 - New / total infections: 148 / 137,448 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,875.0 - Mean R: 0.931 - IFR: 0.94%
202 - 2020-09-10 - New / total infections: 148 / 137,596 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,876.3 - Mean R: 0.932 - IFR: 0.94%
203 - 2020-09-11 - New / total infections: 149 / 137,745 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,877.6 - Mean R: 0.933 - IFR: 0.94%
204 - 2020-09-12 - New / total infections: 149 / 137,894 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,878.8 - Mean R: 0.934 - IFR: 0.94%
205 - 2020-09-13 - New / total infections: 150 / 138,044 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,880.1 - Mean R: 0.935 - IFR: 0.94%
206 - 2020-09-14 - New / total infections: 150 / 138,194 - Hospital beds in use: 31 - New / total deaths: 1.26 / 2,881.3 - Mean R: 0.936 - IFR: 0.94%
207 - 2020-09-15 - New / total infections: 151 / 138,345 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,882.6 - Mean R: 0.937 - IFR: 0.94%
208 - 2020-09-16 - New / total infections: 152 / 138,497 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,883.9 - Mean R: 0.938 - IFR: 0.94%
209 - 2020-09-17 - New / total infections: 152 / 138,650 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,885.1 - Mean R: 0.939 - IFR: 0.94%
210 - 2020-09-18 - New / total infections: 153 / 138,803 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,886.4 - Mean R: 0.939 - IFR: 0.94%
211 - 2020-09-19 - New / total infections: 154 / 138,957 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,887.7 - Mean R: 0.940 - IFR: 0.94%
212 - 2020-09-20 - New / total infections: 155 / 139,111 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,888.9 - Mean R: 0.941 - IFR: 0.94%
213 - 2020-09-21 - New / total infections: 156 / 139,267 - Hospital beds in use: 31 - New / total deaths: 1.27 / 2,890.2 - Mean R: 0.942 - IFR: 0.94%
215 - 2020-09-23 - New / total infections: 157 / 139,580 - Hospital beds in use: 32 - New / total deaths: 1.27 / 2,892.8 - Mean R: 0.944 - IFR: 0.94%
216 - 2020-09-24 - New / total infections: 158 / 139,739 - Hospital beds in use: 32 - New / total deaths: 1.28 / 2,894.0 - Mean R: 0.945 - IFR: 0.94%
217 - 2020-09-25 - New / total infections: 159 / 139,897 - Hospital beds in use: 32 - New / total deaths: 1.28 / 2,895.3 - Mean R: 0.946 - IFR: 0.94%
218 - 2020-09-26 - New / total infections: 160 / 140,057 - Hospital beds in use: 32 - New / total deaths: 1.28 / 2,896.6 - Mean R: 0.947 - IFR: 0.94%
219 - 2020-09-27 - New / total infections: 161 / 140,218 - Hospital beds in use: 32 - New / total deaths: 1.28 / 2,897.9 - Mean R: 0.948 - IFR: 0.94%
220 - 2020-09-28 - New / total infections: 162 / 140,380 - Hospital beds in use: 32 - New / total deaths: 1.29 / 2,899.2 - Mean R: 0.949 - IFR: 0.94%
221 - 2020-09-29 - New / total infections: 163 / 140,542 - Hospital beds in use: 32 - New / total deaths: 1.29 / 2,900.5 - Mean R: 0.950 - IFR: 0.94%
222 - 2020-09-30 - New / total infections: 164 / 140,706 - Hospital beds in use: 32 - New / total deaths: 1.29 / 2,901.8 - Mean R: 0.951 - IFR: 0.94%
223 - 2020-10-01 - New / total infections: 165 / 140,871 - Hospital beds in use: 32 - New / total deaths: 1.29 / 2,903.0 - Mean R: 0.952 - IFR: 0.94%
224 - 2020-10-02 - New / total infections: 166 / 141,037 - Hospital beds in use: 33 - New / total deaths: 1.30 / 2,904.3 - Mean R: 0.953 - IFR: 0.94%
225 - 2020-10-03 - New / total infections: 167 / 141,204 - Hospital beds in use: 33 - New / total deaths: 1.30 / 2,905.6 - Mean R: 0.954 - IFR: 0.94%
226 - 2020-10-04 - New / total infections: 168 / 141,372 - Hospital beds in use: 33 - New / total deaths: 1.31 / 2,907.0 - Mean R: 0.955 - IFR: 0.94%
227 - 2020-10-05 - New / total infections: 169 / 141,541 - Hospital beds in use: 33 - New / total deaths: 1.31 / 2,908.3 - Mean R: 0.956 - IFR: 0.94%
228 - 2020-10-06 - New / total infections: 170 / 141,711 - Hospital beds in use: 33 - New / total deaths: 1.31 / 2,909.6 - Mean R: 0.957 - IFR: 0.94%
229 - 2020-10-07 - New / total infections: 172 / 141,883 - Hospital beds in use: 33 - New / total deaths: 1.32 / 2,910.9 - Mean R: 0.958 - IFR: 0.94%
230 - 2020-10-08 - New / total infections: 173 / 142,056 - Hospital beds in use: 34 - New / total deaths: 1.32 / 2,912.2 - Mean R: 0.959 - IFR: 0.94%
231 - 2020-10-09 - New / total infections: 174 / 142,230 - Hospital beds in use: 34 - New / total deaths: 1.33 / 2,913.5 - Mean R: 0.960 - IFR: 0.94%
232 - 2020-10-10 - New / total infections: 175 / 142,405 - Hospital beds in use: 34 - New / total deaths: 1.33 / 2,914.9 - Mean R: 0.960 - IFR: 0.94%
233 - 2020-10-11 - New / total infections: 177 / 142,582 - Hospital beds in use: 34 - New / total deaths: 1.34 / 2,916.2 - Mean R: 0.961 - IFR: 0.94%
234 - 2020-10-12 - New / total infections: 178 / 142,760 - Hospital beds in use: 34 - New / total deaths: 1.34 / 2,917.6 - Mean R: 0.962 - IFR: 0.94%
235 - 2020-10-13 - New / total infections: 179 / 142,939 - Hospital beds in use: 34 - New / total deaths: 1.35 / 2,918.9 - Mean R: 0.963 - IFR: 0.94%
236 - 2020-10-14 - New / total infections: 181 / 143,120 - Hospital beds in use: 35 - New / total deaths: 1.36 / 2,920.3 - Mean R: 0.964 - IFR: 0.94%
237 - 2020-10-15 - New / total infections: 182 / 143,302 - Hospital beds in use: 35 - New / total deaths: 1.36 / 2,921.6 - Mean R: 0.965 - IFR: 0.94%
238 - 2020-10-16 - New / total infections: 184 / 143,485 - Hospital beds in use: 35 - New / total deaths: 1.37 / 2,923.0 - Mean R: 0.966 - IFR: 0.94%
239 - 2020-10-17 - New / total infections: 185 / 143,671 - Hospital beds in use: 35 - New / total deaths: 1.38 / 2,924.4 - Mean R: 0.967 - IFR: 0.94%
240 - 2020-10-18 - New / total infections: 187 / 143,857 - Hospital beds in use: 36 - New / total deaths: 1.38 / 2,925.8 - Mean R: 0.968 - IFR: 0.94%
241 - 2020-10-19 - New / total infections: 188 / 144,045 - Hospital beds in use: 36 - New / total deaths: 1.39 / 2,927.1 - Mean R: 0.969 - IFR: 0.94%
242 - 2020-10-20 - New / total infections: 190 / 144,235 - Hospital beds in use: 36 - New / total deaths: 1.40 / 2,928.5 - Mean R: 0.970 - IFR: 0.94%
243 - 2020-10-21 - New / total infections: 191 / 144,427 - Hospital beds in use: 36 - New / total deaths: 1.40 / 2,929.9 - Mean R: 0.971 - IFR: 0.94%
244 - 2020-10-22 - New / total infections: 193 / 144,620 - Hospital beds in use: 36 - New / total deaths: 1.41 / 2,931.4 - Mean R: 0.972 - IFR: 0.94%
245 - 2020-10-23 - New / total infections: 195 / 144,815 - Hospital beds in use: 37 - New / total deaths: 1.42 / 2,932.8 - Mean R: 0.973 - IFR: 0.94%
246 - 2020-10-24 - New / total infections: 197 / 145,011 - Hospital beds in use: 37 - New / total deaths: 1.43 / 2,934.2 - Mean R: 0.974 - IFR: 0.94%
247 - 2020-10-25 - New / total infections: 198 / 145,210 - Hospital beds in use: 37 - New / total deaths: 1.44 / 2,935.6 - Mean R: 0.975 - IFR: 0.94%
248 - 2020-10-26 - New / total infections: 200 / 145,410 - Hospital beds in use: 38 - New / total deaths: 1.44 / 2,937.1 - Mean R: 0.976 - IFR: 0.94%
249 - 2020-10-27 - New / total infections: 202 / 145,612 - Hospital beds in use: 38 - New / total deaths: 1.45 / 2,938.5 - Mean R: 0.977 - IFR: 0.94%
250 - 2020-10-28 - New / total infections: 204 / 145,816 - Hospital beds in use: 38 - New / total deaths: 1.46 / 2,940.0 - Mean R: 0.978 - IFR: 0.94%
251 - 2020-10-29 - New / total infections: 206 / 146,021 - Hospital beds in use: 38 - New / total deaths: 1.47 / 2,941.5 - Mean R: 0.979 - IFR: 0.94%
252 - 2020-10-30 - New / total infections: 208 / 146,229 - Hospital beds in use: 39 - New / total deaths: 1.48 / 2,943.0 - Mean R: 0.980 - IFR: 0.94%
253 - 2020-10-31 - New / total infections: 210 / 146,439 - Hospital beds in use: 39 - New / total deaths: 1.49 / 2,944.4 - Mean R: 0.981 - IFR: 0.94%
254 - 2020-11-01 - New / total infections: 212 / 146,651 - Hospital beds in use: 39 - New / total deaths: 1.50 / 2,945.9 - Mean R: 0.982 - IFR: 0.94%
-------------------------------------
End of simulation       : 2020-11-01
Total infections        : 146,651
Peak hospital beds used : 566
Total deaths            : 2,946
====================================================
Done - Current time: 2020-10-12 05:40:09.630126