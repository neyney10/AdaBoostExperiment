############################
# Python Ver: 3.8.5.       #
# Date: 17/12/2020.        #
# Author: Ofek Bader.      #
# ------------------------ # 
# Dependencies:            #
# - numpy==1.18.5          #  
# - pandas==1.1.1          #  
############################

''' Results for running the experiment (for tables style: https://ozh.github.io/ascii-tables/)
Configuration:
- 8 iteration of adaboost (8 rules)
- 100 runs of the experiment.

Dataset: Iris
    Emperical Error (over train set)
    ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬───────────┬────────────┐
    │     H1      │     H2      │     H3      │     H4      │     H5      │     H6      │    H7     │     H8     │
    ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼───────────┼────────────┤
    │ 0.047555557 │ 0.054666667 │ 0.041777667 │ 0.039555557 │ 0.023555567 │ 0.023555567 │ 0.0168889 │ 0.01511111 │
    └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴───────────┴────────────┘

    True Error (over test set)
    ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬───────┬─────────────┐
    │     H1      │     H2      │     H3      │     H4      │     H5      │     H6      │  H7   │     H8      │
    ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼───────┼─────────────┤
    │ 0.092444333 │ 0.090666667 │ 0.089333333 │ 0.092888667 │ 0.105333333 │ 0.094666667 │ 0.104 │ 0.089776667 │
    └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴───────┴─────────────┘

Dataset: HCBT
    Emperical Error (over train set)
    ┌─────────────┬─────────────┬────────────┬────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    │     H1      │     H2      │     H3     │     H4     │     H5      │     H6      │     H7      │     H8      │
    ├─────────────┼─────────────┼────────────┼────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
    │ 0.375383333 │ 0.403102567 │ 0.34051282 │ 0.34356154 │ 0.320128203 │ 0.327179487 │ 0.292205127 │ 0.308720513 │
    └─────────────┴─────────────┴────────────┴────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

    True Error (over test set)
    ┌────────────┬────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    │     H1     │     H2     │     H3      │     H4      │     H5      │     H6      │     H7      │     H8      │
    ├────────────┼────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
    │ 0.47690513 │ 0.51071795 │ 0.465666667 │ 0.463579487 │ 0.454346153 │ 0.482048717 │ 0.448194873 │ 0.478971793 │
    └────────────┴────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

Raw Printouts:
[[0.375383333 0.47690513 ] 
 [0.403102567 0.51071795 ] 
 [0.34051282  0.465666667]
 [0.34356154  0.463579487]
 [0.320128203 0.454346153]
 [0.327179487 0.482048717]
 [0.292205127 0.448194873]
 [0.308720513 0.478971793]]
[[0.047555557 0.092444333]
 [0.054666667 0.090666667]
 [0.041777667 0.089333333]
 [0.039555557 0.092888667]
 [0.023555567 0.105333333]
 [0.023555567 0.094666667]
 [0.0168889   0.104      ]
 [0.01511111  0.089776667]]


Do we see overfitting?
Yes, we can see that the emperical error is decreasing but the true error is not.
'''

# Notice: written following some principals of OOP instead of functional.

# imports #
from data_source import HCBTDataSource, IrisDataSource
from experiment import Experiment

# main run #
print("Starting...")

# declare data sources
irisDataSource = IrisDataSource('data/iris.data')
hcbtDataSource = HCBTDataSource('data/HC_Body_Temperature.txt')
data_sources = [hcbtDataSource, irisDataSource]

# declare experiment for each data source
experiments = list()
for data_source in data_sources:
    experiment = Experiment(data_source, 8, 100)
    experiments.append(experiment)

# run experiments and store each experiment's result in a list.
results = list()
for experiment in experiments:
    result = experiment.result()
    results.append(result)

# print results
for result in results:
    print(result)



