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
    ┌───────────┬───────┬─────────┬────────────┬───────────┬──────────┬───────────┬───────────┐
    │    H1     │  H2   │   H3    │     H4     │    H5     │    H6    │    H7     │    H8     │
    ├───────────┼───────┼─────────┼────────────┼───────────┼──────────┼───────────┼───────────┤
    │ 4.266667% │ 5.2%  │ 3.3333% │  3.066667% │  2.66667% │ 1.86667% │ 1.866667% │ 1.733333% │
    └───────────┴───────┴─────────┴────────────┴───────────┴──────────┴───────────┴───────────┘

    True Error (over test set)
    ┌─────────┬─────┬──────┬────────────┬───────┬─────┬──────┬───────────┐
    │   H1    │ H2  │  H3  │     H4     │  H5   │ H6  │  H7  │    H8     │
    ├─────────┼─────┼──────┼────────────┼───────┼─────┼──────┼───────────┤
    │ 9.7333% │ 10% │ 9.6% │ 10.266667% │ 0.92% │ 10% │ 8.8% │ 9.733333% │
    └─────────┴─────┴──────┴────────────┴───────┴─────┴──────┴───────────┘

Dataset: HCBT
    Emperical Error (over train set)
    ┌─────────┬───────┬─────┬────────┬───────┬─────┬───────┬────────┐
    │   H1    │  H2   │ H3  │   H4   │  H5   │ H6  │  H7   │   H8   │
    ├─────────┼───────┼─────┼────────┼───────┼─────┼───────┼────────┤
    │ 36.615% │ 39.7% │ 32% │ 33.53% │ 30.5% │ 32% │ 29.2% │ 30.77% │
    └─────────┴───────┴─────┴────────┴───────┴─────┴───────┴────────┘

    True Error (over test set)
    ┌────────┬───────┬───────┬────────┬────────┬────────┬────────┬────────┐
    │   H1   │  H2   │  H3   │   H4   │   H5   │   H6   │   H7   │   H8   │
    ├────────┼───────┼───────┼────────┼────────┼────────┼────────┼────────┤
    │ 48.61% │ 52.6% │ 47.7% │ 48.92% │ 46.15% │ 49.23% │ 44.92% │ 49.23% │
    └────────┴───────┴───────┴────────┴────────┴────────┴────────┴────────┘

Do we see overfitting?
No, we dont see overfitting as the True Error isn't that far from the Empirical Error.
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
