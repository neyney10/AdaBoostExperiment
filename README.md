# AdaBoost Experiment
## Files
- main.py - the program's entry point, run this file in order to start the experiments, the details of the experiments and what we are trying to achieve along with the results can be found in main.py header.
- rules.py - contains all rules used such as LineRule for 2D-line that have ```classify``` method that returns 1 (for positive) or -1 (for negative), the file contains also Line Rules Generator/Finder in order to compute all possible line classifiers for given data.
- data_source.py - contains data sources such as IRIS and HC-Body-Temp, along with parsing and transforming the data to fit the adaboost constraints (such as labels {-1,1}).
- experiment.py - the experiment to execute/run.
- test.py - contains few tests for LineRule classes to test that lines classifiy 2D points correctly.
- adaboost.py - the adaboost algorithm.