import numpy as np
from adaboost import AdaBoost
from rules import AllPossibleLineRulesFinder, BoostedRule

class Experiment:
    def __init__(self, data_source, adaboost_iteration_amount, experiment_iteration_amount) -> None:
        self.data_source = data_source
        self.adaboost_iteration_amount = adaboost_iteration_amount
        self.experiment_iteration_amount = experiment_iteration_amount


    def result(self):
        '''
        Computes and returns the experiment result
        What is the experiment? 
        1. Given a data source with 2d data points and a label, divide the data randomly into two sets:
            - train set 50%
            - test set 50%
        2. Compute all possible bi-directional classifier line rules.
        3. Run AdaBoost algorithm with adaboost_iteration_amount best rules (iterations) to get rule-weights.
        4. Use the rules with rule weights to get boosted rule, for each k in [1,2,...,adaboost_iteration_amount]
            create a boosted rule from all <k first rules.
        5. Calculate true and emperical error for each k.
        6. Run steps 1-5 experiment_iteration_amount of times.
        7. Get average true and emperical error over all iterations.
        '''

        # read data
        data = self.data_source.read()

        # generate rules
        rule_finder = AllPossibleLineRulesFinder(data) # [['c2','c3']]
        rules = rule_finder.find()
        
        results = np.zeros((self.adaboost_iteration_amount,2)) # For each k out of adaboost_iteration_amount we store a sum of emperical and true error
        for i in range(self.experiment_iteration_amount):
            true_err, emp_error = self._single_run(rules, data)
            results[:,0] += true_err
            results[:,1] += emp_error
        
        # compute average of each error sum
        results /= self.experiment_iteration_amount

        return results

    def _single_run(self, rules, data):
        # split data
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        train_data, test_data = np.vsplit(shuffled_data, 2) # split vertically.
        # run AdaBoost
        adaboost = AdaBoost(train_data, rules, self.adaboost_iteration_amount)
        best_rules, weights = adaboost.boost()
        # test model performance.
        ## calculate true error (over test set) & calculate empirical error (over train set)
        empirical_errors = list()
        true_errors = list()
        for k in range(1,1+len(best_rules)):
            boosted_rule = BoostedRule(best_rules[:k], weights)
            test_results = test_data.apply(lambda row: boosted_rule.classify(row) == row[2], axis=1, result_type='reduce')
            true_error = len(test_results[test_results == False]) / len(test_results)
            true_errors.append(true_error)
            train_results = train_data.apply(lambda row: boosted_rule.classify(row) == row[2], axis=1, result_type='reduce')
            empirical_error = len(train_results[train_results == False]) / len(train_results)
            empirical_errors.append(empirical_error)

        return (true_errors, empirical_errors)