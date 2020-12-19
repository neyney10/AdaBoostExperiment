import numpy as np
import sys

class AdaBoost:
    '''
    Accepts 2d-data (with labels) and binary classifier rules.
    The data is 3-column with first and second colums as x,y and the third as a label.
    '''
    def __init__(self, data, rules, num_of_iterations) -> None:
        self.data = data
        self.rules = rules
        self.num_of_iterations = num_of_iterations

    def boost(self):
        # initialize point weights.
        weights = np.full(len(self.data.index), 1/len(self.data.index))
        # initialize empty best rules found list.
        best_rules = list()
        rules_weight = list()
        # iterate.
        for i in range(self.num_of_iterations):
            # compute weighted error for each rule.
            min_weighted_error_rule = self.rules[0]
            min_weighted_error = sys.maxsize
            for rule in self.rules:
                weighted_error = 0
                classification_results = self.data.apply(lambda row: rule.classify(row) == row[2], axis=1, result_type='reduce')
                for index in classification_results[classification_results == False].index:
                        weighted_error += weights[index]
                if weighted_error <= min_weighted_error:
                    min_weighted_error = weighted_error
                    min_weighted_error_rule = rule
            best_rules.append(min_weighted_error_rule)
            # compute classifier weight based on min weighted error.
            min_weighted_error = 0.999 if min_weighted_error >= 1 else min_weighted_error # fix min_weighted_error invalid >=1 because of precision error.
            min_weighted_error = 0.001 if min_weighted_error <= 0 else min_weighted_error # fix min_weighted_error invalid <=0 because of precision error.
            classifier_weight = 0.5 * np.log((1-min_weighted_error)/(min_weighted_error))
            rules_weight.append(classifier_weight)
            # update point weights.
            for index,point_with_label in self.data.iterrows():
                weights[index] *= (np.e**(-classifier_weight * min_weighted_error_rule.classify(point_with_label) * point_with_label[2]))
                weights /= np.sum(weights) # normalize by Z
        
        return (best_rules, rules_weight)