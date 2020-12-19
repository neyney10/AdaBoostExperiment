############################
# Python Ver: 3.8.5.       #
# Date: 13/12/2020.        #
# Author: Ofek Bader.      #
############################

# imports #
from rules import AllPossibleLineRulesFinder, LineRule
import pandas as pd
import unittest


class TestLineRule(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_classification_1(self):
        point = [0,0]
        rule = LineRule(1,1)
        
        self.assertEqual(rule.classify(point), -1, "Should classify it as -1")

    def test_classification_2(self):
        point = [2,2]
        rule = LineRule(1,1)
        
        self.assertEqual(rule.classify(point), -1, "Should classify it as -1")

    def test_classification_2(self):
        point = [0.5,2]
        rule = LineRule(1,1)
        
        self.assertEqual(rule.classify(point), +1, "Should classify it as +1")

    def test_classification_4(self):
        point = [1,1]
        rule = LineRule(0.5,0)
        
        self.assertEqual(rule.classify(point), +1, "Should classify it as +1")


if __name__ == '__main__':
    # main run #
    # load data
    data = pd.read_csv('data/iris.data', 
                        header=None, 
                        index_col=False, 
                        names=['c1','c2','c3','label'], 
                        usecols=[1,2,3])

    data_points_only = data[['c2','c3']]
    unittest.main()
