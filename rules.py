import numpy as np

class LineRule:
    '''
    m = slope
    y0 = y value of the point (x,y) of the intersection with Y-axis.
    direction = how to classify positive+ (above or below line), 
                1 - if above = positive.
                -1 - if above = negative.
    '''
    def __init__(self, m, y0, direction = 1) -> None:
        self.m = m
        self.y0 = y0
        self.direction = direction

    def classify(self, point):
        classification = 0
        if point[1] >= self.m*point[0] + self.y0:
            classification = +1
        else:
            classification = -1
        
        return classification * self.direction

class LineRuleParallelToYAxis:
    def __init__(self, x0, direction = 1) -> None:
        self.x0 = x0
        self.direction = direction

    def classify(self, point):
        classification = 0
        if point[1] >= self.x0:
            classification = +1
        else:
            classification = -1

        return classification * self.direction

class AllPossibleLineRulesFinder:
    '''
    Accepts 2D-data.
    '''
    def __init__(self, data) -> None:
        self.data = data

    def find(self) -> list:
        ''' 
        Outline:
        1. get 2 points (by iterating on all possible pairs)
        2. compute line (for each possible pair)
        '''
        lines = list()
        for i1,p1 in enumerate(self.data.values):
            for i2,p2 in enumerate(self.data.values):
                if i1 == i2:
                    continue
                median_point = [np.average([p1[0],p2[0]]), np.average([p1[1],p2[1]])]
                for rule_direction in [1,-1]:
                    rule = None # placeholder for a line rule to generate
                    if p1[0]-p2[0] == 0: # if m=infinity, i.e parallel to Y-axis
                        rule = self._orthogonal_parallel_to_y_line(median_point, rule_direction)
                    elif p1[1]-p2[1] == 0: # if m=0, i.e parallel to X-axis
                        rule = self._orthogonal_parallel_to_x_line(median_point, rule_direction)
                    else:
                        m = (p1[1]-p2[1])/(p1[0]-p2[0])
                        rule = self._orthogonal_line(m, median_point, rule_direction)
                    lines.append(rule)
            
        return lines

    def _orthogonal_line(self, m, intersect_at, rule_direction) -> LineRule:
        new_m = -(1/m)
        new_y0 = intersect_at[1] - new_m * intersect_at[0]

        return LineRule(new_m, new_y0, rule_direction)

    def _orthogonal_parallel_to_y_line(self, intersect_at, rule_direction) -> LineRule:
        new_m = 0
        new_y0 = intersect_at[1]

        return LineRule(new_m, new_y0, rule_direction)

    def _orthogonal_parallel_to_x_line(self, intersect_at, rule_direction) -> LineRuleParallelToYAxis:
        x0 = intersect_at[0]

        return LineRuleParallelToYAxis(x0, rule_direction)


class BoostedRule:
    def __init__(self, rules, weights) -> None:
        self.rules = rules
        self.weights = weights

    def classify(self, point):
        weighted_vote = 0
        for rule, weight in zip(self.rules, self.weights):
            weighted_vote += weight * rule.classify(point)

        return np.sign(weighted_vote)