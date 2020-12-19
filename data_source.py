import pandas as pd

class IrisDataSource:
    def __init__(self, filepath) -> None:
        self.filepath = filepath

    def read(self):
        data = pd.read_csv(self.filepath,  #'data/iris.data'
                    header=None, 
                    index_col=False, 
                    names=['c1','c2','c3','c4','label'], 
                    usecols=[1,2,4])

        data = data[data['label'] != 'Iris-setosa']
        data.loc[data['label'] == 'Iris-versicolor', 'label'] = +1
        data.loc[data['label'] == 'Iris-virginica', 'label'] = -1

        return data


class HCBTDataSource: # HC Body temperature
    def __init__(self, filepath) -> None:
        self.filepath = filepath

    def read(self):
        data = pd.read_table(self.filepath, #'data/HC_Body_Temperature.txt', 
                    header=None,
                    index_col=False,
                    names=['c1', 'label' ,'c2'],
                    delimiter=r"\s+")

        # reorder columns so that label is the third column
        data = data[data.columns[[0,2,1]]]

        # change label 2 --> -1
        data.loc[data['label'] == 2, 'label'] = -1

        return data