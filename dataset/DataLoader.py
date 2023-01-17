import os
import pandas as pd


class DataLoader :

    DATADIR : str
    TRAINDIR : str
    TESTDIR : str
    DEVDIR : str


    def __init__(self):
        self.DATADIR  = "./dataset/data_preprocessed"
        self.TRAINDIR = os.path.join(self.DATADIR, "train.csv")
        self.TESTDIR  = os.path.join(self.DATADIR, "test.csv")
        self.DEVDIR = os.path.join(self.DATADIR, "dev.csv")

    def load_data_from_csv(self,path):
        data_time_series = pd.read_csv(path, sep=",")
        X = data_time_series.drop(columns=['Y'])
        Y = data_time_series['Y']
        return X,Y

    def data_retriever(self,dir):
        data_path = ""
        if dir == "train" : 
            data_path = self.TRAINDIR
        elif dir == "test" :
            data_path = self.TESTDIR
        elif dir == "dev" :
            data_path = self.DEVDIR
        else :
            print("Error: wrong dir")
            return 
        return self.load_data_from_csv(data_path)

print("Loading data...")
data_loader = DataLoader()
print(data_loader.data_retriever("test"))