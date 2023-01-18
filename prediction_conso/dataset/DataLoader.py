import os
import pandas as pd
from preprocess.Preprocess import Preprocess

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
        self.preprocessor = Preprocess()

    # def load_data_from_csv(self,path):
    #     data_time_series = pd.read_csv(path, sep=",")
    #     X = data_time_series.drop(columns=['Y'])
    #     Y = data_time_series['Y']
    #     return X,Y

    def data_retriever(self,dir):
        data_path = ""
        if dir == "train" : 
            data_path = self.TRAINDIR
            X = self.preprocessor.X_train
            Y = self.preprocessor.Y_train
        elif dir == "test" :
            data_path = self.TESTDIR
            X = self.preprocessor.X_test
            Y = self.preprocessor.Y_test
        elif dir == "dev" :
            data_path = self.DEVDIR
            X = self.preprocessor.X_dev
            Y = self.preprocessor.Y_dev
        else :
            print("Error: wrong dir")
            return 
        return X,Y

# print("Loading data...")
# data_loader = DataLoader()
# print(data_loader.data_retriever("test"))