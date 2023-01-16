import os
import pandas as pd


class DataLoader :

    DATADIR : str
    TRAINDIR : str
    TESTDIR : str
    DEVDIR : str

    columns = ['total', 
       '0:00', '0:30', '1:00', '1:30', '2:00', '2:30', '3:00',
       '3:30', '4:00', '4:30', '5:00', '5:30', '6:00', '6:30', '7:00', '7:30',
       '8:00', '8:30', '9:00', '9:30', '10:00', '10:30', '11:00', '11:30',
       '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30',
       '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30',
       '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30']

    def __init__(self):
        self.DATADIR  = "./dataset/data"
        self.ANNOTATIONSDIR = os.path.join(self.DATADIR, "annotations")
        self.TRAINDIR = os.path.join(self.DATADIR, "images/train")
        self.TESTDIR  = os.path.join(self.DATADIR, "images/test")
        self.DEVDIR = os.path.join(self.DATADIR, "images/dev")

    def load_data_from_csv(self,path):
        data_houses = pd.read_csv(path, sep=",")
        X = data_houses.drop(columns=self.columns)
        Y = data_houses[self.columns]
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
print(data_loader.data_retriever("train"))