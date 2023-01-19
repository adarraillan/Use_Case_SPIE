import os
import pandas as pd
import numpy as np

class DataLoader :

    DATADIR : str
    TRAINDIR : str
    TESTDIR : str
    DEVDIR : str
    Y_means = []
    Y_stds = []
    X_means = []
    X_stds = []

    def __init__(self):
        self.DATADIR  = "./prediction_conso/dataset/data_preprocessed"
    
    def standardize_X(self,X):
        for i in range(X.shape[2]):
            mean = np.mean(X[:, :, i])
            # if i == 0 or i ==1:
            #     print("mean : ",mean)
            std = np.std(X[:, :, i])
            if std != 0 :
                X[:, :, i] = (X[:, :, i]-mean)/std
            else :
                X[:, :, i] = (X[:, :, i]-mean)
            self.X_means.append(mean)
            self.X_stds.append(std)
        return X

    def standardize_Y(self,Y):
        for i in range(Y.shape[1]):
            mean = np.mean(Y[:, i])
            std = np.std(Y[:, i])
            if std != 0 :
                Y[:, i] = (Y[:, i]-mean)/std
            else : 
                Y[:, i] = (Y[:, i]-mean)
            self.Y_means.append(mean)
            self.Y_stds.append(std)
        return Y

    def load_data_from_npy(self,path_X,path_Y):
        X = np.load(path_X) 
        Y = np.load(path_Y)
        print("X shape : ",X.shape)
        return self.standardize_X(X),self.standardize_Y(Y)

    def data_retriever(self,dir):
        path_X = ""
        path_Y = ""
        if dir == "train" : 
            path_X = self.DATADIR+'/X_train.npy'
            path_Y = self.DATADIR+'/Y_train.npy'
        elif dir == "test" :
            path_X = self.DATADIR+'/X_test.npy'
            path_Y = self.DATADIR+'/Y_test.npy'
        elif dir == "dev" :
            path_X = self.DATADIR+'/X_dev.npy'
            path_Y = self.DATADIR+'/Y_dev.npy'
        else :
            print("Error: wrong dir")
            return None
        return self.load_data_from_npy(path_X,path_Y)

# print("Loading data...")
# data_loader = DataLoader()
# print(data_loader.data_retriever("test"))