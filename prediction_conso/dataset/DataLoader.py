import os
import pandas as pd
import numpy as np

class DataLoader :

    DATADIR : str
    TRAINDIR : str
    TESTDIR : str
    DEVDIR : str
    X_train : np.array
    Y_train : np.array
    X_test :  np.array
    Y_test : np.array
    X_dev : np.array
    Y_dev : np.array
    X : np.array
    Y : np.array
    last_months : np.array
    Y_means : list() = []
    Y_stds : list() = []
    X_means : list() = []
    X_stds : list() = []

    def __init__(self):
        self.DATADIR  = "./dataset/data_preprocessed"
        self.X_train,self.Y_train = self.data_retriever("train")
        self.X_test,self.Y_test = self.data_retriever("test")
        self.X_dev,self.Y_dev = self.data_retriever("dev")
        self.X, self.Y = self.data_retriever("all")
        self.last_month = self.data_retriever("last_months")
        self.X_means, self.X_stds = self.get_list_mean_std_X_train(self.X_train)
        self.Y_means,self.Y_stds = self.get_list_mean_std_Y_train(self.Y_train)
    
    # Get the list of means and stds for each feature of X_train
    def get_list_mean_std_X_train(self,X):
        means = []
        stds = []
        for i in range(X.shape[2]):
            mean = np.mean(X[:, :, i])
            std = np.std(X[:, :, i])
            means.append(mean)
            stds.append(std)
        return means,stds

    #Get the list of mean and stds for each feature of Y_train
    def get_list_mean_std_Y_train(self,Y):
        means = []
        stds = []
        for i in range(Y.shape[1]):
            mean = np.mean(Y[:, i])
            std = np.std(Y[:, i])
            means.append(mean)
            stds.append(std)
        return means,stds

    # Standardise a X matrix with the means and stds of X_train
    def standardize_X(self,X):
        for i in range(X.shape[2]):
            if self.X_stds[i] != 0 :
                X[:, :, i] = (X[:, :, i]-self.X_means[i])/self.X_stds[i]
            else :
                X[:, :, i] = (X[:, :, i]-self.X_means[i])
        return X

    # Standardise a Y matrix with the means and stds of Y_train
    def standardize_Y(self,Y):
        for i in range(Y.shape[1]):
            if self.Y_stds[i] != 0 :
                Y[:, i] = (Y[:, i]-self.Y_means[i])/self.Y_stds[i]
            else : 
                Y[:, i] = (Y[:, i]-self.Y_means[i])
        return Y

    def load_data_from_npy(self,path_X,path_Y):
        X = np.load(path_X) 
        Y = np.load(path_Y)
        return X,Y

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
        elif dir == "all" :
            path_X = self.DATADIR+'/X.npy'
            path_Y = self.DATADIR+'/Y.npy'
        else :
            last_months = np.load(self.DATADIR+'/last_months.npy') 
            return last_months 
        return self.load_data_from_npy(path_X,path_Y)

# print("Loading data...")
# data_loader = DataLoader()
# X_test = data_loader.standardize_X(data_loader.X_test)
# Y_test = data_loader.standardize_Y(data_loader.Y_test)
# print(X_test)