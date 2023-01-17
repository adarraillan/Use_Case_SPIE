import os,sys
import numpy as np
from PIL import Image
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime as dt
import random as rd

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

class Preprocess:

    PATH_DATA = "./dataset/data/"
    PATH_DATA_PROCESSED = "./dataset/data_preprocessed/"
    # data :pd.DataFrame
    infoclimate : pd.DataFrame
    data_time_series : pd.DataFrame()
    train : pd.DataFrame
    test : pd.DataFrame
    dev : pd.DataFrame
    columns = ['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','d21','d22','d23','d24','d25','d26','d27','d28','d29']

    def __init__(self):
        # self.data = self.load_files()
        self.infoclimate = self.load_infoclimate()
        self.data_time_series = self.load_files_time_series()
        self.train, self.test, self.dev = self.split_data_time_series()
        self.save_data()

    def load_infoclimate(self):
        df = pd.read_csv("./dataset/infoclimat.csv", sep=",")
        df['date'] = pd.to_datetime(df['dh_utc']).dt.strftime('%Y-%m-%d')
        df.drop(columns=['dh_utc'], inplace=True)
        df['index_date'] = df['date']
        df = df.set_index(['index_date'])
        # select the list of dates that are in the dataset
        dates = pd.unique(df[['date']].values.ravel())
        #compute the average temperature for each date
        df_date_climate = pd.DataFrame(columns=["date","avg_temp"])
        for date in dates:
            df_date = df[df['date'] == date]
            avg_temp = df_date["temperature"].mean() 
            pd.DataFrame(columns = ["date","avg_temp"], data = [[date,avg_temp]])
            df_date_climate = pd.concat([df_date_climate,pd.DataFrame(columns = ["date","avg_temp"], data = [[date,avg_temp]])],axis=0)
        
        # Turn dates into the index
        df_date_climate['date'] = pd.to_datetime(df_date_climate['date']).dt.strftime('%Y-%d-%m')
        df_date_climate = df_date_climate.set_index(['date'])
        return df_date_climate

    def load_files(self):
        df = pd.DataFrame(columns=self.columns)

        for folder in os.listdir(self.PATH_DATA):
            # print("FOLDER :",folder)
            type_home = folder[5:6]
            end = folder[6:]
            [surface, nb_people] = end.split("-")

            # Get all the data from the files of the folder in a dataframe
            df_temps = pd.DataFrame()
            for file in os.listdir(self.PATH_DATA + folder)[0:2]:
                # print("FILE :",file)
                df_temps = pd.read_csv(self.PATH_DATA + folder + "/" + file, sep=",",skiprows=1)
                df_temps.rename(columns = {'Unnamed: 0':'date', 'Unnamed: 1':'total'}, inplace = True)
                df_temps['type'] = type_home
                df_temps['nb_inhabitant'] = nb_people
                df_temps['surface'] = surface
                df = pd.concat([df,df_temps])
        df['index_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        df = df.set_index(['index_date'])
        return df


    def load_files_time_series(self):
        decalage = 10
        length=29
        df_file = pd.DataFrame()
        list_houses = []
        X = []
        Y = []
        for folder in os.listdir(self.PATH_DATA):
            print("FOLDER :",folder)
            type_home = folder[5:6]
            end = folder[6:]
            [surface, nb_people] = end.split("-")

            # Get all the data from the files of the folder in a dataframe
            df_temps = pd.DataFrame()
            
            for house in os.listdir(self.PATH_DATA + folder)[0:2]:

                # Read file and change column names
                df_temps = pd.read_csv(self.PATH_DATA + folder + "/" + house, sep=",",skiprows=1)
                df_temps.rename(columns = {'Unnamed: 0':'date', 'Unnamed: 1':'total'}, inplace = True)
                df_temps['type']=type_home
                df_temps['nb_inhabitant']=nb_people
                df_temps['surface']=surface
                
                #Set index of the dataframe
                df_temps['index_date'] = pd.to_datetime(df_temps['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
                df_temps = df_temps.set_index(['index_date'])

                # Sort data by increasing dates
                df_temps = df_temps.sort_index()

                # Concatenate the climate information to the dataframe
                df_temps = df_temps.join(self.infoclimate, on='index_date')

                # Tirer au hasard une date de début
                start_index = rd.randint(0, 10)
                start_date = pd.to_datetime(df_temps.index[start_index])
                
                i=0

                # Pour chaque date de début, prendre les 30 jours suivants            
                for start in (start_date + dt.timedelta(decalage*i) for i in range(df_temps.shape[0]//decalage)):
                    d_start =(start + dt.timedelta(0)).strftime('%Y-%m-%d')
                    d_end = (start + dt.timedelta(length)).strftime('%Y-%m-%d')
                    if pd.to_datetime(d_end) < pd.to_datetime(df_temps.index[-1]) :
                        list_month = df_temps.loc[d_start:d_end] 
                    else:
                        break
                    target = self.get_chunck(df_temps,start+dt.timedelta(length),1)
                    # print(len(list_month))
                    rows = [ row for row in list_month.to_numpy()]
                    X.append(rows)
                    Y.append([target.to_numpy()])
                    # list_month=[]
                    # if start_date + dt.timedelta(31) > pd.to_datetime(df_temps.index[-1]):
                    #     break
                    # for single_date in (start_date + dt.timedelta(n) for n in range(30)) :
                    #     if single_date in df_temps.index:
                    #         list_conso_day = list(df_temps.loc[single_date.date().isoformat()])
                    #         list_month.append(list_conso_day)
                    # # Add the cosumption of the following day which will be the target value        
                    # list_month.append(list(df_temps[self.columns].loc[(start_date+dt.timedelta(31)).date().isoformat()]))
                    # list_houses.append(list_month)   
            
            # df_file = pd.concat([df_file,pd.DataFrame(list_houses)],axis=0)  
        X = pd.DataFrame(data=X,columns = self.columns)
        Y = pd.DataFrame(data=Y,columns=['Y'])
        out = pd.concat([X,Y],axis=1)
        print(out.shape)
        print(out.head())
        # out = pd.DataFrame(data = list_houses, columns=['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','d21','d22','d23','d24','d25','d26','d27','d28','d29','Y'])  
        return out

    def get_chunck(self,df,start_date,length):
        start =(start_date + dt.timedelta(0)).strftime('%Y-%m-%d')
        end = (start_date + dt.timedelta(length)).strftime('%Y-%m-%d')
        list = df.loc[start:end] 
        return list

    def split_data_time_series(self):
        # Split data in train, test and dev
        X = self.data_time_series.drop(columns=['Y'])
        Y = self.data_time_series['Y']        
        X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.33, random_state=42)
        train = pd.concat([X_train,Y_train], axis=1)
        X_test, X_dev, Y_test, Y_dev=train_test_split(X_test,Y_test, test_size=0.5, random_state=42)
        test = pd.concat([X_test,Y_test], axis=1)
        dev = pd.concat([X_dev,Y_dev], axis=1)
        return train, test, dev

    def split_data(self):
        X = self.data.drop(columns=self.columns)
        Y = self.data[self.columns]        
        X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.33, random_state=42)
        train = pd.concat([X_train,y_train], axis=1)
        X_test, X_dev, y_test, y_dev=train_test_split(X_test,y_test, test_size=0.3, random_state=42)
        test = pd.concat([X_test,y_test], axis=1)
        dev = pd.concat([X_dev,y_dev], axis=1)
        print(train.shape)
        print(test.shape)
        print(dev.shape) 
        return train, test, dev

    def save_data(self):
        self.data_time_series.to_csv(self.PATH_DATA_PROCESSED + "data_processed.csv", index=False)
        self.train.to_csv(self.PATH_DATA_PROCESSED + "train.csv", index=False)
        self.test.to_csv(self.PATH_DATA_PROCESSED + "test.csv", index=False)
        self.dev.to_csv(self.PATH_DATA_PROCESSED + "dev.csv", index=False)
    
    def preprocess_data(self):
        print(self.data.head())
        # Look for missing values
        print("Missing values : ", self.data.isnull().sum())
        # Look for duplicates
        print("Duplicates : ",self.data.duplicated().sum())
        # Look for outliers
        print(self.data.describe())
        pass

preproc = Preprocess()

# preproc.save_data()
# preproc.load_files()
# X_train, X_test, y_train, y_test = preproc.split_data_into_csv()
# print(X_train, X_test, y_train, y_test)
# preproc.preprocess_data()
