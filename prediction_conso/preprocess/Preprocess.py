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
    columns = ['total', 
       '0:00', '0:30', '1:00', '1:30', '2:00', '2:30', '3:00',
       '3:30', '4:00', '4:30', '5:00', '5:30', '6:00', '6:30', '7:00', '7:30',
       '8:00', '8:30', '9:00', '9:30', '10:00', '10:30', '11:00', '11:30',
       '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30',
       '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30',
       '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30']

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
            type = folder[5:6]
            end = folder[6:]
            [surface, nb_people] = end.split("-")

            # Get all the data from the files of the folder in a dataframe
            df_temps = pd.DataFrame()
            for file in os.listdir(self.PATH_DATA + folder)[0:2]:
                # print("FILE :",file)
                df_temps = pd.read_csv(self.PATH_DATA + folder + "/" + file, sep=",",skiprows=1)
                df_temps.rename(columns = {'Unnamed: 0':'date', 'Unnamed: 1':'total'}, inplace = True)
                df_temps['type'] = type
                df_temps['nb_inhabitant'] = nb_people
                df_temps['surface'] = surface
                df = pd.concat([df,df_temps])
        df['index_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        df = df.set_index(['index_date'])
        return df


    def load_files_time_series(self):
        decalage = 10
        df_file = pd.DataFrame()
        list_houses = []
        for folder in os.listdir(self.PATH_DATA):
            print("FOLDER :",folder)
            type = folder[5:6]
            end = folder[6:]
            [surface, nb_people] = end.split("-")

            # Get all the data from the files of the folder in a dataframe
            df_temps = pd.DataFrame()
            
            for house in os.listdir(self.PATH_DATA + folder)[0:10]:

                # Read file and change column names
                df_temps = pd.read_csv(self.PATH_DATA + folder + "/" + house, sep=",",skiprows=1)
                df_temps.rename(columns = {'Unnamed: 0':'date', 'Unnamed: 1':'total'}, inplace = True)
                df_temps['index_date'] = pd.to_datetime(df_temps['date'], format='%m/%d/%Y')
                df_temps['type']=type
                df_temps['nb_inhabitant']=nb_people
                df_temps['surface']=surface
                df_temps = df_temps.set_index(['index_date'])

                # Trier les données par date croissantes
                df_temps = df_temps.sort_index()

                # Tirer au hasard une date de début
                start_index = rd.randint(0, 10)
                
                i=0
                # Pour chaque date de début, prendre les 30 jours suivants            
                for start_date in (df_temps.index[start_index] + dt.timedelta(decalage*i) for i in range(df_temps.shape[0]//decalage)):
                    list_month = []
                    if start_date + dt.timedelta(31) > df_temps.index[-1]:
                        break
                    for single_date in (start_date + dt.timedelta(n) for n in range(30)) :
                        # print(single_date.date().strftime(format='%Y-%d-%m'))
                        if single_date.date().strftime(format='%Y-%d-%m') in self.infoclimate.index :
                            temperature = self.infoclimate['avg_temp'].loc[single_date.date().strftime(format='%Y-%d-%m')]
                        else:
                            temperature = None
                        if single_date in df_temps.index:
                            list_conso_day = list(df_temps.loc[single_date.date().isoformat()])
                            list_conso_day.append(temperature)
                            list_month.append(list_conso_day)
                    # Add the cosumption of the following day which will be the target value        
                    list_month.append(list(df_temps[self.columns].loc[(start_date+dt.timedelta(31)).date().isoformat()]))
                    list_houses.append(list_month)   
            # df_file = pd.concat([df_file,pd.DataFrame(list_houses)],axis=0)  
        out = pd.DataFrame(data = list_houses, columns=['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','d21','d22','d23','d24','d25','d26','d27','d28','d29','Y'])  
        return out

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
