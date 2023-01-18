import os,sys
import numpy as np
from PIL import Image
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    X_train : pd.DataFrame()
    Y_train : pd.DataFrame()
    X_test :  pd.DataFrame()
    Y_test : pd.DataFrame()
    X_dev : pd.DataFrame()
    Y_dev : pd.DataFrame()
    columns = ['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','d21','d22','d23','d24','d25','d26','d27','d28','d29']
    nb_house = 2
    mean : list
    std : list

    def __init__(self):
        # self.data = self.load_files()
        self.infoclimate = self.load_infoclimate()
        self.X, self.Y = self.load_files_time_series()
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_dev, self.Y_dev= self.split_data_time_series()
        # self.save_data()
        # self.preprocess_data()

#     def save_data(self):
#         self.X_train.tofile(self.PATH_DATA_PROCESSED + "X_train.csv", sep=",")
# #
# Methods to create time series csv files
#
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

    def load_files_time_series(self):
        decalage = 10
        length=29
        df_file = pd.DataFrame()
        list_houses = []
        X = []
        Y = []
        for folder in os.listdir(self.PATH_DATA)[0:4]:
            print("PREPROCESSING FOLDER :",folder)
            if folder[5:6] == 'M' :
                type_home = 0
            else : 
                type_home = 1
            end = folder[6:]
            [surface, nb_people] = end.split("-")

            # Get all the data from the files of the folder in a dataframe
            df_temps = pd.DataFrame()
            
            for house in os.listdir(self.PATH_DATA + folder)[0:self.nb_house]:

                # Read file, change column names and add some columns
                df_temps = pd.read_csv(self.PATH_DATA + folder + "/" + house, sep=",",skiprows=1)
                df_temps.rename(columns = {'Unnamed: 0':'date', 'Unnamed: 1':'total'}, inplace = True)
                df_temps['type']=type_home
                df_temps['nb_inhabitant']=nb_people
                df_temps['surface']=surface

                #Set index of the dataframe
                df_temps['index_date'] = pd.to_datetime(df_temps['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
                df_temps = df_temps.set_index(['index_date'])

                # Add the cos and sin of day and year : cyclical continuous features
                df_temps['Seconds'] = pd.to_datetime(df_temps.index).map(pd.Timestamp.timestamp)
                day = 60*60*24
                year = 365.2425*day
                df_temps['Day sin'] = np.sin(df_temps['Seconds'] * (2* np.pi / day))
                df_temps['Day cos'] = np.cos(df_temps['Seconds'] * (2 * np.pi / day))
                df_temps['Year sin'] = np.sin(df_temps['Seconds'] * (2 * np.pi / year))
                df_temps['Year cos'] = np.cos(df_temps['Seconds'] * (2 * np.pi / year))

                # Drop the date column
                df_temps = df_temps.drop(columns=['date'])    

                # Sort data by increasing dates
                df_temps = df_temps.sort_index()

                # Concatenate the climate information to the dataframe
                df_temps = df_temps.join(self.infoclimate, on='index_date')

                # Tirer au hasard une date de début
                start_index = rd.randint(0, 10)
                start_date = pd.to_datetime(df_temps.index[start_index])
                
                i=0
                # For each start date, get the 30 following days            
                for start in (start_date + dt.timedelta(decalage*i) for i in range(df_temps.shape[0]//decalage)):
                    d_start =(start + dt.timedelta(0)).strftime('%Y-%m-%d')
                    d_end = (start + dt.timedelta(length)).strftime('%Y-%m-%d')
                    if pd.to_datetime(d_end) < pd.to_datetime(df_temps.index[-1]) :
                        list_month = df_temps.loc[d_start:d_end] 
                    else:
                        break

                    row = np.array([ r.astype(np.float64) for r in list_month.to_numpy()])
                    # row.insert(0,d_start)
                    X.append(row)

                    target = self.get_chunck(df_temps,start+dt.timedelta(length+1),0)
                    target = target.drop(columns=['type','nb_inhabitant','surface','avg_temp','Seconds','Day sin','Day cos','Year sin','Year cos'])
                    Y.append(np.array(target.to_numpy().astype(np.float64)))

        X = np.array(X)
        Y = np.array(Y)
        print(type(Y[0][0][0]))
        # print(len(X))
        # print()
        # print(len(Y))
        # val = Y

        # while type(val) == list:
        #     print(len(val), type(val))
        #     val = val[0]
        # print(len(val),type(val),val)
                
        # print(np.array(X).shape)            
        # X = pd.DataFrame(data=X,columns = self.columns)
        # print(X.shape) 
        # Y = pd.DataFrame(data=Y,columns=['Y'])
        # print(Y.shape) 
        # out = pd.concat([X,Y],axis=1)
        # out = np.concatenate((X,Y),axis=1)
        #Set the start date as index
        # out = out.set_index(['start'])
        # Sort by increasing dates
        # out = out.sort_index()
        # out = pd.DataFrame(data = out, columns=self.columns)
        # print(out.shape)
        # print(out.shape)
        return X,Y

    def get_chunck(self,df,start_date,length):
        start =(start_date + dt.timedelta(0)).strftime('%Y-%m-%d')
        end = (start_date + dt.timedelta(length)).strftime('%Y-%m-%d')
        list = df.loc[start:end] 
        return list

    def split_data_time_series(self):
        # Split data in train, test and dev
        # X = self.data_time_series.drop(columns=['Y'])
        # Y = self.data_time_series['Y']
        X = self.X
        Y = self.Y
        end_train = round(self.X.shape[0]*0.70)
        end_test = round(X.shape[0]*0.85)
        X_train, Y_train = X[:end_train], Y[:end_train]   
        X_test, Y_test = X[end_train+1:end_test], Y[end_train+1:end_test]
        X_dev, Y_dev = X[end_test+1:], Y[end_test+1:]
        return  X_train, Y_train, X_test, Y_test, X_dev, Y_dev


    def preprocess_data(self):
        X_train = np.array(pd.concat([self.X_train,self.Y_train],axis=1))
        X_test = np.array(self.X_test)
        X_dev = np.array(self.X_dev)

        # Scale data
        self.train_mean = np.mean(X_train[:, :, 0])
        self.train_std = np.std(self.X_train[:, :, 0])
        # print(self.train_mean,self.train_std)
        pass

#
# Methods no longer used
#

    def load_files(self):
        df = pd.DataFrame(columns=self.columns)

        for folder in os.listdir(self.PATH_DATA):
            # print("FOLDER :",folder)
            type_home = folder[5:6]
            end = folder[6:]
            [surface, nb_people] = end.split("-")

            # Get all the data from the files of the folder in a dataframe
            df_temps = pd.DataFrame()
            for file in os.listdir(self.PATH_DATA + folder)[0:self.nb_house]:
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

    def split_data(self):
        X = self.data.drop(columns=self.columns)
        Y = self.data[self.columns]        
        X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.33, random_state=42)
        train = pd.concat([X_train,y_train], axis=1)
        X_test, X_dev, y_test, y_dev=train_test_split(X_test,y_test, test_size=0.3, random_state=42)
        test = pd.concat([X_test,y_test], axis=1)
        dev = pd.concat([X_dev,y_dev], axis=1)
        # print(train.shape)
        # print(test.shape)
        # print(dev.shape) 
        return train, test, dev
 
  


# praproc = Preprocess()
