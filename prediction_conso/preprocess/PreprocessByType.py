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
    X_train : np.array
    Y_train : np.array
    X_test :  np.array
    Y_test : np.array
    X_dev : np.array
    Y_dev : np.array
    X : np.array
    Y : np.array
    last_months : np.array
    columns = ['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','d21','d22','d23','d24','d25','d26','d27','d28','d29']
    nb_house = 100
    mean : list
    std : list

    def __init__(self):
        # self.data = self.load_files()
        self.infoclimate = self.load_infoclimate()
        

    def save_data(self,X,Y,X_train, Y_train, X_test, Y_test, X_dev, Y_dev,last_months,path):
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path+'X_train.npy', X_train) # save
        np.save(path+'Y_train.npy', Y_train)
        np.save(path+'X_test.npy', X_test) # save
        np.save(path+'Y_test.npy', Y_test)
        np.save(path+'X_dev.npy', X_dev) # save
        np.save(path+'Y_dev.npy', Y_dev)
        np.save(path+'X.npy', X) # save X
        np.save(path+'Y.npy', Y) # save Y
        np.save(path+'last_months.npy',last_months) # save Y

    def load_files_all_types(self) :
        for folder in os.listdir(self.PATH_DATA):
            print("PREPROCESSING FOLDER :",folder)
            X,Y,last_months = self.load_files_time_series_one_type(folder)
            X_train, Y_train, X_test, Y_test, X_dev, Y_dev= self.split_data_time_series(X,Y)
            self.save_data(X,Y,X_train, Y_train, X_test, Y_test, X_dev, Y_dev,last_months,'./dataset/data_preprocessed/'+folder+'/')


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



    def preprocess_house_file(self,folder,house):
        # Read file, change column names and add some columns
        df_temps = pd.read_csv(self.PATH_DATA + folder + "/" + house, sep=",",skiprows=1)
        df_temps.rename(columns = {'Unnamed: 0':'date', 'Unnamed: 1':'total'}, inplace = True)

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
        # df_temps = df_temps.drop(columns=['date']) 

        # Sort data by increasing dates
        df_temps = df_temps.sort_index()

        # Concatenate the climate information to the dataframe
        df_temps = df_temps.join(self.infoclimate, on='index_date')
        
        # Fill the missing values with the previous value
        df_temps['avg_temp'] = df_temps['avg_temp'].fillna(method='ffill')

        return df_temps


    def load_files_time_series_one_type(self,folder):
        decalage = 10
        length=29
        X = []
        Y = []
        last_months = []
  
        # Get all the data from the files of the folder in a dataframe
        df_temps = pd.DataFrame()
            
        for house in os.listdir(self.PATH_DATA + folder)[0:self.nb_house]:

            df_temps = self.preprocess_house_file(folder, house)

            # Tirer au hasard une date de début
            start_index = rd.randint(0, 10)
            start_date = pd.to_datetime(df_temps.index[start_index])
                
            # For each start date, get the 30 following days            
            for start in (start_date + dt.timedelta(decalage*i) for i in range(df_temps.shape[0]//decalage)):
                d_start =(start + dt.timedelta(0)).strftime('%Y-%m-%d')
                d_end = (start + dt.timedelta(length)).strftime('%Y-%m-%d')
                if pd.to_datetime(d_end) < pd.to_datetime(df_temps.index[-1]) :
                    list_month = df_temps.loc[d_start:d_end] 
                    list_month = list_month.drop(columns=['date'])
                else:
                    break        
                row = np.array([ r.astype(np.float64) for r in list_month.to_numpy()])
                X.append(row)
                target = self.get_chunck(df_temps,start+dt.timedelta(length+1),0)
                target = target.drop(columns=['date','avg_temp','Seconds','Day sin','Day cos','Year sin','Year cos'])
                Y.append(np.array(target.to_numpy().astype(np.float64)))
                
            # Get the last month of the data for the current house
            last_date_of_file = (pd.to_datetime(df_temps.iloc[-1]['date'], format='%m/%d/%Y')).strftime('%Y-%m-%d')
            start_date_month = (pd.to_datetime(last_date_of_file, format='%Y-%m-%d') - dt.timedelta(29)).strftime('%Y-%m-%d')
            temp_df_temp = df_temps
            temp_df_temp = temp_df_temp.drop(columns=['date'])
            list_one_month = temp_df_temp.loc[start_date_month:last_date_of_file]
            row = np.array([ r.astype(np.float64) for r in list_one_month.to_numpy()])
            last_months.append(list_one_month)
        
        last_months = np.array(last_months)
        X = np.array(X)
        Y = np.array(Y).reshape(-1,49)
        return X,Y,last_months

    def get_chunck(self,df,start_date,length):
        start =(start_date + dt.timedelta(0)).strftime('%Y-%m-%d')
        end = (start_date + dt.timedelta(length)).strftime('%Y-%m-%d')
        list = df.loc[start:end] 
        return list

    def split_data_time_series(self,X,Y):
        print("X shape",X.shape)
        end_train = round(X.shape[0]*0.70)
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

preproc = Preprocess()
preproc.load_files_all_types()
