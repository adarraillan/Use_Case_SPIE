import os,sys
import numpy as np
from PIL import Image
import csv
import pandas as pd



class Preprocess:

    PATH_DATA = "./dataset/data/"
    PATH_DATA_PROCESSED = "./dataset/data_preprocessed/"

    def __init__(self):
        pass

    def load_files(self):
        df = pd.DataFrame(columns=["date","type","nb_inhabitant","surface","day",'total', 
       '0:00', '0:30', '1:00', '1:30', '2:00', '2:30', '3:00',
       '3:30', '4:00', '4:30', '5:00', '5:30', '6:00', '6:30', '7:00', '7:30',
       '8:00', '8:30', '9:00', '9:30', '10:00', '10:30', '11:00', '11:30',
       '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30',
       '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30',
       '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30'])

        for folder in os.listdir(self.PATH_DATA):
            print("FOLDER :",folder)
            type = folder[5:6]
            end = folder[6:]
            [surface, nb_people] = end.split("-")

            # Get all the data from the files of the folder in a dataframe
            df_temps = pd.DataFrame()
            for file in os.listdir(self.PATH_DATA + folder)[0:2]:
                print("FILE :",file)
                df_temps = pd.read_csv(self.PATH_DATA + folder + "/" + file, sep=",",skiprows=1)
                df_temps.rename(columns = {'Unnamed: 0':'date', 'Unnamed: 1':'total'}, inplace = True)
                df_temps['type'] = type
                df_temps['nb_inhabitant'] = nb_people
                df_temps['surface'] = surface
                df = pd.concat([df,df_temps])

        df.to_csv(self.PATH_DATA_PROCESSED + "data_processed.csv", index=False)
        return df_temps


preproc = Preprocess()
res = preproc.load_files()
print(res.head)
print(res.columns)