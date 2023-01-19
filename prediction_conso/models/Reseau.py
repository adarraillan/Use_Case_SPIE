from models._Model import Model
from models.lstm import Lstm
from dataset.DataLoader import DataLoader
import datetime
import numpy as np
import os

class Reseau :

    prediction_by_type = dict()
    PS : np.array
    number_of_houses = [672,916,926,680,756,761,429,823,872,157,683,879,863,1014,896,659,1167,694,1268,842,861,754,326]
    dict_number_of_houses = dict()

    def __init__(self):
        self.dict_number_of_houses = self.get_dict_number_houses()
        self.prediction_by_type = self.predict_each_type()
        self.PS = self.final_prediction_PS()  

    def get_dict_number_houses(self):
        dict_numbers = dict()
        i=0
        for folder in os.listdir("./dataset/data"):
            dict_numbers[folder] = self.number_of_houses[i]
            i = i+1
        return dict_numbers

    def final_prediction_PS(self):
        total = np.array(np.sum(list(self.prediction_by_type.values()),axis=0))
        return total

    def predict_each_type(self):
        dict_all_predictions = dict()
        for folder in os.listdir("./dataset/data"):
            dataloader = DataLoader(folder)
            X = dataloader.X
            Y = dataloader.Y
            model = Model(folder)
            last_months = dataloader.last_months
            number_of_houses = self.dict_number_of_houses[folder]
            predictions = self.prediction_one_type(model,last_months,number_of_houses)
            dict_all_predictions[folder]=predictions
        return dict_all_predictions

    def prediction_one_type(self,model,data_last_months,number_of_houses) : 
        predictions = model.predict_batch(data_last_months)
        sum_predictions = np.sum(predictions,axis=0)
        val = number_of_houses/len(data_last_months)
        final_prediction = np.array(list(map(lambda x: x * val , sum_predictions)))
        return final_prediction
  