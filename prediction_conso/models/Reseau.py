from models._Model import Model
from models.lstm import Lstm
from dataset.DataLoader import DataLoader
import datetime
import numpy as np

class Reseau :

    nombre_type_habitation  : dict
    list_houses_by_type : dict
    X : np.array
    Y : np.array
    last_months : np.array
    dataloader : DataLoader
    model : Model
    X_columns = ['total', '0:00', '0:30', '1:00', '1:30', '2:00', '2:30', '3:00', '3:30',
       '4:00', '4:30', '5:00', '5:30', '6:00', '6:30', '7:00', '7:30', '8:00',
       '8:30', '9:00', '9:30', '10:00', '10:30', '11:00', '11:30', '12:00',
       '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00',
       '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00',
       '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30', 'type',
       'nb_inhabitant', 'surface', 'Seconds', 'Day sin', 'Day cos', 'Year sin',
       'Year cos', 'avg_temp']
    predictions_by_type = dict()


    def __init__(self,model):
        self.nombre_type_habitation = {'A-15-1':672,'A-25-1':916,'A-30-2':926,'A-50-2':680,'A-50-3':756,'A-100-3':761,'A-110-5':761,'A-120-4':429,'A-130-4':823,'A-150-6':872,'M-50-2':157,'M-65-3':683,'M-80-2':879,'M-85-3':863,'M-100-3':1014,'M-110-4':896,'M-120-5':659,'M-135-3':1167,'M-140-5':694,'M-150-4':1268,'M-160-5':842,'M-170-6':861,'M-200-6':754}
        self.dataloader = DataLoader()
        self.X = self.dataloader.X
        self.Y = self.dataloader.Y
        self.last_months = self.dataloader.last_months
        self.list_houses_by_type = self.get_list_habitations_by_type()
        self.model = model
        self.predictions_by_type = self.prediction_by_type()

    # Get the list of habitation of X that are of type 'type' and have a surface of 'surface' and a number of habitants of 'habitants' and 
    def get_list_habitations_by_type(self):
        dict_last_months_by_type = dict({'A-15-1' : [],'A-25-1':[],'A-30-2':[],'A-50-2':[],'A-50-3':[],'A-100-3':[],'A-110-5':[],'A-120-4':[],'A-130-4':[],'A-150-6':[],'M-50-2':[],'M-65-3':[],'M-80-2':[],'M-85-3':[],'M-100-3':[],'M-110-4':[],'M-120-5':[],'M-135-3':[],'M-140-5':[],'M-150-4':[],'M-160-5':[],'M-170-6':[],'M-200-6':[]})
        for i in range(len(self.last_months)) : 
            row = self.last_months[i]
            type = int(row[0][-9])
            surface = int(row[0][-7])
            habitants = int(row[0][-8])
            if type == 0 :
                type_str = 'M'
            elif type == 1 : 
                type_str = 'A'
            dict_last_months_by_type[type_str+'-'+str(surface)+'-'+str(habitants)].append(row)
        return dict_last_months_by_type
    
    def prediction_by_type(self) : 
        sum_predictions_by_type = dict()
        predictions_by_type = dict()
        for type, nombre in self.nombre_type_habitation.items() :
            # print(self.list_houses_by_type[self.type_habitation[i]])
            predictions = self.model.predict_batch(np.array(self.list_houses_by_type[type]))
            # print("prediciton :",predictions)
            sum_predictions_by_type[type] = np.sum(predictions,axis=0)
            val = nombre/len(self.list_houses_by_type[type])
            predictions_by_type[type] = map(lambda x: x * val , sum_predictions_by_type[type]) 
        return predictions_by_type

    def prediction_Poste_Source(self) :
        print(np.sum(self.predictions_by_type.values,axis=0))
        return np.sum(self.predictions_by_type.values,axis=0)      
