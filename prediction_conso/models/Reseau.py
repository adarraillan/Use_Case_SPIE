from models._Model import Model
from models.lstm import Lstm
from dataset.DataLoader import DataLoader
import datetime

class Reseau :

    type_habitation : list
    number_habitation : list
    X_columns = ['total', '0:00', '0:30', '1:00', '1:30', '2:00', '2:30', '3:00', '3:30',
       '4:00', '4:30', '5:00', '5:30', '6:00', '6:30', '7:00', '7:30', '8:00',
       '8:30', '9:00', '9:30', '10:00', '10:30', '11:00', '11:30', '12:00',
       '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00',
       '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00',
       '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30', 'type',
       'nb_inhabitant', 'surface', 'Seconds', 'Day sin', 'Day cos', 'Year sin',
       'Year cos', 'avg_temp']


    def __init__(self):
        self.type_habitation = ['A-15-1','A-25-1','A-30-2','A-50-2','A-50-3','A-100-3','A-110-5','A-120-4','A-130-4','A-150-6','M-50-2','M-65-3','M-80-2','M-85-3','M-100-3','M-110-4','M-120-5','M-135-3','M-140-5','M-150-4','M-160-5','M-170-6','M-200-6']
        self.number_habitation = [672,916,926,680,756,761,429,823,872,157,683,879,863,1014,896,659,1167,694,1268,842,861,754,326] 
        self.dataloader = DataLoader()
        self.X = self.dataloader.X
        self.Y = self.dataloader.Y
        self.last_months = self.dataloader.last_months

    # Get the list of habitation of X that are of type 'type' and have a surface of 'surface' and a number of habitants of 'habitants' and 
    def get_list_habitations_by_type(self,type,surface,habitants,end_date):
        list_habitation = []
        end_date_timestamp = datetime.datetime.strptime(end_date, "%d/%m/%Y").timestamp()

        for i in range(len(self.X)) : 
            row = self.X[i]
            # print(" in array  :", int(row[-1][-6]))
            # print("end date : ",int(end_date_timestamp))
            print("Surface : ",int(row[-1][-7]))
            print("habitants : ", int(row[-1][-8]))
            print("type : ", int(row[-1][-9]))
            if int(row[-1][-7]) == surface and int(row[-1][-8]) == habitants and int(row[-1][-9]) == type :
                print("row : ",row)
        return list_habitation
    
    def prediction_Poste_Source(self) : 
        print(len(self.number_habitation))
        print(len(self.type_habitation))
        self.get_list_habitations_by_type(0,80,2,'01/01/2022')
        pass