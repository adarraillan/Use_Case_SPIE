from models._Model import Model
from models.lstm import Lstm
from models.Reseau import Reseau
import os

if __name__ == "__main__":

    # Pour entrainer les modèles
    # for folder in os.listdir("./dataset/data"):
    #     model = Model(folder)
    #     model.train()
    #     model.save()
   

    # Pour print les plots 

    # for folder in os.listdir("./dataset/data"):
    #     model = Model(folder)
    #     print(model.plot_predictions_test())

    # Pour print la prediction finale au poste source
    res = Reseau()
    print("\n\n  ************************* PREDICTION PS *************************\n\n")
    print(res.PS)

   