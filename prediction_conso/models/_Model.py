import sys

sys.path.append("./")

import os
import time
import datetime

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import datetime
# from codecarbon import EmissionsTracker
import csv
import numpy as np
from dataset import DataLoader as dl
# %matplotlib inline


"""
    This class is used to train, evaluate, save, load the model.
    Class variables:
        _model : tf.keras.Model already compiled
        _model_name : str
        _optimizer : str
        _loss : tf.keras.losses
        _metrics : str
        _emission : float = Emission product during the training in kgCO2e
        _eval_info : [Type_logement : accuracy]
"""
class Model:
    _model = None
    _model_name : str = "Base Model"
    _optimizer = "adam"
    _loss = tf.keras.losses.MeanSquaredError
    _metrics = 'accuracy'
    _emission : float = 0
    _eval_infos = None
    _data_loader = dl.DataLoader()


    """
        This function initialize our class
    """
    def __init__(self,model_name,optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics='accuracy'):
        self._model_name = model_name
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics
        

#usefull ?
    # def _build_model(self,load=False):
    #     print( "Building model parent")
    #     if load:
    #         return self.load()
    #     else:
    #         raise NotImplementedError

    """
        This function clean the logs for tensorboard
    """
    def clean_logs(self):
        try:
            os.rmtree('./logs')
        except:
            pass


    """
        This function show the learning rate in a plot
        Function variables:
            history : tf.keras.callbacks.History
            title : str
    """
    def plot_learning_curves(self, history,title):
        acc = history.history["accuracy"]
        loss = history.history["loss"]
        val_acc = history.history["val_accuracy"]
        val_loss = history.history["val_loss"]
        epochs = range(len(acc))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        fig.suptitle(title, fontsize="x-large")
        ax1.plot(epochs, acc, label="Entraînement")
        ax1.plot(epochs, val_acc, label="Validation")
        ax1.set_title("Accuracy - Données entraînement vs. validation.")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax2.plot(epochs, loss, label="Entraînement")
        ax2.plot(epochs, val_loss, label="Validation")
        ax2.set_title("Perte - Données entraînement vs. validation.")
        ax2.set_ylabel('Perte')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        fig.show()
    

    """
        This function train our model and evaluate the carbon emission of this training. It also plot the learning curves and open tensorboard.
        Function variables:
            epochs : int
            batch_size : int
    """
    def train(self, epochs=100, batch_size=100,patience = 10):

        self.clean_logs()

        #use dataloader here

        X_train, Y_train = self._data_loader.data_retriever("train")
        X_dev, Y_dev = self._data_loader.data_retriever("dev")

        #tensorboard
        log_dir = "./prediction_conso/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        earlyStoping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        cp1 = tf.keras.callbacks.ModelCheckpoint(filepath='./prediction_conso/models/best_model.h5', monitor='val_loss', save_best_only=True)

        #tracking emissions
        # tracker = EmissionsTracker()
        # tracker.start()

        #model training
        self._model.summary()
        self._model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), batch_size=batch_size, epochs=epochs, callbacks=[earlyStoping_callback,cp1,tensorboard_callback])
        
        # hist = self._model.fit( 
        #     train_data, train_labels, 
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     validation_data=(dev_data, dev_labels), 
        #     callbacks=[earlyStoping_callback,tensorboard_callback])
        

        # self._emission  = tracker.stop()
        # title = f"{self._model_name} - Learning curves"
        # self.plot_learning_curves(hist, title)
        # print(f"Emissions: {self._emission} kgCO2e")

        #see model in tensorboard
        # !tensorboard --logdir ./logs/fit


    """
        This function clean the result folder and create the csv files for the predictions and the evaluation and add the header.
    """
    def clean_result(self):
        try:
            with open('./prediction_conso/results/predictions.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id','date',"hour",'prediction','ground_truth'])
        except:
            print("Error with predictions file")
        try:
            with open('./prediction_conso/results/result_evaluation.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['model_name'])
        except:
            print("Error with result_evaluation file")


    """
        This function evaluate the model and save the results in 2 csv file : in prediction.csv where the predictions are record and in result_evaluation.csv where the results of the evaluation are record.
        Function variables:
            test_dataset_path : str
    """
    # def evaluate(self, test_dataset_path="./prediction_conso/dataset/data_preprocess/test.csv"):
    #     self.clean_result()
    #     df = pd.read_csv(test_dataset_path)
    #     predictions = []
    #     #TODO:
    #     #recuperer test dataset avec dataloader
    #     #pour chaque élément du dataset faire les 2 lignes suivantes:
    #     output = self.predict(ligne)
    #     predictions.append(_id,type_logement,date,hour,output,ground_truth)
    #     pd.DataFrame(predictions).to_csv('./prediction_conso/results/prediction.csv', index=False)
    #     self._eval_infos=[]
    #     for step in predictions:
    #         if step[1] is not in self._eval_infos :
    #             self._eval_info[step[1]]=(0,0)
    #         self._eval_infos[step[1]] += (np.abs(step[4]-step[5]),1)
    #     for a in range(len(self._eval_infos)):
    #         self._eval_infos[a][0] = self._eval_infos[a][0]/self._eval_infos[a][1]
    #     #self._eval_infos type logement accuracy
    #     with open("./prediction_conso/results/result_evaluations.csv", 'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["Date", time.time])
    #         writer.writerow(["Emissions du training en kCO2e", self._emission])
    #         writer.writerow(["Type de logement", "Accuracy"])
    #         for e,v in self._eval_infos:
    #             writer.writerow([e, v])
                

    """
        This function predict the class of an image.
        Function variables:
            data : data to predict
    """
    def predict(self, data):
        return self._model.predict(data)


    """
        This function save the model in model_name.h5 if there is no path specified and the results of the evaluation are writed in a csv file where all the result of our models are written.
        Function variables:
            path : str
    """
    def save(self, path=None):
        if path is None:
            path = "./saved_models/{}.h5".format(self._model_name)
        self._model.save(path)
        if not os.path.exists("./saved_models/saved_models_results.csv"):
            with open("./prediction_conso/saved_models/saved_models_results.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Tous les résultats des modèles"])
        with open("./prediction_conso/saved_models/saved_models_results.csv", 'a') as f:
            writer.writerow(["Nom du modèle",self._model_name])
            writer = csv.writer(f)
            writer.writerow(["Date", time.time])
            writer.writerow(["Emissions du training en kgCO2e", self._emission])
            writer.writerow(["Type de logement", "Accuracy"])
            for e,v in self._eval_infos:
                writer.writerow([e, v])


    """
        This function load a model.
        Function variables:
            path : str
    """
    def load(self, path=None):
        if path is None:
            path = "saved_models/{}.h5".format(self._model_name)
        self._model = tf.keras.models.load_model(path)


    """
        This function return the model
    """
    def get_model(self):
        return self._model


    """
        This function return the model name
    """
    def get_model_name(self):
        return self._model_name


    """
        This function set the model name
    """
    def set_model_name(self, model_name):
        self._model_name = model_name


    """
        This function print some useful informations about the model.
    """
    def infos_model(self):
        self._model.summary()
        if self._emission != 0:
            print("Emissions during training: {} kgCO2e".format(self._emission))
        if self._eval_infos is not None:
            for e,v in self._eval_infos:
                print("Accuracy for {} : {}".format(e,v))
        # !tensorboard --logdir ./logs/fit

    

# """
#     Main function
# """
# if __name__ == "__main__":
#     model = Model("base_model")
#     model.train("dataset/data/annotations/train.csv", "data/annotations/dev.csv")
#     model.infos_model()
#     model.evaluate("dataset/data/annotations/test.csv")
#     model.save()


    

