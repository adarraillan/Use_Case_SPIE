import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from models._Model import Model
import os

#lstm model with tensorflow keras
class Lstm(Model):
    _model_name = "lstm"

    def __init__(self) -> None:
        super().__init__(self._model_name)
        self._optimizer = Adam(learning_rate=0.0001)
        self._loss = tf.keras.losses.MeanSquaredError()
        self._metrics = RootMeanSquaredError()
        self._model = self._build_model()
    
    
    def _build_model(self, load = False):
        if self.train == False and os.isfile('./saved_models/'+self._model_name+'.h5'):
            return self.load()
        else:
            model = models.Sequential()
            model.add(layers.InputLayer((30,57)))
            model.add(layers.LSTM(64))
            model.add(layers.Dense(8, 'relu'))
            model.add(layers.Dense(48, activation='linear'))
            model.compile(loss=self._loss, optimizer=self._optimizer, metrics=[self._metrics])
            return model