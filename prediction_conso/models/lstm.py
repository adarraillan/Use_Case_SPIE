import tensorflow as tf
from tensorflow import keras
print(keras.__version__)
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from models._Model import Model


#lstm model with tensorflow keras
class Lstm(Model):
    _model_name = "lstm"

    def __init__(self,load = False) -> None:
        super().__init__(self._model_name)
        self._optimizer = "adam"
        self._loss = tf.keras.losses.MeanSquaredError()
        self._metrics = RootMeanSquaredError()
        self._model = self._build_model(load)
    
    
    def _build_model(self, load = False):
        model = models.Sequential()
        model.add(layers.InputLayer((30,69)))
        model.add(layers.LSTM(64))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(64))
        model.add(layers.Dense(49, activation='linear'))
        model.compile(loss=self._loss, optimizer=self._optimizer, metrics=[self._metrics])
        if load:
            model=tf.keras.models.load_model("./prediction_conso/saved_models/best_model.h5")
        return model