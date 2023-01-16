import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from _Model import Model


#lstm model with tensorflow keras
class lstm(Model):
    _model_name = "lstm"


    def __init__(self) -> None:
        super().__init__()
        self._optimizer = "adam"
        self._loss = tf.keras.losses.MeanSquaredError
        self._metrics = 'accuracy'
        self._model = self._build_model()
        

    
    def _build_model(self, load = False):
        if load:
            return self.load()
        else:
            model = models.Sequential()
            model.add(layers.LSTM(32, input_shape=(None, 1)))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss=self._loss, optimizer=self._optimizer, metrics=[self._metrics])
            return model