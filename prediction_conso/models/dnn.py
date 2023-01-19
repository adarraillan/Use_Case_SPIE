import tensorflow as tf
from tensorflow import keras
print(keras.__version__)
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from models._Model import Model


#lstm model with tensorflow keras
class Dnn(Model):
    _model_name = "dnn"

    def __init__(self,load = False) -> None:
        super().__init__(self._model_name)
        self._optimizer = tf.keras.optimizers.Adam(lr=0.2, decay=0.1)
        self._loss = tf.keras.losses.MeanSquaredError()
        self._metrics = RootMeanSquaredError()
        self._model = self._build_model(load)
    
    
    def _build_model(self, load = False):
        model = models.Sequential()
        model.add(layers.InputLayer((30,69)))
        model.add(layers.Flatten())

        model.add(layers.Dense(512,activation='LeakyReLU', kernel_regularizer=regularizers.L1L2(0.01,0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(512, activation='LeakyReLU', kernel_regularizer=regularizers.L1L2(0.01,0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(256, activation='LeakyReLU', kernel_regularizer=regularizers.L1L2(0.01,0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Dense(256, activation='LeakyReLU', kernel_regularizer=regularizers.L1L2(0.01,0.01)))
        keras.layers.BatchNormalization(),
        model.add(layers.Dropout(0.1))

        model.add(layers.Dense(128, activation='LeakyReLU', kernel_regularizer=regularizers.L1L2(0.01,0.01)))
        keras.layers.BatchNormalization(),
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(64, activation='LeakyReLU',kernel_regularizer=regularizers.L1L2(0.01,0.01)))
        model.add(layers.Dense(49, activation='linear'))


        model.compile(loss=self._loss, optimizer=self._optimizer , metrics = [self._metrics])

        if load:
            model=tf.keras.models.load_model("./prediction_conso/saved_models/best_model.h5")
        return model