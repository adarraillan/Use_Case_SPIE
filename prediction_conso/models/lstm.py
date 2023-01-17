import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from models._Model import Model


#lstm model with tensorflow keras
class lstm(Model):
    _model_name = "lstm"

    def __init__(self,embedding_dim = 50, vocab_size = 7*48) -> None:
        super().__init__()
        self._optimizer = "adam"
        self._loss = tf.keras.losses.MeanSquaredError
        self._metrics = 'accuracy'
        self._model = self._build_model()
        

    
    def _build_model(self, load = False):
        if load:
            return self.load("./prediction_conso/saved_models/"+self._model_name+".h5")
        else:
            inputs = tf.keras.Input(shape=(None,))
            lstm = tf.keras.layers.LSTM(32)(inputs)
            output = tf.keras.layers.Dense(48, activation='sigmoid')(lstm)
            model = tf.keras.Model(inputs=inputs, outputs=output)
            model.compile(loss=self._loss, optimizer=self._optimizer, metrics=[self._metrics])
            return model

if __name__ == "__main__":
    model = lstm()
    model.infos_model()