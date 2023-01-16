import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from _Model import Model


#lstm model with tensorflow keras
class lstm(Model):
    _model_name = "lstm"
    _embedding_dim = 50
    _vocab_size = 48*7
    _max_relev_size = 48*7

    def __init__(self,embedding_dim = 50, vocab_size = 7*48) -> None:
        super().__init__()
        self._optimizer = "adam"
        self._loss = tf.keras.losses.MeanSquaredError
        self._metrics = 'accuracy'
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._model = self._build_model()
        

    
    def _build_model(self, load = False):
        if load:
            return self.load()
        else:
            model = tf.models.Sequential(name = "lstm")
            model.add(tf.layers.Embedding(input_dim=self._vocab_size, output_dim=self._embedding_dim,embeddings_regularizer=tf.keras.regularizers.l1(0.0005)))
            model.add(tf.layers.LSTM(32))
            model.add(tf.layers.Dense(48, activation='sigmoid'))
            model.compile(loss=self._loss, optimizer=self._optimizer, metrics=[self._metrics])
            return model

if __name__ == "__main__":
    model = lstm()
    model.infos_model()