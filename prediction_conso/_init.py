from models._Model import Model
from models.lstm import Lstm
from models.dnn import Dnn
import numpy as np

if __name__ == "__main__":
    model = Dnn()
    model.infos_model()
    model.train(epochs=50)
    X_train, Y_train = model._data_loader.data_retriever("train")
    print("X_train shape : ",np.array([X_train[0]]).shape)
    print("Prediction : ",model._model.predict(np.array([X_train[0]])))
    print("Real value : ",Y_train[0])
    print(model.plot_predictions_test())
    