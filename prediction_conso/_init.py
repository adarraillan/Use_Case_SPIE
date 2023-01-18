from models._Model import Model
from models.lstm import Lstm

if __name__ == "__main__":
    model = Lstm()
    # model.infos_model()
    model.train()
    
    