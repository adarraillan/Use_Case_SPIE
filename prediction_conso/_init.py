from models._Model import Model
from models.lstm import lstm

if __name__ == "__main__":
    model = lstm()
    model.infos_model()
    model.train()
    
    