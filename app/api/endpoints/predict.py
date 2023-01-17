from fastapi import APIRouter

from app.api.endpoints.utils import InputDataPredict
import app.main as main


router = APIRouter()

    
@router.post("/")
async def prediction(data: InputDataPredict):
    """Gives the model's prediction about the input data
    Args:
        data (InputDataPredict): Input data for the prediction of the consommation

    Returns:
        Dict[str, float]: Prediction of the consommation
    """
    data = [list(data.__dict__.values())]
    prediction = main.base_model.model.predict(data)
    return {"Prediction de la consommation" : float(prediction[0][0])}