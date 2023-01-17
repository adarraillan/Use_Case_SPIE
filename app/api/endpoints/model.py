from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
import json
from csv import writer

#TODO
from app.models_._Model import Model
from app.api.endpoints.utils import InputData
import app.main as main

router = APIRouter()


@router.get("/")
async def get_model():
    """Returns the model serialization as json

    Returns:
        Dict[str, Dict]: Nested objects representing layers and parameters
    """
    model_json = main.base_model.model.to_json()
    return json.loads(model_json)

    
#TODO
@router.get("/description")
async def get_model_infos():
    """Returns basic informations about the model

    Returns:
        Dict[str, str]: Layers description, loss, optimizer and metric
    """
    return {}


@router.post("/retrain")
async def retrain_model():
    """Retrains the model using the data saved on disk

    Returns:
        Dict[str, str]: Message confirming that the retrain was successfull
    """
    main.base_model.train()
    main.base_model.save()
    return {'message' : "Model successfully retrained and saved"}


@router.post("/save")
async def save_model():
    """Saves the model on disk

    Returns:
        Dict[str, str]: Message confirming that the save was successfull
    """
    main.base_model.save()
    return {'message' : "Model successfully saved"}


@router.post("/evaluate")
async def evaluate_model():
    """Evaluates the model using the data saved on disk

    Returns:
        Dict[str, float]: Loss and metric values
    """
    return main.base_model.evaluate()