from fastapi import FastAPI
from app.api.api import api_router
from app.models_._Model import Model

app = FastAPI(title="Consommation Prediction")

"""
    Load the model
"""
print("Loading the model...")
model = Model(model_name="bas_model",load=True)
print("Model loaded")

app.include_router(api_router, prefix="/api")