from fastapi import APIRouter
from app.api.endpoints import predict, model


api_router = APIRouter()


api_router.include_router(predict.router, prefix="/predict", tags=["predict"])
api_router.include_router(model.router, prefix="/model", tags=["model"])