from fastapi import APIRouter

import src.predictions.crud as crud
from src.predictions.deps import SessionDep
from src.predictions.schemas import AddPredictionSchema

router = APIRouter(prefix="/api/v1/predictions", tag="Предсказания")


@router.post("/", summary="Добавить предсказание")
async def add_prediction(data: AddPredictionSchema, db: SessionDep):
    return crud.add_prediction(data, db)
