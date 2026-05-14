from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from src.predictions.models import Prediction
from src.predictions.schemas import AddPredictionSchema


async def add_prediction(data: AddPredictionSchema, db: AsyncSession):
    new_predict = Prediction(**data.model_dump())
    db.add(new_predict)
    try:
        await db.commit()
        await db.refresh(new_predict)
        return new_predict
    except SQLAlchemyError as e:
        db.rollback()
        raise e
