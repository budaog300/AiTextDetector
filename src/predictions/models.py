from sqlalchemy import String, Text, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Default


class Prediction(Default):
    __tablename__ = "predictions"

    text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    target_label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    manager_name: Mapped[str] = mapped_column(String(50), nullable=False)
