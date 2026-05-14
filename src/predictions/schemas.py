from pydantic import BaseModel, Field


class TextRequestSchema(BaseModel):
    text: str = Field(..., min_length=100, description="Исходный текст")
    manager_name: str = Field(..., description="Название менеджера")
    model_name: str = Field(..., description="Название модели")
    target_label: int | None = Field(
        default=None, ge=0, le=1, description="Целевой класс"
    )


class AddPredictionSchema(BaseModel):
    text: str = Field(..., min_length=100, description="Исходный текст")
    predicted_label: int = Field(
        ..., description="Предсказанный класс (0 = Human, 1 = AI)"
    )
    confidence: float = Field(..., ge=0, le=1, description="Уверенность в ответе")
    target_label: int | None = Field(
        default=None, ge=0, le=1, description="Целевой класс"
    )
    manager_name: str = Field(..., description="Название менеджера")
    model_name: str = Field(..., description="Название модели")
