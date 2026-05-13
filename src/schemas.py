from pydantic import BaseModel, Field


class TextRequestSchema(BaseModel):
    text: str = Field(..., min_length=100, description="Анализ текста")
    manager: str = Field(..., description="Название менеджера")
    model: str = Field(..., description="Название модели")
