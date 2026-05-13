import time
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.model_service import BertModelManager, SklearnModelManager
from src.model_service.service import TextClassificationService
from src.schemas import TextRequestSchema


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = TextClassificationService()
    app.state.service.initialize("bert", BertModelManager("./models/bert"))
    app.state.service.initialize(
        "sklearn_v1", SklearnModelManager("./models/sklearn/v1")
    )
    app.state.service.initialize(
        "sklearn_v2", SklearnModelManager("./models/sklearn/v2")
    )
    yield
    app.state.service.unload("bert")
    app.state.service.unload("sklearn_v1")
    app.state.service.unload("sklearn_v2")


app = FastAPI(title="Детектор AI-текстов", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)

app.mount("/static", StaticFiles(directory="./src/static"), name="static")

templates = Jinja2Templates(directory="./src/templates")


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-TIME-HEADER"] = str(duration)
    return response


@app.get("/health", summary="Проверка сервера")
async def health():
    return {"status": "ok"}


@app.get("/client")
async def client(request: Request):
    return templates.TemplateResponse(request=request, name="client.html", context={})


@app.get("/managers", summary="Получить все менеджеры")
async def managers():
    return {"managers": app.state.service.get_available_managers()}


@app.get("/models/{manager_name}", summary="Получить все модели менеджера")
async def get_models(manager_name: str):
    return {"models": app.state.service.get_manager_models(manager_name)}


@app.post("/predict", summary="Анализ текста")
async def predict(data: TextRequestSchema):
    response = app.state.service.predict(
        manager_name=data.manager,
        text=data.text,
        model_name=data.model,
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
