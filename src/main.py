import time
from fastapi import FastAPI, Request
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
    app.state.service.initialize("sklearn", SklearnModelManager("./models/sklearn/v2"))
    yield
    app.state.service.unload("bert")
    app.state.service.unload("sklearn")


app = FastAPI(title="Детектор AI-текстов", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)


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


@app.get("/client", summary="Клиентский интерфейс приложения")
async def get_client():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>AI Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>

<body class="bg-slate-50 min-h-screen flex items-center justify-center p-6">

<div class="w-full max-w-2xl bg-white p-6 rounded-2xl shadow">

    <h1 class="text-2xl font-bold mb-4">🔍 AI Detector</h1>

    <!-- MANAGER -->
    <label class="text-sm">Выбор типа модели</label>
    <select id="manager" class="w-full border p-2 rounded mb-3">
        <option value="bert">BERT</option>
        <option value="sklearn">Sklearn</option>
    </select>

    <!-- MODEL -->
    <label class="text-sm">Выбор модели</label>
    <select id="model" class="w-full border p-2 rounded mb-3"></select>

    <!-- TEXT -->
    <textarea id="text" class="w-full h-40 border p-3 rounded mb-3"
        placeholder="Введите текст..."></textarea>

    <button onclick="predict()"
        class="w-full bg-blue-600 text-white p-3 rounded hover:bg-blue-700">
        Анализировать
    </button>

    <!-- RESULT -->
    <div id="result" class="mt-5 hidden">
        <h2 id="label" class="text-xl font-bold"></h2>
        <p id="conf"></p>
    </div>

</div>

<script>

async function loadModels() {
    const manager = document.getElementById("manager").value;

    const res = await fetch(`/models/${manager}`);
    const data = await res.json();

    const select = document.getElementById("model");
    select.innerHTML = "";

    if (!data.models) return;

    data.models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.innerText = m;
        select.appendChild(opt);
    });
}

document.getElementById("manager").addEventListener("change", loadModels);

async function predict() {
    const text = document.getElementById("text").value;
    const manager = document.getElementById("manager").value;
    const model = document.getElementById("model").value;

    const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            text: text,
            manager: manager,
            model: model
        })
    });

    const data = await res.json();

    document.getElementById("result").classList.remove("hidden");
    document.getElementById("label").innerText = data.label;
    document.getElementById("conf").innerText =
        "Confidence: " + (data.confidence * 100).toFixed(1) + "%";
}

window.onload = async () => {
    await loadModels();
};

</script>

</body>
</html>
""")


@app.get("/managers", summary="Получить все менеджеры")
async def managers():
    return app.state.service.get_available_managers()


@app.get("/models/{manager_name}", summary="Получить все модели менеджера")
async def get_models(manager_name: str):
    return app.state.service.get_manager_models(manager_name)


@app.post("/predict", summary="Анализ текста")
async def predict(data: TextRequestSchema):
    response = app.state.service.predict(
        manager_name=data.manager,
        text=data.text,
        model_name=data.model_name,
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
