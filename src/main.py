import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.inference import predict_text
from src.schemas import TextRequestSchema


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


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
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Text Detector</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Inter', sans-serif; }
                .glass { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); }
            </style>
        </head>
        <body class="bg-slate-50 min-h-screen flex items-center justify-center p-4">

            <div class="max-w-2xl w-full glass p-8 rounded-2xl shadow-2xl border border-slate-200">
                <h1 class="text-3xl font-bold text-slate-800 mb-2 flex items-center gap-2">
                    🔍 AI Detector
                </h1>
                <p class="text-slate-500 mb-6">Проверьте, написан ли текст человеком или нейросетью.</p>

                <textarea id="textInput" 
                    class="w-full h-48 p-4 border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all resize-none mb-4 text-slate-700"
                    placeholder="Вставьте текст для анализа (минимум 100 символов)..."></textarea>

                <button onclick="analyzeText()" id="btn"
                    class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-xl transition-all shadow-lg active:scale-[0.98]">
                    Проверить текст
                </button>

                <!-- Блок результатов (скрыт по умолчанию) -->
                <div id="resultBox" class="mt-8 hidden border-t pt-6">
                    <div class="flex justify-between items-end mb-2">
                        <span class="text-sm font-medium text-slate-500">Результат анализа:</span>
                        <span id="label" class="text-2xl font-bold"></span>
                    </div>
                    
                    <div class="w-full bg-slate-200 rounded-full h-4 mb-2">
                        <div id="confidenceBar" class="h-4 rounded-full transition-all duration-1000" style="width: 0%"></div>
                    </div>
                    
                    <div class="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400">
                        <span id="confValue">0%</span>
                    </div>
                </div>

                <div id="loader" class="mt-4 hidden text-center text-blue-600 font-medium animate-pulse">
                    Анализирую паттерны текста...
                </div>
            </div>

            <script>
                async function analyzeText() {
                    const text = document.getElementById('textInput').value;
                    const btn = document.getElementById('btn');
                    const resultBox = document.getElementById('resultBox');
                    const loader = document.getElementById('loader');
                    const label = document.getElementById('label');
                    const confBar = document.getElementById('confidenceBar');
                    const confValue = document.getElementById('confValue');
                    const text_length = 100;

                    if (text.length < text_length) {
                        alert('Пожалуйста, введите более длинный текст');
                        return;
                    }

                    // Визуальное состояние загрузки
                    btn.disabled = true;
                    btn.classList.add('opacity-50');
                    loader.classList.remove('hidden');
                    resultBox.classList.add('hidden');

                    try {
                        // Замени /predict на твой реальный эндпоинт, если он отличается
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ text: text })
                        });

                        const data = await response.json();

                        // Обработка данных
                        const isAI = data.label === 'AI';
                        const confidence = (data.confidence * 100).toFixed(1);

                        label.innerText = isAI ? '🤖 Это ИИ' : '👨‍💻 Это человек';
                        label.className = `text-2xl font-bold ${isAI ? 'text-red-600' : 'text-green-600'}`;
                        
                        // Настройка прогресс-бара
                        confValue.innerText = `Уверенность: ${confidence}%`;                        
                        confBar.style.width = `${confidence}%`;
                        confBar.className = `h-4 rounded-full transition-all duration-1000 ${isAI ? 'bg-red-500' : 'bg-green-500'}`;
                        resultBox.classList.remove('hidden');
                    } catch (error) {
                        alert('Ошибка при соединении с сервером');
                        console.error(error);
                    } finally {
                        btn.disabled = false;
                        btn.classList.remove('opacity-50');
                        loader.classList.add('hidden');
                    }
                }
            </script>
        </body>
        </html>
        """)


@app.post("/predict", summary="Анализ текста")
async def predict(data: TextRequestSchema):
    response = predict_text(data.text)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
