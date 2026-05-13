async function loadManagers() {
    try {
        const response = await fetch(`/managers`);
        const data = await response.json();
        console.log(data);

        const managerSelect = document.getElementById("manager");

        data.managers.forEach(manager => {
            const option = document.createElement("option");

            option.value = manager;
            option.innerText = manager;

            managerSelect.appendChild(option);
        });
    }
    catch (error) {
        console.error("Ошибка загрузки менеджеров:", error);
    }
}


async function loadModels() {
    const manager = document.getElementById("manager").value;

    try {
        const response = await fetch(`/models/${manager}`);
        const data = await response.json();

        const modelSelect = document.getElementById("model");
        modelSelect.innerHTML = "";

        if (!data.models) return;

        data.models.forEach(model => {
            const option = document.createElement("option");

            option.value = model;
            option.innerText = model;

            modelSelect.appendChild(option);
        });

    } catch (error) {
        console.error("Ошибка загрузки моделей:", error);
    }
}


async function predictText() {
    const text = document.getElementById("text").value;
    const manager = document.getElementById("manager").value;
    const model = document.getElementById("model").value;

    const loader = document.getElementById("loader");
    const result = document.getElementById("result");
    const button = document.getElementById("predictBtn");

    if (text.length < 10) {
        alert("Введите текст");
        return;
    }

    loader.classList.remove("hidden");
    result.classList.add("hidden");

    button.disabled = true;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text,
                manager: manager,
                model: model
            })
        });

        const data = await response.json();

        const isAI = data.label === "AI";
        const confidence = (data.confidence * 100).toFixed(2);

        const label = document.getElementById("label");
        const confText = document.getElementById("confidence");
        const confBar = document.getElementById("confidenceBar");

        label.innerText = isAI
            ? "🤖 AI Generated"
            : "👨 Human Written";

        label.className = isAI
            ? "text-2xl font-bold text-red-600 mb-2"
            : "text-2xl font-bold text-green-600 mb-2";

        confText.innerText = `Уверенность: ${confidence}%`;

        confBar.style.width = `${confidence}%`;

        confBar.className = isAI
            ? "h-4 rounded-full bg-red-500 transition-all duration-700"
            : "h-4 rounded-full bg-green-500 transition-all duration-700";

        result.classList.remove("hidden");

    } catch (error) {
        console.error(error);
        alert("Ошибка запроса");
    } finally {
        loader.classList.add("hidden");
        button.disabled = false;
    }
}


document
    .getElementById("manager")
    .addEventListener("change", loadModels);


window.onload = async () => {
    await loadManagers();
    await loadModels();
};