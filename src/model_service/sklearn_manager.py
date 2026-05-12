import joblib
import gc
from pathlib import Path
import numpy as np
import pymorphy3
import nltk
from nltk.tokenize import word_tokenize

from src.model_service import BaseModelManager

# nltk.download("punkt")


class SklearnModelManager(BaseModelManager):
    def __init__(self, folder_path: str = "./models/sklearn/v1"):
        self.folder = folder_path
        self.models = {}
        self.morph = pymorphy3.MorphAnalyzer()
        self.load()

    def load(self) -> dict:
        model_folders = [p for p in Path(self.folder).iterdir()]

        for model_path in model_folders:
            model = joblib.load(model_path)
            self.models[model_path.stem] = model

        return self.models

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        tokens = word_tokenize(text)
        clean_tokens = []
        for word in tokens:
            if word in {".", ",", "!", "?"}:
                clean_tokens.append(word)
            else:
                clean_tokens.append(self.morph.parse(word)[0].normal_form)

        return " ".join(clean_tokens)

    def predict(self, text: str, model_name: str) -> dict:
        model = self.models[model_name]
        cleaned = self._preprocess(text)

        prediction = model.predict([cleaned])[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([cleaned])[0]
            conf = max(prob)
        else:
            score = model.decision_function([cleaned])[0]
            conf = float(1 / (1 + np.exp(-score)))

        label = "AI" if prediction == 1 else "Human"
        # conf = probability[1] if prediction == 1 else probability[0]

        return {"label": label, "confidence": float(conf)}

    def unload(self):
        self.models.clear()
        gc.collect()
