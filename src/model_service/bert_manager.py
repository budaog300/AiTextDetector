import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.model_service import BaseModelManager


class BertModelManager(BaseModelManager):
    def __init__(self, folder_path: str = "./models/bert"):
        self.folder = folder_path
        self.models = {}
        self.tokenizers = {}
        self.devices = {}
        self.load()

    def load(self) -> dict:
        model_folders = [
            p
            for p in Path(self.folder).rglob("*")
            if p.is_dir() and (p / "config.json").exists()
        ]

        for model_path in model_folders:
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()

            key = model_path.stem
            self.models[key] = model
            self.tokenizers[key] = tokenizer
            self.devices[key] = device

        return self.models

    def predict(self, text: str, model_name: str) -> dict:
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        device = self.devices[model_name]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(np.max(probs))

        label = "AI" if pred == 1 else "Human"

        return {
            "label": label,
            "confidence": conf,
        }

    def unload(self):
        self.models.clear()
        self.tokenizers.clear()
        gc.collect()
        torch.cuda.empty_cache()
