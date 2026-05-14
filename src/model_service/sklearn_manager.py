import joblib
import gc
from pathlib import Path
import numpy as np
import pymorphy3
import nltk
from nltk.tokenize import word_tokenize
from sklearn.compose import ColumnTransformer
import pandas as pd
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Doc

from src.model_service import BaseModelManager

# nltk.download("punkt")


class SklearnModelManager(BaseModelManager):
    def __init__(self, folder_path: str = "./models/sklearn/v1"):
        self.folder = folder_path
        self.models = {}
        self.morph = pymorphy3.MorphAnalyzer()
        self.segmenter = Segmenter()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.syntax_parser = NewsSyntaxParser(emb)
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

    def _get_chunk_linguistics(self, text: str):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)

        tokens = doc.tokens
        total = len(tokens)
        if total == 0:
            return {
                "noun_ratio": 0,
                "verb_ratio": 0,
                "adj_ratio": 0,
                "avg_word_len": 0,
                "perf_verb_ratio": 0,
                "syntax_depth_avg": 0,
            }

        pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "PUNCT": 0, "PRON": 0}
        perf_verbs = 0
        syntax_rels = 0
        word_lengths = []

        for t in tokens:
            if t.pos in pos_counts:
                pos_counts[t.pos] += 1
            if t.pos != "PUNCT":
                word_lengths.append(len(t.text))
            if t.pos == "VERB" and t.feats and t.feats.get("Aspect") == "Perf":
                perf_verbs += 1
            if t.head_id and t.head_id != "0":
                syntax_rels += 1

        return {
            "noun_ratio": pos_counts["NOUN"] / total,
            "verb_ratio": pos_counts["VERB"] / total,
            "adj_ratio": pos_counts["ADJ"] / total,
            "pron_ratio": pos_counts["PRON"] / total,
            "perf_verb_ratio": perf_verbs / (pos_counts["VERB"] + 1e-5),
            "punct_ratio": pos_counts["PUNCT"] / total,
            "avg_syntax_links": syntax_rels / total,
            "avg_word_len": (
                sum(word_lengths) / len(word_lengths) if word_lengths else 0
            ),
        }

    def predict(self, text: str, model_name: str) -> dict:
        model = self.models[model_name]
        cleaned = self._preprocess(text)

        preprocessor = model.named_steps.get("preprocessor")

        if isinstance(preprocessor, ColumnTransformer):
            features = self._get_chunk_linguistics(text)
            features["cleaned_text"] = cleaned
            X = pd.DataFrame([features])
        else:
            X = [cleaned]

        pred = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0]
            conf = max(prob)
        else:
            score = model.decision_function(X)[0]
            conf = float(1 / (1 + np.exp(-score)))

        # label = "AI" if pred == 1 else "Human"
        # conf = probability[1] if pred == 1 else probability[0]

        return {"label": pred, "confidence": float(conf)}

    def unload(self):
        self.models.clear()
        gc.collect()
