import joblib
import pymorphy3
import nltk
from nltk.tokenize import word_tokenize

# nltk.download("punkt")
morph = pymorphy3.MorphAnalyzer()

# 1. Загружаем модель
model = joblib.load("./models/ai_detector_model.pkl")


def clean_text(text: str):
    text = text.lower()
    tokens = word_tokenize(text)
    clean_tokens = []
    for word in tokens:
        if word in ".,!?":
            clean_tokens.append(word)
        else:
            clean_tokens.append(morph.parse(word)[0].normal_form)

    return " ".join(clean_tokens)


def predict_text(raw_text):
    cleaned = clean_text(raw_text)

    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0]

    label = "AI" if prediction == 1 else "Human"
    conf = probability[1] if prediction == 1 else probability[0]

    return {"label": label, "confidence": float(conf)}


if __name__ == "__main__":
    with open("./src/text.txt", "r", encoding="utf-8") as f:
        test_msg = f.read()
    print(predict_text(test_msg))
