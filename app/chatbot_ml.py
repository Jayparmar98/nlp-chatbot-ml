import json
import os
import random
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "intent_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

def get_response(message: str) -> str:
    X = vectorizer.transform([message.lower()])
    intent = model.predict(X)[0]

    for item in intents["intents"]:
        if item["tag"] == intent:
            return random.choice(item["responses"])

    return "Sorry, I didn't understand that."
