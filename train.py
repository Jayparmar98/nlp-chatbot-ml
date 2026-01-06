import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open("app/intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Save model
joblib.dump(model, "app/model/intent_model.pkl")
joblib.dump(vectorizer, "app/model/vectorizer.pkl")

print("✅ Model trained and saved successfully")
