from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot_ml import get_response

app = FastAPI(title="ML NLP Chatbot API")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "ML Chatbot API is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    return {"reply": get_response(request.message)}
