# FastAPI application for serving the model

from fastapi import FastAPI
from pydantic import BaseModel
from src.sentiment import predict_sentiment
import json
import os

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

app = FastAPI(title="Sentiment Monitoring API")

class TextInput(BaseModel):
    text: str


# Lightweight monitoring

stats = {
    "total_requests": 0,
    "prediction_counts": {"negative": 0, "neutral": 0, "positive": 0}
}

def log_prediction(label: str):
    stats["total_requests"] += 1
    if label in stats["prediction_counts"]:
        stats["prediction_counts"][label] += 1

def save_stats():
    with open("stats.json", "w") as f:
        json.dump(stats, f)

# Endpoints

@app.get("/")
def home():
    return {"message": "Welcome to MachineInnovators Sentiment API"}

@app.post("/predict")
def predict(input_data: TextInput):
    result = predict_sentiment(input_data.text)
    log_prediction(result["sentiment"])
    save_stats()
    return result


@app.get("/stats")
def get_stats():
    return {
        "model_version": MODEL_VERSION,
        **stats
    }