# FastAPI application for serving the model

from fastapi import FastAPI
from pydantic import BaseModel
from src.sentiment import predict_sentiment
import json
import os

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_METADATA_PATH = os.getenv("MODEL_METADATA_PATH", "src/models/model_v1/metadata.json")

# Load reference metrics at startup - offline metrics
try:
    with open(MODEL_METADATA_PATH, "r") as f:
        model_metadata = json.load(f)
except FileNotFoundError:
    model_metadata = {}

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

        "reference_metrics":{
            "accuracy": model_metadata.get("validation_accuracy"),
            "num_train_samples": model_metadata.get("num_train_samples"),
            "num_val_samples": model_metadata.get("num_val_samples"),
        }
        **stats
    }