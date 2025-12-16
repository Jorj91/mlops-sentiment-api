# FastAPI application for serving the model
# This API exposes prediction and monitoring endpoints for online reputation analysis

from fastapi import FastAPI
from pydantic import BaseModel
from src.sentiment import predict_sentiment
import json
import os
from huggingface_hub import hf_hub_download

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_REPO_ID = os.getenv("HF_MODEL_REPO", "jorj91/sentiment-model") # HF Model hub repo containing trained model + metadata

# Load reference offline evaluation metrics at startup - offline metrics from model hub

try:
    metadata_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="metadata.json",
        repo_type="model"
    )
    with open(metadata_path, "r") as f:
        model_metadata = json.load(f)
except Exception as e:
    print(f"Warning: could not load model metadata: {e}")
    model_metadata = {}

### FastAPI Initialization

app = FastAPI(title="Sentiment Monitoring API")

class TextInput(BaseModel):
    text: str


# lightweight online monitoring statistics

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

def compute_sentiment_percentages():
    total = stats["total_requests"]
    if total == 0:
        return {label: 0.0 for label in stats["prediction_counts"]}

    return {
        label: round((count / total)*100, 2)
        for label, count in stats["prediction_counts"].items()
    }

# API Endpoints

# welcome endpoint
@app.get("/")
def home():
    return {"message": "Welcome to MachineInnovators Sentiment API!"}


# perform snetiment analysis on single input sentence
@app.post("/predict")
def predict(input_data: TextInput):
    result = predict_sentiment(input_data.text)
    log_prediction(result["sentiment"])
    save_stats()
    return result

# monitoring endpoint
@app.get("/stats")
def get_stats():
    return {
        "model_version": MODEL_VERSION,

        # offline evaluation metrics computed during training (reference only)
        "reference_metrics":{
            "accuracy": model_metadata.get("validation_accuracy"),
            "num_train_samples": model_metadata.get("num_train_samples"),
            "num_val_samples": model_metadata.get("num_val_samples"),
        },

        # online monitoring metrics
        "total_requests": stats["total_requests"],
        "prediction_counts": stats["prediction_counts"],
        "prediction_percentages": compute_sentiment_percentages()

    }