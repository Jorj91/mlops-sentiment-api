import sys
import os

# Add project root to sys.path so imports work in CI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app import app
from src.sentiment import predict_sentiment

# UNIT TESTS on the model

# single sentence test. It verifies that output contains expected keys + sentiment label is one of the allowed classes
def test_sentiment_single():
    text = "I love AI!"
    result = predict_sentiment(text)

    # Check keys
    assert "sentiment" in result
    assert "probabilities" in result

    # Check sentiment value
    assert result["sentiment"] in ["positive", "neutral", "negative"]


# multiple sentences test
from src.sentiment import predict_sentiment

def test_sentiment_multiple():
    examples = [
        "I love AI!",
        "This product is terrible.",
        "Could be better.",
        "I'm not sure how I feel about this.",
        "Best purchase ever!"
    ]
    
    for text in examples:
        result = predict_sentiment(text)
        assert "sentiment" in result
        assert "probabilities" in result
        assert result["sentiment"] in ["positive", "neutral", "negative"]

# API TESTS for /stats

client = TestClient(app) # create a test client for the FastAPI app

def test_stats_endpoint():
    resp = client.get("/stats")
    assert resp.status_code == 200 # check that endpoint is reachable
    data = resp.json()

    # check that response includes monitoring fields
    assert "total_requests" in data
    assert "prediction_counts" in data

    for label in ["negative", "neutral", "positive"]:
        assert label in data["prediction_counts"]