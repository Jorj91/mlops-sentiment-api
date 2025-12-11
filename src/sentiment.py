

import os

USE_FAKE_MODEL = os.getenv("USE_FAKE_MODEL", "false").lower() == "true"


# FAKE MODEL (CI / docker test)

if USE_FAKE_MODEL:

    LABELS = ["negative", "neutral", "positive"]

    def predict_sentiment(text: str) -> dict:
        return {
            "text": text,
            "sentiment": "positive",
            "probabilities": {
                "negative": 0.05,
                "neutral": 0.05,
                "positive": 0.90
            }
        }
    

# REAL MODEL (hugging face pre-trained)
  
else:
    # Use Hugging Face Hub model by default
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import numpy as np

    # Default: hub model. Can be overridden with MODEL_PATH env var (e.g. local folder or other model id)
    MODEL_PATH = os.getenv("MODEL_PATH", "cardiffnlp/twitter-roberta-base-sentiment-latest")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    LABELS = ["negative", "neutral", "positive"]

    def predict_sentiment(text: str) -> dict:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits[0].cpu().numpy()
            probs = np.exp(scores) / np.exp(scores).sum()

        label = LABELS[np.argmax(probs)]

        return {
            "text": text,
            "sentiment": label,
            "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        }