'''
This module provides a function `predict_sentiment` that can run in two modes:
- Fake model mode (used during CI/CD and fast testing)
- Real model mode (using HF pretrained model)
'''

import os

USE_FAKE_MODEL = os.getenv("USE_FAKE_MODEL", "false").lower() == "true"


# FAKE MODEL (CI / docker test)

if USE_FAKE_MODEL:

    LABELS = ["negative", "neutral", "positive"]

    # fake sentiment predictor that always returns a fixed positive prediction (intentional).
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
    

# REAL MODEL (HF pre-trained)
  
else:
    # Use Hugging Face Hub model by default
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import logging
    logging.set_verbosity_error()
    import torch
    import numpy as np

    # Default: HF Hub model ID. Or, can be overridden with local folder path
    MODEL_PATH = os.getenv("MODEL_PATH", "cardiffnlp/twitter-roberta-base-sentiment-latest")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval() # inference mode

    LABELS = ["negative", "neutral", "positive"]

    def predict_sentiment(text: str) -> dict:

        # predict sentiment for a single input test.

        inputs = tokenizer(text, return_tensors="pt", truncation=True) # Tokenize input text 

        with torch.no_grad():
            outputs = model(**inputs) # run inference with model
            scores = outputs.logits[0].cpu().numpy()
            probs = np.exp(scores) / np.exp(scores).sum() #  softmax to convert logits to probabilities

        # get label with highest prob
        label = LABELS[np.argmax(probs)]

        return {
            "text": text,
            "sentiment": label,
            "probabilities": {LABELS[i]: round(float(probs[i]),2) for i in range(len(LABELS))}
        }