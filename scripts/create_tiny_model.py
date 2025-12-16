"""
Create a lightweight local copy of a pretrained model for CI/testing purposes.
The model is downloaded once and saved to a temporary directory (/tmp).
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Pretrained sentiment analysis model from HF Hub
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# HF cache directory (default to /tmp/huggingface if not set)
HF_HOME = os.getenv("HF_HOME", "/tmp/huggingface")
os.environ["HF_HOME"] = HF_HOME

# directory to save the tiny model (default to /tmp/local_model if not set)
MODEL_DIR = os.getenv("TINY_MODEL_DIR", "/tmp/local_model")
os.makedirs(MODEL_DIR, exist_ok=True)


print(f"Downloading model {MODEL_ID} with cache in {HF_HOME}")
print(f"Saving tiny model to {MODEL_DIR}")

# donwload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, ignore_mismatched_sizes=True)

# Save model and tokenizer locally for CI usage
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"Tiny model saved successfully at {MODEL_DIR}")
