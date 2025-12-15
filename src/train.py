# Purpose: to have a model trained on a small dataset to test local pipeline, API, and integration 
# (small fine-tuning is performed on 100 train examples + 50 validation examples (just a demo run)).
# The model used is cardiffnlp/twitter-roberta-base-sentiment-latest (already pre-trained on Twitter sentiment).
import warnings
from transformers import logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset


MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
#MODEL_DIR = "src/models/model_v1"
MODEL_DIR = os.getenv("MODEL_PATH", "src/models/model_v1")
LABELS = ["negative", "neutral", "positive"]

def train_model():
    print("Step 1: Loading TweetEval dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")

    # Take small subset for demo
    train_data = dataset["train"].select(range(100))
    val_data = dataset["validation"].select(range(50))

    print("Step 2: Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, ignore_mismatched_sizes=True)

    print("Step 3: Tokenizing dataset...")
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)
        
    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Step 4: Fine-tuning model (1 epoch for demo)...")
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print("Step 4: Fine-tuning completed.")


    print("Step 4b: Evaluating model on validation set...")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Validation accuracy: {val_accuracy:.4f}")


    # Step 5: Save model and metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    metadata = {"status": "trained", 
                "model_id": MODEL_ID,
                "validation_accuracy": round(val_accuracy, 4),
                "num_train_samples": len(train_data),
                "num_val_samples": len(val_data)
                }
    
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    
    print(f"Step 5: Model saved to {MODEL_DIR}")
    print("Training pipeline executed successfully.")
    return {"status": "ok"}

if __name__ == "__main__":
    train_model()
