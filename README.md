---
title: Sentiment Monitoring API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Sentiment Monitoring API - MLOps Project

## Project Overview
This project implements an end-to-end MLOps pipeline for sentiment analysis to monitor online reputation using social media text.
The solution follows MLOps best practices, covering model training, CI/CD, deployment, and monitoring.

Company scenario: MachineInnovators Inc.
Goal: Automatically analyze sentiment (positive / neutral / negative) from social media text and monitor trends over time.


## Model
- Pretrained model: cardiffnlp/twitter-roberta-base-sentiment-latest (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- Dataset: TweetEval (public dataset for sentiment classification)
- Task: positive / neutral / negative classification
- Fine-tuning: demo fine-tuning on a small subset (100 training samples and 50 validation samples)
- Offline evaluation: validation accuracy computed during training. Offline metrics are stored as model metadata and exposed via the API.

## MLOps Pipeline (CI/CD)
- Automated CI/CD pipeline implemented using GitHub Actions

### CI (on push / pull request)
- Install dependencies
- Run unit tests
- Run API tests
- Use a tiny model for fast CI
- Build Docker Image

### CD (on Git tag 'v*')
- Train model
- Compute validation accuracy
- Upload model to Hugging Face Model Hub
- Deploy application to Hugging Face Spaces
- version-aware deployment with Git tags


## API Endpoints
- POST /predict
    Predict sentiment for a single sentence.

- GET /stats
    Monitoring endpoint, exposing: 
        model version
        offline reference metrics (validation accuracy)
        online monitoring of sentiment distribution over incoming requests

## Deployment
- Hugging Face Space (Docker-based)
- Public API endpoint
- Live Demo: https://jorj91-sentiment-api.hf.space/docs


## Future Improvements
- Automatic retraining based on new labeled data
- Data drift detection
- Advanced monitoring dashboards (e.g. Prometheus / Grafana)