# Facial Expression Recognition API

This repository contains the pipeline and API for facial expression recognition. It handles class imbalance in stratified datasets and displays CNNs (Baseline CNN and EfficientNet-B0) via a FastAPI backend.

## Repository Map
```text
.
├── Facial_Expression_Recognition/
│   ├── app/
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── ml/
│   │   ├── routes/
│   │   └── schemas/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── train.py
├── data/
├── docker/
├── models/
│   ├── baseline_cnn_best_weights.pth
│   └── effnet.pth
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## Environment Setup

This project uses `uv`

### Local Installation

1. Clone the repository:

```bash
git clone https://github.com/Tessastudentrug/Applied-ML.git
cd Applied-ML
```

2. Sync the environment:

```bash
uv sync
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Docker Installation

To run the API in a containerized environment:

```bash
docker compose up --build
```

## Running the API

The backend is built with FastAPI. To start the server locally:

```bash
uv run uvicorn Facial_Expression_Recognition.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API documentation will be automatically available at:

```text
http://localhost:8000/docs
```

## Example API Calls

### 1. Health Check

Verify the API and model registry are running correctly.

```bash
curl --request GET "http://localhost:8000/health"
```

### 2. Model Inference

Send an image for facial expression classification.

```bash
curl --request POST "http://localhost:8000/predict" \
     --header "Content-Type: application/json" \
     --data '{"image_data": "<base64_encoded_string>", "model_type": "effnet"}'
```

## Model Training

The training pipeline handles dataset stratification, preprocessing, and model evaluation. To retrain the models, execute the training script:

```bash
uv run Facial_Expression_Recognition/train.py
```

## Testing

To run the test suite and verify data preprocessing and API routing:

```bash
uv run pytest tests/
```
