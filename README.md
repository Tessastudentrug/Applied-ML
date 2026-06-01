# Facial Expression Recognition API

This repository contains the pipeline and API for facial expression recognition. It handles class imbalance in stratified datasets and displays CNNs (Baseline CNN and EfficientNet-B0) via a FastAPI backend.

## Repository Map
```text
.
├── Facial_Expression_Recognition
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── app
│   │   ├── __pycache__
│   │   │   ├── config.cpython-311.pyc
│   │   │   └── main.cpython-311.pyc
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── ml
│   │   │   ├── __pycache__
│   │   │   │   ├── loader.cpython-311.pyc
│   │   │   │   └── registry.cpython-311.pyc
│   │   │   ├── loader.py
│   │   │   └── registry.py
│   │   ├── routes
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── health.cpython-311.pyc
│   │   │   │   └── models.cpython-311.pyc
│   │   │   ├── health.py
│   │   │   └── models.py
│   │   └── schemas
│   │       ├── __init__.py
│   │       ├── error.py
│   │       ├── health.py
│   │       └── models.py
│   ├── data
│   │   ├── __init__.py
│   │   └── data.py
│   ├── features
│   │   ├── ExpW_preprocessor.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── preprocessing.cpython-311.pyc
│   │   └── preprocessing.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── cnn.cpython-311.pyc
│   │   │   └── effnet.cpython-311.pyc
│   │   ├── cnn.py
│   │   └── effnet.py
│   ├── train.py
│   └── training
│       ├── __init__.py
│       └── trainer.py
├── README.md
├── __init__.py
├── __pycache__
│   └── main.cpython-312.pyc
├── data
│   └── ADD_SAMPLEDATA.txt
├── docker
│   ├── Dockerfile
│   ├── pyproject_docker.toml
│   └── uv_docker.lock
├── docker-compose.yml
├── main.py
├── models
│   ├── baseline_cnn_best_weights.pth
│   └── effnet.pth
├── pyproject.toml
├── tests
│   ├── __init__.py
│   ├── data
│   │   └── __init__.py
│   ├── features
│   │   └── __init__.py
│   ├── models
│   │   └── __init__.py
│   └── test_main.py
└── uv.lock
```

# Getting Started

1. Clone the repository:

```bash
git clone https://github.com/Tessastudentrug/Applied-ML.git
cd Applied-ML
```

2. Choose Docker or full functionality

## Docker Installation (API only)
This project allows the option to run the API service as a Docker container. The Docker service image is meant for inference only and does not include the training pipeline and its dependencies. This makes the API significantly lighter than the full deployment environment.

To build and start the container:
```bash
docker compose up --build
```
The API will start automatically after finishing building. The API documentation will be available at `http://localhost:8000/docs` unless configured differently in `docker-compose.yml`


## Full installation
This project uses `uv` for dependency management. For instructions on how to install `uv`, visit their [official documentation](https://docs.astral.sh/uv/getting-started/installation/). 

### Install all project dependencies:

```bash
uv sync
```

### Running the API

The backend is built with FastAPI. To start the server locally run from root:

```bash
uv run uvicorn Facial_Expression_Recognition.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API documentation will be automatically available at:

```text
http://localhost:8000/docs
```


### Model Training

The training pipeline handles dataset stratification, preprocessing, and model evaluation. To retrain the models, execute the training script:

To train the baseline CNN:
```bash
uv run Facial_Expression_Recognition/train.py --model cnn
```

To train EfficientNet-B0:
```bash
uv run Facial_Expression_Recognition/train.py --model effnet
```

### Kaggle Authentication
Training data is downloaded automatically using KaggleHub when running `train.py`. Before running the training pipeline, configure your Kaggle API token.

You can generate a free Kaggle API token from the [Kaggle account settings page](https://www.kaggle.com/settings/api)

Run the following commands to place the token where KaggleHub expects it:

``` bash
mkdir -p ~/.kaggle 
echo '<YOUR_KAGGLE_TOKEN>' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token 
```

### Testing

To run the test suite and verify data preprocessing and API routing:

```bash
uv run pytest tests/
```


# Example API Calls

### 1. Health Check

Verify the API and model registry are running correctly.

```bash
curl --request GET "http://localhost:8000/health"
```

### 2. Model Inference

Send an image for facial expression classification.

```bash
curl --request POST "http://localhost:8000/models/{model_id}/predict" \
     --header "Content-Type: application/json" \
     --data '{"image_data": "<base64_encoded_string>"}'
```

### 3. Retrieve Models
Retrieves a list of all available models. The id's returned can be used in `/models/{model_id}/predict`
```bash
curl "http://localhost:8000/models" \
```