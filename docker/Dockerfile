FROM python:3.11-slim

WORKDIR /usr/src/app

# Only including the bare minimum to run it
COPY docker/pyproject_docker.toml ./pyproject.toml 
COPY docker/uv_docker.lock ./uv.lock
COPY Facial_Expression_Recognition/app ./Facial_Expression_Recognition/app
COPY Facial_Expression_Recognition/features/preprocessing.py ./Facial_Expression_Recognition/features/preprocessing.py
COPY Facial_Expression_Recognition/features/__init__.py ./Facial_Expression_Recognition/features/__init__.py
COPY Facial_Expression_Recognition/models ./Facial_Expression_Recognition/models
COPY Facial_Expression_Recognition/__init__.py ./Facial_Expression_Recognition/__init__.py
COPY models ./models

RUN pip install uv
RUN uv sync --frozen --no-dev --no-install-project

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "Facial_Expression_Recognition.app.main:app", "--host", "0.0.0.0", "--port", "8000"]