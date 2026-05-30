from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import Facial_Expression_Recognition.app.routes as routes
from Facial_Expression_Recognition.app.ml.registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model_registry = ModelRegistry()
    app.state.model_registry.load(device=device)
    yield
    app.state.model_registry.clear()


app = FastAPI(
    title="Facial Expression Recognition API",
    description="RESTful API for classifying emotions from face images.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your local HTML file to talk to the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(routes.health_router)
app.include_router(routes.models_router)
