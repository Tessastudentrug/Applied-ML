from fastapi import APIRouter, Request
from pydantic import BaseModel
from Facial_Expression_Recognition.app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])



@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    registry = request.app.state.model_registry

    return {"status": "ok", "models_loaded": registry.list()}
