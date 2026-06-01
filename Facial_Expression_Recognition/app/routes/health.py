"""
Health check endpoint for monitoring API and model availability.
"""

from fastapi import APIRouter, Request

from Facial_Expression_Recognition.app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Return the API health status and a list of loaded models.

    Args:
        request: FastAPI request containing application state.

    Returns:
        HealthResponse containing API status and loaded models.
    """
    registry = request.app.state.model_registry

    return HealthResponse(status="ok", models_loaded=registry.list())
