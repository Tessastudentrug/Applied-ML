from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(request: Request):
    registry = request.app.state.model_registry

    return {"status": "ok", "models_loaded": registry.list()}
