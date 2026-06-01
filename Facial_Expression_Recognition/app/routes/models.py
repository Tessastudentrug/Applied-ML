import io
import os
import torch
from fastapi import APIRouter, HTTPException, Request, UploadFile
from PIL import Image
from Facial_Expression_Recognition.features.preprocessing import get_eval_transform
from Facial_Expression_Recognition.app.schemas.models import PredictionResponse, ModelsResponse, Emotion 
from Facial_Expression_Recognition.app.schemas.error import ErrorResponse
from Facial_Expression_Recognition.app.config import MAX_FILESIZE

EMOTIONS = list(Emotion)

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/{model_id}/predict", response_model=PredictionResponse, 
    summary="Predict facial emotion",
    description="Upload a image of a face and predict the emotion using your selected model",
    responses={
    404: {'model': ErrorResponse, "description": 'Model not found'},
    413: {'model': ErrorResponse, "description": 'File too large'},
    415: {'model': ErrorResponse, "description": 'Unsupported Mediatype'},
    422: {"description": 'Unprocessable Entity'}, # FastAPI has different error schema
    500: {'model': ErrorResponse, "description": 'Internal Error'},})
async def predict_emotion(model_id: str, request: Request, file: UploadFile) -> PredictionResponse:
    """
    Accepts an image upload and returns the predicted emotion.
    """
    registry = request.app.state.model_registry

    if model_id not in registry.list():
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found. " 
              + f"Available models: {registry.list()}",
        )

    # Use HTTP 415 (Unsupported Mediatype) if the user uploads a non-image file
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415, detail="Invalid file type. Please upload an image."
        )

    try:
        # Read the file bytes
        contents = await file.read()
        if len(contents) > int(os.getenv("MAX_FILESIZE", MAX_FILESIZE)):
            raise HTTPException(
                status_code=413,
                detail="File exceeds maximum size."
            )

        # Convert to Grayscale ("L") to match your training pipeline
        image = Image.open(io.BytesIO(contents)).convert("L")

        # Dyamic image size: 224 for Effnet, 64 for CNN
        img_size = 224 if model_id == "effnet" else 64
        transform = get_eval_transform(image_size=img_size)
        tensor = transform(image).unsqueeze(0).to(registry.device)
        model = registry.get(model_id)

        # Make the prediction without tracking gradients
        with torch.inference_mode():
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs, 1)

        emotion = EMOTIONS[predicted_idx.item()]

        # Return a clean JSON response
        #  (HTTP 200 is default for successful FastAPI returns)

        return PredictionResponse(
            filename=file.filename,
            predicted_emotion=emotion,
        )

    except HTTPException:
        raise

    except Exception as e:
        # Use HTTP 500 (Internal Server Error) if the processing crashes
        raise HTTPException(
            status_code=500, detail=f"An error occurred processing the image: {str(e)}"
        ) from e


@router.get("",
    summary="List available models",
    description="Returns model id's that can be used for prediction",
    response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    registry = request.app.state.model_registry
    return ModelsResponse(models=registry.list())
