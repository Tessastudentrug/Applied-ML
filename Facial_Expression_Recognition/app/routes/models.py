import io

import torch
from fastapi import APIRouter, HTTPException, Request, UploadFile
from PIL import Image

from Facial_Expression_Recognition.app.config import EMOTION_LABELS
from Facial_Expression_Recognition.features.preprocessing import get_eval_transform

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/{model_id}/predict")
async def predict_emotion(model_id: str, request: Request, file: UploadFile):
    """
    Accepts an image upload and returns the predicted emotion.
    """
    registry = request.app.state.model_registry

    if model_id not in registry.list():
        raise HTTPException(
            status_code=404,
            detail=f"""Model {model_id} not found.
              Available models: {registry.list()}""",
        )

    # Use HTTP 400 (Bad Request) if the user uploads a non-image file
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image."
        )

    try:
        # Read the file bytes
        contents = await file.read()

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

        emotion = EMOTION_LABELS[predicted_idx.item()]

        # Return a clean JSON response
        #  (HTTP 200 is default for successful FastAPI returns)

        return {
            "filename": file.filename,
            "predicted_emotion": emotion,
            "status_code": 200,
        }

    except Exception as e:
        # Use HTTP 500 (Internal Server Error) if the processing crashes
        raise HTTPException(
            status_code=500, detail=f"An error occurred processing the image: {str(e)}"
        ) from e


@router.get("")
async def list_models(request: Request):
    registry = request.app.state.model_registry
    return registry.list()
