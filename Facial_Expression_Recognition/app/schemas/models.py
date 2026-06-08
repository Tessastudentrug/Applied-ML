"""
Pydantic schemas and enums for model inference and model listing endpoints.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Emotion(str, Enum):
    """
    Supported emotion classes predicted by the model.
    """

    angry = "angry"
    disgust = "disgust"
    fear = "fear"
    happy = "happy"
    sad = "sad"
    surprise = "surprise"
    neutral = "neutral"
class PredictionResponse(BaseModel):
    """
    Response schema for emotion prediction endpoint.
    """

    filename: str = Field(
        example="angry_man.jpg", description="Filename of image to classify"
    )
    predicted_emotion: Emotion = Field(
        example="angry", description="Predicted emotion"
    )
    inference_time_ms: float = Field(
        example=42.15, description="Time taken to run model inference in milliseconds"
    )


class ModelsResponse(BaseModel):
    """
    Response schema for listing available models.
    """

    models: list[str] = Field(
        example=["cnn", "effnet"], description="Models currently implemented"
    )
