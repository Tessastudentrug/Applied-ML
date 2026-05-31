from enum import Enum
from pydantic import BaseModel

class Emotion(str, Enum):
    angry = "angry",
    disgust="disgust",
    fear="fear", 
    happy="happy", 
    sad="sad", 
    surprise="surprise", 
    neutral="neutral"

class PredictionResponse(BaseModel):
    filename: str
    predicted_emotion: Emotion
    status_code: int

class ModelsResponse(BaseModel):
    models: list[str]


