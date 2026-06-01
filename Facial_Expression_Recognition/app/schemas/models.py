from enum import Enum
from pydantic import BaseModel, Field

class Emotion(str, Enum):
    angry = "angry"
    disgust="disgust"
    fear="fear"
    happy="happy" 
    sad="sad" 
    surprise="surprise" 
    neutral="neutral"

class PredictionResponse(BaseModel):
    filename: str = Field(example="angry_man.jpg", description="Filename of image to classify")
    predicted_emotion: Emotion = Field(example="angry", description="Predicted emotion")

class ModelsResponse(BaseModel):
    models: list[str] = Field(example=["cnn", "effnet"], description="Models currently implemented")


