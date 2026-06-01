from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(example="ok", description="Current API health status")
    models_loaded: list[str] = Field(
        example=["cnn", "effnet"], description="Models currently implemented"
    )
