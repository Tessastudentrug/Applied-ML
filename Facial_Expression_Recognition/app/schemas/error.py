from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str = Field(
        example="Model logreg not found. Available models:" " ['cnn', 'effnet']"
    )
