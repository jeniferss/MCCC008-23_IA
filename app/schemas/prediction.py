from pydantic import BaseModel


class TextInput(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    labels: list[str]
    scores: dict[str, float]
