from fastapi import APIRouter, Depends

from schemas.prediction import TextInput, PredictionResponse
from services.prediction_service import PredictionService, get_prediction_service

router = APIRouter()


@router.post("/analyze", response_model=PredictionResponse)
async def analyze_character(
        text_input: TextInput,
        service: PredictionService = Depends(get_prediction_service)
):
    text = text_input.text
    result = service.predict(text)

    return PredictionResponse(
        labels=result["labels"],
        scores=result["scores"]
    )


@router.get("/model-info")
async def get_model_info(service: PredictionService = Depends(get_prediction_service)):
    return service.get_model_info()
