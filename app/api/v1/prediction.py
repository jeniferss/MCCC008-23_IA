from fastapi import APIRouter

from schemas.prediction import TextInput, PredictionResponse

from services.prediction_service import predict_traits_snippet, predict_traits_snippet2

router = APIRouter()


@router.post("/analyze", response_model=PredictionResponse)
async def analyze_character(
        text_input: TextInput,
):
    text = text_input.text
    result = predict_traits_snippet2(snippet=text)

    return PredictionResponse(
        labels=result["labels"],
        scores=result["scores"]
    )
