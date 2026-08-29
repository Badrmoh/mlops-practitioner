from fastapi import APIRouter, status, Depends, Request
from prodml.predict import DurationPredictor, PredictSettings
from prodml.api.schemas import PredictRequest, PredictBatchRequest, PredictionSchema

router = APIRouter(prefix="/predict", tags=["Prediction"])

def get_predictor(request: Request) -> DurationPredictor:
    return request.app.state.predictor

@router.post("", status_code=status.HTTP_200_OK)
async def predict(
        payload: PredictRequest,
        request: Request
    ) -> PredictionSchema:
    """
    Predict endpoint to get model predictions.

    Args:
        payload (PredictRequest): The request model containing the input features.
        request (Request): The FastAPI request object.
    Returns:
        dict: A dictionary containing the prediction results.
    """

    predictor = request.app.state.predictor
    features_dict = payload.data.model_dump()
    response = predictor.predict(features_dict)
    return PredictionSchema(**response)


@router.post("/batch", status_code=status.HTTP_200_OK)
async def predict_batch(payload: PredictBatchRequest, request: Request) -> list[PredictionSchema]:
    """
    Predict batch endpoint to get model predictions for multiple inputs.

    Args:
        payload (PredictBatchRequest): The request model containing the list of input features.
        request (Request): The FastAPI request object.
    Returns:
        list[dict]: A list of dictionaries containing the batch prediction results.
    """

    predictor = request.app.state.predictor
    response = predictor.predict_batch([item.model_dump() for item in payload.data])
    return [PredictionSchema(**item) for item in response]