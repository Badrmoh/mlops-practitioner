import os
import hashlib

from datetime import datetime

from fastapi import APIRouter, status, Request
from prodml.api.routes import metadata
from prodml.predict import PredictSettings
from prodml.api.schemas import MetadataSchema

settings = PredictSettings()

router = APIRouter()

@router.get("/metadata", response_model=MetadataSchema, status_code=status.HTTP_200_OK)
async def get_metadata(request: Request) -> MetadataSchema:
    """
    Metadata endpoint to retrieve model metadata.

    Args:
        request (Request): The FastAPI request object.
    Returns:
        MetadataSchema: A schema containing the model metadata.
    """

    return request.app.state.predictor.metadata