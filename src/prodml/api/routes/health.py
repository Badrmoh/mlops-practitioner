from uuid import uuid4
from fastapi import APIRouter, status
from contextlib import asynccontextmanager
from prodml.logging_config import set_correlation_id

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify that the API is running.

    Returns:
        dict: A dictionary containing the status of the API.
    """
    return {"status": "ok"}