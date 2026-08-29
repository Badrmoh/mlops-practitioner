
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class PredictionSchema(BaseModel):
    """
    Schema for input features.
    """

    PU_DO: str
    Trip_Distance: float = Field(...,
        gt=0,
        description="Distance of the trip in miles"
    )
    Prediction: Optional[float] = Field(
        None,
        description=(
            "Optional field for the predicted duration." 
            "This field is not required in the request but can be included in the response."
        )
    )

class PredictRequest(BaseModel):
    """
    Request model for prediction endpoint.
    """

    data: PredictionSchema = Field(
        ...,
        description="Single input data for prediction"
    )


class PredictBatchRequest(BaseModel):
    """
    Request model for batch prediction endpoint.
    """

    data: list[PredictionSchema] = Field(
        ...,
        min_length=1,
        description="List of input data for batch prediction"
    )

class MetadataSchema(BaseModel):
    """
    Schema for metadata information.
    """

    model_name: str
    model_version: str
    feature_names: list[str]
    training_framework: str
    training_date: datetime
    artifact_hash: str
