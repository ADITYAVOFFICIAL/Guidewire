# app/models/prediction.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.core.config import settings
class PredictionInput(BaseModel):
    """
    Input data structure for predicting pod failure.
    Matches the features expected by the model.
    Define fields based on the *raw* data needed for feature engineering.
    """
    pod_name: str = Field(..., description="Name of the pod")
    namespace: Optional[str] = Field(settings.TARGET_NAMESPACE, description="Namespace of the pod")

    # Raw features needed for engineering (examples)
    cpu_milli: Optional[float] = Field(None, description="Requested CPU in millicores")
    memory_mib: Optional[float] = Field(None, description="Requested Memory in MiB")
    num_gpu: Optional[int] = Field(0, description="Number of GPUs requested")
    gpu_milli: Optional[float] = Field(0, description="GPU millicores requested (if sharing)")
    qos: Optional[str] = Field(None, description="Quality of Service class")
    # Add other raw fields if your feature engineering needs them (e.g., gpu_spec)

    # Allow extra fields if needed, or keep strict
    # class Config:
    #     extra = 'allow'

class PredictionOutput(BaseModel):
    pod_name: str
    namespace: str
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    prediction_timestamp: str