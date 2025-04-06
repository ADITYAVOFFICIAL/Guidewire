# app/models/prediction.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.core.config import settings
import time # Import time

class PredictionInput(BaseModel):
    """
    Input data structure for predicting pod failure.
    Includes basic pod specs and critical timestamps.
    """
    pod_name: str = Field(..., description="Name of the pod being predicted")
    namespace: Optional[str] = Field(settings.TARGET_NAMESPACE, description="Namespace of the pod")

    # --- Pod Specification ---
    cpu_milli: Optional[float] = Field(None, description="Requested CPU in millicores")
    memory_mib: Optional[float] = Field(None, description="Requested Memory in MiB")
    num_gpu: Optional[int] = Field(0, description="Number of GPUs requested")
    gpu_milli: Optional[float] = Field(0, description="GPU millicores requested (if sharing)")
    qos: Optional[str] = Field(None, description="Quality of Service class")
    # Add other raw fields if your feature engineering needs them (e.g., gpu_spec)

    # --- Timestamps (Critical for accurate time-based features) ---
    # Ideally provided by the caller monitoring the pod. Unix timestamp (seconds).
    creation_time: Optional[float] = Field(None, description="Pod creation timestamp (Unix seconds)")
    scheduled_time: Optional[float] = Field(None, description="Pod scheduled timestamp (Unix seconds, optional)")

    # Allow extra fields if needed, or keep strict
    # class Config:
    #     extra = 'allow'

class PredictionOutput(BaseModel):
    pod_name: str
    namespace: str
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    prediction_timestamp: str # Consider making this a datetime object or ISO string