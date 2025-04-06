# app/core/config.py
import os
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, validator
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    APP_NAME: str = "Pod Remediation Agent"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # Remediation Settings
    PREDICTION_THRESHOLD: float = Field(0.75, description="Probability threshold to trigger remediation")
    ACTION_MODE: str = Field("recommend", description="'recommend' or 'automate' actions")
    TARGET_NAMESPACE: str = Field("default", description="Default Kubernetes namespace to operate within")

    # Model/Feature Paths (relative to where the app runs, e.g., inside container)
    # --- Ensure these files are copied to these locations in your Docker image ---
    MODEL_FILE_PATH: str = "/Users/adityaverma/Documents/GitHub/Guidewire/app/ml_models/xgboost_pod_failure_model_best.json"
    FEATURE_FILE_PATH: str = "/Users/adityaverma/Documents/GitHub/Guidewire/app/ml_models/features.joblib"
    # --- ADDED: Path for node data file ---
    # Make sure this file is also available at this path in the running environment
    NODE_FILE_PATH: str = "/Users/adityaverma/Documents/GitHub/Guidewire/app/data/openb_node_list_all_node.csv" # Adjust path if needed

    # Kubernetes Config - leave blank to use in-cluster or default kubeconfig
    KUBE_CONFIG_PATH: Optional[str] = None

    # Groq API Key for LLM recommendations
    GROQ_API_KEY: Optional[SecretStr] = Field(None, description="API Key for Groq service")

    @validator('ACTION_MODE')
    def validate_action_mode(cls, v):
        if v not in ['recommend', 'automate']:
            raise ValueError("ACTION_MODE must be either 'recommend' or 'automate'")
        return v

    class Config:
        env_file = '.env' # Load environment variables from .env file
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields from environment

settings = Settings()

# --- ADDED: Check for Groq API Key after loading settings ---
if not settings.GROQ_API_KEY and settings.ACTION_MODE == "recommend": # Only warn if needed
    logger.warning("GROQ_API_KEY environment variable not set. LLM-based recommendations will be disabled.")
elif not settings.GROQ_API_KEY and settings.ACTION_MODE == "automate":
     logger.warning("GROQ_API_KEY environment variable not set. LLM cannot be used to inform potential automated actions.")