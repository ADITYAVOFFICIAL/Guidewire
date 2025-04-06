# app/api/v1/endpoints/remediation.py
import logging
from fastapi import APIRouter, HTTPException, status, Depends # Added Depends
from app.models.prediction import PredictionInput, PredictionOutput
from app.models.remediation import RemediationResponse, RemediationAction
# Import the instantiated services directly
from app.services.prediction_service import prediction_service
from app.services.remediation_service import remediation_service
from app.core.config import settings
import time # Use time module directly
import pandas as pd # Keep pandas import if needed elsewhere, but not directly here

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Optional: Dependency for checking service readiness ---
async def check_services_ready():
    if prediction_service.model is None or prediction_service.features is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready (model or features not loaded)."
        )
    # Add checks for other critical dependencies if needed
    # e.g., if k8s_service is required for automate mode:
    # if settings.ACTION_MODE == "automate" and not k8s_service.is_available():
    #    raise HTTPException(
    #        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #        detail="Kubernetes service is not available, cannot operate in 'automate' mode."
    #    )

@router.post(
    "/predict-and-remediate",
    response_model=RemediationResponse,
    # Add the dependency check to the endpoint
    dependencies=[Depends(check_services_ready)]
)
async def predict_and_remediate_pod(
    input_data: PredictionInput
) -> RemediationResponse:
    """
    Receives pod data, predicts failure probability using the loaded model
    and engineered features, determines remediation actions (potentially using LLM),
    and executes or recommends actions based on the configured ACTION_MODE.
    """
    start_time_ns = time.perf_counter_ns() # More precise timing
    namespace = input_data.namespace or settings.TARGET_NAMESPACE # Use default if not provided

    logger.info(f"Received prediction request for pod: {input_data.pod_name} in namespace: {namespace}")

    # 1. Predict Failure Probability and get features used
    #    Handles feature engineering internally.
    #    Returns probability (float) and features_df (DataFrame) or None, None on failure.
    probability, features_df = prediction_service.predict(input_data)

    # Handle prediction failure
    if probability is None or features_df is None:
        # Log the specific reason if possible (already logged in prediction_service)
        logger.error(f"Prediction failed for pod {input_data.pod_name}. Cannot proceed with remediation.")
        # Return a specific response indicating prediction failure
        # Use status code 500 or 503? 500 seems appropriate if prediction logic failed.
        # However, the endpoint itself worked, so maybe return a 200 OK with error details in the body.
        # Let's stick to the RemediationResponse model but indicate error clearly.
        error_action = RemediationAction(
            action_type="error",
            reason="Prediction failed",
            recommendation="Could not generate failure probability. Check service logs.",
            command=None,
            executed=False,
            error="Prediction service returned None."
        )
        return RemediationResponse(
             pod_name=input_data.pod_name,
             namespace=namespace,
             failure_probability=-1.0, # Use a sentinel value for probability error
             triggered=False,
             actions_determined=[error_action],
             action_mode=settings.ACTION_MODE
        )

    logger.info(f"Prediction for {input_data.pod_name}: Probability={probability:.4f}")

    # 2. Determine & Process Remediation Actions using the features_df
    #    This step now receives the features used for prediction.
    processed_actions = remediation_service.process_remediation(
        pod_name=input_data.pod_name,
        namespace=namespace,
        probability=probability,
        features_df=features_df # Pass the features DataFrame
    )

    # 3. Construct the response
    is_triggered = probability > settings.PREDICTION_THRESHOLD
    response = RemediationResponse(
        pod_name=input_data.pod_name,
        namespace=namespace,
        failure_probability=probability,
        triggered=is_triggered,
        actions_determined=processed_actions,
        action_mode=settings.ACTION_MODE
    )

    duration_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000 # Duration in milliseconds
    logger.info(f"Processed remediation request for {input_data.pod_name} in {duration_ms:.2f} ms. Triggered: {is_triggered}. Actions determined: {len(processed_actions)}")

    return response