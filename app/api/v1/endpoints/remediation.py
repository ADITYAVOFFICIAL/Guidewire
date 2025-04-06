# app/api/v1/endpoints/remediation.py
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.prediction import PredictionInput # PredictionOutput not used directly here
from app.models.remediation import RemediationResponse, RemediationAction
# Import the instantiated services directly
from app.services.prediction_service import prediction_service
from app.services.remediation_service import remediation_service
from app.core.config import settings
import time
import pandas as pd # Needed by remediation_service now

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Optional: Dependency for checking service readiness ---
async def check_services_ready():
    """Dependency to check if essential services are ready."""
    if prediction_service.model is None or prediction_service.features is None:
        logger.critical("Prediction service check failed: Model or features not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready (model or features not loaded)."
        )
    # Add checks for other critical dependencies if needed
    # e.g., if k8s_service is required for automate mode:
    # from app.services.kubernetes_service import k8s_service # Import instance
    # if settings.ACTION_MODE == "automate" and not k8s_service.is_available():
    #    logger.critical("Kubernetes service check failed: Client not available for 'automate' mode.")
    #    raise HTTPException(
    #        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #        detail="Kubernetes service is not available, cannot operate in 'automate' mode."
    #    )
    logger.debug("Service readiness check passed.")

@router.post(
    "/predict-and-remediate",
    response_model=RemediationResponse,
    summary="Predict Pod Failure and Determine Remediation",
    description="""
Receives pod metric and spec data, predicts failure probability using a pre-trained ML model,
determines remediation actions based on ML probability AND explicit rules (e.g., high scheduling latency),
potentially enhances recommendations with an LLM, and returns recommended or executed actions
based on the configured `ACTION_MODE`.
    """,
    # Add the dependency check to the endpoint
    dependencies=[Depends(check_services_ready)]
)
async def predict_and_remediate_pod(
    input_data: PredictionInput
) -> RemediationResponse:
    """
    Endpoint to handle pod failure prediction and remediation determination.
    """
    start_time_ns = time.perf_counter_ns()
    # Use default namespace from settings if not provided in the input
    namespace = input_data.namespace or settings.TARGET_NAMESPACE

    logger.info(f"Received prediction request for pod: {input_data.pod_name} in namespace: {namespace}")

    # 1. Predict Failure Probability and get features used
    #    Handles feature engineering internally.
    #    Returns probability (float) and features_df (DataFrame) or None, None on failure.
    try:
        probability, features_df = prediction_service.predict(input_data)
    except Exception as pred_err:
        logger.error(f"Unhandled error during prediction for {input_data.pod_name}: {pred_err}", exc_info=True)
        # Return a specific error response
        error_action = RemediationAction(
            action_type="error",
            reason="Prediction service internal error",
            recommendation="Could not generate failure probability due to an unexpected error. Check service logs.",
            command=None,
            executed=False,
            error=f"Prediction Exception: {str(pred_err)}"
        )
        return RemediationResponse(
             pod_name=input_data.pod_name,
             namespace=namespace,
             failure_probability=-1.0, # Sentinel value for error
             triggered=False,
             actions_determined=[error_action],
             action_mode=settings.ACTION_MODE
        )


    # Handle prediction failure (if predict service returns None gracefully)
    if probability is None or features_df is None:
        logger.error(f"Prediction service returned None for pod {input_data.pod_name}. Cannot proceed.")
        error_action = RemediationAction(
            action_type="error",
            reason="Prediction failed",
            recommendation="Could not generate failure probability. Prediction service returned None. Check service logs.",
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

    logger.info(f"Prediction for {input_data.pod_name}: ML Probability={probability:.4f}")

    # 2. Determine & Process Remediation Actions using the updated service
    #    This now includes rule checks (e.g., latency) inside determine_actions
    #    and handles execution/logging based on ACTION_MODE.
    try:
        processed_actions = remediation_service.process_remediation(
            pod_name=input_data.pod_name,
            namespace=namespace,
            probability=probability,
            features_df=features_df # Pass features DataFrame for rules and LLM
        )
    except Exception as rem_err:
        logger.error(f"Unhandled error during remediation processing for {input_data.pod_name}: {rem_err}", exc_info=True)
        # Return probability but indicate remediation error
        error_action = RemediationAction(
            action_type="error",
            reason="Remediation service internal error",
            recommendation="Failure probability generated, but remediation processing failed. Check service logs.",
            command=None,
            executed=False,
            error=f"Remediation Exception: {str(rem_err)}"
        )
        # Determine trigger status based *only* on ML probability if remediation fails
        ml_triggered_on_error = probability > settings.PREDICTION_THRESHOLD
        return RemediationResponse(
             pod_name=input_data.pod_name,
             namespace=namespace,
             failure_probability=probability, # Report ML probability even if remediation failed
             triggered=ml_triggered_on_error, # Base trigger only on ML prob here
             actions_determined=[error_action], # Add the error action
             action_mode=settings.ACTION_MODE
        )


    # 3. Construct the final response
    #    Set 'triggered' to True if EITHER the ML probability exceeded the threshold
    #    OR if any remediation actions were determined (which implies a rule was met).
    ml_triggered = probability > settings.PREDICTION_THRESHOLD
    # Check if there are any actions determined that are NOT errors
    rules_or_ml_actions_exist = any(action.action_type != "error" for action in processed_actions)
    is_triggered = ml_triggered or rules_or_ml_actions_exist # Overall trigger status

    response = RemediationResponse(
        pod_name=input_data.pod_name,
        namespace=namespace,
        failure_probability=probability, # Always report the actual ML probability
        triggered=is_triggered,          # Report the combined trigger status
        actions_determined=processed_actions, # Actions determined/processed by the service
        action_mode=settings.ACTION_MODE
    )

    duration_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000
    # Updated log message for clarity
    logger.info(
        f"Processed request for {input_data.pod_name} in {duration_ms:.2f} ms. "
        f"ML_Prob={probability:.4f} (Thresh={settings.PREDICTION_THRESHOLD}), "
        f"ML_Trig={ml_triggered}, Actions_Exist={rules_or_ml_actions_exist}, "
        f"Overall_Trig={is_triggered}. Actions: {len(processed_actions)}"
    )

    return response