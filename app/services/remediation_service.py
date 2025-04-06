# app/services/remediation_service.py
import logging
import pandas as pd
from typing import List, Optional
from groq import Groq, RateLimitError, APIError # Import Groq client
from app.core.config import settings
from app.models.remediation import RemediationAction
from app.services.kubernetes_service import k8s_service # Import the instance

logger = logging.getLogger(__name__)

class RemediationService:
    def __init__(self):
        self.groq_client = None
        if settings.GROQ_API_KEY:
            try:
                # Use .get_secret_value() for SecretStr
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY.get_secret_value())
                logger.info("Groq client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
                self.groq_client = None # Ensure it's None if init fails
        else:
            logger.warning("Groq API key not set. LLM recommendations will be disabled.")

    def _get_llm_recommendation(self, pod_name: str, namespace: str, probability: float, features_df: Optional[pd.DataFrame]) -> Optional[str]:
        """Calls Groq API to get remediation recommendations based on features."""
        if not self.groq_client:
            return "LLM recommendations disabled (API key not configured)."
        if features_df is None or features_df.empty:
             logger.warning(f"Cannot generate LLM recommendation for {pod_name}: Features DataFrame is missing or empty.")
             return "Could not generate recommendation: Missing feature data."

        # --- Select key features for the prompt ---
        key_features_for_prompt = [
            'cpu_milli', 'memory_mib', 'cpu_x_mem',
            'scheduling_latency_sec', 'scheduled_time_missing',
            'recent_failure_rate', # If available and accurate
            'cluster_cpu_ratio', 'cluster_mem_ratio', # If available and accurate
            # Include relevant QoS flags if they were important
            'qos_BE', 'qos_LS', 'qos_Burstable', 'qos_Guaranteed' # Ensure these match column names after get_dummies
        ]
        # Add others like 'num_gpu', 'gpu_milli' if relevant

        feature_values_text = "Key Feature Values:\n"
        try:
            # Use .iloc[0] as features_df should contain only one row for the pod
            for feature in key_features_for_prompt:
                if feature in features_df.columns:
                    value = features_df[feature].iloc[0]
                    # Format based on type (e.g., float vs int)
                    if isinstance(value, float):
                        # Handle special cases like latency=-1.0
                        if feature == 'scheduling_latency_sec' and value < 0:
                             feature_values_text += f"- {feature}: Not Available/Missing\n"
                        else:
                             feature_values_text += f"- {feature}: {value:.3f}\n"
                    else:
                        feature_values_text += f"- {feature}: {value}\n"
                else:
                    # Check for potential QoS features that might not exist if input QoS was different
                    if feature.startswith('qos_'):
                         feature_values_text += f"- {feature}: 0 (Not Applicable)\n"
                    else:
                         feature_values_text += f"- {feature}: (Not Available)\n" # Indicate if a key feature wasn't present
        except IndexError:
             logger.error(f"Error accessing feature values for {pod_name}. features_df might be malformed.")
             return "Could not generate recommendation: Error processing feature data."
        except Exception as e:
             logger.error(f"Unexpected error formatting features for LLM prompt: {e}", exc_info=True)
             return "Could not generate recommendation: Error processing feature data."


        prompt = f"""
        Act as an expert Kubernetes Site Reliability Engineer (SRE).
        A machine learning model predicts a high probability of failure ({probability:.4f}) for a pod.
        Additionally, specific rules might have triggered based on observed metrics.
        Your task is to provide concise, actionable troubleshooting and remediation steps based *primarily* on the predicted probability AND the provided feature values AND any explicit rule triggers mentioned below. Focus on root causes suggested by the data.

        Pod Information:
        - Name: {pod_name}
        - Namespace: {namespace}
        - Predicted Failure Probability (ML Model): {probability:.4f} (Threshold for ML action: {settings.PREDICTION_THRESHOLD})

        {feature_values_text}

        Analysis & Recommendations:
        Based *specifically* on the probability and the feature values above, what are the most likely reasons for the predicted failure, and what are the top 2-3 specific `kubectl` commands or investigation steps an operator should take *first*? Be concise and prioritize actions. If latency is high, mention scheduling investigation steps.

        Example format:
        Likely Cause(s): [Brief explanation based on features AND probability, e.g., High memory request, High latency observed, High ML probability suggests resource issues]
        Recommended Actions:
        1. `kubectl logs {pod_name} -n {namespace}` (Check for immediate errors)
        2. `kubectl describe pod {pod_name} -n {namespace}` (Check events, resource status, node placement, scheduling details)
        3. [Another specific action based on features/probability, e.g., Check node condition if cluster ratios are high OR Check scheduler logs if latency is high]
        """

        try:
            logger.info(f"Querying Groq for recommendations for pod {pod_name}...")
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful Kubernetes SRE assistant providing troubleshooting advice based on ML predictions and metric rules."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-specdec", # Or "llama3-70b-8192"
                temperature=0.3,
                max_tokens=300, # Increased slightly for potentially more detail
            )
            recommendation_text = chat_completion.choices[0].message.content
            logger.info(f"Received recommendation from Groq for pod {pod_name}.")
            return recommendation_text.strip()

        except RateLimitError:
            logger.warning(f"Groq API rate limit exceeded while getting recommendation for {pod_name}.")
            return "Recommendation generation failed due to API rate limiting."
        except APIError as e:
            logger.error(f"Groq API error getting recommendation for {pod_name}: Status={e.status_code} Response={e.response}", exc_info=True)
            return f"Recommendation generation failed due to Groq API error: {e.message}"
        except Exception as e:
            logger.error(f"Unexpected error calling Groq API for {pod_name}: {e}", exc_info=True)
            return "Recommendation generation failed due to an unexpected internal error."


    def determine_actions(self, pod_name: str, namespace: str, probability: float, features_df: Optional[pd.DataFrame]) -> list[RemediationAction]:
        """
        Determines remediation actions based on probability, explicit rules (like latency),
        and potentially LLM analysis.
        """
        logger.info(f"Determining remediation for {pod_name} in {namespace} (ML Probability: {probability:.4f})")
        actions = []
        triggered_by_ml = probability > settings.PREDICTION_THRESHOLD
        triggered_by_rule = False # Flag for rule-based triggers

        # --- Rule 1: High Scheduling Latency ---
        # Define your latency threshold in seconds (e.g., 10 minutes = 600 seconds)
        latency_threshold_sec = 600
        latency = -1.0 # Default value if not found or error

        if features_df is not None and 'scheduling_latency_sec' in features_df.columns:
            try:
                # Ensure we get a number, handle potential non-numeric if conversion failed earlier
                latency_val = pd.to_numeric(features_df['scheduling_latency_sec'].iloc[0], errors='coerce')
                if pd.notna(latency_val):
                    latency = float(latency_val)
                else:
                    logger.warning(f"Could not convert scheduling_latency_sec to numeric for {pod_name}. Value: {features_df['scheduling_latency_sec'].iloc[0]}")
            except IndexError:
                 logger.warning(f"Could not extract latency for {pod_name} from features_df (IndexError).")
            except Exception as e:
                 logger.error(f"Error extracting or converting latency for {pod_name}: {e}", exc_info=True)

        # Check if latency is valid (>= 0) and exceeds threshold
        if latency >= 0 and latency >= latency_threshold_sec:
             logger.warning(f"RULE TRIGGERED: High scheduling latency ({latency:.0f}s >= {latency_threshold_sec}s) detected for {pod_name}. Adding specific action.")
             actions.append(RemediationAction(
                 action_type='high_latency_detected', # Specific type for this rule
                 reason=f'Scheduling latency ({latency:.0f}s) exceeded threshold ({latency_threshold_sec}s). ML Prediction: {probability:.2f}.',
                 recommendation=(
                     f"Investigate potential scheduling issues for pod '{pod_name}'. "
                     f"Check scheduler events (`kubectl get events --sort-by=.metadata.creationTimestamp`), node availability/taints/conditions (`kubectl describe node <node-name>`), "
                     f"resource quotas (`kubectl get resourcequota -n {namespace}`), admission controllers, and pod priority/preemption."
                 ),
                 # Example commands focus on scheduling info
                 command=f"kubectl describe pod {pod_name} -n {namespace} && kubectl get events -n {namespace} --sort-by=.metadata.creationTimestamp",
                 executed=False, # Will be handled by process_remediation if automated
                 error=None
             ))
             triggered_by_rule = True # Mark that a rule triggered an action

        # --- Actions based on ML Prediction Threshold ---
        if triggered_by_ml:
            logger.info(f"ML TRIGGERED: Probability {probability:.4f} is above threshold {settings.PREDICTION_THRESHOLD}. Adding ML-based actions.")
            # Action: Basic Checks (Always add if ML triggered)
            actions.append(RemediationAction(
                action_type='basic_investigation_ml', # Differentiate from latency rule if needed
                reason=f'High ML failure probability ({probability:.2f} > {settings.PREDICTION_THRESHOLD:.2f})',
                recommendation=f"ML model predicts high failure risk for '{pod_name}'. Start investigation with logs and events.",
                command=f"kubectl logs {pod_name} -n {namespace} --tail=50 && kubectl describe pod {pod_name} -n {namespace}" # Added tail
            ))

            # Action: Get LLM Recommendation (Only if ML triggered)
            llm_recommendation_text = self._get_llm_recommendation(pod_name, namespace, probability, features_df)

            if llm_recommendation_text:
                actions.append(RemediationAction(
                    action_type='llm_recommendation_ml',
                    reason='LLM analysis based on high ML probability and feature values.',
                    recommendation=llm_recommendation_text,
                    command=None
                ))
            else:
                 # Fallback if LLM fails or is disabled
                 actions.append(RemediationAction(
                    action_type='llm_failed_fallback_ml',
                    reason='High ML probability, but LLM recommendation failed or is disabled.',
                    recommendation="LLM analysis unavailable. Perform standard pod troubleshooting (logs, events, resource usage, node status).",
                    command=None
                ))

            # --- Add potential ML-based AUTOMATION rules here if needed ---
            # Example: (Make sure the action_type matches what process_remediation looks for)
            # You would need reliable features_df data for this
            # try:
            #     high_cpu_pressure = features_df['cluster_cpu_ratio'].iloc[0] > 0.9 if features_df is not None and 'cluster_cpu_ratio' in features_df else False
            #     very_high_prob = probability > (settings.PREDICTION_THRESHOLD + 0.1) # e.g., > 0.85 if threshold is 0.75
            # except (KeyError, IndexError):
            #     high_cpu_pressure = False
            #     very_high_prob = False

            # if very_high_prob and high_cpu_pressure:
            #     actions.append(RemediationAction(
            #         action_type='restart_pod_due_to_pressure_ml', # Specific type
            #         reason=f'Very high ML failure probability ({probability:.2f}) and high cluster CPU pressure.',
            #         recommendation=f"Consider restarting pod '{pod_name}' due to likely resource contention.",
            #         command=f"kubectl delete pod {pod_name} -n {namespace}"
            #     ))

        # --- Logging Summary ---
        if not actions:
             logger.info(f"Neither ML threshold nor explicit rules triggered for {pod_name}. No actions determined.")
        else:
             # Remove potential duplicate actions (e.g., if latency rule and ML rule both suggest describe pod) - simple approach
             unique_actions = []
             seen_recommendations = set()
             for action in actions:
                 # Consider actions unique based on type and recommendation text
                 action_key = (action.action_type, action.recommendation)
                 if action_key not in seen_recommendations:
                     unique_actions.append(action)
                     seen_recommendations.add(action_key)
             actions = unique_actions # Use the deduplicated list

             logger.info(f"Determined {len(actions)} potential actions for {pod_name} (ML Trigger: {triggered_by_ml}, Rule Trigger: {triggered_by_rule}).")

        return actions


    def process_remediation(self, pod_name: str, namespace: str, probability: float, features_df: Optional[pd.DataFrame]) -> List[RemediationAction]:
        """Determines actions (incl. rules) and executes/logs based on mode."""
        # Call the updated determine_actions
        determined_actions = self.determine_actions(pod_name, namespace, probability, features_df)
        processed_actions: List[RemediationAction] = []

        if not determined_actions:
             logger.info(f"No actions were determined for {pod_name}. Nothing to process.")
             return [] # No actions to process

        if settings.ACTION_MODE == "recommend":
            logger.info(f"Action Mode: 'recommend'. Logging recommendations for {pod_name}.")
            for action in determined_actions:
                # Log recommendations including the new high_latency_detected type
                logger.info(f"-> Recommendation ({action.action_type}): {action.recommendation} (Reason: {action.reason})")
                if action.command:
                     logger.info(f"   Suggested Command: {action.command}")
                action.executed = False # Ensure false in recommend mode
                action.error = None # Ensure no error in recommend mode
                processed_actions.append(action)

        elif settings.ACTION_MODE == "automate":
            logger.warning(f"Action Mode: 'automate'. Evaluating actions for potential execution for {pod_name}.")
            action_executed_flag = False
            for action in determined_actions:
                 # --- Automation Logic ---
                 # Define explicitly which action types are safe/intended for automation
                 executable_action_types = [
                     'restart_pod_due_to_pressure_ml', # Example ML-based rule
                     # 'high_latency_detected' # <-- Explicitly NOT automating this by default
                     # Add other action_types intended for automation here
                 ]

                 if action.action_type in executable_action_types:
                     logger.warning(f"AUTOMATING ACTION: '{action.action_type}' for pod {pod_name} based on rule.")
                     if not k8s_service.is_available():
                          logger.error(f"Cannot execute action '{action.action_type}': Kubernetes client is not available.")
                          action.executed = False
                          action.error = "Kubernetes client unavailable."
                     # Example: Check for delete command specifically for restart actions
                     elif action.command and action.command.strip().startswith("kubectl delete pod"):
                          logger.info(f"Executing command: {action.command}")
                          success = k8s_service.delete_pod(pod_name, namespace)
                          action.executed = success
                          if not success:
                              action.error = "Failed to execute delete via K8s API."
                              logger.error(f"Automated deletion failed for pod {pod_name}.")
                          else:
                              action.error = None
                              logger.info(f"Successfully executed automated deletion for pod {pod_name}.")
                              action_executed_flag = True
                              # break # Optional: Stop after first successful automated action
                     else:
                          logger.error(f"Cannot execute action '{action.action_type}': Command is missing, invalid, or not a 'kubectl delete pod' command ('{action.command}').")
                          action.executed = False
                          action.error = "Invalid or missing command for automation."
                     processed_actions.append(action) # Add action regardless of execution success

                 else:
                     # Log non-automated actions/recommendations even in automate mode
                     logger.info(f"-> Recommendation / Skipped Action ({action.action_type}): {action.recommendation} (Reason: {action.reason})")
                     if action.command:
                          logger.info(f"   Suggested Command: {action.command}")
                     # Ensure executed is False for non-automated actions
                     action.executed = False
                     action.error = None
                     processed_actions.append(action)

            # Log summary messages for automate mode
            ml_prob_triggered = probability > settings.PREDICTION_THRESHOLD
            latency_rule_triggered = any(a.action_type == 'high_latency_detected' for a in determined_actions)

            if not action_executed_flag:
                if ml_prob_triggered and not any(a.action_type in executable_action_types for a in determined_actions if a.action_type.endswith('_ml')):
                     logger.warning(f"High ML probability ({probability:.2f}) detected for {pod_name}, but no *matching automated ML actions* were defined or triggered.")
                elif latency_rule_triggered:
                     logger.info(f"High latency detected for {pod_name}, but the action type ('high_latency_detected') is not configured for automation.")
                elif not ml_prob_triggered and not latency_rule_triggered:
                     logger.info(f"No conditions met threshold/rules for automated actions for {pod_name}.")
                # Add other conditions if needed

        else:
             logger.error(f"Invalid ACTION_MODE '{settings.ACTION_MODE}'. No actions processed.")
             # Still return the determined actions, but mark as not processed
             for action in determined_actions:
                 action.executed = False
                 action.error = f"Invalid ACTION_MODE '{settings.ACTION_MODE}'"
                 processed_actions.append(action)


        return processed_actions

# Instantiate the service (singleton pattern)
remediation_service = RemediationService()