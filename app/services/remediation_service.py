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
        # Choose features known to be important from training analysis,
        # or features that are logically relevant to pod failure.
        key_features_for_prompt = [
            'cpu_milli', 'memory_mib', 'cpu_x_mem',
            'scheduling_latency_sec', 'scheduled_time_missing',
            'recent_failure_rate', # If available and accurate
            'cluster_cpu_ratio', 'cluster_mem_ratio', # If available and accurate
            # Include relevant QoS flags if they were important
            'qos_BE', 'qos_LS', 'qos_Burstable', 'qos_Guaranteed'
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
                        feature_values_text += f"- {feature}: {value:.3f}\n"
                    else:
                        feature_values_text += f"- {feature}: {value}\n"
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
        A machine learning model predicts a high probability of failure for the following pod.
        Your task is to provide concise, actionable troubleshooting and remediation steps based *primarily* on the predicted probability and the provided feature values. Focus on root causes suggested by the features.

        Pod Information:
        - Name: {pod_name}
        - Namespace: {namespace}
        - Predicted Failure Probability: {probability:.4f} (Threshold for action: {settings.PREDICTION_THRESHOLD})

        {feature_values_text}

        Analysis & Recommendations:
        Based *specifically* on the probability and the feature values above, what are the most likely reasons for the predicted failure, and what are the top 2-3 specific `kubectl` commands or investigation steps an operator should take *first*? Be concise and prioritize actions.

        Example format:
        Likely Cause(s): [Brief explanation based on features, e.g., High memory request, high cluster memory pressure]
        Recommended Actions:
        1. `kubectl logs {pod_name} -n {namespace}` (Check for immediate errors)
        2. `kubectl describe pod {pod_name} -n {namespace}` (Check events, resource status, node placement)
        3. [Another specific action based on features, e.g., Check node condition if cluster ratios are high]
        """

        try:
            logger.info(f"Querying Groq for recommendations for pod {pod_name}...")
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful Kubernetes SRE assistant providing troubleshooting advice."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192", # Or other suitable model like llama3-70b-8192 if more power needed
                temperature=0.3, # Lower temperature for more focused, less creative recommendations
                max_tokens=250, # Adjust as needed
                # top_p=0.9, # Optional: Adjust nucleus sampling
                # stop=None # Optional: Define stop sequences if needed
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
        """Determines remediation actions based on probability and features, potentially using LLM."""
        logger.info(f"Determining remediation for {pod_name} in {namespace} (Probability: {probability:.4f})")
        actions = []
        triggered = probability > settings.PREDICTION_THRESHOLD

        if not triggered:
            logger.info(f"Probability {probability:.4f} is below threshold {settings.PREDICTION_THRESHOLD}. No remediation actions determined.")
            return []

        # --- Action Strategy ---
        # 1. Always recommend basic checks for high probability.
        # 2. Get LLM recommendation based on features for more specific insights.
        # 3. Define potential automatable actions based on rules (optional, use with caution).

        # Action 1: Basic Checks (Always add if triggered)
        actions.append(RemediationAction(
            action_type='basic_investigation',
            reason=f'High failure probability ({probability:.2f} > {settings.PREDICTION_THRESHOLD:.2f})',
            recommendation=f"Investigate pod '{pod_name}'. Start with logs and describe.",
            command=f"kubectl logs {pod_name} -n {namespace} && kubectl describe pod {pod_name} -n {namespace}"
        ))

        # Action 2: Get LLM Recommendation
        llm_recommendation_text = self._get_llm_recommendation(pod_name, namespace, probability, features_df)

        if llm_recommendation_text:
            actions.append(RemediationAction(
                action_type='llm_recommendation',
                reason='LLM analysis based on high probability and feature values.',
                recommendation=llm_recommendation_text,
                command=None # LLM provides text; specific commands might be extracted or inferred
            ))
        else:
             # Fallback if LLM fails or is disabled
             actions.append(RemediationAction(
                action_type='llm_failed_fallback',
                reason='High failure probability, but LLM recommendation failed or is disabled.',
                recommendation="LLM analysis unavailable. Perform standard pod troubleshooting (logs, events, resource usage, node status).",
                command=None
            ))

        # Action 3: Define Potential Automatable Actions (Example Rule)
        # CAUTION: Only automate actions you are confident about.
        # Example: If probability is *very* high AND cluster pressure is high, suggest restart.
        # This requires reliable cluster state features.
        # high_cpu_pressure = features_df['cluster_cpu_ratio'].iloc[0] > 0.9 if features_df is not None and 'cluster_cpu_ratio' in features_df else False
        # very_high_prob = probability > (settings.PREDICTION_THRESHOLD + 0.1) # e.g., > 0.85 if threshold is 0.75

        # if very_high_prob and high_cpu_pressure:
        #     actions.append(RemediationAction(
        #         action_type='restart_pod_due_to_pressure', # Specific type for automation logic
        #         reason=f'Very high failure probability ({probability:.2f}) and high cluster CPU pressure.',
        #         recommendation=f"Consider restarting pod '{pod_name}' due to likely resource contention.",
        #         command=f"kubectl delete pod {pod_name} -n {namespace}" # The command to execute if automated
        #     ))

        logger.info(f"Determined {len(actions)} potential actions for {pod_name}.")
        return actions

    def process_remediation(self, pod_name: str, namespace: str, probability: float, features_df: Optional[pd.DataFrame]) -> List[RemediationAction]:
        """Determines actions and executes/logs based on mode and action types."""
        determined_actions = self.determine_actions(pod_name, namespace, probability, features_df)
        processed_actions: List[RemediationAction] = []

        if not determined_actions:
             # This case is handled in determine_actions, but double-check
             return []

        if settings.ACTION_MODE == "recommend":
            logger.info(f"Action Mode: 'recommend'. Logging recommendations for {pod_name}.")
            for action in determined_actions:
                logger.info(f"-> Recommendation ({action.action_type}): {action.recommendation} (Reason: {action.reason})")
                # Command is just informational in recommend mode
                if action.command:
                     logger.info(f"   Suggested Command: {action.command}")
                processed_actions.append(action) # Keep original action object

        elif settings.ACTION_MODE == "automate":
            logger.warning(f"Action Mode: 'automate'. Evaluating actions for potential execution for {pod_name}.")
            action_executed_flag = False
            for action in determined_actions:
                 # --- Automation Logic ---
                 # Execute ONLY specific, predefined action types deemed safe for automation.
                 # NEVER execute based on raw LLM text directly.
                 if action.action_type == 'restart_pod_due_to_pressure': # Match the specific type from determine_actions
                     logger.warning(f"AUTOMATING ACTION: '{action.action_type}' for pod {pod_name} based on rule.")
                     if not k8s_service.is_available():
                          logger.error(f"Cannot execute action '{action.action_type}': Kubernetes client is not available.")
                          action.executed = False
                          action.error = "Kubernetes client unavailable."
                     elif action.command and action.command.startswith("kubectl delete pod"): # Basic safety check
                          success = k8s_service.delete_pod(pod_name, namespace)
                          action.executed = success
                          if not success:
                              action.error = "Failed to execute delete via K8s API."
                              logger.error(f"Automated deletion failed for pod {pod_name}.")
                          else:
                              logger.info(f"Successfully executed automated deletion for pod {pod_name}.")
                              action_executed_flag = True
                              # Decide if you want to stop after the first successful automated action
                              # break
                     else:
                          logger.error(f"Cannot execute action '{action.action_type}': Command is missing or invalid for automation ('{action.command}').")
                          action.executed = False
                          action.error = "Invalid or missing command for automation."
                     processed_actions.append(action)
                 else:
                     # Log non-automated actions/recommendations even in automate mode
                     logger.info(f"-> Recommendation ({action.action_type}): {action.recommendation} (Reason: {action.reason})")
                     if action.command:
                          logger.info(f"   Suggested Command: {action.command}")
                     # Ensure executed is False for non-automated actions
                     action.executed = False
                     processed_actions.append(action)

            if probability > settings.PREDICTION_THRESHOLD and not action_executed_flag:
                 logger.warning(f"High probability ({probability:.2f}) detected for {pod_name}, but no automated actions were triggered or executed based on current rules.")

        else:
             logger.error(f"Invalid ACTION_MODE '{settings.ACTION_MODE}'. No actions processed.")

        return processed_actions

# Instantiate the service (singleton pattern)
remediation_service = RemediationService()