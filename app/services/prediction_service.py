# app/services/prediction_service.py
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb # Import lightgbm if you might load its models
import joblib
import logging
from app.core.config import settings
from app.models.prediction import PredictionInput
# Import the k8s service instance
from app.services.kubernetes_service import k8s_service
from typing import Optional, Tuple, List, Any # Added Any for generic model type
import os
import time # Keep for logging/timing if needed

logger = logging.getLogger(__name__)

# --- Load Node Data for Cluster Capacity ---
# This section remains the same, loading node capacity data
total_cluster_cpu = 1.0
total_cluster_mem = 1.0
try:
    # Check if the NODE_FILE_PATH is set and exists
    if settings.NODE_FILE_PATH and os.path.exists(settings.NODE_FILE_PATH):
        nodes_df = pd.read_csv(settings.NODE_FILE_PATH)
        total_cluster_cpu = nodes_df['cpu_milli'].sum()
        total_cluster_mem = nodes_df['memory_mib'].sum()
        # Prevent division by zero if sums are zero or negative
        if total_cluster_cpu <= 0:
            logger.warning(f"Total cluster CPU from {settings.NODE_FILE_PATH} is <= 0. Using default 1.0.")
            total_cluster_cpu = 1.0
        if total_cluster_mem <= 0:
            logger.warning(f"Total cluster Memory from {settings.NODE_FILE_PATH} is <= 0. Using default 1.0.")
            total_cluster_mem = 1.0
        logger.info(f"Successfully loaded node data. Total Capacity: CPU={total_cluster_cpu}m, Mem={total_cluster_mem}MiB")
    else:
        logger.warning(f"Node data file path '{settings.NODE_FILE_PATH}' not configured or file not found. Using default capacity values (1.0). Cluster ratios will be inaccurate.")
        total_cluster_cpu = 1.0
        total_cluster_mem = 1.0
except Exception as e:
    logger.error(f"Failed to load or process node data from '{settings.NODE_FILE_PATH}': {e}. Using default capacity values (1.0).", exc_info=True)
    total_cluster_cpu = 1.0
    total_cluster_mem = 1.0
# --- End Node Data Loading ---


class PredictionService:
    def __init__(self, model_path: str, feature_path: str):
        # Use a more generic type hint for the model
        self.model: Optional[Any] = None
        self.features: Optional[List[str]] = None
        self._load_model_and_features(model_path, feature_path)

    def _load_model_and_features(self, model_path: str, feature_path: str):
        """Loads the ML model and the list of feature names."""
        logger.info(f"Attempting to load model from: {model_path}")
        logger.info(f"Attempting to load features from: {feature_path}")
        try:
            # --- Load Model ---
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            if model_path.endswith(".joblib"):
                self.model = joblib.load(model_path)
                model_type = type(self.model).__name__
                logger.info(f"Loaded Joblib model ({model_type}) from {model_path}")
                # Basic check for classifier interface
                if not hasattr(self.model, 'predict_proba') or not callable(getattr(self.model, 'predict_proba')):
                    raise TypeError(f"Loaded object from {model_path} of type {model_type} does not have a callable 'predict_proba' method.")

            elif model_path.endswith(".json"): # XGBoost JSON format
                self.model = xgb.XGBClassifier()
                # Ensure XGBoost model uses JSON for loading if it's a JSON file
                self.model.load_model(model_path)
                logger.info(f"Loaded XGBoost model from JSON file: {model_path}")

            # Example for LightGBM .txt model (requires careful handling)
            # elif model_path.endswith(".txt"):
            #     # Load the booster first
            #     booster = lgb.Booster(model_file=model_path)
            #     # Wrap it in the scikit-learn interface if needed for consistency
            #     self.model = lgb.LGBMClassifier()
            #     self.model.booster_ = booster
            #     self.model._n_features = booster.num_feature() # Important for sklearn wrapper
            #     # self.model._classes = ... # Need to know classes from training
            #     # self.model._n_classes = ... # Need to know num classes
            #     logger.info(f"Loaded LightGBM model from txt file: {model_path}")
            #     # Add predict_proba check if using wrapper
            #     if not hasattr(self.model, 'predict_proba'):
            #          raise TypeError("Loaded LightGBM model wrapper missing predict_proba.")

            else:
                raise ValueError(f"Unsupported model file type: {model_path}. Expected .joblib or .json")

            # --- Load Features ---
            if not os.path.exists(feature_path):
                 raise FileNotFoundError(f"Features file not found at: {feature_path}")
            self.features = joblib.load(feature_path)
            if not isinstance(self.features, list):
                 raise TypeError(f"Expected a list of feature names from {feature_path}, got {type(self.features)}")
            logger.info(f"Loaded feature list ({len(self.features)} features) from {feature_path}")
            logger.debug(f"Features loaded: {self.features}")

        except FileNotFoundError as e:
            logger.error(f"Initialization failed: {e}")
            self.model = None
            self.features = None
        except (TypeError, ValueError) as e:
             logger.error(f"Initialization failed due to type/value error: {e}")
             self.model = None
             self.features = None
        except Exception as e:
            logger.error(f"Failed to load model/features due to unexpected error: {e}", exc_info=True)
            self.model = None
            self.features = None

    def _engineer_features(self, input_data: PredictionInput) -> Optional[pd.DataFrame]:
        """
        Replicates feature engineering using dynamic data where possible.
        Fetches cluster state from KubernetesService.
        Uses timestamps from input_data if provided.
        Matches the feature engineering logic used during training.
        """
        logger.debug(f"Engineering features for pod: {input_data.pod_name}")
        if self.features is None:
            logger.error("Feature list not loaded. Cannot engineer features.")
            return None

        # --- Fetch Dynamic Cluster State ---
        # Pass the pod name and namespace to potentially exclude it from totals
        cluster_state = k8s_service.get_cluster_state(
            namespace=input_data.namespace, # Fetch for the specific namespace or all if None/default
            exclude_pod_name=input_data.pod_name
        )
        # If fetching fails, cluster_state will contain default placeholder values
        logger.debug(f"Fetched cluster state for {input_data.pod_name}: {cluster_state}")

        # --- Start Feature Engineering ---
        data = {}

        # 1. Base Numeric Features (Handle None values -> Use defaults similar to training imputation)
        # Using median values from training might be better, but defaults are simpler for now.
        # Let's assume training imputed with median, but here we use reasonable defaults if None.
        default_cpu = 1000 # Example default, adjust if needed
        default_mem = 1024 # Example default
        data['cpu_milli'] = input_data.cpu_milli if input_data.cpu_milli is not None else default_cpu
        data['memory_mib'] = input_data.memory_mib if input_data.memory_mib is not None else default_mem
        data['num_gpu'] = input_data.num_gpu if input_data.num_gpu is not None else 0
        data['gpu_milli'] = input_data.gpu_milli if input_data.gpu_milli is not None else 0

        # 2. Categorical Features (One-Hot Encode QoS)
        qos = input_data.qos if input_data.qos else 'Unknown'
        # Ensure all possible QoS columns from the loaded features list are created
        possible_qos_features = [f for f in self.features if f.startswith('qos_')]
        logger.debug(f"Possible QoS features from loaded list: {possible_qos_features}")
        for qos_feature in possible_qos_features:
            # Extract the QoS value part (e.g., 'BE' from 'qos_BE')
            expected_qos_value = qos_feature.split('qos_', 1)[-1]
            data[qos_feature] = 1 if qos == expected_qos_value else 0
        logger.debug(f"Input QoS '{qos}' resulted in one-hot encoded data: {[ (k,v) for k,v in data.items() if k.startswith('qos_')]}")

        # Placeholder for gpu_spec_provided (assuming it was 0/1 in training)
        # This needs actual input if the model used it meaningfully.
        data['gpu_spec_provided'] = 0

        # 3. Interaction Features
        data['cpu_x_mem'] = data['cpu_milli'] * data['memory_mib']

        # 4. Time-Based Features (Use input timestamps if available)
        creation_unix_ts = input_data.creation_time
        scheduled_unix_ts = input_data.scheduled_time

        # Default values if timestamps are missing
        data['scheduling_latency_sec'] = -1.0
        data['scheduled_time_missing'] = 1
        creation_dt = pd.Timestamp.now(tz='UTC') # Use current time as fallback for cyclical features

        if creation_unix_ts is not None:
            try:
                creation_dt = pd.to_datetime(creation_unix_ts, unit='s', utc=True) # Ensure UTC
                # Calculate latency only if creation time is valid
                if scheduled_unix_ts is not None:
                     try:
                         scheduled_dt = pd.to_datetime(scheduled_unix_ts, unit='s', utc=True)
                         latency = (scheduled_dt - creation_dt).total_seconds()
                         # Match training: ensure non-negative latency, else keep -1
                         data['scheduling_latency_sec'] = max(0.0, latency)
                         data['scheduled_time_missing'] = 0
                     except (ValueError, TypeError):
                         logger.error(f"Invalid scheduled_time format for {input_data.pod_name}: {scheduled_unix_ts}. Treating as missing.")
                         data['scheduling_latency_sec'] = -1.0 # Fallback to -1 if scheduled time is invalid
                         data['scheduled_time_missing'] = 1
                else:
                     # Scheduled time not provided, keep defaults
                     data['scheduling_latency_sec'] = -1.0
                     data['scheduled_time_missing'] = 1

            except (ValueError, TypeError):
                 logger.error(f"Invalid creation_time format for {input_data.pod_name}: {creation_unix_ts}. Using current time for cyclical features.")
                 # Keep latency/missing defaults, use current time only for hour/day features
                 creation_dt = pd.Timestamp.now(tz='UTC')
                 data['scheduling_latency_sec'] = -1.0
                 data['scheduled_time_missing'] = 1
        else:
            logger.warning(f"Pod creation_time not provided for {input_data.pod_name}. Using current time for cyclical features. Latency features set to defaults.")
            # Keep latency/missing defaults

        # Calculate cyclical time components from creation_dt (best effort)
        data['creation_hour'] = creation_dt.hour
        data['creation_dayofweek'] = creation_dt.dayofweek
        data['hour_sin'] = np.sin(2 * np.pi * data['creation_hour'] / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * data['creation_hour'] / 24.0)
        data['dayofweek_sin'] = np.sin(2 * np.pi * data['creation_dayofweek'] / 7.0)
        data['dayofweek_cos'] = np.cos(2 * np.pi * data['creation_dayofweek'] / 7.0)
        logger.debug(f"Time features for {input_data.pod_name}: Latency={data['scheduling_latency_sec']}, MissingSched={data['scheduled_time_missing']}, Hour={data['creation_hour']}")


        # 5. Cluster State Features (Use fetched data)
        data['cluster_active_pods'] = cluster_state.get('active_pods', 0) # Use .get for safety
        data['cluster_pending_pods'] = cluster_state.get('pending_pods', 0)
        data['cluster_req_cpu'] = cluster_state.get('total_req_cpu', 0.0)
        data['cluster_req_mem'] = cluster_state.get('total_req_mem', 0.0)
        data['cluster_req_gpu_milli'] = cluster_state.get('total_req_gpu_milli', 0.0)
        data['recent_failure_rate'] = cluster_state.get('recent_failure_rate', 0.0)

        # Calculate ratios using fetched totals and loaded node capacity
        data['cluster_cpu_ratio'] = data['cluster_req_cpu'] / total_cluster_cpu if total_cluster_cpu > 0 else 0
        data['cluster_mem_ratio'] = data['cluster_req_mem'] / total_cluster_mem if total_cluster_mem > 0 else 0
        # Add GPU ratio if needed/trained and if total capacity is known
        # data['cluster_gpu_ratio'] = ...

        # --- End Feature Engineering ---
        logger.debug(f"Raw engineered data dict for {input_data.pod_name}: {data}")

        # Create DataFrame and validate/reorder against loaded features
        try:
            pod_df = pd.DataFrame([data])
            # Ensure all columns expected by the model exist, fill missing with 0 or median if appropriate
            missing_features = [f for f in self.features if f not in pod_df.columns]
            if missing_features:
                logger.warning(f"Engineered features missing columns expected by model: {missing_features}. Filling with 0.")
                for f in missing_features:
                    pod_df[f] = 0 # Or use a more sophisticated imputation if needed

            # Ensure columns are in the exact order expected by the model
            # Also drops any extra columns generated during engineering but not in self.features
            pod_df = pod_df[self.features]

            # --- Final Data Type Conversion & NaN check ---
            # Convert all columns to numeric, coercing errors (should ideally not happen here)
            for col in pod_df.columns:
                 pod_df[col] = pd.to_numeric(pod_df[col], errors='coerce')

            # Final check for NaNs introduced by coercion or missed earlier
            if pod_df.isnull().values.any():
                 nan_cols = pod_df.columns[pod_df.isnull().any()].tolist()
                 logger.warning(f"NaNs detected after final numeric conversion for {input_data.pod_name} in columns: {nan_cols}. Filling with 0.")
                 # Consider using median from training data if available, otherwise 0 is a fallback
                 pod_df.fillna(0, inplace=True)

            logger.debug(f"Feature engineering successful for pod {input_data.pod_name}. Final shape: {pod_df.shape}")
            return pod_df

        except KeyError as e:
            logger.error(f"Feature mismatch during final reordering: Cannot find expected feature '{e}'. Expected features: {self.features}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error during feature finalization or DataFrame creation: {e}", exc_info=True)
            return None


    def predict(self, input_data: PredictionInput) -> Optional[Tuple[float, Optional[pd.DataFrame]]]:
        """
        Engineer features (using dynamic data) and predict failure probability.
        Returns probability (float) and the feature DataFrame used for prediction.
        """
        if self.model is None or self.features is None:
            logger.error("Model or features not loaded. Cannot predict.")
            return None, None

        # 1. Engineer features
        # This now returns a DataFrame with columns in the exact order expected by the model
        pod_features_df = self._engineer_features(input_data)
        if pod_features_df is None:
             logger.error(f"Feature engineering failed for pod {input_data.pod_name}. Cannot predict.")
             return None, None # Return None for both probability and DataFrame

        # Log the final features being sent to the model for debugging
        logger.debug(f"Features DataFrame going into predict_proba for {input_data.pod_name}:\n{pod_features_df.to_string()}")
        logger.debug(f"DataFrame dtypes:\n{pod_features_df.dtypes}")


        # 2. Predict probability
        try:
            # The predict_proba method is standard for scikit-learn classifiers (like RF)
            # and XGBoost's scikit-learn wrapper.
            proba = self.model.predict_proba(pod_features_df)
            # Probability of the positive class (failure, index 1)
            failure_probability = float(proba[0, 1])

            # Clamp probability just in case (shouldn't be necessary with valid models)
            failure_probability = max(0.0, min(1.0, failure_probability))

            logger.info(f"Prediction successful for pod {input_data.pod_name}. Probability: {failure_probability:.4f}")
            # Return the probability AND the dataframe used to make the prediction
            return failure_probability, pod_features_df

        except ValueError as ve:
             # Log specific ValueError, often related to feature mismatch or NaNs if checks failed
             logger.error(f"Prediction failed for pod {input_data.pod_name} due to ValueError: {ve}. Check feature consistency and NaNs.", exc_info=True)
             return None, pod_features_df # Return None for prob, but df might be useful for debug
        except AttributeError as ae:
             # E.g., if the loaded model doesn't have predict_proba
             logger.error(f"Prediction failed for pod {input_data.pod_name} due to AttributeError: {ae}. Is the loaded model a valid classifier?", exc_info=True)
             return None, pod_features_df
        except Exception as e:
            logger.error(f"Prediction failed unexpectedly for pod {input_data.pod_name}: {e}", exc_info=True)
            return None, pod_features_df # Return None for prob, df for debug

# --- Instantiate the service ---
# This will now load the model specified in settings.MODEL_FILE_PATH,
# handling either .joblib or .json based on the file extension.
prediction_service = PredictionService(
    model_path=settings.MODEL_FILE_PATH,
    feature_path=settings.FEATURE_FILE_PATH
)