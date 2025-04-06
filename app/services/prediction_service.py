# app/services/prediction_service.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from app.core.config import settings
from app.models.prediction import PredictionInput
from typing import Optional, Tuple, List # Import Tuple, List
import os
logger = logging.getLogger(__name__)

# --- Load Node Data for Cluster Capacity ---
# This runs once when the module is imported.
# Ensure the NODE_FILE_PATH points to a valid file in your runtime environment.
total_cluster_cpu = 1.0 # Default to avoid division by zero
total_cluster_mem = 1.0 # Default to avoid division by zero
try:
    # Use the path from settings
    nodes_df = pd.read_csv(settings.NODE_FILE_PATH)
    total_cluster_cpu = nodes_df['cpu_milli'].sum()
    total_cluster_mem = nodes_df['memory_mib'].sum()
    logger.info(f"Successfully loaded node data. Total Capacity: CPU={total_cluster_cpu}m, Mem={total_cluster_mem}MiB")
    # Optional: Add GPU capacity if needed by features
    # total_cluster_gpu_count = nodes_df['gpu'].sum()
except FileNotFoundError:
    logger.error(f"Node data file not found at '{settings.NODE_FILE_PATH}'. Using default capacity values (1.0). Check NODE_FILE_PATH setting and file existence.")
except Exception as e:
    logger.error(f"Failed to load or process node data from '{settings.NODE_FILE_PATH}': {e}. Using default capacity values (1.0).", exc_info=True)


class PredictionService:
    def __init__(self, model_path: str, feature_path: str):
        self.model: Optional[xgb.XGBClassifier] = None
        self.features: Optional[List[str]] = None
        self._load_model_and_features(model_path, feature_path)

    def _load_model_and_features(self, model_path, feature_path):
        """Loads the ML model and feature list at startup."""
        try:
            # Load Model
            if model_path.endswith(".json"):
                self.model = xgb.XGBClassifier()
                # Check if file exists before loading
                if not os.path.exists(model_path):
                     raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")
                self.model.load_model(model_path)
                logger.info(f"Loaded XGBoost model from {model_path}")
            # Add elif for other model types (.joblib, .txt for lgbm) if needed
            # elif model_path.endswith(".txt"): # Example for LightGBM native
            #     self.model = lgb.Booster(model_file=model_path) # Adjust loading method
            #     logger.info(f"Loaded LightGBM model from {model_path}")
            else:
                # Fallback or error for unsupported types
                raise ValueError(f"Unsupported model file type: {model_path}. Expected .json (XGBoost)")

            # Load Features
            if not os.path.exists(feature_path):
                 raise FileNotFoundError(f"Features file not found at: {feature_path}")
            self.features = joblib.load(feature_path)
            if not isinstance(self.features, list):
                 raise TypeError(f"Expected a list of feature names from {feature_path}, got {type(self.features)}")
            logger.info(f"Loaded feature list ({len(self.features)} features) from {feature_path}")

        except FileNotFoundError as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            # Set to None so predictions will fail clearly
            self.model = None
            self.features = None
        except Exception as e:
            logger.error(f"Failed to load model/features: {e}", exc_info=True)
            self.model = None
            self.features = None

    def _engineer_features(self, input_data: PredictionInput) -> Optional[pd.DataFrame]:
        """
        Replicates feature engineering from training.
        CRITICAL: This MUST accurately reflect the steps taken during training,
                  especially regarding handling of time and cluster state.
                  Using placeholders or current time will lead to inaccurate predictions.
        """
        logger.debug(f"Engineering features for pod: {input_data.pod_name}")
        if self.features is None:
            logger.error("Feature list not loaded. Cannot engineer features.")
            return None

        # --- Feature Engineering Logic ---
        # This needs to precisely match your training script's feature engineering
        # using the fields available in PredictionInput.

        data = {}

        # 1. Base Numeric Features (Handle potential None values)
        # Use defaults similar to training imputation if None, or raise error if required
        default_cpu = 1000 # Example default, align with training median/mean if used
        default_mem = 1024 # Example default
        data['cpu_milli'] = input_data.cpu_milli if input_data.cpu_milli is not None else default_cpu
        data['memory_mib'] = input_data.memory_mib if input_data.memory_mib is not None else default_mem
        data['num_gpu'] = input_data.num_gpu if input_data.num_gpu is not None else 0
        data['gpu_milli'] = input_data.gpu_milli if input_data.gpu_milli is not None else 0

        # 2. Categorical Features (One-Hot Encode QoS)
        qos = input_data.qos if input_data.qos else 'Unknown' # Handle None QoS
        # Ensure these categories match exactly those created by pd.get_dummies in training
        possible_qos_features = [f for f in self.features if f.startswith('qos_')]
        for qos_feature in possible_qos_features:
            # Assumes feature name is 'qos_ActualValue'
            expected_qos_value = qos_feature.split('qos_')[-1]
            data[qos_feature] = 1 if qos == expected_qos_value else 0
        # Handle 'gpu_spec_provided' - needs input if used in training
        # If 'gpu_spec' was a feature in training input, add it to PredictionInput
        data['gpu_spec_provided'] = 0 # Placeholder - adjust if gpu_spec is used

        # 3. Interaction Features
        data['cpu_x_mem'] = data['cpu_milli'] * data['memory_mib']

        # 4. Time-Based Features
        # !! CRITICAL !! Using current time is a POOR proxy for creation time.
        # Ideally, PredictionInput should include the actual 'creation_time' timestamp.
        # If not possible, this introduces significant potential error.
        try:
            # If creation_time is provided in input_data (BEST)
            # creation_timestamp = pd.to_datetime(input_data.creation_time, unit='s', errors='coerce')
            # If not, use current time (WORST - document this limitation)
            creation_timestamp = pd.Timestamp.now(tz='UTC') # Use timezone-aware timestamp
            logger.warning(f"Using current time for feature engineering for pod {input_data.pod_name} as creation_time was not provided. This may reduce accuracy.")
        except AttributeError:
             creation_timestamp = pd.Timestamp.now(tz='UTC')
             logger.warning(f"Using current time for feature engineering for pod {input_data.pod_name}. Add 'creation_time' to PredictionInput for better accuracy.")

        if pd.isna(creation_timestamp):
             logger.error(f"Could not determine timestamp for pod {input_data.pod_name}. Using defaults for time features.")
             data['creation_hour'] = -1
             data['creation_dayofweek'] = -1
             # Set cyclical features to neutral (0 or 1 depending on encoding) or mean
             data['hour_sin'], data['hour_cos'] = 0.0, 1.0 # Example defaults
             data['dayofweek_sin'], data['dayofweek_cos'] = 0.0, 1.0 # Example defaults
        else:
             # Ensure timestamp is timezone-aware (e.g., UTC) if needed for consistency
             # creation_timestamp = creation_timestamp.tz_convert('UTC') # Example
             data['creation_hour'] = creation_timestamp.hour
             data['creation_dayofweek'] = creation_timestamp.dayofweek
             # Cyclical encoding (matches training)
             data['hour_sin'] = np.sin(2 * np.pi * data['creation_hour'] / 24.0)
             data['hour_cos'] = np.cos(2 * np.pi * data['creation_hour'] / 24.0)
             data['dayofweek_sin'] = np.sin(2 * np.pi * data['creation_dayofweek'] / 7.0)
             data['dayofweek_cos'] = np.cos(2 * np.pi * data['creation_dayofweek'] / 7.0)

        # 5. Scheduling Features
        # !! CRITICAL !! These require actual data about the pod's scheduling.
        # Placeholder values will lead to inaccurate predictions.
        # Consider getting this from Kubernetes API or monitoring.
        data['scheduling_latency_sec'] = 0 # Placeholder
        data['scheduled_time_missing'] = 1 # Placeholder (assuming not scheduled yet if predicting pre-run)

        # 6. Cluster State Features
        # !! CRITICAL !! These require real-time cluster state information.
        # This is the hardest part to replicate accurately. Options:
        #   - Query Prometheus/metrics server.
        #   - Query Kubernetes API (can be slow/intensive).
        #   - Maintain state in a separate cache/database.
        # Using static placeholders makes the model highly unreliable.
        logger.warning(f"Using PLACEHOLDER values for cluster state features for pod {input_data.pod_name}! Predictions will likely be inaccurate.")
        data['cluster_active_pods'] = 50       # Placeholder
        data['cluster_pending_pods'] = 5        # Placeholder
        data['cluster_req_cpu'] = 50000       # Placeholder
        data['cluster_req_mem'] = 100000      # Placeholder
        data['cluster_req_gpu_milli'] = 1000    # Placeholder (if used)
        # Use the globally loaded cluster capacities
        data['cluster_cpu_ratio'] = data['cluster_req_cpu'] / total_cluster_cpu if total_cluster_cpu > 0 else 0
        data['cluster_mem_ratio'] = data['cluster_req_mem'] / total_cluster_mem if total_cluster_mem > 0 else 0
        data['recent_failure_rate'] = 0.05     # Placeholder

        # --- End Feature Engineering ---

        # Create DataFrame
        try:
            pod_df = pd.DataFrame([data])
        except Exception as e:
             logger.error(f"Failed to create DataFrame from engineered features: {e}", exc_info=True)
             return None

        # Ensure all expected features are present and in the correct order
        missing_features = [f for f in self.features if f not in pod_df.columns]
        if missing_features:
            logger.error(f"Feature engineering failed: Missing expected features: {missing_features}")
            # Optionally add missing features with default values (e.g., 0 or NaN)
            # for feature in missing_features:
            #     pod_df[feature] = 0 # Or np.nan, then handle imputation if needed
            # logger.warning(f"Added missing features with default value 0: {missing_features}")
            # If adding defaults isn't appropriate, return None
            return None


        extra_features = [f for f in pod_df.columns if f not in self.features]
        if extra_features:
            logger.warning(f"Extra features generated that are not in the loaded feature list: {extra_features}. Dropping them.")
            pod_df = pod_df.drop(columns=extra_features)

        # Reorder columns precisely to match the order used during training
        try:
            pod_df = pod_df[self.features]
            logger.debug(f"Feature engineering successful for pod {input_data.pod_name}. Shape: {pod_df.shape}")
            return pod_df
        except KeyError as e:
            logger.error(f"Feature mismatch during final reordering: Cannot find expected feature '{e}' in generated DataFrame. Check feature engineering logic against loaded features.", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error during feature reordering: {e}", exc_info=True)
            return None


    def predict(self, input_data: PredictionInput) -> Optional[Tuple[float, Optional[pd.DataFrame]]]:
        """
        Engineer features and predict failure probability.
        Returns:
            Tuple containing:
            - float: Predicted failure probability (0.0 to 1.0).
            - pd.DataFrame: The DataFrame of features used for the prediction.
            Returns None, None if prediction fails at any step.
        """
        if self.model is None or self.features is None:
            logger.error("Model or features not loaded. Cannot predict.")
            return None, None

        try:
            # 1. Engineer features
            pod_features_df = self._engineer_features(input_data)
            if pod_features_df is None:
                 logger.error(f"Feature engineering failed for pod {input_data.pod_name}. Cannot predict.")
                 return None, None # Error during feature engineering

            # 2. Predict probability
            # Ensure DataFrame is suitable for XGBoost (e.g., no unexpected object types)
            # You might need explicit type conversions if issues arise:
            # pod_features_df = pod_features_df.astype(float) # Example

            proba = self.model.predict_proba(pod_features_df)
            # proba is likely [[prob_class_0, prob_class_1]], we want prob_class_1
            failure_probability = float(proba[0, 1])

            logger.info(f"Prediction successful for pod {input_data.pod_name}. Probability: {failure_probability:.4f}")
            return failure_probability, pod_features_df # Return probability and features

        except ValueError as ve: # Catch potential issues like feature mismatch during predict
             logger.error(f"Prediction failed for pod {input_data.pod_name} due to ValueError: {ve}. This often indicates a mismatch between training features and prediction input features/types.", exc_info=True)
             return None, None
        except Exception as e:
            logger.error(f"Prediction failed unexpectedly for pod {input_data.pod_name}: {e}", exc_info=True)
            return None, None

# Instantiate the service (singleton pattern)
# Ensure MODEL_FILE_PATH and FEATURE_FILE_PATH in settings point to the correct files
# within your running environment (e.g., Docker container)
prediction_service = PredictionService(
    model_path=settings.MODEL_FILE_PATH,
    feature_path=settings.FEATURE_FILE_PATH
)