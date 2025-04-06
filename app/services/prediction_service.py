# app/services/prediction_service.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from app.core.config import settings
from app.models.prediction import PredictionInput
# Import the k8s service instance
from app.services.kubernetes_service import k8s_service
from typing import Optional, Tuple, List
import os
import time # Keep for logging/timing if needed

logger = logging.getLogger(__name__)

# --- Load Node Data for Cluster Capacity (Keep as is) ---
total_cluster_cpu = 1.0
total_cluster_mem = 1.0
try:
    nodes_df = pd.read_csv(settings.NODE_FILE_PATH)
    total_cluster_cpu = nodes_df['cpu_milli'].sum()
    total_cluster_mem = nodes_df['memory_mib'].sum()
    if total_cluster_cpu <= 0: total_cluster_cpu = 1.0 # Avoid division by zero
    if total_cluster_mem <= 0: total_cluster_mem = 1.0 # Avoid division by zero
    logger.info(f"Successfully loaded node data. Total Capacity: CPU={total_cluster_cpu}m, Mem={total_cluster_mem}MiB")
except FileNotFoundError:
    logger.error(f"Node data file not found at '{settings.NODE_FILE_PATH}'. Using default capacity values (1.0).")
except Exception as e:
    logger.error(f"Failed to load or process node data from '{settings.NODE_FILE_PATH}': {e}. Using default capacity values (1.0).", exc_info=True)


class PredictionService:
    def __init__(self, model_path: str, feature_path: str):
        self.model: Optional[xgb.XGBClassifier] = None
        self.features: Optional[List[str]] = None
        self._load_model_and_features(model_path, feature_path)

    def _load_model_and_features(self, model_path, feature_path):
        # (Keep the existing _load_model_and_features method as it was)
        # ... (ensure it handles FileNotFoundError etc.) ...
        try:
            # Load Model
            if model_path.endswith(".json"):
                self.model = xgb.XGBClassifier()
                if not os.path.exists(model_path):
                     raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")
                self.model.load_model(model_path)
                logger.info(f"Loaded XGBoost model from {model_path}")
            else:
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
            self.model = None
            self.features = None
        except Exception as e:
            logger.error(f"Failed to load model/features: {e}", exc_info=True)
            self.model = None
            self.features = None

    def _engineer_features(self, input_data: PredictionInput) -> Optional[pd.DataFrame]:
        """
        Replicates feature engineering using dynamic data where possible.
        Fetches cluster state from KubernetesService.
        Uses timestamps from input_data if provided.
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

        # --- Start Feature Engineering ---
        data = {}

        # 1. Base Numeric Features (Handle None values)
        default_cpu = 1000
        default_mem = 1024
        data['cpu_milli'] = input_data.cpu_milli if input_data.cpu_milli is not None else default_cpu
        data['memory_mib'] = input_data.memory_mib if input_data.memory_mib is not None else default_mem
        data['num_gpu'] = input_data.num_gpu if input_data.num_gpu is not None else 0
        data['gpu_milli'] = input_data.gpu_milli if input_data.gpu_milli is not None else 0

        # 2. Categorical Features (One-Hot Encode QoS)
        qos = input_data.qos if input_data.qos else 'Unknown'
        possible_qos_features = [f for f in self.features if f.startswith('qos_')]
        for qos_feature in possible_qos_features:
            expected_qos_value = qos_feature.split('qos_')[-1]
            data[qos_feature] = 1 if qos == expected_qos_value else 0
        data['gpu_spec_provided'] = 0 # Placeholder - adjust if gpu_spec is used

        # 3. Interaction Features
        data['cpu_x_mem'] = data['cpu_milli'] * data['memory_mib']

        # 4. Time-Based Features (Use input timestamps if available)
        creation_unix_ts = input_data.creation_time
        scheduled_unix_ts = input_data.scheduled_time

        if creation_unix_ts is None:
            logger.warning(f"Pod creation_time not provided for {input_data.pod_name}. Using current time for time features - ACCURACY REDUCED.")
            creation_dt = pd.Timestamp.now(tz='UTC')
            # Cannot calculate accurate latency without creation time
            data['scheduling_latency_sec'] = -1 # Indicate unknown latency
            data['scheduled_time_missing'] = 1 # Assume missing if creation time is missing
        else:
            try:
                creation_dt = pd.to_datetime(creation_unix_ts, unit='s', utc=True) # Ensure UTC
            except (ValueError, TypeError):
                 logger.error(f"Invalid creation_time format for {input_data.pod_name}: {creation_unix_ts}. Using current time.")
                 creation_dt = pd.Timestamp.now(tz='UTC')
                 data['scheduling_latency_sec'] = -1
                 data['scheduled_time_missing'] = 1

            # Calculate scheduling latency if scheduled_time is also provided
            if scheduled_unix_ts is not None:
                 try:
                     scheduled_dt = pd.to_datetime(scheduled_unix_ts, unit='s', utc=True)
                     latency = (scheduled_dt - creation_dt).total_seconds()
                     # Handle potential negative latency if clocks are skewed or data is wrong
                     data['scheduling_latency_sec'] = max(0, latency)
                     data['scheduled_time_missing'] = 0
                 except (ValueError, TypeError):
                     logger.error(f"Invalid scheduled_time format for {input_data.pod_name}: {scheduled_unix_ts}. Treating as missing.")
                     data['scheduling_latency_sec'] = -1 # Or latency based on creation only? Needs definition.
                     data['scheduled_time_missing'] = 1
            else:
                 # Scheduled time not provided
                 data['scheduling_latency_sec'] = -1 # Indicate unknown latency
                 data['scheduled_time_missing'] = 1

        # Calculate time components from creation_dt
        data['creation_hour'] = creation_dt.hour
        data['creation_dayofweek'] = creation_dt.dayofweek
        data['hour_sin'] = np.sin(2 * np.pi * data['creation_hour'] / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * data['creation_hour'] / 24.0)
        data['dayofweek_sin'] = np.sin(2 * np.pi * data['creation_dayofweek'] / 7.0)
        data['dayofweek_cos'] = np.cos(2 * np.pi * data['creation_dayofweek'] / 7.0)

        # 5. Cluster State Features (Use fetched data)
        data['cluster_active_pods'] = cluster_state['active_pods']
        data['cluster_pending_pods'] = cluster_state['pending_pods']
        data['cluster_req_cpu'] = cluster_state['total_req_cpu']
        data['cluster_req_mem'] = cluster_state['total_req_mem']
        data['cluster_req_gpu_milli'] = cluster_state['total_req_gpu_milli']
        data['recent_failure_rate'] = cluster_state['recent_failure_rate']

        # Calculate ratios using fetched totals and loaded node capacity
        data['cluster_cpu_ratio'] = data['cluster_req_cpu'] / total_cluster_cpu if total_cluster_cpu > 0 else 0
        data['cluster_mem_ratio'] = data['cluster_req_mem'] / total_cluster_mem if total_cluster_mem > 0 else 0
        # Add GPU ratio if needed/trained
        # data['cluster_gpu_ratio'] = ...

        # --- End Feature Engineering ---

        # Create DataFrame and validate/reorder (Keep this logic)
        try:
            pod_df = pd.DataFrame([data])
        except Exception as e:
             logger.error(f"Failed to create DataFrame from engineered features: {e}", exc_info=True)
             return None

        missing_features = [f for f in self.features if f not in pod_df.columns]
        if missing_features:
            logger.error(f"Feature engineering failed: Missing expected features: {missing_features}")
            return None

        extra_features = [f for f in pod_df.columns if f not in self.features]
        if extra_features:
            logger.warning(f"Extra features generated: {extra_features}. Dropping them.")
            pod_df = pod_df.drop(columns=extra_features)

        try:
            pod_df = pod_df[self.features] # Ensure correct order
            # --- Data Type Conversion (Crucial for XGBoost) ---
            # Ensure all columns are numeric before prediction
            for col in pod_df.columns:
                 pod_df[col] = pd.to_numeric(pod_df[col], errors='coerce')
            # Handle potential NaNs introduced by coerce (e.g., fill with 0 or median)
            if pod_df.isnull().values.any():
                 logger.warning(f"NaNs detected after numeric conversion for {input_data.pod_name}. Filling with 0.")
                 pod_df.fillna(0, inplace=True)

            logger.debug(f"Feature engineering successful for pod {input_data.pod_name}. Shape: {pod_df.shape}")
            return pod_df
        except KeyError as e:
            logger.error(f"Feature mismatch during final reordering: Cannot find expected feature '{e}'.", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error during feature finalization: {e}", exc_info=True)
            return None


    def predict(self, input_data: PredictionInput) -> Optional[Tuple[float, Optional[pd.DataFrame]]]:
        """
        Engineer features (using dynamic data) and predict failure probability.
        Returns probability and the feature DataFrame used.
        """
        if self.model is None or self.features is None:
            logger.error("Model or features not loaded. Cannot predict.")
            return None, None

        try:
            # 1. Engineer features (now uses dynamic data)
            pod_features_df = self._engineer_features(input_data)
            if pod_features_df is None:
                 logger.error(f"Feature engineering failed for pod {input_data.pod_name}. Cannot predict.")
                 return None, None

            # 2. Predict probability
            proba = self.model.predict_proba(pod_features_df)
            failure_probability = float(proba[0, 1])

            logger.info(f"Prediction successful for pod {input_data.pod_name}. Probability: {failure_probability:.4f}")
            return failure_probability, pod_features_df

        except ValueError as ve:
             logger.error(f"Prediction failed for pod {input_data.pod_name} due to ValueError: {ve}.", exc_info=True)
             return None, None
        except Exception as e:
            logger.error(f"Prediction failed unexpectedly for pod {input_data.pod_name}: {e}", exc_info=True)
            return None, None

# Instantiate the service
prediction_service = PredictionService(
    model_path=settings.MODEL_FILE_PATH,
    feature_path=settings.FEATURE_FILE_PATH
)