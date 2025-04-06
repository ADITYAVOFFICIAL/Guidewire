# app/services/kubernetes_service.py
import logging
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from app.core.config import settings
import os
import time # For failure rate time window
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class KubernetesService:
    def __init__(self):
        self.core_api: Optional[client.CoreV1Api] = None
        self.apps_api: Optional[client.AppsV1Api] = None
        self._load_config()

    def _load_config(self):
        """Loads Kubernetes configuration."""
        try:
            # Prioritize in-cluster config
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes config.")
            # Then check explicit path from settings
            elif settings.KUBE_CONFIG_PATH and os.path.exists(settings.KUBE_CONFIG_PATH):
                config.load_kube_config(config_file=settings.KUBE_CONFIG_PATH)
                logger.info(f"Loaded Kubernetes config from: {settings.KUBE_CONFIG_PATH}")
            # Fallback to default kubeconfig location
            else:
                config.load_kube_config()
                logger.info("Loaded default Kubernetes config (kubeconfig).")

            self.core_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
            logger.info("Kubernetes API clients initialized.")

        except config.ConfigException as e:
             logger.warning(f"Could not load Kubernetes config (normal if not in-cluster or no kubeconfig): {e}")
             self.core_api = None
             self.apps_api = None
        except Exception as e:
            logger.error(f"Unexpected error configuring Kubernetes client: {e}", exc_info=True)
            self.core_api = None
            self.apps_api = None

    def is_available(self) -> bool:
        """Check if K8s clients are initialized."""
        return self.core_api is not None # Only need CoreV1Api for current functions

    def get_cluster_state(self, namespace: Optional[str] = None, exclude_pod_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches current cluster state by querying the K8s API.
        WARNING: Can be slow on large clusters. Consider caching for production.

        Args:
            namespace: If specified, only scan this namespace. Otherwise, scans all.
                       Use settings.TARGET_NAMESPACE if appropriate.
            exclude_pod_name: The name of the pod being predicted, to exclude its
                              own resource requests from the cluster totals.

        Returns:
            A dictionary containing cluster state metrics, or defaults if API unavailable/fails.
            Keys: 'active_pods', 'pending_pods', 'total_req_cpu', 'total_req_mem',
                  'total_req_gpu_milli', 'recent_failure_rate'
        """
        if not self.is_available():
            logger.warning("K8s client not available. Returning default cluster state.")
            return self._get_default_cluster_state()

        state = {
            'active_pods': 0,
            'pending_pods': 0,
            'failed_pods_recent': 0, # Pods failed within the window
            'total_pods_recent': 0,   # Pods created within the window
            'total_req_cpu': 0.0,
            'total_req_mem': 0.0,
            'total_req_gpu_milli': 0.0, # Assuming simple sum for now
        }
        # Define time window for failure rate (e.g., last 1 hour)
        failure_window_seconds = 3600
        now = time.time()
        window_start_time = now - failure_window_seconds

        try:
            logger.debug(f"Fetching pod list for cluster state (namespace: {namespace or 'all'})...")
            if namespace:
                pod_list = self.core_api.list_namespaced_pod(namespace, timeout_seconds=10) # Add timeout
            else:
                # Warning: Listing all pods can be very resource intensive
                logger.warning("Listing pods for all namespaces. This can be slow/resource intensive.")
                pod_list = self.core_api.list_pod_for_all_namespaces(timeout_seconds=30) # Longer timeout

            logger.debug(f"Processing {len(pod_list.items)} pods for cluster state.")
            for pod in pod_list.items:
                # Skip the pod being predicted if specified
                if exclude_pod_name and pod.metadata.name == exclude_pod_name and pod.metadata.namespace == namespace:
                    continue

                pod_status = pod.status.phase
                pod_creation_timestamp = pod.metadata.creation_timestamp

                # Calculate recent failure rate components
                if pod_creation_timestamp:
                    pod_created_unix = pod_creation_timestamp.timestamp()
                    if pod_created_unix >= window_start_time:
                         state['total_pods_recent'] += 1
                         if pod_status == 'Failed':
                              state['failed_pods_recent'] += 1

                # Count active/pending and sum resources for Running pods
                if pod_status == 'Running':
                    state['active_pods'] += 1
                    # Sum resource requests from all containers in the pod
                    if pod.spec.containers:
                        for container in pod.spec.containers:
                            if container.resources and container.resources.requests:
                                requests = container.resources.requests
                                state['total_req_cpu'] += self._parse_resource(requests.get('cpu', '0'))
                                state['total_req_mem'] += self._parse_resource(requests.get('memory', '0'))
                                # Simplified GPU handling - assumes nvidia.com/gpu is the key
                                state['total_req_gpu_milli'] += self._parse_resource(requests.get('nvidia.com/gpu', '0')) * 1000 # Assuming whole GPUs requested

                elif pod_status == 'Pending':
                    state['pending_pods'] += 1

            # Calculate recent failure rate (avoid division by zero)
            if state['total_pods_recent'] > 0:
                 state['recent_failure_rate'] = state['failed_pods_recent'] / state['total_pods_recent']
            else:
                 state['recent_failure_rate'] = 0.0

            # Remove temporary keys
            del state['failed_pods_recent']
            del state['total_pods_recent']

            logger.info(f"Fetched cluster state: Active={state['active_pods']}, Pending={state['pending_pods']}, FailRate={state['recent_failure_rate']:.3f}")
            return state

        except ApiException as e:
            logger.error(f"Kubernetes API error fetching cluster state: {e.status} - {e.reason}", exc_info=True)
            return self._get_default_cluster_state()
        except Exception as e:
            logger.error(f"Unexpected error fetching cluster state: {e}", exc_info=True)
            return self._get_default_cluster_state()

    def _get_default_cluster_state(self) -> Dict[str, Any]:
        """Returns placeholder values if K8s API is unavailable."""
        return {
            'active_pods': 50, # Placeholder
            'pending_pods': 5,  # Placeholder
            'total_req_cpu': 50000.0, # Placeholder
            'total_req_mem': 100000.0, # Placeholder
            'total_req_gpu_milli': 1000.0, # Placeholder
            'recent_failure_rate': 0.05 # Placeholder
        }

    def _parse_resource(self, value: str) -> float:
        """Parses Kubernetes resource strings (CPU, Memory) into a base unit (milliCPU, MiB). Simple version."""
        value = value.lower()
        try:
            if value.endswith('m'): # milliCPU
                return float(value[:-1])
            if value.endswith('ki'): # Memory KiB -> MiB
                return float(value[:-2]) / 1024
            if value.endswith('mi'): # Memory MiB
                return float(value[:-2])
            if value.endswith('gi'): # Memory GiB -> MiB
                return float(value[:-2]) * 1024
            if value.endswith('ti'): # Memory TiB -> MiB
                return float(value[:-2]) * 1024 * 1024
            # Add other units like 'k', 'm', 'g', 't' for memory if needed
            # Handle whole CPU cores
            return float(value) * 1000 # Assume whole number is CPU cores -> milliCPU
        except ValueError:
            logger.warning(f"Could not parse resource value '{value}', returning 0.")
            return 0.0


    def delete_pod(self, pod_name: str, namespace: str) -> bool:
        """Deletes a specific pod in a namespace."""
        if not self.is_available():
            logger.error("K8s client not available. Cannot delete pod.")
            return False
        try:
            logger.info(f"Attempting to delete pod '{pod_name}' in namespace '{namespace}'...")
            # Use graceful deletion (default grace period)
            self.core_api.delete_namespaced_pod(name=pod_name, namespace=namespace)
            logger.info(f"Pod '{pod_name}' deletion initiated successfully.")
            return True
        except ApiException as e:
            if e.status == 404:
                 logger.warning(f"Pod '{pod_name}' not found in namespace '{namespace}'. Assuming already deleted.")
                 return True # Treat not found as success in this context
            logger.error(f"Kubernetes API error deleting pod '{pod_name}': {e.status} - {e.reason}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting pod '{pod_name}': {e}", exc_info=True)
            return False

# Instantiate the service (singleton pattern)
k8s_service = KubernetesService()