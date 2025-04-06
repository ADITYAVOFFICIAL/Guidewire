# app/services/kubernetes_service.py
import logging
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from app.core.config import settings
import os
logger = logging.getLogger(__name__)

class KubernetesService:
    def __init__(self):
        self.core_api = None
        self.apps_api = None
        self._load_config()

    def _load_config(self):
        """Loads Kubernetes configuration."""
        try:
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes config.")
            elif settings.KUBE_CONFIG_PATH:
                config.load_kube_config(config_file=settings.KUBE_CONFIG_PATH)
                logger.info(f"Loaded Kubernetes config from: {settings.KUBE_CONFIG_PATH}")
            else:
                config.load_kube_config()
                logger.info("Loaded default Kubernetes config (kubeconfig).")

            self.core_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
            logger.info("Kubernetes API clients initialized.")

        except Exception as e:
            logger.error(f"Failed to configure Kubernetes client: {e}", exc_info=True)
            # The service can still run in recommend mode without a client
            self.core_api = None
            self.apps_api = None

    def is_available(self) -> bool:
        """Check if K8s clients are initialized."""
        return self.core_api is not None and self.apps_api is not None

    def delete_pod(self, pod_name: str, namespace: str) -> bool:
        """Deletes a specific pod in a namespace."""
        if not self.is_available():
            logger.error("K8s client not available. Cannot delete pod.")
            return False
        try:
            logger.info(f"Attempting to delete pod '{pod_name}' in namespace '{namespace}'...")
            self.core_api.delete_namespaced_pod(name=pod_name, namespace=namespace)
            logger.info(f"Pod '{pod_name}' deleted successfully.")
            return True
        except ApiException as e:
            logger.error(f"Kubernetes API error deleting pod '{pod_name}': {e.body}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting pod '{pod_name}': {e}", exc_info=True)
            return False

    # --- Add other K8s interaction functions as needed ---
    # Example: Scaling (requires finding owner, more complex)
    # def scale_deployment(self, deployment_name: str, namespace: str, replicas: int) -> bool:
    #     if not self.is_available(): return False
    #     try:
    #         scale_body = {"spec": {"replicas": replicas}}
    #         self.apps_api.patch_namespaced_deployment_scale(
    #             name=deployment_name, namespace=namespace, body=scale_body
    #         )
    #         logger.info(f"Scaled deployment '{deployment_name}' to {replicas} replicas.")
    #         return True
    #     except ApiException as e:
    #         logger.error(f"API error scaling deployment '{deployment_name}': {e.body}", exc_info=True)
    #         return False
    #     except Exception as e:
    #         logger.error(f"Unexpected error scaling deployment '{deployment_name}': {e}", exc_info=True)
    #         return False

# Instantiate the service (could use dependency injection in FastAPI later)
k8s_service = KubernetesService()