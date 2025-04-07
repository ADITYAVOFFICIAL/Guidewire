# V7.3 - Kubernetes Realistic Simulator - Enhanced Realism
import pandas as pd
import numpy as np
import datetime
import random
import math
import time
import copy
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import matplotlib.pyplot as plt # Import for plotting

# --- Constants ---
STATE_NORMAL = 'Normal'
STATE_PRECURSOR = 'Precursor'
STATE_RECOVERING = 'Recovering'

COND_CPU_PRESSURE = 'CPUPressure'
COND_MEM_PRESSURE = 'MemoryPressure'
COND_DISK_PRESSURE = 'DiskPressure'
COND_PID_PRESSURE = 'PIDPressure'
COND_NET_UNAVAILABLE = 'NetworkUnavailable'
COND_MAINTENANCE = 'Maintenance'
COND_NOT_READY = 'NotReady'

# --- Configuration V7.3 ---
N_DATAPOINTS = 1000000 # Target number of data points (Adjust as needed)
TIME_STEP = datetime.timedelta(minutes=1)
START_TIME = datetime.datetime(2025, 4, 1, 0, 0, 0)
SEED = 42

# --- Apply Seed ---
random.seed(SEED)
np.random.seed(SEED)

# --- Initial Cluster State & Node Pools---
INITIAL_PODS_PER_NODE_TARGET = 16
NODE_CAPACITY = {
    'standard': {'cpu_cores': 4000, 'memory_mib': 16000, 'max_pods': 70, 'cost': 1.0},
    'high-mem': {'cpu_cores': 4000, 'memory_mib': 32000, 'max_pods': 50, 'cost': 1.5},
    'compute-opt': {'cpu_cores': 8000, 'memory_mib': 16000, 'max_pods': 60, 'cost': 1.8}
}
INITIAL_NODE_COUNTS = {'standard': 8, 'high-mem': 3, 'compute-opt': 2}
MAX_NODES_TOTAL = 50
NODE_HEADROOM_FACTOR = 0.90

# --- Application Profiles (V7.3 - Added Pool Preferences) ---
# <<< CHANGE START >>>
APP_PROFILES = {
    'web-frontend': {
        'target_replicas_ratio': 0.4, 'cpu_request': 150, 'cpu_limit': 500,
        'mem_request': 300, 'mem_limit': 800, 'hpa_cpu_target': 65, 'hpa_mem_target': 70,
        'disk_io_factor': 0.5, 'net_io_factor': 1.2, 'base_restart_prob': 0.0001,
        'preferred_pool': 'standard', 'avoid_pool': 'high-mem' # Example preference
    },
    'database': {
        'target_replicas_ratio': 0.2, 'cpu_request': 500, 'cpu_limit': 1500,
        'mem_request': 2000, 'mem_limit': 4000, 'hpa_cpu_target': None, 'hpa_mem_target': None,
        'disk_io_factor': 2.5, 'net_io_factor': 0.8, 'base_restart_prob': 0.00005,
        'pv_required': True, 'preferred_pool': 'high-mem' # Example preference
    },
    'batch-job': {
        'target_replicas_ratio': 0.4, 'cpu_request': 800, 'cpu_limit': 2000,
        'mem_request': 1000, 'mem_limit': 2500, 'hpa_cpu_target': 80, 'hpa_mem_target': None,
        'disk_io_factor': 1.0, 'net_io_factor': 0.5, 'base_restart_prob': 0.0002,
        'preferred_pool': 'compute-opt', 'avoid_pool': 'high-mem' # Example preference
    }
}
# <<< CHANGE END >>>
total_ratio = sum(p['target_replicas_ratio'] for p in APP_PROFILES.values());
for app in APP_PROFILES: APP_PROFILES[app]['target_replicas_ratio'] /= max(0.01, total_ratio) # Normalize safely

# --- Autoscaling Config ---
HPA_CPU_THRESHOLD_SCALE_UP = 70; HPA_CPU_THRESHOLD_SCALE_DOWN = 40
HPA_MEM_THRESHOLD_SCALE_UP = 75; HPA_MEM_THRESHOLD_SCALE_DOWN = 45
HPA_SCALE_FACTOR = 0.20; HPA_FLAP_PROBABILITY = 0.03
HPA_REACTION_DELAY_STEPS_MIN = 1; HPA_REACTION_DELAY_STEPS_MAX = 5
HPA_EFFECTIVENESS_REDUCTION_FACTOR = 0.8; HPA_SCALE_FAIL_PROBABILITY = 0.015
HPA_COOLDOWN_PERIOD_STEPS = 5; MIN_PODS_PER_APP = 2;
MIN_PODS_FACTOR = 0.3 # *** ADDED MISSING CONSTANT ***
# <<< CHANGE START >>>
REPLICA_UPDATE_DELAY_STEPS = 2 # Delay before current_replicas reflects target_replicas change
# <<< CHANGE END >>>
CA_PENDING_PODS_THRESHOLD = 5; CA_NODE_UTILIZATION_LOW_THRESHOLD = 30
CA_NODE_PRESSURE_HIGH_THRESHOLD = 80; CA_SCALE_UP_COOLDOWN_STEPS = 12; CA_SCALE_DOWN_COOLDOWN_STEPS = 25
CA_NODE_PROVISION_DELAY_STEPS_MIN = 5; CA_NODE_PROVISION_DELAY_STEPS_MAX = 15
CA_NODE_FAIL_PROVISION_PROB = 0.02

# --- Control Plane Simulation ---
ETCD_BASE_LATENCY = 15; ETCD_LATENCY_STDDEV = 5
ETCD_SPIKE_PROB = 0.0005; ETCD_SPIKE_DURATION = (3, 8); ETCD_SPIKE_LATENCY = 150
ETCD_LEADER_ELECTION_PROB = 0.00005; ETCD_LEADER_ELECTION_DURATION = (5, 10); ETCD_ELECTION_LATENCY = 300
API_LATENCY_ETCD_FACTOR = 2.5; API_LATENCY_LOAD_FACTOR = 0.05

# --- Persistent Volume Simulation ---
INITIAL_PV_COUNT = 50; MAX_PV_COUNT = 150
PV_CREATE_RATE = 0.05; PV_ATTACH_ERROR_PROB_BASE = 0.001
PV_ATTACH_ERROR_PROB_API_FACTOR = 0.0005

# --- Event Simulation Config ---
EVENT_PROBABILITIES = { 'Event_Deployment': 0.0003, 'Event_NodeMaintenance': 0.00006, 'Event_ConfigChange': 0.00015, 'Event_StoragePressure': 0.0001, 'Event_NetworkSaturation': 0.00008, 'Event_ExternalDependency': 0.00015, 'Event_KubeletIssue': 0.0001, 'Event_ManualIntervention': 0.00004, 'Event_NetworkPartition': 0.00003 }
EVENT_DURATIONS = { 'Event_Deployment': (8, 25), 'Event_NodeMaintenance': (50, 150), 'Event_ConfigChange': (4, 12), 'Event_StoragePressure': (20, 60), 'Event_NetworkSaturation': (15, 50), 'Event_ExternalDependency': (15, 45), 'Event_KubeletIssue': (10, 40), 'Event_ManualIntervention': 1, 'Event_NetworkPartition': (20, 70) }
# <<< CHANGE START >>>
# Added specific log rate impacts
EVENT_IMPACTS = {
    'Event_Deployment': {'Pod_Restart_Rate': 0.6, 'App_Latency_Avg': 60, 'API_Server_Request_Latency_Avg': 35, 'WarningLogRate': 6, 'APIErrorLogRate': 0.1},
    'Event_NodeMaintenance': {'node_specific': True}, # Specific node impacts handled elsewhere
    'Event_ConfigChange': {'Cluster_CPU_Usage_Avg': (0.75, 1.25), 'Cluster_Memory_Usage_Avg': (0.85, 1.15), 'App_Error_Rate_Avg': (0.9, 3.5), 'Pod_Restart_Rate': (0.9, 2.2), 'WarningLogRate': (1.5, 6.0)},
    'Event_StoragePressure': {'Cluster_Disk_IO_Rate': 150, 'Evicted_Pod_Count_Rate': 0.1, 'Num_Pods_Pending': 5, 'WarningLogRate': 8, 'Storage_IOPS_Saturation_Percent': 40, 'node_pressure': COND_DISK_PRESSURE, 'EvictionLogRate': 0.15},
    'Event_NetworkSaturation': {'Cluster_Network_Errors_Rate': 2, 'Cluster_Network_Latency_Avg': 30, 'Network_Bandwidth_Saturation_Percent': 50, 'WarningLogRate': 10, 'NetworkErrorLogRate': 2.5},
    'Event_ExternalDependency': {'App_Latency_Avg': 180, 'App_Error_Rate_Avg': 10, 'WarningLogRate': 12},
    'Event_KubeletIssue': {'node_specific': True, 'impact': {'Pod_Restart_Rate': 1.0, 'App_Latency_Avg': 20, 'WarningLogRate': 5, 'ErrorLogRate': 1.5}},
    'Event_ManualIntervention': {'is_intervention': True},
    'Event_NetworkPartition': {'node_specific': True, 'num_nodes_factor': 0.3, 'impact': {'Cluster_Network_Errors_Rate': 5, 'Cluster_Network_Latency_Avg': 100, 'API_Server_Request_Latency_Avg': 80, 'NetworkErrorLogRate': 6.0}, 'node_condition_add': (COND_NET_UNAVAILABLE, 1.0)}
}
# <<< CHANGE END >>>
MANUAL_INTERVENTION_ACTIONS = ['clear_pending_pods', 'force_node_ready', 'scale_down_app:web-frontend', 'scale_down_app:batch-job', 'restart_api_server', 'delete_failing_pods:Impending_Pod_Failure']
POST_EVENT_FAIL_BOOST_FACTOR = 1.7; POST_EVENT_FAIL_BOOST_DURATION = 50

# --- Configuration Drift ---
DRIFT_CONFIG = { 'Cluster_CPU_Usage_Avg': (1.25, 0.00001), 'Cluster_Memory_Usage_Avg': (1.25, 0.00001), 'App_Latency_Avg': (1.30, 0.000015), 'App_Error_Rate_Avg': (1.50, 0.00002)}
DRIFT_APPLICATION_INTERVAL = 100
drift_factors: Dict[str, float] = {metric: 1.0 for metric in DRIFT_CONFIG} # Global drift state

# --- Noise Config ---
OUTLIER_PROBABILITY = 0.003; OUTLIER_MAGNITUDE_FACTOR = 4.0
STUCK_VALUE_PROBABILITY = 0.001; ZERO_DROP_PROBABILITY = 0.0005
last_value_cache: Dict[str, float] = {} # Global cache for stuck values

# --- Normal Operating Ranges (Mean, StdDev) (V7.3 - Added specific log rates) ---
# <<< CHANGE START >>>
NORMAL_RANGES = {
    'Cluster_CPU_Usage_Avg': (35, 6), 'Cluster_CPU_Usage_Max': (65, 12), 'Cluster_Memory_Usage_Avg': (45, 6), 'Cluster_Memory_Usage_Max': (75, 12),
    'MaxNodePressurePercent': (35, 12), 'AvgCPURequestCommitRatio': (0.70, 0.20), 'AvgMemoryRequestCommitRatio': (0.80, 0.20),
    'CPU_Throttling_Rate': (0.5, 0.7), 'Cluster_Disk_IO_Rate': (150, 50), 'Storage_IOPS_Saturation_Percent': (15, 10), 'Storage_Throughput_Saturation_Percent': (20, 12),
    'Cluster_Network_Bytes_In': (8e6, 3.5e6), 'Cluster_Network_Bytes_Out': (3.5e6, 2.5e6), 'Cluster_Network_Latency_Avg': (2.0, 1.0), 'Cluster_Network_Errors_Rate': (0.3, 0.3),
    'Network_Bandwidth_Saturation_Percent': (25, 15), 'Num_Pods_Running': (sum(INITIAL_NODE_COUNTS.values()) * INITIAL_PODS_PER_NODE_TARGET, 40),
    'Num_Pods_Pending': (1, 2.0), 'Pod_Restart_Rate': (0.15, 0.15), 'Evicted_Pod_Count_Rate': (0.02, 0.08), 'ImagePullBackOff_Count_Rate': (0.07, 0.14),
    'Num_Ready_Nodes': (sum(INITIAL_NODE_COUNTS.values()), 0), 'Num_NotReady_Nodes': (0, 0), 'API_Server_Request_Latency_Avg': (70, 30),
    'Etcd_DB_Size': (700, 30), 'Etcd_Request_Latency_Avg': (ETCD_BASE_LATENCY, ETCD_LATENCY_STDDEV), 'App_Error_Rate_Avg': (2.0, 2.0), 'App_Latency_Avg': (140, 40),
    'ErrorLogRate': (0.7, 0.7), 'WarningLogRate': (2.8, 1.4),
    'OOMKillLogRate': (0.01, 0.05), 'NetworkErrorLogRate': (0.05, 0.1), 'EvictionLogRate': (0.02, 0.06), 'APIErrorLogRate': (0.03, 0.08), # New specific log rates
    'PV_Attached_Count': (INITIAL_PV_COUNT, 5), 'PV_Attach_Error_Rate': (0.001, 0.002)
}
# <<< CHANGE END >>>
ORIGINAL_MEANS = {k: v[0] for k, v in NORMAL_RANGES.items() if k in DRIFT_CONFIG}

# --- Failure Configuration (V7.3 - Added specific log impacts) ---
# <<< CHANGE START >>>
FAILURE_IMPACTS = {
    'Impending_Resource_Exhaustion_CPU': {
        'primary': {'node_cpu_usage_factor': (1.5, 'ease-in-out'), 'node_condition_add': (COND_CPU_PRESSURE, 0.6), 'node_impact_scale_factor':'cpu_request'},
        'secondary': {'CPU_Throttling_Rate': (15, 'linear', 1.8, 0.1), 'App_Latency_Avg': (90, 'linear', 1.0, 0.2), 'API_Server_Request_Latency_Avg': (60, 'linear', 1.0, 0.3), 'WarningLogRate': (15, 'linear', 1.0, 0.2), 'App_Error_Rate_Avg': (8, 'linear', 1.2, 0.4)},
        'cascade_trigger': {'Impending_API_Slowdown': (0.08, 1.6), 'Impending_Pod_Failure:web-frontend': (0.06, 1.3)}
    },
    'Impending_Resource_Exhaustion_Memory': {
        'primary': {'node_mem_usage_factor': (1.6, 'ease-in-out'), 'node_condition_add': (COND_MEM_PRESSURE, 0.7), 'node_impact_scale_factor':'mem_request'},
        'secondary': {'Evicted_Pod_Count_Rate': (0.8, 'spike', 2.5, 0.5), 'Num_Pods_Pending': (15, 'linear', 1.0, 0.6), 'ErrorLogRate': (8, 'linear', 1.0, 0.4), 'Pod_Restart_Rate': (5, 'linear', 1.0, 0.3), 'OOMKillLogRate': (0.5, 'linear', 1.0, 0.4), 'EvictionLogRate': (0.6, 'spike', 1.0, 0.5)}, # Added OOM/Eviction logs
        'cascade_trigger': {'Impending_Pod_Failure:database': (0.1, 1.4), 'Impending_Node_Failure': (0.05, 1.6)}
    },
    'Impending_Memory_Leak': {
        'target_app_type': 'random', 'primary': {'app_mem_usage_factor': (2.5, 'linear')},
        'secondary': {'Pod_Restart_Rate': (2, 'linear', 1.0, 0.6), 'Evicted_Pod_Count_Rate': (0.2, 'linear', 1.5, 0.8), 'OOMKillLogRate': (0.3, 'linear', 1.0, 0.7)}, # Added OOM logs
        'cascade_trigger': {'Impending_Resource_Exhaustion_Memory': (0.20, 1.0)}
    },
    'Impending_Pod_Failure': {
        'target_app_type': 'random', 'primary': {'app_restart_rate_increase': (15, 'ease-in-out'), 'app_pending_increase': (25, 'linear')},
        'secondary': {'App_Error_Rate_Avg': (20, 'linear', 1.0, 0.1), 'App_Latency_Avg': (70, 'linear', 1.0, 0.1), 'ErrorLogRate': (12, 'linear', 1.0, 0)},
        'tertiary': {'API_Server_Request_Latency_Avg': (35, 'linear', 1.0, 0.3)},
        'cascade_trigger': {'Impending_Service_Disruption': (0.12, 1.2), 'Impending_API_Slowdown': (0.05, 1.1)}
    },
    'Impending_Network_Issue': {
        'primary': {'Cluster_Network_Latency_Avg': (100, 'ease-in-out'), 'Cluster_Network_Errors_Rate': (12, 'linear'), 'Network_Bandwidth_Saturation_Percent': (45, 'linear')},
        'secondary': {'App_Latency_Avg': (180, 'linear', 1.0, 0.1), 'App_Error_Rate_Avg': (12, 'linear', 1.0, 0.2), 'Pod_Restart_Rate': (2.0, 'linear', 1.0, 0.3), 'WarningLogRate': (20, 'linear', 1.0, 0.1), 'API_Server_Request_Latency_Avg': (50, 'linear', 1.0, 0.4), 'NetworkErrorLogRate': (15, 'linear', 1.0, 0.1)}, # Added Network logs
        'tertiary': {'node_condition_add': (COND_NET_UNAVAILABLE, 0.2, 0.5)},
        'cascade_trigger': {'Impending_Node_Failure': (0.1, 1.4), 'Impending_Pod_Failure': (0.08, 1.2), 'Event_NetworkPartition': (0.05, 1.0)}
    },
    'Impending_Node_Failure': {
        'node_specific': True, 'primary': {'node_set_not_ready': (True, 'spike')},
        'secondary': {'Num_Pods_Pending': (40, 'linear', 1.0, 0.1), 'Evicted_Pod_Count_Rate': (1.2, 'spike', 1.0, 0.1), 'WarningLogRate': (30, 'spike', 1.0, 0), 'EvictionLogRate': (1.0, 'spike', 1.0, 0.1)}, # Added Eviction logs
        'tertiary': {'API_Server_Request_Latency_Avg': (45, 'linear', 1.0, 0.2)},
        'cascade_trigger': {'Impending_API_Slowdown': (0.1, 1.0)}
    },
    'Impending_Service_Disruption': {
        'target_app_type': 'random', 'primary': {'app_error_rate_increase': (40, 'ease-in-out'), 'app_latency_increase': (300, 'ease-in-out')},
        'secondary': {'ErrorLogRate': (35, 'linear', 1.0, 0), 'Pod_Restart_Rate': (3.0, 'linear', 1.0, 0.2)},
        'cascade_trigger': {'Impending_Pod_Failure': (0.1, 1.3)}
    },
    'Impending_API_Slowdown': {
        'primary': {'API_Server_Request_Latency_Avg': (350, 'ease-in-out'), 'Etcd_Request_Latency_Avg': (100, 'ease-in-out')},
        'secondary': {'Num_Pods_Pending': (18, 'linear', 1.0, 0.1), 'Etcd_DB_Size': (120, 'linear', 1.0, 0.3), 'WarningLogRate': (18, 'linear', 1.0, 0.1), 'CPU_Throttling_Rate': (3, 'linear', 1.0, 0.2), 'PV_Attach_Error_Rate': (0.01, 'linear', 1.0, 0.2), 'APIErrorLogRate': (0.8, 'linear', 1.0, 0.1)}, # Added API logs
        'cascade_trigger': {'Impending_Pod_Failure': (0.15, 1.2), 'Impending_Node_Failure': (0.06, 1.1)}
    },
    'Impending_PV_Attach_Storm': {
        'primary': {'PV_Attach_Error_Rate': (0.05, 'spike')},
        'secondary': {'Num_Pods_Pending': (25, 'linear', 1.0, 0.1), 'WarningLogRate': (15, 'linear', 1.0, 0), 'ErrorLogRate': (5, 'linear', 1.0, 0.1)}, # Added Error logs
        'cascade_trigger': {'Impending_Pod_Failure:database': (0.1, 1.0)}
    }
}
# <<< CHANGE END >>>
FAILURE_TYPES = list(FAILURE_IMPACTS.keys())
BASE_FAILURE_PROBABILITY = 0.00055
MIN_PRECURSOR_DURATION = 30; MAX_PRECURSOR_DURATION = 180; RECOVERY_DURATION_FACTOR = 0.85; INTERMITTENT_RECOVERY_PROB = 0.07

# --- Class Balance Maintenance ---
CLASS_BALANCE_TARGET_RATIO = 0.8; FAILURE_TYPE_COUNTS = defaultdict(int); TOTAL_FAILURES_TRIGGERED = 0

# --- Helper Function Definitions (Defined Before Use) ---
def get_seasonal_multiplier(timestamp: datetime.datetime, daily_amp: float, weekly_amp: float) -> float:
    """Calculates a seasonal multiplier based on time of day and week."""
    SECONDS_PER_DAY = 24 * 60 * 60; SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
    time_of_day_sec = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    day_of_week_sec = timestamp.weekday() * SECONDS_PER_DAY + time_of_day_sec
    daily_cycle = math.sin(2 * math.pi * time_of_day_sec / SECONDS_PER_DAY - math.pi / 2)
    weekly_cycle = math.sin(2 * math.pi * day_of_week_sec / SECONDS_PER_WEEK - math.pi / 2)
    return 1.0 + daily_amp * daily_cycle + weekly_amp * weekly_cycle

def apply_trend(base_value: float, progress: float, delta: float, shape: str, exponent: float = 1.0) -> float:
    """Applies a trend delta based on progress (0 to 1), shape, and exponent."""
    effective_delta = 0.0; progress = max(0.0, min(1.0, progress))
    if shape == 'linear': effective_delta = delta * progress
    elif shape == 'ease-in-out': p = -2 * progress**3 + 3 * progress**2; effective_delta = delta * p
    elif shape == 'spike': effective_delta = delta
    else: effective_delta = delta * progress
    if exponent != 1.0 and effective_delta != 0: effective_delta = math.copysign(abs(effective_delta) ** exponent, effective_delta)
    result = base_value + effective_delta
    if base_value >= 0 and (base_value + effective_delta) < 0 and delta < 0 : return 0.0
    elif base_value >= 0 and delta >=0 : return max(0.0, result)
    return result

def add_noise_and_outliers(metric: str, value: float, stddev: float, step: int) -> float:
    """Adds noise, potential outliers, and handles stuck/zero drop values."""
    global last_value_cache
    if random.random() < STUCK_VALUE_PROBABILITY and metric in last_value_cache: return last_value_cache[metric]
    base_noise = stddev * 0.4; noise = np.random.normal(0, max(0.01, base_noise))
    final_value = value + noise
    if random.random() < OUTLIER_PROBABILITY: outlier = np.random.normal(0, stddev * OUTLIER_MAGNITUDE_FACTOR) * random.choice([-1, 1]); final_value += outlier
    is_rate = 'Rate' in metric or 'Count' in metric; is_percent = 'Percent' in metric or '%' in metric or 'Ratio' in metric
    if is_rate and random.random() < ZERO_DROP_PROBABILITY: final_value = 0.0
    final_value = max(0.0, final_value)
    if is_percent : final_value = min(100.0 if ('Percent' in metric or '%' in metric) else 5.0, final_value)
    last_value_cache[metric] = final_value; return final_value

def generate_base_value(metric: str, timestamp: datetime.datetime, current_n_nodes_total: int, target_total_pods: float, not_ready_node_ids: Set[int]) -> float:
    """Generates a baseline value for a metric, considering drift and seasonality."""
    global drift_factors
    mean_drift_factor = drift_factors.get(metric, 1.0)
    original_mean = ORIGINAL_MEANS.get(metric)
    if metric in NORMAL_RANGES:
        base_mean, stddev = NORMAL_RANGES[metric]
        if original_mean is not None: mean = original_mean * mean_drift_factor
        else: mean = base_mean
    else: return 0.0 # Metric not defined in NORMAL_RANGES
    initial_node_count = sum(INITIAL_NODE_COUNTS.values())
    if metric == 'Num_Pods_Running': mean = target_total_pods; stddev = mean * 0.05
    elif metric in ['Cluster_Network_Bytes_In', 'Cluster_Network_Bytes_Out', 'Cluster_Disk_IO_Rate', 'ErrorLogRate', 'WarningLogRate', 'OOMKillLogRate', 'NetworkErrorLogRate', 'EvictionLogRate', 'APIErrorLogRate']: # <<< CHANGE: Added specific log rates >>>
         if initial_node_count > 0: mean = mean * (current_n_nodes_total / initial_node_count)
    multiplier = 1.0;
    if 'CPU' in metric or 'Memory' in metric or 'Pressure' in metric or 'Saturation' in metric: multiplier = get_seasonal_multiplier(timestamp, 0.15, 0.10)
    elif 'Network' in metric: multiplier = get_seasonal_multiplier(timestamp, 0.20, 0.15)
    elif 'App_' in metric or 'LogRate' in metric: multiplier = get_seasonal_multiplier(timestamp, 0.25, 0.05)
    adjusted_mean = mean * multiplier
    # Use Poisson for rates/counts, Normal for others
    if metric in ['Num_Pods_Pending', 'Pod_Restart_Rate', 'Cluster_Network_Errors_Rate', 'App_Error_Rate_Avg', 'Evicted_Pod_Count_Rate', 'ImagePullBackOff_Count_Rate', 'CPU_Throttling_Rate', 'ErrorLogRate', 'WarningLogRate', 'PV_Attach_Error_Rate', 'OOMKillLogRate', 'NetworkErrorLogRate', 'EvictionLogRate', 'APIErrorLogRate']: # <<< CHANGE: Added specific log rates >>>
        val = np.random.poisson(max(0.1, adjusted_mean))
    elif metric == 'Num_NotReady_Nodes': return float(len(not_ready_node_ids))
    elif metric == 'Num_Ready_Nodes': return float(current_n_nodes_total - len(not_ready_node_ids))
    else: val = np.random.normal(adjusted_mean, stddev);
    # Apply constraints
    if 'Ratio' in metric: val = max(0.1, min(3.0, val))
    elif 'Pressure' in metric or 'Saturation' in metric or 'Usage' in metric: val = max(0, min(100, val))
    elif metric == 'Etcd_Request_Latency_Avg': val = max(5, val)
    elif metric == 'API_Server_Request_Latency_Avg': val = max(10, val)
    return max(0.0, val) # Ensure non-negative

def apply_config_drift():
    """Applies random drift to baseline parameters."""
    global drift_factors
    for metric, (max_drift, change_stddev) in DRIFT_CONFIG.items():
        change = np.random.normal(0, change_stddev)
        current_factor = drift_factors[metric]
        new_factor = current_factor + change
        min_factor = 1.0 / max_drift
        drift_factors[metric] = max(min_factor, min(max_drift, new_factor))

def get_dynamic_failure_factor(metrics: Dict[str, float], boost_factor: float = 1.0) -> float: # DEFINED HERE
    """Calculates a dynamic factor to adjust failure probability based on cluster health."""
    cpu = metrics.get('Cluster_CPU_Usage_Avg', 35); mem = metrics.get('Cluster_Memory_Usage_Avg', 45); api_lat = metrics.get('API_Server_Request_Latency_Avg', 70); node_press = metrics.get('MaxNodePressurePercent', 35); restart_rate = metrics.get('Pod_Restart_Rate', 0.15)
    factor = 1.0; factor += 0.7 * (cpu > 80) + 0.7 * (mem > 85); factor += 0.5 * (api_lat > 250); factor += 0.4 * (node_press > 75); factor += 0.3 * (restart_rate > 5)
    return min(factor * boost_factor, 4.5)

# --- Classes ---
class Node:
    """Represents a simulated Kubernetes node."""
    def __init__(self, node_id: int, pool_name: str):
        self.id: int = node_id; self.pool: str = pool_name; self.capacity: Dict[str, Any] = NODE_CAPACITY[pool_name]; self.ready: bool = True; self.conditions: Set[str] = set()
        self.cpu_usage: float = 0.0; self.mem_usage: float = 0.0; self.disk_io: float = 0.0; self.net_in: float = 0.0; self.net_out: float = 0.0
        self.cpu_request_total: float = 0.0; self.mem_request_total: float = 0.0; self.pods: Dict[str, int] = defaultdict(int)
        self.cpu_over_limit: bool = False; self.mem_over_limit: bool = False

    def update_requests_from_pods(self):
        """Recalculates total CPU/Mem requests based on assigned pods."""
        self.cpu_request_total = sum(APP_PROFILES[app]['cpu_request'] * count for app, count in self.pods.items())
        self.mem_request_total = sum(APP_PROFILES[app]['mem_request'] * count for app, count in self.pods.items())

    def update_metrics(self, cluster_metrics_base: Dict[str, float], step: int, sim_state: 'ClusterState'):
        """Calculates node metrics based on pods, cluster load, events, failures."""
        profile = self.capacity; app_pods = self.pods; current_time = sim_state.timestamps[step]
        # 1. Base usage
        base_cpu_usage = sum(APP_PROFILES[app]['cpu_request'] * (1 + random.uniform(-0.1, 0.3)) * count for app, count in app_pods.items())
        base_mem_usage = sum(APP_PROFILES[app]['mem_request'] * (1 + random.uniform(-0.05, 0.15)) * count for app, count in app_pods.items())
        node_disk_factor = sum(APP_PROFILES[app]['disk_io_factor'] * count for app, count in app_pods.items()); node_net_factor = sum(APP_PROFILES[app]['net_io_factor'] * count for app, count in app_pods.items())
        total_disk_factor_approx = max(1, sum(APP_PROFILES[app]['disk_io_factor'] * a_data.current_replicas for app, a_data in sim_state.apps.items())); total_net_factor_approx = max(1, sum(APP_PROFILES[app]['net_io_factor'] * a_data.current_replicas for app, a_data in sim_state.apps.items()))
        base_disk_io = node_disk_factor * cluster_metrics_base['Cluster_Disk_IO_Rate'] / total_disk_factor_approx if total_disk_factor_approx else 0
        base_net_in = node_net_factor * cluster_metrics_base['Cluster_Network_Bytes_In'] / total_net_factor_approx if total_net_factor_approx else 0
        base_net_out = node_net_factor * cluster_metrics_base['Cluster_Network_Bytes_Out'] / total_net_factor_approx if total_net_factor_approx else 0

        # 2. Apply Failure Impacts
        node_cpu_factor = 1.0; node_mem_factor = 1.0
        if sim_state.current_state in [STATE_PRECURSOR, STATE_RECOVERING] and sim_state.active_failure_type:
            failure_details = FAILURE_IMPACTS[sim_state.active_failure_type]
            progress = sim_state.progress
            if sim_state.active_failure_target_node == self.id or not failure_details.get('node_specific', False):
                impact_scale = 1.0; scale_basis = failure_details.get('node_impact_scale_factor')
                if scale_basis == 'cpu_request': impact_scale = max(0.5, min(2.0, self.cpu_request_total / max(1, sim_state.avg_node_cpu_request)))
                elif scale_basis == 'mem_request': impact_scale = max(0.5, min(2.0, self.mem_request_total / max(1, sim_state.avg_node_mem_request)))

                impact = failure_details.get('primary',{}).get('node_cpu_usage_factor')
                if impact: delta, shape = impact; node_cpu_factor = apply_trend(1.0, progress, (delta - 1.0) * impact_scale, shape)
                impact = failure_details.get('primary',{}).get('node_mem_usage_factor')
                if impact: delta, shape = impact; node_mem_factor = apply_trend(1.0, progress, (delta - 1.0) * impact_scale, shape)
                # Add Condition...
                impact = failure_details.get('primary',{}).get('node_condition_add') or failure_details.get('tertiary', {}).get('node_condition_add')
                if impact:
                    condition, prob = impact[0], impact[1]; delay_frac = impact[2] if len(impact) > 2 else 0.0
                    if sim_state.step_in_state >= sim_state.precursor_duration_actual * delay_frac:
                        if random.random() < prob * max(progress, 0.1): self.conditions.add(condition)

            # Set node NotReady...
            impact = failure_details.get('primary',{}).get('node_set_not_ready')
            if impact and sim_state.active_failure_target_node == self.id:
                 should_be_not_ready, shape = impact
                 if shape == 'spike' and sim_state.current_state == STATE_PRECURSOR and sim_state.step_in_state == 0: self.ready = not should_be_not_ready; self.conditions.add(COND_NOT_READY)
                 elif shape != 'spike':
                      is_not_ready = (should_be_not_ready and progress > 0.05)
                      self.ready = not is_not_ready
                      if is_not_ready: self.conditions.add(COND_NOT_READY)
                      else: self.conditions.discard(COND_NOT_READY)


        self.cpu_usage = base_cpu_usage * node_cpu_factor; self.mem_usage = base_mem_usage * node_mem_factor; self.disk_io = base_disk_io; self.net_in = base_net_in; self.net_out = base_net_out

        # 3. Apply Event Impacts
        if sim_state.current_state.startswith('Event_'):
            event_info = sim_state.active_event_info; target_nodes = event_info.get('target_nodes', [])
            if self.id in target_nodes:
                impact_data = event_info.get('impact_data', {}); node_pressure_cond = impact_data.get('node_pressure'); node_condition_impact = impact_data.get('node_condition_add')
                if node_pressure_cond: self.conditions.add(node_pressure_cond)
                if node_condition_impact and isinstance(node_condition_impact, tuple) and len(node_condition_impact) >= 2:
                    condition, prob = node_condition_impact
                    if random.random() < prob:
                        self.conditions.add(condition)
        # 4. Update Conditions based on usage
        existing_conditions = self.conditions.copy(); self.conditions = set(c for c in existing_conditions if c not in [COND_CPU_PRESSURE, COND_MEM_PRESSURE])
        cpu_usage_percent = 100 * self.cpu_usage / max(1, self.capacity['cpu_cores']); mem_usage_percent = 100 * self.mem_usage / max(1, self.capacity['memory_mib'])
        self.cpu_over_limit = self.cpu_usage > self.capacity['cpu_cores']; self.mem_over_limit = self.mem_usage > self.capacity['memory_mib']
        if cpu_usage_percent > 90 or self.cpu_over_limit: self.conditions.add(COND_CPU_PRESSURE)
        if mem_usage_percent > 90 or self.mem_over_limit: self.conditions.add(COND_MEM_PRESSURE)
        if not self.ready: self.conditions.add(COND_NOT_READY)
        else: self.conditions.discard(COND_NOT_READY)

    def can_schedule(self, app_profile: Dict) -> bool:
        """Checks if a pod of the given profile can be scheduled on this node."""
        # Basic checks: node readiness and critical conditions
        if not self.ready or self.conditions.intersection({COND_DISK_PRESSURE, COND_PID_PRESSURE, COND_NET_UNAVAILABLE, COND_MAINTENANCE}):
            return False
        # Resource checks
        fits_cpu = (self.cpu_request_total + app_profile['cpu_request']) < self.capacity['cpu_cores'] * NODE_HEADROOM_FACTOR
        fits_mem = (self.mem_request_total + app_profile['mem_request']) < self.capacity['memory_mib'] * NODE_HEADROOM_FACTOR
        fits_pods = sum(self.pods.values()) < self.capacity['max_pods']

        return fits_cpu and fits_mem and fits_pods

class Application:
    """Represents a simulated application type."""
    def __init__(self, name: str):
        self.name: str = name; self.profile: Dict = APP_PROFILES[name]
        self.target_replicas: int = 0; self.current_replicas: int = 0; self.pending_pods: int = 0; self.restarts_rate: float = 0.0
        # <<< CHANGE START >>>
        # Attributes for delayed replica updates
        self.intended_replicas: int = 0
        self.replica_change_apply_step: int = -1
        # <<< CHANGE END >>>

class ClusterState:
    """Manages the overall state of the simulated cluster."""
    def __init__(self, start_time: datetime.datetime, num_datapoints: int, time_step: datetime.timedelta):
        self.current_step: int = 0; self.current_time: datetime.datetime = start_time; self.timestamps: List[datetime.datetime] = [start_time + i * time_step for i in range(num_datapoints)]
        self.nodes: Dict[int, Node] = {}; self.apps: Dict[str, Application] = {}; self.next_node_id: int = 0; self.current_n_nodes_total: int = 0; self.target_total_pods: float = 0; self.pv_count: int = INITIAL_PV_COUNT; self.not_ready_node_ids: Set[int] = set()
        self.etcd_latency: float = ETCD_BASE_LATENCY; self.etcd_state: str = STATE_NORMAL; self.etcd_state_end_step: int = -1
        self.current_state: str = STATE_NORMAL; self.state_end_step: int = -1; self.event_end_step: int = -1; self.active_failure_type: Optional[str] = None; self.precursor_start_step: int = -1; self.precursor_duration_actual: int = 1; self.step_in_state: int = 0; self.progress: float = 0.0; self.active_deltas: Dict[str, float] = {}; self.active_event_info: Dict[str, Any] = {}; self.active_failure_target_app: Optional[str] = None; self.active_failure_target_node: Optional[int] = None; self.previous_state: str = STATE_NORMAL; self.previous_failure_type: Optional[str] = None; self.previous_failure_target_app: Optional[str] = None; self.previous_failure_target_node: Optional[int] = None
        self.last_hpa_action_step: Dict[str, int] = defaultdict(lambda: -HPA_COOLDOWN_PERIOD_STEPS * 2); self.last_hpa_fail_step: Dict[str, int] = defaultdict(lambda: -1)
        self.last_ca_scale_up_step: int = -CA_SCALE_UP_COOLDOWN_STEPS * 2; self.last_ca_scale_down_step: int = -CA_SCALE_DOWN_COOLDOWN_STEPS * 2
        self.pending_node_provisioning: Dict[int, List[Tuple[int, str]]] = defaultdict(list); self.failed_provision_attempts: Dict[int, int] = defaultdict(int)
        self.post_event_boost_end_step: int = -1; self.last_intervention_step: int = -1000
        self.last_step_metrics: Dict[str, float] = {}; self.avg_node_cpu_request: float = 1.0; self.avg_node_mem_request: float = 1.0

    def initialize_cluster(self):
        """Initializes nodes and apps based on config."""
        self.nodes = {}; self.apps = {app_name: Application(app_name) for app_name in APP_PROFILES}; self.current_n_nodes_total = 0; self.not_ready_node_ids = set(); self.next_node_id = 0
        for pool_name, count in INITIAL_NODE_COUNTS.items():
            for _ in range(count):
                node_id = self.next_node_id; self.next_node_id += 1; self.nodes[node_id] = Node(node_id, pool_name); self.current_n_nodes_total += 1
        initial_total_pods_target = self.current_n_nodes_total * INITIAL_PODS_PER_NODE_TARGET; self.target_total_pods = initial_total_pods_target; remaining_pods = initial_total_pods_target
        app_names = list(APP_PROFILES.keys()); random.shuffle(app_names)
        for i, app_name in enumerate(app_names):
            profile = APP_PROFILES[app_name]
            if i == len(app_names) - 1: app_target = max(MIN_PODS_PER_APP, int(remaining_pods))
            else: app_target = max(MIN_PODS_PER_APP, int(initial_total_pods_target * profile['target_replicas_ratio'])); app_target = min(app_target, remaining_pods - MIN_PODS_PER_APP * (len(app_names) - 1 - i))
            self.apps[app_name].target_replicas = app_target
            self.apps[app_name].current_replicas = app_target
            # <<< CHANGE START >>>
            self.apps[app_name].intended_replicas = app_target # Initialize intended replicas
            # <<< CHANGE END >>>
            remaining_pods -= app_target
        all_node_ids = list(self.nodes.keys()); pod_assignments = [];
        for app_name, app_data in self.apps.items(): pod_assignments.extend([app_name] * app_data.current_replicas)
        random.shuffle(pod_assignments)
        for idx, app_name in enumerate(pod_assignments):
             if not all_node_ids: break; node_id = all_node_ids[idx % len(all_node_ids)]; self.nodes[node_id].pods[app_name] += 1
        for node in self.nodes.values(): node.update_requests_from_pods()
        self._update_avg_node_requests(); print(f"Initialized {self.current_n_nodes_total} nodes and {initial_total_pods_target} pods.")

    def _update_avg_node_requests(self):
        """Calculates average request load per node."""
        total_cpu_req = sum(n.cpu_request_total for n in self.nodes.values()); total_mem_req = sum(n.mem_request_total for n in self.nodes.values())
        num_nodes = len(self.nodes); self.avg_node_cpu_request = total_cpu_req / max(1, num_nodes); self.avg_node_mem_request = total_mem_req / max(1, num_nodes)

    def get_ready_nodes(self) -> List[int]: return [nid for nid, ndata in self.nodes.items() if ndata.ready]
    def get_total_capacity(self, resource_type: str) -> float:
        attr = 'cpu_cores' if resource_type == 'cpu' else 'memory_mib' if resource_type == 'mem' else None
        if attr is None: return 0.0
        total_cap = sum(node.capacity[attr] for node in self.nodes.values() if node.ready)
        return max(1.0, total_cap)

# --- Main Simulator Class ---
class Simulator:
    """Orchestrates the Kubernetes simulation steps."""
    def __init__(self, num_datapoints: int, start_time: datetime.datetime, time_step: datetime.timedelta):
        self.cluster_state: ClusterState = ClusterState(start_time, num_datapoints, time_step)
        self.cluster_state.initialize_cluster()
        # <<< CHANGE START >>>
        self.metrics_to_record: List[str] = sorted(list(NORMAL_RANGES.keys())) # Ensure all defined metrics are recorded
        # <<< CHANGE END >>>

    def run_step(self, i: int) -> Dict[str, Any]:
        """Runs a single simulation step."""
        cs = self.cluster_state; cs.current_step = i; cs.current_time = cs.timestamps[i]
        step_metrics: Dict[str, Any] = {}

        # --- Order of Operations ---
        if i > 0 and i % DRIFT_APPLICATION_INTERVAL == 0: self._apply_config_drift()
        self._update_node_provisioning()
        self._update_node_readiness_from_events()
        self._update_control_plane(step_metrics)
        pods_to_schedule_hpa = self._run_hpa() # HPA decides target replicas
        self._run_ca()
        self._update_simulation_state()
        pods_to_schedule_restarts = self._get_restarting_pods() # Calculates pods needing restart based on current state
        # <<< CHANGE START >>>
        # Apply delayed replica changes BEFORE scheduling new pods
        self._apply_delayed_replica_changes()
        # <<< CHANGE END >>>
        all_pods_to_schedule = pods_to_schedule_hpa + pods_to_schedule_restarts
        scheduled_count, pending_count = self._simulate_scheduling(all_pods_to_schedule) # Schedule new/restarted pods
        self._update_all_node_metrics(step_metrics) # Update node usage based on *current* pod assignments
        self._aggregate_cluster_metrics(step_metrics) # Aggregate node metrics
        self._update_misc_cluster_metrics(step_metrics) # Update other cluster metrics (PVs, base values, etc.)
        self._apply_cluster_event_impacts(step_metrics) # Apply event impacts to cluster metrics
        self._apply_cluster_failure_impacts(step_metrics) # Apply failure impacts to cluster metrics
        final_row = self._apply_noise_and_finalize(step_metrics) # Add noise, finalize row
        cs.last_step_metrics = final_row.copy();
        # <<< CHANGE: Removed direct update here, handled by _apply_delayed_replica_changes >>>
        # self._update_app_replicas();
        cs._update_avg_node_requests()

        return final_row

    # --- Helper methods for Simulator class ---
    def _apply_config_drift(self): apply_config_drift()
    def _update_node_provisioning(self):
        cs = self.cluster_state; newly_ready_nodes_info = []; steps_to_remove = []
        for step_ready, nodes_info in list(cs.pending_node_provisioning.items()):
            if cs.current_step >= step_ready: newly_ready_nodes_info.extend(nodes_info); steps_to_remove.append(step_ready)
        for step in steps_to_remove: del cs.pending_node_provisioning[step]
        num_actually_ready = 0
        for node_id, pool_name in newly_ready_nodes_info:
            if random.random() < CA_NODE_FAIL_PROVISION_PROB: cs.failed_provision_attempts[cs.current_step] += 1
            else: cs.nodes[node_id] = Node(node_id, pool_name); cs.current_n_nodes_total += 1; num_actually_ready +=1

    def _update_node_readiness_from_events(self):
        cs = self.cluster_state
        if cs.current_state != 'Event_NodeMaintenance' and cs.previous_state == 'Event_NodeMaintenance':
             target_nodes = cs.active_event_info.get('target_nodes', [])
             for nid in target_nodes:
                  if nid in cs.nodes: cs.nodes[nid].ready = True; cs.nodes[nid].conditions.discard(COND_MAINTENANCE); cs.nodes[nid].conditions.discard(COND_NOT_READY)
        if cs.current_state != 'Event_NetworkPartition' and cs.previous_state == 'Event_NetworkPartition':
             target_nodes = cs.active_event_info.get('target_nodes', [])
             impact_data = EVENT_IMPACTS['Event_NetworkPartition']; condition_impact = impact_data.get('node_condition_add'); condition_name = condition_impact[0] if condition_impact else COND_NET_UNAVAILABLE
             for nid in target_nodes:
                 if nid in cs.nodes: cs.nodes[nid].conditions.discard(condition_name)

    def _update_control_plane(self, step_metrics: Dict):
        cs = self.cluster_state; # ... Control Plane Logic ...
        if cs.etcd_state == STATE_NORMAL: cs.etcd_latency = max(5, np.random.normal(ETCD_BASE_LATENCY, ETCD_LATENCY_STDDEV));
        if random.random() < ETCD_SPIKE_PROB: cs.etcd_state = 'Spike'; cs.etcd_state_end_step = cs.current_step + random.randint(*ETCD_SPIKE_DURATION)
        elif random.random() < ETCD_LEADER_ELECTION_PROB: cs.etcd_state = 'Election'; cs.etcd_state_end_step = cs.current_step + random.randint(*ETCD_LEADER_ELECTION_DURATION)
        elif cs.etcd_state == 'Spike': cs.etcd_latency = ETCD_SPIKE_LATENCY + np.random.normal(0, 20)
        elif cs.etcd_state == 'Election': cs.etcd_latency = ETCD_ELECTION_LATENCY + np.random.normal(0, 50)
        if cs.current_step == cs.etcd_state_end_step: cs.etcd_state = STATE_NORMAL
        step_metrics['Etcd_Request_Latency_Avg'] = cs.etcd_latency
        api_base_latency = NORMAL_RANGES['API_Server_Request_Latency_Avg'][0]; api_load_latency = API_LATENCY_LOAD_FACTOR * sum(a.current_replicas for a in cs.apps.values()) / 100; api_etcd_impact = max(0, (cs.etcd_latency - ETCD_BASE_LATENCY) * API_LATENCY_ETCD_FACTOR); api_stddev = NORMAL_RANGES['API_Server_Request_Latency_Avg'][1]
        step_metrics['API_Server_Request_Latency_Avg'] = max(20, api_base_latency + api_load_latency + api_etcd_impact + np.random.normal(0, api_stddev))


    def _run_hpa(self) -> List[str]:
        cs = self.cluster_state; pods_to_schedule_hpa = []; api_latency_ms = cs.last_step_metrics.get('API_Server_Request_Latency_Avg', 70); node_pressure = cs.last_step_metrics.get('MaxNodePressurePercent', 35); hpa_effectiveness_factor = 1.0 - HPA_EFFECTIVENESS_REDUCTION_FACTOR * max(0, min(1.0, (api_latency_ms - 150)/300)); hpa_effectiveness_factor *= (1.0 - HPA_EFFECTIVENESS_REDUCTION_FACTOR * max(0, min(1.0, (node_pressure - 75)/25)))
        for app_name, app_data in cs.apps.items(): # ... HPA Logic ...
            profile = app_data.profile; hpa_cpu_target = profile.get('hpa_cpu_target'); hpa_mem_target = profile.get('hpa_mem_target');
            if (hpa_cpu_target is None and hpa_mem_target is None) or (cs.current_step - cs.last_hpa_action_step[app_name]) <= HPA_COOLDOWN_PERIOD_STEPS: continue
            # Use last step's metrics for HPA decision
            app_cpu_usage_ratio = cs.last_step_metrics.get('AvgCPURequestCommitRatio', 0.7) * 100
            app_mem_usage_ratio = cs.last_step_metrics.get('AvgMemoryRequestCommitRatio', 0.8) * 100
            scale_decision = 0;
            if hpa_cpu_target is not None and app_cpu_usage_ratio > hpa_cpu_target: scale_decision = 1
            elif hpa_mem_target is not None and app_mem_usage_ratio > hpa_mem_target: scale_decision = 1
            elif (hpa_cpu_target is None or app_cpu_usage_ratio < HPA_CPU_THRESHOLD_SCALE_DOWN) and \
                 (hpa_mem_target is None or app_mem_usage_ratio < HPA_MEM_THRESHOLD_SCALE_DOWN): scale_decision = -1
            if scale_decision != 0 and (cs.current_step - cs.last_hpa_fail_step[app_name]) > HPA_COOLDOWN_PERIOD_STEPS:
                 if random.random() < HPA_SCALE_FAIL_PROBABILITY: cs.last_hpa_fail_step[app_name] = cs.current_step; continue
                 if random.random() < HPA_FLAP_PROBABILITY: scale_decision *= -1
                 current_replicas = app_data.current_replicas; delta = max(1, int(current_replicas * HPA_SCALE_FACTOR * hpa_effectiveness_factor));
                 # <<< CHANGE START >>>
                 # Decide the *intended* target, don't change current_replicas yet
                 new_intended_replicas = app_data.intended_replicas
                 if scale_decision == 1: new_intended_replicas = current_replicas + delta
                 elif scale_decision == -1: new_intended_replicas = max(MIN_PODS_PER_APP, current_replicas - delta)
                 # <<< CHANGE END >>>

                 total_max_pods_cluster = sum(NODE_CAPACITY[n.pool]['max_pods'] for n in cs.nodes.values()); other_app_pods = sum(a.target_replicas for name, a in cs.apps.items() if name != app_name); app_max_pods = max(MIN_PODS_PER_APP, total_max_pods_cluster - other_app_pods);
                 # <<< CHANGE START >>>
                 new_intended_replicas = min(new_intended_replicas, app_max_pods)
                 if new_intended_replicas != app_data.intended_replicas:
                     # Set the intention and the step when it should be applied
                     cs.apps[app_name].intended_replicas = new_intended_replicas
                     cs.apps[app_name].replica_change_apply_step = cs.current_step + REPLICA_UPDATE_DELAY_STEPS
                     cs.last_hpa_action_step[app_name] = cs.current_step
                     # If scaling up, immediately add pods to be scheduled (they might become pending)
                     actual_change = new_intended_replicas - current_replicas
                     if actual_change > 0:
                         pods_to_schedule_hpa.extend([app_name] * actual_change)
                 # <<< CHANGE END >>>
        return pods_to_schedule_hpa

    def _run_ca(self):
        cs = self.cluster_state; last_metrics = cs.last_step_metrics; pending_pods = last_metrics.get('Num_Pods_Pending', 0); cpu_avg = last_metrics.get('Cluster_CPU_Usage_Avg', 35); mem_avg = last_metrics.get('Cluster_Memory_Usage_Avg', 45); node_pressure = last_metrics.get('MaxNodePressurePercent', 35); api_latency_ms = last_metrics.get('API_Server_Request_Latency_Avg', 70); api_delay_factor = 1.0 + max(0, (api_latency_ms - 100) / 200)
        # Scale Up...
        needs_scale_up = pending_pods > CA_PENDING_PODS_THRESHOLD
        if needs_scale_up and (cs.current_step - cs.last_ca_scale_up_step) > CA_SCALE_UP_COOLDOWN_STEPS and cs.current_n_nodes_total < MAX_NODES_TOTAL and not cs.pending_node_provisioning:
            pool_to_add = 'standard'; # Pool selection...
            if mem_avg > 75 and len([n for n in cs.nodes.values() if n.pool == 'high-mem']) < MAX_NODES_TOTAL * 0.4: pool_to_add = 'high-mem'
            elif len([n for n in cs.nodes.values() if n.pool == 'standard']) < INITIAL_NODE_COUNTS['standard'] // 2 : pool_to_add = 'standard'
            max_pods_in_pool = NODE_CAPACITY[pool_to_add]['max_pods']; nodes_to_add = max(1, math.ceil(pending_pods / max(1, max_pods_in_pool))); nodes_to_add = min(nodes_to_add, MAX_NODES_TOTAL - cs.current_n_nodes_total)
            if nodes_to_add > 0: # Provision...
                 provision_delay = random.randint(CA_NODE_PROVISION_DELAY_STEPS_MIN, CA_NODE_PROVISION_DELAY_STEPS_MAX); provision_ready_step = cs.current_step + int(provision_delay * api_delay_factor); new_nodes_info = []
                 for _ in range(nodes_to_add): new_node_id = cs.next_node_id; cs.next_node_id += 1; new_nodes_info.append((new_node_id, pool_to_add)) # Use cs.next_node_id
                 cs.pending_node_provisioning[provision_ready_step].extend(new_nodes_info); cs.last_ca_scale_up_step = cs.current_step
        # Scale Down...
        elif cpu_avg < CA_NODE_UTILIZATION_LOW_THRESHOLD and mem_avg < CA_NODE_UTILIZATION_LOW_THRESHOLD and node_pressure < CA_NODE_PRESSURE_HIGH_THRESHOLD and cs.current_n_nodes_total > sum(INITIAL_NODE_COUNTS.values()) and (cs.current_step - cs.last_ca_scale_down_step) > CA_SCALE_DOWN_COOLDOWN_STEPS and not cs.pending_node_provisioning:
             pool_candidates = []; # Pool selection...
             if len([n for n in cs.nodes.values() if n.pool == 'compute-opt']) > INITIAL_NODE_COUNTS['compute-opt']: pool_candidates.append('compute-opt')
             if len([n for n in cs.nodes.values() if n.pool == 'high-mem']) > INITIAL_NODE_COUNTS['high-mem']: pool_candidates.append('high-mem')
             if len([n for n in cs.nodes.values() if n.pool == 'standard']) > INITIAL_NODE_COUNTS['standard']: pool_candidates.append('standard')
             pool_to_remove_from = random.choice(pool_candidates) if pool_candidates else None
             removable_nodes = [nid for nid, n in cs.nodes.items() if n.pool == pool_to_remove_from and n.ready and not n.conditions];
             if removable_nodes: # Check safety...
                  node_to_remove_id = random.choice(removable_nodes); safe_to_remove = True
                  if 'database' in cs.nodes[node_to_remove_id].pods and cs.apps['database'].current_replicas <= MIN_PODS_PER_APP: safe_to_remove = False
                  if safe_to_remove: # Remove...
                       del cs.nodes[node_to_remove_id]; cs.current_n_nodes_total -= 1; cs.last_ca_scale_down_step = cs.current_step
                       # Adjust total target pods...
                       total_max_pods_cluster = sum(NODE_CAPACITY[n.pool]['max_pods'] for n in cs.nodes.values()); min_total_pods_overall = max(len(cs.apps)*MIN_PODS_PER_APP, int(cs.current_n_nodes_total * INITIAL_PODS_PER_NODE_TARGET * MIN_PODS_FACTOR))
                       cs.target_total_pods = max(min_total_pods_overall, min(cs.target_total_pods, total_max_pods_cluster))

    def _update_simulation_state(self):
        global TOTAL_FAILURES_TRIGGERED, FAILURE_TYPE_COUNTS
        cs = self.cluster_state; i = cs.current_step
        cs.previous_state = cs.current_state; cs.previous_failure_type = cs.active_failure_type; cs.previous_failure_target_app = cs.active_failure_target_app; cs.previous_failure_target_node = cs.active_failure_target_node
        is_event_active = cs.current_state.startswith('Event_')
        # End states...
        if i == cs.state_end_step and cs.current_state not in [STATE_NORMAL] and not is_event_active: cs.current_state = STATE_NORMAL; cs.state_end_step = -1; cs.precursor_start_step = -1;
        elif i == cs.event_end_step and is_event_active: cs.current_state = STATE_NORMAL; cs.event_end_step = -1; cs.post_event_boost_end_step = i + POST_EVENT_FAIL_BOOST_DURATION; cs.active_event_info = {}
        # Post-Recovery/Intervention Checks...
        if (cs.previous_state == STATE_RECOVERING or cs.previous_state == 'Event_ManualIntervention') and cs.current_state == STATE_NORMAL and cs.previous_failure_type:
            severity = 0; current_active_deltas = copy.deepcopy(cs.active_deltas)
            if current_active_deltas: max_abs_delta = max(abs(d) for d in current_active_deltas.values() if isinstance(d, (int, float))) if current_active_deltas else 0; severity = min(2.0, max(0.5, max_abs_delta / 50.0))
            cs.active_deltas = {} # Clear after use
            cascade_options = FAILURE_IMPACTS[cs.previous_failure_type].get('cascade_trigger', {}); cascaded = False
            if cascade_options: # Check cascade...
                 for next_failure_raw, (base_prob, severity_factor) in cascade_options.items():
                      next_failure = next_failure_raw; target_app = None;
                      if ':' in next_failure_raw: next_failure, target_app_req = next_failure_raw.split(':', 1); target_app = target_app_req if target_app_req in APP_PROFILES else cs.previous_failure_target_app
                      if random.random() < base_prob * (severity ** severity_factor):
                           cs.active_failure_type = next_failure; cs.active_failure_target_app = target_app; cs.active_failure_target_node = None; precursor_duration = random.randint(MIN_PRECURSOR_DURATION, MAX_PRECURSOR_DURATION); cs.precursor_start_step = i; cs.state_end_step = i + precursor_duration; cs.current_state = STATE_PRECURSOR; cs.active_deltas = {};
                           print(f"Step {i}: CASCADE! {cs.previous_failure_type}(Sev:{severity:.1f}) -> {cs.active_failure_type}{f':{target_app}' if target_app else ''}, ends step {cs.state_end_step}")
                           cascaded = True; FAILURE_TYPE_COUNTS[cs.active_failure_type] += 1; TOTAL_FAILURES_TRIGGERED += 1; break
            if not cascaded and random.random() < INTERMITTENT_RECOVERY_PROB: # Check intermittent...
                 cs.active_failure_type = cs.previous_failure_type; cs.active_failure_target_app = cs.previous_failure_target_app; cs.active_failure_target_node = cs.previous_failure_target_node; precursor_duration = random.randint(MIN_PRECURSOR_DURATION // 2, MAX_PRECURSOR_DURATION // 2); cs.precursor_start_step = i; cs.state_end_step = i + precursor_duration; cs.current_state = STATE_PRECURSOR; cs.active_deltas = {}
                 # print(f"Step {i}: INTERMITTENT! Relapse {cs.active_failure_type}, ends {cs.state_end_step}")
                 cascaded = True; FAILURE_TYPE_COUNTS[cs.active_failure_type] += 1; TOTAL_FAILURES_TRIGGERED += 1
            if not cascaded: cs.active_failure_type = None; cs.active_failure_target_app = None; cs.active_failure_target_node = None;
        elif cs.current_state != STATE_RECOVERING and cs.current_state != STATE_PRECURSOR: # Clear targets...
            cs.active_failure_type = None; cs.active_failure_target_app = None; cs.active_failure_target_node = None;
            if cs.previous_state == STATE_PRECURSOR: cs.active_deltas = {}

        # Start new failure?
        fail_boost = POST_EVENT_FAIL_BOOST_FACTOR if i < cs.post_event_boost_end_step else 1.0
        dynamic_fail_prob_base = BASE_FAILURE_PROBABILITY * get_dynamic_failure_factor(cs.last_step_metrics, fail_boost) # Use helper
        if cs.current_state == STATE_NORMAL and random.random() < dynamic_fail_prob_base:
            selected_failure_type = None; target_app = None; target_node = None; # Select failure type with balance...
            if TOTAL_FAILURES_TRIGGERED > 50:
                avg_freq = max(1, TOTAL_FAILURES_TRIGGERED) / len(FAILURE_TYPES)
                weights = []
                total_weight = 0
                for ft in FAILURE_TYPES: 
                    count = FAILURE_TYPE_COUNTS[ft]
                    weight = 1.0
                    if count < avg_freq * CLASS_BALANCE_TARGET_RATIO: 
                        weight = 1.0 + (avg_freq * CLASS_BALANCE_TARGET_RATIO - count) / max(1, avg_freq * CLASS_BALANCE_TARGET_RATIO)
                    weights.append(weight)
                    total_weight += weight
                norm_weights = [w / max(0.01, total_weight) for w in weights]; selected_failure_type = random.choices(FAILURE_TYPES, weights=norm_weights, k=1)[0]
            else: selected_failure_type = random.choice(FAILURE_TYPES)
            failure_details = FAILURE_IMPACTS[selected_failure_type]; # Determine target app/node...
            target_app_req = failure_details.get('target_app_type');
            if target_app_req == 'random': target_app = random.choice(list(APP_PROFILES.keys()))
            elif target_app_req in APP_PROFILES: target_app = target_app_req
            if failure_details.get('node_specific'):
                available_nodes = [nid for nid, ndata in cs.nodes.items() if ndata.ready];
                if available_nodes: target_node = random.choice(available_nodes)
                else: selected_failure_type = None
            if selected_failure_type: # Start precursor...
                 cs.active_failure_type = selected_failure_type; cs.active_failure_target_app = target_app; cs.active_failure_target_node = target_node; precursor_duration = random.randint(MIN_PRECURSOR_DURATION, MAX_PRECURSOR_DURATION);
                 if cs.active_failure_type == 'Impending_Memory_Leak': precursor_duration *= 1.5
                 cs.precursor_start_step = i; cs.state_end_step = i + int(precursor_duration); cs.current_state = STATE_PRECURSOR; cs.active_deltas = {}
                 FAILURE_TYPE_COUNTS[cs.active_failure_type] += 1; TOTAL_FAILURES_TRIGGERED += 1

        # Precursor -> Recovery transition
        elif cs.current_state == STATE_PRECURSOR and i == cs.state_end_step:
             cs.precursor_duration_actual = max(1, cs.state_end_step - cs.precursor_start_step)
             recovery_duration = max(10, int(cs.precursor_duration_actual * RECOVERY_DURATION_FACTOR));
             cs.state_end_step = i + recovery_duration; cs.current_state = STATE_RECOVERING
             # Store final deltas correctly
             cs.active_deltas = copy.deepcopy(cs.active_deltas) # Store the calculated deltas from the last precursor step

        # Start new event?
        elif cs.current_state == STATE_NORMAL: # Event selection...
             rand_event = random.random(); cumulative_prob = 0; event_found = False; shuffled_event_items = list(EVENT_PROBABILITIES.items()); random.shuffle(shuffled_event_items)
             for state_name, base_prob in shuffled_event_items:
                cumulative_prob += base_prob
                if rand_event < cumulative_prob: # Check cooldowns/preconditions...
                     if state_name == 'Event_ManualIntervention' and (i - cs.last_intervention_step) < 500: continue
                     if (state_name in ['Event_NodeMaintenance', 'Event_KubeletIssue', 'Event_NetworkPartition']) and not any(nd.ready for nd in cs.nodes.values()): continue
                     duration_range = EVENT_DURATIONS[state_name]
                     if isinstance(duration_range, tuple):
                         duration = random.randint(*duration_range)
                     else:
                         # Handle the case where duration_range is an integer
                         duration = duration_range
                     
                     cs.current_state = state_name; cs.event_end_step = i + duration; cs.active_deltas = {}; cs.active_event_info = {'impact_data': EVENT_IMPACTS[state_name]}; cs.active_failure_target_node = None; cs.active_failure_target_app = None;
                     # Handle event specifics...
                     if state_name == 'Event_ConfigChange': cs.active_deltas = {m: random.uniform(*r) for m, r in EVENT_IMPACTS[state_name].items() if isinstance(r, tuple)}
                     elif EVENT_IMPACTS[state_name].get('node_specific'):
                          num_nodes_factor = EVENT_IMPACTS[state_name].get('num_nodes_factor', 1 / max(1, cs.current_n_nodes_total)); num_nodes_to_affect = max(1, int(cs.current_n_nodes_total * num_nodes_factor)); available_nodes = [nid for nid, ndata in cs.nodes.items() if ndata.ready];
                          if len(available_nodes) >= num_nodes_to_affect: cs.active_event_info['target_nodes'] = random.sample(available_nodes, num_nodes_to_affect)
                          else: cs.current_state = STATE_NORMAL; continue
                          if state_name == 'Event_NodeMaintenance':
                               for nid in cs.active_event_info['target_nodes']: cs.nodes[nid].ready = False; cs.nodes[nid].conditions.add(COND_MAINTENANCE)
                     elif state_name == 'Event_ManualIntervention': cs.last_intervention_step = i; cs.active_event_info['action'] = random.choice(MANUAL_INTERVENTION_ACTIONS)
                     event_found = True; break

        # Update progress calculation...
        cs.progress = 0.0; cs.step_in_state = 0
        if cs.current_state in [STATE_PRECURSOR, STATE_RECOVERING]:
            cs.step_in_state = i - cs.precursor_start_step
            if cs.current_state == STATE_PRECURSOR:
                 cs.precursor_duration_actual = max(1, cs.state_end_step - cs.precursor_start_step)
                 cs.progress = min(1.0, cs.step_in_state / cs.precursor_duration_actual)
            elif cs.current_state == STATE_RECOVERING:
                 if cs.precursor_duration_actual <= 1: # Estimate if missed
                      approx_rec_start = cs.state_end_step - max(10, int(1 * RECOVERY_DURATION_FACTOR))
                      cs.precursor_duration_actual = max(1, approx_rec_start - cs.precursor_start_step)
                 recovery_start_step = cs.state_end_step - max(10, int(cs.precursor_duration_actual * RECOVERY_DURATION_FACTOR))
                 recovery_duration_actual = max(1, cs.state_end_step - recovery_start_step); step_in_recovery = i - recovery_start_step
                 cs.progress = max(0.0, 1.0 - (step_in_recovery / recovery_duration_actual))


    def _get_restarting_pods(self) -> List[str]:
        cs = self.cluster_state; restarting_pods = []; explicit_restart_increase = 0
        total_base_restart_rate = sum(APP_PROFILES[app]['base_restart_prob'] * a_data.current_replicas for app, a_data in cs.apps.items()); eviction_restarts = cs.last_step_metrics.get('Evicted_Pod_Count_Rate', 0) * 0.8
        if cs.current_state in [STATE_PRECURSOR, STATE_RECOVERING] and cs.active_failure_type:
            failure_details = FAILURE_IMPACTS[cs.active_failure_type]; impact = failure_details.get('secondary', {}).get('Pod_Restart_Rate') or failure_details.get('primary', {}).get('app_restart_rate_increase');
            if impact: 
                delta, shape = impact[0], impact[1]
                increase = apply_trend(0, cs.progress, delta, shape)
                if cs.active_failure_target_app and cs.active_failure_target_app in cs.apps: 
                    explicit_restart_increase += increase * cs.apps[cs.active_failure_target_app].current_replicas
                else: 
                    explicit_restart_increase += increase
        total_expected_restarts = total_base_restart_rate + eviction_restarts + explicit_restart_increase
        num_restarts = np.random.poisson(max(0, total_expected_restarts))
        if num_restarts > 0: # Distribute...
            app_weights = {name: data.current_replicas * (APP_PROFILES[name]['base_restart_prob'] + data.restarts_rate) for name, data in cs.apps.items()}; total_weight = sum(app_weights.values())
            if total_weight > 0:
                app_probs = {name: w / total_weight for name, w in app_weights.items()}; restarting_apps = np.random.choice(list(app_probs.keys()), size=num_restarts, p=list(app_probs.values())); restarting_pods.extend(restarting_apps)
                # <<< CHANGE START >>>
                # Don't decrement current_replicas here; scheduling handles placement
                # for app_name in restarting_apps:
                #     if cs.apps[app_name].current_replicas > 0: cs.apps[app_name].current_replicas -= 1
                # <<< CHANGE END >>>
        return restarting_pods


    def _simulate_scheduling(self, pods_to_schedule: List[str]) -> Tuple[int, int]:
        cs = self.cluster_state; scheduled_count = 0; pending_count = 0
        # <<< CHANGE START >>>
        # Enhanced scoring with pool affinity
        def score_node(node_id: int, app_name: str) -> float:
            node = cs.nodes[node_id]
            profile = APP_PROFILES[app_name]
            score = 1000.0 # Base score

            # Basic readiness and condition checks (negative infinity if unschedulable)
            if not node.ready or node.conditions.intersection({COND_DISK_PRESSURE, COND_PID_PRESSURE, COND_NET_UNAVAILABLE, COND_MAINTENANCE}):
                return -float('inf')

            # Penalize based on current load (prefer less loaded nodes - bin packing tendency)
            score -= sum(node.pods.values()) * 5 # Penalize for existing pods
            score -= (node.cpu_usage / max(1, node.capacity['cpu_cores'])) * 100 # Penalize CPU usage %
            score -= (node.mem_usage / max(1, node.capacity['memory_mib'])) * 100 # Penalize Memory usage %

            # Penalize for pressure conditions
            if COND_CPU_PRESSURE in node.conditions: score -= 200
            if COND_MEM_PRESSURE in node.conditions: score -= 200

            # Reward for existing pods of the same app (basic affinity)
            if node.pods.get(app_name, 0) > 0: score += 50

            # Apply pool preferences/avoidances
            preferred_pool = profile.get('preferred_pool')
            avoid_pool = profile.get('avoid_pool')
            if preferred_pool and node.pool == preferred_pool:
                score += 100 # Bonus for preferred pool
            if avoid_pool and node.pool == avoid_pool:
                score -= 150 # Penalty for avoided pool

            return score
        # <<< CHANGE END >>>

        random.shuffle(pods_to_schedule) # Avoid bias in scheduling order

        for app_name in pods_to_schedule:
            profile = APP_PROFILES[app_name]; scheduled = False
            schedulable_nodes = [nid for nid in cs.get_ready_nodes() if cs.nodes[nid].can_schedule(profile)]

            if not schedulable_nodes:
                cs.apps[app_name].pending_pods += 1; pending_count += 1; continue

            # Score and sort nodes
            scored_nodes = sorted([(nid, score_node(nid, app_name)) for nid in schedulable_nodes], key=lambda item: item[1], reverse=True)

            if not scored_nodes or scored_nodes[0][1] == -float('inf'):
                 cs.apps[app_name].pending_pods += 1; pending_count += 1; continue # No suitable node found

            best_node_id = scored_nodes[0][0]; node_data = cs.nodes[best_node_id]

            # Assign pod to node and update node state immediately
            node_data.pods[app_name] += 1;
            node_data.update_requests_from_pods(); # Recalculate requests
            # Approximate immediate usage increase (can be refined further)
            node_data.cpu_usage += profile['cpu_request'] * 1.1;
            node_data.mem_usage += profile['mem_request'] * 1.05;
            # <<< CHANGE START >>>
            # Increment current replicas immediately upon successful scheduling attempt
            # The target/intended replica logic is handled by HPA and _apply_delayed_replica_changes
            cs.apps[app_name].current_replicas += 1
            # <<< CHANGE END >>>
            scheduled_count += 1; scheduled = True

            # If somehow scheduling failed despite finding a node (shouldn't happen with current logic)
            if not scheduled: cs.apps[app_name].pending_pods += 1; pending_count += 1

        return scheduled_count, pending_count


    def _update_all_node_metrics(self, step_metrics: Dict):
        cs = self.cluster_state
        base_cluster_metrics = {m: generate_base_value(m, cs.current_time, cs.current_n_nodes_total, cs.target_total_pods, cs.not_ready_node_ids) for m in ['Cluster_Disk_IO_Rate', 'Cluster_Network_Bytes_In', 'Cluster_Network_Bytes_Out']}
        for node_id, node_data in cs.nodes.items(): node_data.update_metrics(base_cluster_metrics, cs.current_step, cs)

    def _aggregate_cluster_metrics(self, step_metrics: Dict):
        cs = self.cluster_state
        total_cpu_usage = 0; total_mem_usage = 0; total_disk_io = 0; total_net_in = 0; total_net_out = 0; total_cpu_requests = 0; total_mem_requests = 0; total_cpu_capacity = 0; total_mem_capacity = 0; max_node_cpu_perc = 0; max_node_mem_perc = 0; num_ready = 0; node_pressures = []
        for node_id, n in cs.nodes.items():
            if n.ready:
                num_ready += 1; node_cap = NODE_CAPACITY[n.pool]; total_cpu_capacity += node_cap['cpu_cores']; total_mem_capacity += node_cap['memory_mib']; total_cpu_usage += n.cpu_usage; total_mem_usage += n.mem_usage; total_disk_io += n.disk_io; total_net_in += n.net_in; total_net_out += n.net_out; total_cpu_requests += n.cpu_request_total; total_mem_requests += n.mem_request_total
                node_cpu_perc = 100 * n.cpu_usage / max(1, node_cap['cpu_cores']); node_mem_perc = 100 * n.mem_usage / max(1, node_cap['memory_mib']); max_node_cpu_perc = max(max_node_cpu_perc, node_cpu_perc); max_node_mem_perc = max(max_node_mem_perc, node_mem_perc); node_pressures.append(max(node_cpu_perc, node_mem_perc))
        step_metrics['Cluster_CPU_Usage_Avg'] = 100 * total_cpu_usage / max(1, total_cpu_capacity) if total_cpu_capacity else 0; step_metrics['Cluster_CPU_Usage_Max'] = max_node_cpu_perc; step_metrics['Cluster_Memory_Usage_Avg'] = 100 * total_mem_usage / max(1, total_mem_capacity) if total_mem_capacity else 0; step_metrics['Cluster_Memory_Usage_Max'] = max_node_mem_perc; step_metrics['MaxNodePressurePercent'] = max(node_pressures) if node_pressures else 0; step_metrics['AvgCPURequestCommitRatio'] = total_cpu_usage / max(1, total_cpu_requests) if total_cpu_requests else 0; step_metrics['AvgMemoryRequestCommitRatio'] = total_mem_usage / max(1, total_mem_requests) if total_mem_requests else 0; step_metrics['Cluster_Disk_IO_Rate'] = total_disk_io; step_metrics['Cluster_Network_Bytes_In'] = total_net_in; step_metrics['Cluster_Network_Bytes_Out'] = total_net_out

    def _update_misc_cluster_metrics(self, step_metrics: Dict):
        cs = self.cluster_state; i = cs.current_step; api_latency_ms = step_metrics['API_Server_Request_Latency_Avg'] # ... Update PVs, Base Metrics, Throttling, Restarts ...
        # PV Simulation...
        pods_needing_pv = sum(a.current_replicas for app, a in cs.apps.items() if APP_PROFILES[app].get('pv_required')); pv_deficit = max(0, pods_needing_pv - cs.pv_count);
        if pv_deficit > 0 and random.random() < PV_CREATE_RATE * pv_deficit: cs.pv_count = min(MAX_PV_COUNT, cs.pv_count + 1)
        step_metrics['PV_Attached_Count'] = cs.pv_count; pv_attach_error_prob = PV_ATTACH_ERROR_PROB_BASE + max(0, api_latency_ms - 100) * PV_ATTACH_ERROR_PROB_API_FACTOR
        # Add failure impact on PV errors...
        pv_error_increase = 0.0
        impact = None
        if cs.current_state == STATE_PRECURSOR and cs.active_failure_type == 'Impending_API_Slowdown':
            impact = FAILURE_IMPACTS['Impending_API_Slowdown'].get('secondary',{}).get('PV_Attach_Error_Rate')
            if impact: 
                delta, shape, delay_frac = impact[0], impact[1], impact[3]
                pv_error_increase = apply_trend(0, cs.progress, delta, shape)
        elif cs.current_state == STATE_PRECURSOR and cs.active_failure_type == 'Impending_PV_Attach_Storm':
            impact = FAILURE_IMPACTS['Impending_PV_Attach_Storm'].get('primary',{}).get('PV_Attach_Error_Rate')
            if impact:
                delta, shape = impact[0], impact[1]
                pv_error_increase = apply_trend(0, cs.progress, delta, shape)
        pv_attach_error_prob += pv_error_increase
        step_metrics['PV_Attach_Error_Rate'] = np.random.poisson(max(0, pv_attach_error_prob * pods_needing_pv));
        # Update Base Metrics...
        # <<< CHANGE START >>>
        # Include new specific log rates
        base_metrics_to_generate = [
            'App_Error_Rate_Avg', 'App_Latency_Avg', 'ErrorLogRate', 'WarningLogRate',
            'Cluster_Network_Latency_Avg', 'Cluster_Network_Errors_Rate', 'Etcd_DB_Size',
            'CPU_Throttling_Rate', 'Evicted_Pod_Count_Rate', 'ImagePullBackOff_Count_Rate',
            'Storage_IOPS_Saturation_Percent', 'Storage_Throughput_Saturation_Percent',
            'Network_Bandwidth_Saturation_Percent',
            'OOMKillLogRate', 'NetworkErrorLogRate', 'EvictionLogRate', 'APIErrorLogRate'
        ]
        for metric in base_metrics_to_generate:
             if metric not in step_metrics: # Generate if not already set by specific logic (e.g., control plane)
                 step_metrics[metric] = generate_base_value(metric, cs.current_time, cs.current_n_nodes_total, cs.target_total_pods, cs.not_ready_node_ids)
        # <<< CHANGE END >>>

        # Throttling/Evictions (Base calculation, failures might add more)
        step_metrics['CPU_Throttling_Rate'] += sum(1 for n in cs.nodes.values() if n.cpu_over_limit) * np.random.normal(2, 0.5)
        step_metrics['Evicted_Pod_Count_Rate'] += sum(1 for n in cs.nodes.values() if n.mem_over_limit) * np.random.normal(0.2, 0.1)

        # Pod Restarts (Base calculation, failures might add more)
        base_restart_rate = sum(APP_PROFILES[app]['base_restart_prob'] * a_data.current_replicas for app, a_data in cs.apps.items())
        explicit_restart_increase = 0
        if cs.current_state in [STATE_PRECURSOR, STATE_RECOVERING] and cs.active_failure_type:
            failure_details = FAILURE_IMPACTS[cs.active_failure_type]
            impact = failure_details.get('secondary', {}).get('Pod_Restart_Rate') or failure_details.get('primary', {}).get('app_restart_rate_increase')
            if impact:
                delta, shape = impact[0], impact[1]
                increase = apply_trend(0, cs.progress, delta, shape)
                if cs.active_failure_target_app and cs.active_failure_target_app in cs.apps:
                    explicit_restart_increase += increase * cs.apps[cs.active_failure_target_app].current_replicas
                else:
                    explicit_restart_increase += increase
        # Ensure Pod_Restart_Rate exists before adding to it
        if 'Pod_Restart_Rate' not in step_metrics:
             step_metrics['Pod_Restart_Rate'] = generate_base_value('Pod_Restart_Rate', cs.current_time, cs.current_n_nodes_total, cs.target_total_pods, cs.not_ready_node_ids)

        step_metrics['Pod_Restart_Rate'] += base_restart_rate + step_metrics['Evicted_Pod_Count_Rate'] + explicit_restart_increase


    def _apply_cluster_event_impacts(self, step_metrics: Dict):
        cs = self.cluster_state;
        if cs.current_state.startswith('Event_'):
            event_info = cs.active_event_info; impact_data = event_info.get('impact_data', {}); target_nodes = event_info.get('target_nodes', []); num_target_nodes = len(target_nodes) if target_nodes else 1;
            cluster_impacts = impact_data if not impact_data.get('node_specific') else impact_data.get('impact', {})
            for metric, delta_or_range in cluster_impacts.items():
                 # <<< CHANGE START >>>
                 # Ensure metric exists before applying impact
                 if metric not in step_metrics:
                     # Generate a base value if missing, except for tuple-based multipliers
                     if not isinstance(delta_or_range, tuple):
                         step_metrics[metric] = generate_base_value(metric, cs.current_time, cs.current_n_nodes_total, cs.target_total_pods, cs.not_ready_node_ids)
                     else:
                         continue # Cannot apply multiplicative range if base doesn't exist

                 if isinstance(delta_or_range, (int, float)): # Additive
                     scale = num_target_nodes / max(1, cs.current_n_nodes_total) if impact_data.get('node_specific') and 'LogRate' in metric else 1.0
                     step_metrics[metric] = step_metrics.get(metric, 0) + delta_or_range * (random.random() * 0.5 + 0.75) * scale
                 elif isinstance(delta_or_range, tuple) and len(delta_or_range) == 2: # Multiplicative range (ConfigChange)
                     step_metrics[metric] *= random.uniform(delta_or_range[0], delta_or_range[1])
                 # <<< CHANGE END >>>


    def _apply_cluster_failure_impacts(self, step_metrics: Dict):
        cs = self.cluster_state;
        if cs.current_state in [STATE_PRECURSOR, STATE_RECOVERING] and cs.active_failure_type:
            failure_details = FAILURE_IMPACTS[cs.active_failure_type]; progress = cs.progress
            for impact_level in ['primary', 'secondary', 'tertiary']:
                 for metric, impact_data in failure_details.get(impact_level, {}).items():
                     if not any(keyword in metric for keyword in ['node_', 'app_']): # Apply cluster-wide metrics
                          # <<< CHANGE START >>>
                          # Ensure metric exists before applying trend
                          if metric not in step_metrics:
                               step_metrics[metric] = generate_base_value(metric, cs.current_time, cs.current_n_nodes_total, cs.target_total_pods, cs.not_ready_node_ids)
                          # <<< CHANGE END >>>

                          delta, shape = impact_data[0], impact_data[1]; exponent = impact_data[3] if len(impact_data) > 3 else 1.0; delay_frac = impact_data[2] if len(impact_data) > 2 else 0.0
                          if cs.step_in_state >= cs.precursor_duration_actual * delay_frac:
                              effective_step = cs.step_in_state - int(cs.precursor_duration_actual * delay_frac); effective_duration = max(1, cs.precursor_duration_actual * (1.0 - delay_frac)); current_progress = min(1.0, effective_step / effective_duration) if cs.current_state == STATE_PRECURSOR else progress
                              base_value = step_metrics.get(metric) # Already ensured it exists
                              trended_value = apply_trend(base_value, current_progress, delta, shape, exponent); step_metrics[metric] = trended_value
                              if cs.current_state == STATE_PRECURSOR: # Store delta...
                                   actual_delta = trended_value - base_value; is_spike = shape == 'spike';
                                   if is_spike and cs.step_in_state == int(cs.precursor_duration_actual * delay_frac) : cs.active_deltas[metric] = delta
                                   else: cs.active_deltas[metric] = actual_delta

    def _apply_noise_and_finalize(self, step_metrics: Dict) -> Dict[str, Any]:
        cs = self.cluster_state; final_row = {'Timestamp': cs.current_time, 'Target_Failure_Type': cs.active_failure_type if cs.current_state == STATE_PRECURSOR else STATE_NORMAL}
        for metric in self.metrics_to_record:
            current_val = step_metrics.get(metric);
            # <<< CHANGE START >>>
            # Generate base value if metric is missing for any reason
            if current_val is None:
                current_val = generate_base_value(metric, cs.current_time, cs.current_n_nodes_total, cs.target_total_pods, cs.not_ready_node_ids)
            # <<< CHANGE END >>>

            stddev = NORMAL_RANGES[metric][1]; noisy_val = add_noise_and_outliers(metric, current_val, stddev, cs.current_step)
            # Clipping/Rounding...
            if '%' in metric or 'Usage' in metric or 'Pressure' in metric or 'Saturation' in metric: final_row[metric] = max(0.0, min(100.0, noisy_val))
            elif 'Ratio' in metric: final_row[metric] = max(0.01, min(5.0, noisy_val))
            elif 'Count' in metric or metric in ['Num_Pods_Running', 'Num_Pods_Pending', 'PV_Attached_Count', 'Num_Ready_Nodes', 'Num_NotReady_Nodes']: # <<< CHANGE: Added node counts here >>>
                final_row[metric] = round(max(0, noisy_val))
            elif 'LogRate' in metric: # Ensure log rates are non-negative counts (after noise)
                 final_row[metric] = round(max(0, noisy_val))
            else: final_row[metric] = max(0.0, noisy_val)

        # Final Sanity Checks & Overrides...
        final_row['Num_Ready_Nodes'] = cs.current_n_nodes_total - len(cs.not_ready_node_ids);
        final_row['Num_NotReady_Nodes'] = len(cs.not_ready_node_ids);
        final_row['Num_Pods_Running'] = sum(a.current_replicas for a in cs.apps.values());
        final_row['Num_Pods_Pending'] = sum(a.pending_pods for a in cs.apps.values());
        return final_row

    # <<< CHANGE START >>>
    # Renamed from _update_app_replicas to reflect delayed application
    def _apply_delayed_replica_changes(self):
        """Applies intended replica changes after the configured delay."""
        cs = self.cluster_state
        for app_name, app_data in cs.apps.items():
            # Check if a change is pending and the delay has passed
            if app_data.replica_change_apply_step != -1 and cs.current_step >= app_data.replica_change_apply_step:
                intended = app_data.intended_replicas
                current = app_data.current_replicas
                change = intended - current

                if change < 0: # Handle scale down - remove pods from nodes
                    pods_to_remove = abs(change)
                    nodes_with_app = [(nid, node.pods[app_name]) for nid, node in cs.nodes.items() if app_name in node.pods and node.pods[app_name] > 0]
                    # Prioritize removing from nodes with fewer pods of this app? Or random? Random for now.
                    random.shuffle(nodes_with_app)
                    removed_count = 0
                    for nid, count in nodes_with_app:
                        removable_on_node = min(pods_to_remove - removed_count, count)
                        cs.nodes[nid].pods[app_name] -= removable_on_node
                        if cs.nodes[nid].pods[app_name] == 0:
                            del cs.nodes[nid].pods[app_name] # Clean up defaultdict entry
                        cs.nodes[nid].update_requests_from_pods() # Update node requests
                        removed_count += removable_on_node
                        if removed_count >= pods_to_remove:
                            break
                    # Update current replicas based on how many were actually removed
                    app_data.current_replicas -= removed_count

                # Update current replicas to the intended value (scale-up already handled by scheduling)
                # For scale down, this reflects the state after removal attempts.
                # For scale up, this confirms the target after scheduling attempts.
                # This might not perfectly match if scheduling failed, but reflects the controller's view.
                app_data.current_replicas = intended # Or should it be current + scheduled_count for scale up? Let's sync to intended.

                # Reset the change tracking
                app_data.replica_change_apply_step = -1
    # <<< CHANGE END >>>


# --- Main Execution ---
if __name__ == "__main__":
    simulator = Simulator(N_DATAPOINTS, START_TIME, TIME_STEP)
    results_data = []
    start_time_generation = time.time()
    print(f"Starting data generation V7.3 for {N_DATAPOINTS} steps...") # Version updated
    for i in range(N_DATAPOINTS):
        final_row_data = simulator.run_step(i)
        results_data.append(final_row_data)
        if (i + 1) % 500 == 0: elapsed = time.time() - start_time_generation; print(f"  Generated {i+1}/{N_DATAPOINTS} steps in {elapsed:.2f} seconds...")
    df = pd.DataFrame(results_data); end_time_generation = time.time()
    print(f"\nData generation V7.3 finished in {end_time_generation - start_time_generation:.2f} seconds.") # Version updated

    # <<< CHANGE START >>>
    # Ensure all expected columns, including new log rates, are present
    all_expected_cols = ['Timestamp'] + sorted(NORMAL_RANGES.keys()) + ['Target_Failure_Type']
    # <<< CHANGE END >>>
    for col in all_expected_cols:
        if col not in df.columns: print(f"Warning: Column '{col}' missing."); df[col] = np.nan
    df = df[all_expected_cols]; output_filename = 'synthetic_kubernetes_metrics_realistic_v7_3.csv' # Version updated
    df.to_csv(output_filename, index=False, float_format='%.3f')
    print(f"\nSynthetic dataset generated with {len(df)} rows.")
    print(f"Target Failure Type distribution counts:\n{dict(FAILURE_TYPE_COUNTS)}") # Nicer print format
    print(f"\nFinal Node Count: {simulator.cluster_state.current_n_nodes_total}")
    print(f"\nFinal App Pods (Current Replicas): { {name: d.current_replicas for name, d in simulator.cluster_state.apps.items()} }")
    print(f"\nSample rows:\n{df.head()}")
    print(f"\nDataset saved to {output_filename}")

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt
        print("\nGenerating sample plots V7.3...") # Version updated
        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
        # <<< CHANGE START >>>
        # Added a 7th subplot for specific log rates
        fig, axes = plt.subplots(7, 1, figsize=(18, 23), sharex=True) # Taller figure, shared X
        # <<< CHANGE END >>>

        # Plot 1: Resources & Scaling
        ax = axes[0]
        ax.plot(df['Timestamp'], df['Cluster_CPU_Usage_Avg'], label='CPU Avg (%)', alpha=0.8, linewidth=1)
        ax.plot(df['Timestamp'], df['Cluster_Memory_Usage_Avg'], label='Mem Avg (%)', alpha=0.8, linewidth=1)
        ax.plot(df['Timestamp'], df['MaxNodePressurePercent'], label='Max Node Pressure (%)', color='orange', linestyle='-.', alpha=0.6, linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(df['Timestamp'], df['Num_Ready_Nodes'], label='Ready Nodes', color='green', linestyle='--', alpha=0.7)
        ax2.plot(df['Timestamp'], df['Num_Pods_Running'] / 10, label='Pods Running / 10', color='purple', linestyle=':', alpha=0.7)
        ax.set_title('Resource Usage, Node Pressure & Cluster Scale')
        ax.legend(loc='upper left'); ax2.legend(loc='upper right'); ax.grid(True)

        # Plot 2: Latencies & Errors
        ax = axes[1]
        ax.plot(df['Timestamp'], df['App_Latency_Avg'], label='App Latency (ms)', alpha=0.8, linewidth=1)
        ax.plot(df['Timestamp'], df['API_Server_Request_Latency_Avg'], label='API Latency (ms)', alpha=0.8, linewidth=1)
        ax2 = ax.twinx(); ax2.plot(df['Timestamp'], df['App_Error_Rate_Avg'], label='App Errors (/min)', color='red', alpha=0.6, linewidth=1)
        ax.set_title('Latencies & Errors'); ax.legend(loc='upper left'); ax2.legend(loc='upper right'); ax.grid(True)

        # Plot 3: Pod Status & General Logs
        ax = axes[2]
        ax.plot(df['Timestamp'], df['Pod_Restart_Rate'], label='Pod Restarts (/min)', alpha=0.7, linewidth=1)
        ax.plot(df['Timestamp'], df['Num_Pods_Pending'], label='Pods Pending', alpha=0.7, linewidth=1)
        ax.plot(df['Timestamp'], df['Evicted_Pod_Count_Rate'], label='Eviction Rate', alpha=0.5, linestyle=':', linewidth=1)
        ax2 = ax.twinx(); ax2.plot(df['Timestamp'], df['ErrorLogRate'], label='Error Log Rate', color='darkred', alpha=0.6, linewidth=1); ax2.plot(df['Timestamp'], df['WarningLogRate'], label='Warning Log Rate', color='darkorange', alpha=0.6, linewidth=1)
        ax.set_title('Pod Status & General Log Rates'); ax.legend(loc='upper left'); ax2.legend(loc='upper right'); ax.grid(True)

        # Plot 4: Resource Pressure Ratios & Throttling
        ax = axes[3]
        ax.plot(df['Timestamp'], df['AvgCPURequestCommitRatio'], label='Avg CPU Req Ratio', alpha=0.8, linewidth=1)
        ax.plot(df['Timestamp'], df['AvgMemoryRequestCommitRatio'], label='Avg Mem Req Ratio', alpha=0.8, linewidth=1)
        ax2 = ax.twinx(); ax2.plot(df['Timestamp'], df['CPU_Throttling_Rate'], label='CPU Throttling Rate', color='brown', alpha=0.6, linewidth=1)
        ax.set_title('Resource Pressure Ratios & Throttling'); ax.legend(loc='upper left'); ax2.legend(loc='upper right'); ax.grid(True); ax.set_ylim(bottom=0)

        # Plot 5: IO/Network Saturation & Errors
        ax = axes[4]
        ax.plot(df['Timestamp'], df['Storage_IOPS_Saturation_Percent'], label='Storage IOPS Sat (%)', color='navy', alpha=0.7, linewidth=1)
        ax.plot(df['Timestamp'], df['Network_Bandwidth_Saturation_Percent'], label='Net BW Sat (%)', color='teal', alpha=0.7, linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(df['Timestamp'], df['Cluster_Network_Errors_Rate'], label='Network Error Rate', color='magenta', alpha=0.6, linewidth=1)
        ax2.plot(df['Timestamp'], df['Cluster_Disk_IO_Rate'] / 10, label='Disk IO Rate / 10', color='gray', alpha=0.5, linestyle=':', linewidth=1)
        ax.set_title('Storage/Network Saturation & Network Errors'); ax.legend(loc='upper left'); ax2.legend(loc='upper right'); ax.grid(True); ax.set_ylim(0, 105); ax2.set_ylim(bottom=0)

        # <<< CHANGE START >>>
        # Plot 6: Specific Log Rates
        ax = axes[5]
        ax.plot(df['Timestamp'], df['OOMKillLogRate'], label='OOM Kill Log Rate', color='crimson', alpha=0.7, linewidth=1)
        ax.plot(df['Timestamp'], df['NetworkErrorLogRate'], label='Network Error Log Rate', color='dodgerblue', alpha=0.7, linewidth=1)
        ax.plot(df['Timestamp'], df['EvictionLogRate'], label='Eviction Log Rate', color='darkviolet', alpha=0.7, linewidth=1)
        ax.plot(df['Timestamp'], df['APIErrorLogRate'], label='API Error Log Rate', color='saddlebrown', alpha=0.7, linewidth=1)
        ax.set_title('Specific Log Event Rates'); ax.legend(loc='upper left'); ax.grid(True); ax.set_ylim(bottom=0)

        # Plot 7: Failure Periods Overlay (was Plot 6)
        ax = axes[6]
        # <<< CHANGE END >>>
        ax.plot(df['Timestamp'], df['Etcd_Request_Latency_Avg'], label='Etcd Latency (ms)', color='lightblue', alpha=0.8, linewidth=1)
        ax.set_ylabel("Etcd Latency (ms)")
        ax.legend(loc='upper left')
        ax.set_title('Failure Precursor Periods (Shaded Red)')
        ax.grid(True)
        ax.set_xlabel("Timestamp")

        # Mark failure precursor periods across all subplots
        failures = df[df['Target_Failure_Type'] != STATE_NORMAL]
        if not failures.empty:
            # Use .loc to avoid SettingWithCopyWarning
            failures = failures.copy()
            failures['block'] = (failures['Target_Failure_Type'].ne(failures['Target_Failure_Type'].shift()) | failures['Timestamp'].diff().dt.total_seconds().gt(TIME_STEP.total_seconds() * 1.5)).cumsum()
            unique_failure_blocks = failures.groupby('block')
            for _, group in unique_failure_blocks:
                if not group.empty:
                    start = group['Timestamp'].iloc[0]; end = group['Timestamp'].iloc[-1] + TIME_STEP
                    for plot_ax in axes: plot_ax.axvspan(start, end, color='red', alpha=0.08, label='_nolegend_')
            # Add text annotation for the last failure type shown
            last_failure_group = unique_failure_blocks.get_group(failures['block'].iloc[-1])
            # <<< CHANGE START >>>
            # Annotate on the last subplot (axes[6])
            axes[6].text(last_failure_group['Timestamp'].iloc[0], axes[6].get_ylim()[1]*0.9, f" Failure: {last_failure_group['Target_Failure_Type'].iloc[0]}", color='red', alpha=0.7, ha='left', va='top')
            # <<< CHANGE END >>>

        fig.autofmt_xdate()
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout slightly for taller figure
        plot_filename = "synthetic_data_sample_plots_v7_3.png" # Version updated
        plt.savefig(plot_filename, dpi=150)
        print(f"\nSample plots V7.3 saved to {plot_filename}") # Version updated

    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
    except Exception as e:
        print(f"\nError during plotting: {e}. Skipping plots.")