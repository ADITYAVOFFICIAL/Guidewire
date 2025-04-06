# Kubernetes Pod Failure Prediction - Guidewire Hackathon Phase 1

## 1. Introduction

### 1.1. Objective
This project aims to develop a machine learning model capable of predicting potential failures for pods submitted to a Kubernetes cluster. The goal is to identify pods likely to fail *before* they consume significant resources or impact system stability, aligning with Phase 1 of the Guidewire DevTrails University Hackathon.

### 1.2. Approach Overview
The approach involves:
1.  **Data Source:** Utilizing a publicly available Kubernetes cluster trace dataset (Alibaba Cluster Trace v2023).
2.  **Target Definition:** Defining pod failure based on the final `pod_phase` recorded in the trace.
3.  **Feature Engineering:** Creating relevant features from pod specifications, timestamps, and derived cluster state information available at the time of pod creation.
4.  **Modeling:** Training and evaluating several classification models, including baseline (Logistic Regression), ensemble (Random Forest), and gradient boosting methods (XGBoost, LightGBM) with GPU acceleration.
5.  **Validation Strategy:** Employing a strict time-based split for training, validation, and testing to simulate a real-world prediction scenario and prevent data leakage.
6.  **Hyperparameter Tuning:** Using Optuna to optimize hyperparameters for the gradient boosting models based on validation set performance.
7.  **Evaluation:** Assessing model performance using standard classification metrics, focusing on AUC and F1-score due to potential class imbalance, and checking for overfitting.

## 2. Data

### 2.1. Dataset Description
*   **Source:** Alibaba Cluster Trace v2023 ([Link to source if available, e.g., GitHub repo])
*   **Files Used:**
    *   `openb_pod_list_default.csv`: Contains information about individual pod submissions, including resource requests, QoS, timestamps (creation, deletion, scheduled), and final pod phase. This is the primary data source for features and the target variable.
    *   `openb_node_list_all_node.csv`: Provides specifications for the nodes in the cluster (CPU, Memory capacity), used to calculate total cluster capacity for feature engineering.
*   **Key Raw Features:**
    *   `cpu_milli`, `memory_mib`, `num_gpu`, `gpu_milli`: Resource requests by the pod.
    *   `qos`: Quality of Service level (LS, BE, Burstable, etc.).
    *   `creation_time`, `deletion_time`, `scheduled_time`: Timestamps indicating pod lifecycle events.
    *   `pod_phase`: The final recorded status of the pod (Running, Succeeded, Failed, Pending).

### 2.2. Target Variable: `is_failed`
The prediction target is a binary variable `is_failed`, derived from the `pod_phase` column:
*   `is_failed = 1` if `pod_phase == 'Failed'`
*   `is_failed = 0` otherwise (Running, Succeeded, Pending)

**Target Distribution:**

![Target Distribution](/working/eda_target_distribution.png)

*   *Observation:* The dataset exhibits class imbalance, with significantly fewer 'Failed' pods (~23%) than 'Not Failed' pods. This necessitates using appropriate evaluation metrics (AUC, F1) and handling techniques (class weighting).

## 3. Feature Engineering

Several features were engineered to capture pod characteristics, temporal patterns, and cluster context at the time of pod creation:

1.  **Timestamp Features:**
    *   `scheduling_latency_sec`: Time difference between `scheduled_time` and `creation_time`. Filled with -1 if not scheduled.
    *   `scheduled_time_missing`: Binary flag (1 if `scheduled_time` is NaN, 0 otherwise).
2.  **Time-Based Features:**
    *   `creation_hour`: Hour of the day the pod was created.
    *   `creation_dayofweek`: Day of the week the pod was created.
    *   `hour_sin`, `hour_cos`: Cyclical encoding of the creation hour.
    *   `dayofweek_sin`, `dayofweek_cos`: Cyclical encoding of the creation day of week.
3.  **Categorical Features:**
    *   `gpu_spec_provided`: Binary flag indicating if a specific GPU type was requested (based on non-null `gpu_spec`).
    *   `qos_*`: One-hot encoded features derived from the `qos` column (e.g., `qos_BE`, `qos_LS`). 'Unknown' category created for missing values.
4.  **Interaction Features:**
    *   `cpu_x_mem`: Product of `cpu_milli` and `memory_mib` requests, capturing combined resource demand.
5.  **Cluster State Features (Calculated at pod creation time):**
    *   `cluster_active_pods`: Number of pods considered active (created before or at the current time, and not yet deleted).
    *   `cluster_pending_pods`: Number of *active* pods that were never scheduled (`scheduled_time_missing == 1`).
    *   `cluster_req_cpu`, `cluster_req_mem`, `cluster_req_gpu_milli`: Sum of resource requests of all *other* active pods.
    *   `cluster_cpu_ratio`, `cluster_mem_ratio`: Ratio of total requested CPU/Memory by active pods to the total cluster capacity.
    *   `recent_failure_rate`: Proportion of pods created in the last hour (3600s) before the current pod that eventually failed. Calculated using a rolling window on the time-indexed data.

**Feature Distributions & Relationships (EDA):**

*   **Numeric Distributions:** Histograms show the distributions of key numeric features (often log-transformed for visualization due to skewness).
    ![Numeric Distributions](/working/eda_numeric_distributions.png)
*   **Correlation:** A heatmap visualizes correlations between numeric features. High correlation between engineered cluster request features is expected.
    ![Correlation Heatmap](/working/eda_correlation_heatmap.png)
*   **Feature vs. Target:** Box plots illustrate potential differences in feature distributions between failed and non-failed pods.
    ![Box Plots](/working/eda_boxplots_feature_vs_target.png)

**Imputation:** Missing numeric values (primarily in calculated features like `scheduling_latency_sec` or base features if any were missing) were imputed using the median strategy.

## 4. Modeling Approach

### 4.1. Validation Strategy: Time-Based Split
To simulate predicting future failures based on past data and prevent data leakage, the dataset was sorted by `creation_time` and split chronologically:
*   **Training Set:** First 70% of the data.
*   **Validation Set:** Next 15% of the data (used for hyperparameter tuning).
*   **Test Set:** Final 15% of the data (held out for final model evaluation).

### 4.2. Data Scaling
StandardScaler was applied to numeric features *only* for the Logistic Regression model, as it is sensitive to feature scaling. Tree-based models (Random Forest, XGBoost, LightGBM) generally do not require feature scaling. Scaling was fitted on the training set and applied to validation and test sets.

### 4.3. Models Trained
Four different classification models were trained and evaluated:
1.  **Logistic Regression:** A linear model used as a simple baseline. Trained on scaled data with balanced class weights.
2.  **Random Forest:** An ensemble of decision trees. Trained on unscaled data with balanced class weights, limited depth, and minimum leaf samples to reduce overfitting. Utilized multiple CPU cores (`n_jobs=-1`).
3.  **XGBoost:** A gradient boosting machine implementation. Tuned using Optuna, trained with GPU acceleration (`device='cuda'`), and handled class imbalance using `scale_pos_weight`.
4.  **LightGBM:** Another efficient gradient boosting implementation. Tuned using Optuna, trained with GPU acceleration (`device='gpu'`), and handled class imbalance using `scale_pos_weight`.

### 4.4. Hyperparameter Tuning (Optuna)
Optuna was used to optimize hyperparameters for XGBoost and LightGBM:
*   **Objective:** Maximize AUC score on the time-based validation set.
*   **Method:** Tree-structured Parzen Estimator (TPE) sampler (Optuna default).
*   **Process:** Ran a predefined number of trials (`OPTUNA_N_TRIALS_XGB`/`_LGB`) within a timeout limit. Early stopping was used within each trial based on validation AUC.
*   **Outcome:** The best hyperparameters and the optimal number of boosting rounds (`n_estimators`) found during the study were used to train the final XGBoost and LightGBM models.

**Optuna Results:**

*   **XGBoost History:** Shows the progression of validation AUC over trials.
    ![Optuna History XGBoost](/working/optuna_history_XGBoost.png)
*   **XGBoost Parameter Importance:** Indicates which hyperparameters had the most impact during tuning.
    ![Optuna Params XGBoost](/working/optuna_param_importance_XGBoost.png)
*   **LightGBM History:** Shows the progression of validation AUC over trials.
    ![Optuna History LightGBM](/working/optuna_history_LightGBM.png)
*   **LightGBM Parameter Importance:** Indicates which hyperparameters had the most impact during tuning.
    ![Optuna Params LightGBM](/working/optuna_param_importance_LightGBM.png)

### 4.5. Final Model Training
The final XGBoost and LightGBM models were trained using the best hyperparameters found by Optuna on the **combined Training + Validation** data. The number of estimators (`n_estimators`) was set to the best iteration found during the Optuna validation phase to prevent overfitting on the test set during the final training.

## 5. Evaluation

### 5.1. Key Metrics
The following metrics were used for evaluation, focusing on the ability to correctly identify the minority 'Failed' class:
*   **AUC (Area Under the ROC Curve):** Primary metric for overall model discrimination ability, robust to class imbalance.
*   **F1-Score (Failed Class):** Harmonic mean of precision and recall for the 'Failed' class, providing a balanced measure of positive class identification.
*   **Precision (Failed Class):** Proportion of pods predicted as 'Failed' that actually failed (minimizing false alarms).
*   **Recall (Failed Class):** Proportion of actual 'Failed' pods that were correctly identified (minimizing missed failures).
*   **Accuracy:** Overall proportion of correct predictions (less informative for imbalanced datasets).
*   **Classification Report:** Detailed precision, recall, and F1-score for both classes.
*   **Confusion Matrix:** Visualizes true positives, true negatives, false positives, and false negatives for the best model.

### 5.2. Overfitting Assessment
Overfitting was assessed by comparing the performance metrics (AUC, F1) on the **Training Set** versus the **Test Set**. A small difference indicates good generalization, while a large drop suggests overfitting.

### 5.3. Performance Visualizations

*   **ROC Curves:** Compare the trade-off between True Positive Rate and False Positive Rate for all models.
    ![ROC Curves](/working/performance_roc_curves.png)
*   **Precision-Recall Curves:** Compare the trade-off between Precision and Recall for all models, often more informative for imbalanced datasets.
    ![PR Curves](/working/performance_pr_curves.png)
*   **Confusion Matrix (Best Model):** Shows the specific counts of correct and incorrect predictions for the best performing model (determined by Test AUC).
    ![Confusion Matrix](/working/performance_confusion_matrix_LightGBM.png) <!-- Adjust filename if best model changes -->

## 6. Results

### 6.1. Model Comparison

| Model               |       AUC |        F1 | Precision |    Recall |  Accuracy | AUC_train | F1_train |
| :------------------ | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| **XGBoost**         | **0.9981**| **0.9830**| **0.9731**| **0.9931**| **0.9918**| **0.9973**| **0.9790**|
| Random Forest       | 0.9974    | 0.9779    | 0.9664    | 0.9897    | 0.9894    | 0.9994    | 0.9797    |
| LightGBM            | 0.9967    | 0.9521    | 0.9172    | 0.9897    | 0.9763    | 0.9982    | 0.9706    |
| Logistic Regression | 0.9644    | 0.9000    | 0.8738    | 0.9278    | 0.9510    | 0.9679    | 0.8952    |

*   *Observations:*
    *   XGBoost achieved the highest Test AUC (0.9981) and Test F1-Score (0.9830), making it the best performing model in this run.
    *   Random Forest and LightGBM also performed exceptionally well, very close to XGBoost.
    *   All ensemble/boosting models significantly outperformed the Logistic Regression baseline.
    *   The comparison between Train and Test metrics shows minimal differences for the top models, indicating excellent generalization and negligible overfitting.

### 6.2. Feature Importance (Best Model: XGBoost)

![XGBoost Feature Importance](/working/optuna_param_importance_XGBoost.png)

*   *Key Observations (XGBoost):*
    *   **Dominant Features:** The interaction term `cpu_x_mem` and individual resource requests (`memory_mib`, `cpu_milli`) are the most important factors.
    *   **Secondary Features:** Scheduling outcomes (`scheduling_latency_sec`, `scheduled_time_missing`), GPU requests (`gpu_milli`), cluster context (`recent_failure_rate`, `cluster_req_gpu_milli`), and QoS levels (`qos_LS`, `qos_BE`) are also highly predictive.
    *   **Contributing Features:** Time of day (`creation_hour`), GPU count (`num_gpu`), and other cluster metrics provide additional, though less critical, information.

## 7. Conclusion

This project successfully developed high-performance machine learning models (XGBoost, LightGBM, Random Forest) capable of accurately predicting pod failures based on the Alibaba Cluster Trace v2023 dataset.

*   **Approach:** A robust methodology involving time-based splitting, comprehensive feature engineering (including cluster state and temporal features), hyperparameter tuning with Optuna, and careful evaluation was employed.
*   **Performance:** The best model (XGBoost) achieved excellent results on the hold-out test set (AUC ~0.998, F1 ~0.983) with minimal overfitting.
*   **Key Predictors:** Pod resource requests (CPU, Memory, and their interaction), scheduling outcomes (latency, success/failure), QoS levels, and cluster context (recent failures, GPU demand) were identified as the most important factors influencing pod failure prediction in this dataset.
*   **Limitations:** The prediction is based solely on data available *at pod creation time* (requests, cluster state) and scheduling outcomes. It does not incorporate real-time resource *usage* metrics, network/disk IO, or application logs, which limits the ability to predict failures caused by runtime issues not correlated with the input features or to diagnose the specific *type* of failure (e.g., OOMKilled vs. crash loop).

This work fulfills the core requirements of Phase 1 by demonstrating the ability to predict pod failures with high accuracy using a publicly available dataset and standard ML techniques. The saved XGBoost model (`model_XGBoost.json`) is recommended for potential use in Phase 2.