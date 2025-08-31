# Noir Framework: Detailed Workflow Guide

This document provides a comprehensive, phase-by-phase breakdown of the Noir Framework for blockchain fraud detection. It's designed to give developers, researchers, and technical stakeholders a clear understanding of how the system operates from data ingestion and processing to real-time inference.

---

## Framework Overview

The Noir Framework is a modular machine learning pipeline engineered for the detection of fraudulent activities on the Ethereum blockchain. It integrates various data processing, feature engineering, and machine learning methodologies to identify anomalous and illicit transaction patterns in a real-time environment.

---

### Phase 1: Data Acquisition
*(Scripts in `scripts/fetch/`)

This phase focuses on the programmatic retrieval of raw transaction data and associated wallet lists from various blockchain data sources.

*   **`data_fetch_fraud.py`**: Retrieves historical transaction data for known fraudulent addresses (e.g., from sanction lists, identified scam wallets).
*   **`data_fetch_mixers.py`**: Collects transaction data associated with identified cryptocurrency mixer services, often utilized for transaction obfuscation.
*   **`data_fetch_mixer_recipients.py`**: Fetches transaction histories for addresses identified as recipients of funds from mixer services, indicating potential onward movement of obfuscated assets.
*   **`data_fetch_normal.py`**: Acquires transaction data for a sample of benign (normal) wallet addresses to establish a baseline for legitimate blockchain activity.

**Outcome:** Generation of raw CSV data files containing uncurated transaction records for various categorized wallet sets.

---

### Phase 2: Data Preprocessing
*(Scripts in `scripts/process/`)

This phase involves the consolidation, cleaning, and initial structuring of the acquired raw transaction data to prepare it for feature engineering.

*   **`merge_transaction_sets.py`**: Consolidates disparate raw transaction datasets into a unified master record. This script performs deduplication based on transaction hash and computes global aggregates such as wallet risk labels and a network-wide 95th percentile gas fee.
    *   **Outputs**: `wallet_risk_labels.joblib` (serialized wallet risk classifications) and `global_gas_95th_percentile.joblib` (serialized gas fee threshold).
*   **`merge_mixer_recipient_txns.py`**: Integrates transaction data specifically related to mixer recipients into the aggregated transaction dataset.
*   **`save_wallet_reference.py`**: Generates a standardized reference dataset of unique wallets with their associated labels (fraudulent, suspicious, normal).
*   **`validate_merged_data.py`**: Conducts data quality assurance, including checks for missing values, label distribution, and temporal ranges within the consolidated transaction dataset.

**Outcome:** A clean, merged CSV file (`all_transactions_labeled.csv`) containing processed transaction records, alongside serialized global metadata for downstream processing.

---

### Phase 3: Feature Engineering
*(Scripts in `scripts/features/`)

This is a critical phase where raw transaction data is transformed into a rich set of predictive features, structured in hierarchical layers (L0 to L3), to enhance the discriminative power of machine learning models.

*   **`build_features_l0_aggregate.py` (L0: Aggregate Features)**: Extracts basic, wallet-level aggregate statistics from historical transaction data.
    *   **Key Features**: `total_transactions`, `wallet_age_days`, `avg_tx_value`, `std_tx_interval`.
*   **`build_features_l1_behavior.py` (L1: Behavioral & Temporal Features)**: Derives dynamic behavioral and temporal patterns from a wallet's transaction history.
    *   **Key Features**: `burst_tx_ratio`, `dormant_awaken_count`, `failure_ratio`, `mean_tx_interval_hours`, `std_tx_interval_hours`, `weekend_tx_ratio`, `txn_span_hours`, `night_tx_ratio`.
*   **`build_features_l2_riskflags.py` (L2: Risk Indicators & Heuristics)**: Computes rule-based risk flags and heuristics, often reflecting AML best practices and known fraud patterns.
    *   **Key Features**: `num_fraud_counterparties`, `circular_flow_flag`, `gas_anomaly_flag`, `rapid_bridging_flag`, `smart_contract_misuse_flag`, `mixer_flag`, `mixer_then_bridge_flag`, `mixer_exit_tx_count`, `same_recipient_ratio`.
*   **`build_features_l3_metaai.py` (L3: Meta-AI Features)**: Generates advanced anomaly detection features using unsupervised learning models.
    *   **Key Features**: `anomaly_iso` (Isolation Forest score), `anomaly_dbscan` (DBSCAN cluster assignment/outlier status), `combined_risk_tag` (synthesized behavioral risk tag).
*   **`build_features_l3_xai_tags.py` (L3: Explainable AI (XAI) Tags)**: Assigns human-interpretable reason codes for flagged transactions, synthesizing insights from all prior feature layers.
    *   **Possible XAI Reason Codes**: `burst_tx`, `dormant_awakened`, `fraud_link`, `model_anomaly`, `high_failure_rate`, `smart_contract_misuse`, `rapid_layer_hopping`, `circular_fund_flow`, `unusual_gas_fee`, or `clean` (if no suspicious activity is detected). Multiple tags can be combined using a pipe delimiter (e.g., `burst_tx|model_anomaly`).
*   **`merge_final_features.py`**: Consolidates all generated features from L0, L1, L2, and L3 into a single, comprehensive feature set for each wallet.

**Outcome:** A comprehensive feature dataset (`features_final_all_layers.csv`) for model training and saved serialized unsupervised models/scalers.

---

### Phase 4: Model Training and Evaluation
*(Scripts in `scripts/models/` and `scripts/gnn/`)

This phase encompasses the training, validation, and performance evaluation of various machine learning models for fraud detection.

*   **`prepare_features_for_training.py`**: Performs final preprocessing steps on the aggregated feature set, including feature selection (e.g., dropping highly correlated features) and scaling, to optimize for model training.
*   **`train_supervised_models.py`**: Trains supervised classification models, typically Random Forest and XGBoost, using the labeled feature dataset.
    *   **Key Outputs**: Serialized models (`random_forest_model.joblib`, `xgboost_model.joblib`), `model_feature_columns.joblib` (list of features used by models), classification reports, confusion matrices, and SHAP (SHapley Additive exPlanations) plots for model interpretability.
*   **`build_graph_dataset.py`**: Constructs a graph representation of wallet interactions, forming the basis for Graph Neural Network analysis.
*   **`prep_gnn_input.py`**: Prepares the graph dataset and node features into a format compatible with PyTorch Geometric for GNN training.
*   **`train_gnn_model.py`**: Trains a Graph Convolutional Network (GCN) for node-level fraud prediction, leveraging the relational structure of the blockchain network. Generates t-SNE embeddings for visualization of wallet clusters.
*   **`analyse_predictions.py`**: Compares and evaluates the prediction performance of the various trained models (supervised and GNN) through classification reports and agreement rates.

**Outcome:** Trained and serialized machine learning models, comprehensive performance metrics, and visual artifacts for model interpretability and data exploration.

---

### Phase 5: Analysis and Reporting
*(Scripts in `scripts/stats/`, `scripts/phase2/`)

This phase involves generating analytical summaries and reports based on the processed data, extracted features, and model outputs, providing insights into the framework's performance and data characteristics.

*   **`transaction_stats_summary.py`**: Provides high-level statistical summaries of the raw transaction datasets.
*   **`analyze_feature_stats.py`**: Performs statistical analysis on the engineered features, identifying missing values, low-variance features, and highly correlated feature pairs.
*   **`analyse_graphdata.py`**: Provides insights into the structural properties of the constructed wallet interaction graph.
*   **`analyze_feature_flags.py`**: Analyzes the prevalence and distribution of L2 risk flags and L3 XAI tags across different label classes, offering insights into their discriminative power.

**Outcome:** Analytical reports, statistical summaries, and visualizations documenting data characteristics and feature/model performance.

---

### Real-time Live Fraud Capture Module
*(Scripts in `scripts/live/`, specifically `realtime_inference_service.py` and `stream_transactions.py`)

This module facilitates the real-time monitoring and detection of fraudulent blockchain transactions, acting as an active inference endpoint for the trained models.

*   **`stream_transactions.py`**: Establishes and maintains a continuous WebSocket connection to an Ethereum node (e.g., Alchemy), streaming pending transactions in real-time.
    *   **Key Aspects**: Implements robust reconnection logic with exponential backoff to ensure service resilience against network instabilities and temporary connection disruptions.
*   **`realtime_inference_service.py`**: Acts as the real-time inference engine, loading pre-trained models and processing live transaction streams.
    *   Upon receiving a new transaction:
        1.  Updates the in-memory wallet transaction history for behavioral context.
        2.  Calculates L0, L1, L2, and L3 features for the incoming transaction.
        3.  Performs fraud prediction using the loaded supervised models.
        4.  Generates XAI `reason_code` outputs to provide transparent justification for predictions.
        5.  Logs detected fraudulent transactions, including wallet address, transaction hash, prediction, and XAI reason, to `output/flagged_transactions.csv`.
    *   **Key Aspects**: Incorporates comprehensive error handling for incoming transaction data (e.g., validation of 'from' addresses to prevent `NoneType` attribute errors), ensuring operational stability.

**Outcome:** A continuously operating service that provides immediate fraud alerts, explainable prediction rationales, and a persistent audit trail of flagged transactions.

---

## How to Use This Guide

This guide serves as a technical reference. For a high-level project overview, consult the `README.md`. For interactive demonstrations of the workflow and live fraud capture, refer to the Jupyter notebooks (`demo_workflow_walkthrough.ipynb`, `demo_live_fraud_capture.ipynb`) located in the `notebooks/` directory. Each script discussed herein can be found in the `scripts/` directory. 