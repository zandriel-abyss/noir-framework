# Noir Framework – Blockchain Fraud Detection

## Overview

**Noir** is a modular machine learning framework for detecting fraud and enabling AML compliance across Ethereum and Layer-2 blockchain ecosystems. It combines behavioral wallet analysis, unsupervised and supervised ML models, symbolic reasoning, and explainable AI to flag malicious activity and distinguish it from privacy-preserving behavior.

## Key Objectives

- Detect anomalous wallet behavior in a decentralized ecosystem
- Identify fraudulent vs. suspicious vs. normal activity patterns
- Support real-time compliance needs via wallet risk scoring
- Provide transparent, explainable decision rationale for investigators
- Enable modular experimentation across models, rules, and data

---

##  Core Modules

### 1. Data Collection

Scripts for pulling wallet lists and transaction histories from:

- **Fraudulent Addresses**: OFAC, Etherscan "phish"/"hack", Lazarus group datasets
- **Mixers & Obfuscation**: Tornado Cash-linked wallets
- **Normal Wallets**: Etherscan Rich List
- **Transactions**: via Etherscan API (mainnet + L2)

### 2. Wallet Feature Engineering Pipeline

Behavioral signals are extracted from raw wallet transactions in four layers:

####  L0 – Aggregate Features

Basic summary stats across a wallet’s activity history:

- `total_transactions`: Total count of transactions
- `wallet_age_days`: Age since wallet creation
- `avg_tx_value`: Mean transaction value
- `std_tx_interval`: Std deviation of time between txs

####  L1 – Behavioral & Temporal Features

Captures wallet dynamics and unique interaction patterns:

- `burst_tx_ratio`: Ratio of txs in burst clusters (sudden spikes)
- `dormant_awaken_count`: # of dormant → active shifts
- `failure_ratio`: Ratio of failed transactions (from L1)
- `mean_tx_interval_hours`: Average time between transactions
- `std_tx_interval_hours`: Standard deviation of time between transactions
- `weekend_tx_ratio`: Proportion of transactions on weekends
- `txn_span_hours`: Total duration of wallet activity in hours
- `night_tx_ratio`: Proportion of transactions during night hours

#### L2 – Risk Indicators & Heuristics

Derived flags based on AML intuition and fraud research:

- `num_fraud_counterparties`: Known bad addresses interacted with
- `circular_flow_flag`: Funds loop back to self or aliases
- `gas_anomaly_flag`: Flag for unusually high gas fees
- `rapid_bridging_flag`: Indicator for rapid fund movement across layers
- `smart_contract_misuse_flag`: Flag for suspicious smart contract interactions
- `mixer_flag`: Indicates interaction with known mixer services
- `mixer_then_bridge_flag`: Indicates mixer interaction followed by bridging activity
- `mixer_exit_tx_count`: Number of transactions exiting mixer services
- `same_recipient_ratio`: Ratio of transactions to the same recipient

#### L3 – AI & XAI Meta-Signals

Post-model and symbolic enrichment:

- `anomaly_iso`, `anomaly_dbscan`: Anomaly scores from Isolation Forest / DBSCAN models.
- `combined_risk_tag`: Synthesized behavioral risk tag.
- `xai_reason_code`: Human-readable justification for flags (e.g., “burst_tx|fraud_link|model_anomaly”, “dormant_awakened”, “unusual_gas_fee”).

---

##  Modeling & Risk Scoring Engine

- **Unsupervised Models**:  
  - DBSCAN (cluster outliers)  
  - Isolation Forest (outlier scores)  

- **Supervised Models**:  
  - Random Forest, XGBoost  
  - Trained on fraud/normal/suspicious labels

- **GNN Modeling** (Optional):  
  - Graph-based classifier over wallet network  
  - Supports node-level fraud prediction using PyTorch Geometric

- **Explainability Tools**:  
  - SHAP (tree explainer) for local/global feature impact  
  - Symbolic rules + reason codes for interpretable tagging

---

##  Evaluation Highlights

- **F1-score**: ~0.56–0.80 across label classes in test sets
- **Imbalanced Dataset Handling**: Weighted loss, stratified split
- **Confusion Matrix & Classification Reports**: Tracked per run
- **3D t-SNE Embeddings**: Visual map of wallet clusters

---

##  Repository Structure

```
noir-framework/
│
├── datasource/
│   ├── raw/              ← ofac, hackers, mixers, etc.
│   └── processed/         ← cleaned and feature datasets
│
├── scripts/
│   ├── fetch/             ← data_fetch_*.py scripts
│   ├── features/          ← build_features_l*.py scripts
│   ├── gnn/               ← GNN prep and training
│   └── models/            ← train_supervised_models.py, shap_analysis.py
│
├── notebooks/             ← Exploratory analysis and debug notebooks
├── output/                ← Final predictions, visuals, reports
├── docs/                  ← README, evaluation summary, research logs
├── requirements.txt
└── .gitignore
```

---

## ▶ Quick Start

```bash
# Clone and set up environment
git clone https://github.com/zandriel-abyss/noir-framework.git
cd noir-framework
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your API key in a .env file
ETHERSCAN_API_KEY=your_etherscan_key
```

## 🛠️ Usage Examples

```bash
# 1. Fetch transaction data
python3 scripts/fetch/data_fetch_fraud.py
python3 scripts/fetch/data_fetch_mixers.py
python3 scripts/fetch/data_fetch_mixer_recipients.py
python3 scripts/fetch/data_fetch_normal.py

# 2. Build wallet features
python3 scripts/features/build_features_l0_aggregate.py
python3 scripts/features/build_features_l1_behavior.py
python3 scripts/features/build_features_l2_riskflags.py
python3 scripts/features/build_features_l3_metaai.py
python3 scripts/features/build_features_l3_xai_tags.py

# 3. Merge & model
python3 scripts/features/merge_final_features.py
python3 scripts/models/train_supervised_models.py

# 4. GNN pipeline (optional)
python3 scripts/gnn/build_graph_dataset.py
python3 scripts/gnn/prep_gnn_input.py
python3 scripts/gnn/train_gnn_model.py
python3 scripts/gnn/analyse_predictions.py

# 5. Run the new demo notebooks for a guided walkthrough and live fraud capture
# For full data pipeline: jupyter lab demo_workflow_walkthrough.ipynb
# For live fraud detection: jupyter lab demo_live_fraud_capture.ipynb
```

---

##  Data Insights Summary

- **Label Distribution**: Skewed toward fraud, matching real-world attack dynamics
- **Top Features**: `total_transactions`, `wallet_age_days`, `burst_tx_ratio`, `num_fraud_counterparties`
- **Key Correlations**: Rule-based tags often overlap with anomaly scores and known wallet interactions
- **Model Agreement**: Supervised + unsupervised models align on ~90% of suspicious outliers

---

##  Roadmap

- [x] L0–L3 features pipeline
- [x] Supervised & unsupervised ML models
- [x] Explainable outputs (SHAP + symbolic)
- [x] GNN wallet scoring module
- [x] Real-time sandbox scoring API
- [ ] DAO-integrated fraud reporting (experimental)

---

##  References

- Farrugia, S., et al. (2020). Detection of Illicit Accounts over Ethereum.
- Taher, S. S., et al. (2024). Advanced Fraud Detection in Blockchain Transactions.
- Song, K., et al. (2024). Money Laundering Subgraphs.
- Weber, M., et al. (2019). AML via Graph Convolutions.
- Chainalysis (2023–2024). Crypto Crime Reports.
- BIS (2023). Project Aurora & Hertha.

---

##  Author

**Jzacksline Soosaiya**  
GitHub: [zandriel-abyss](https://github.com/zandriel-abyss/noir-framework)  
Master’s Thesis – University of Amsterdam (2025)

For visual demos, model artifacts, or research collaboration, contact via GitHub.

---