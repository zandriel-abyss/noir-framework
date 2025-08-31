# Noir Framework – Progress Snapshot

## Current Milestone: Real-time Inference Module Operational; Model Evaluation and Graph Analysis Complete

- **L0 – Aggregate Statistics**: Basic wallet-level summaries derived from raw transaction data.
    - Key Features: `total_transactions`, `wallet_age_days`, `avg_tx_value`, `std_tx_interval`.

- **L1 – Behavioral Patterns**: Metrics capturing dynamic and temporal behavioral patterns of wallets.
    - Key Features: `burst_tx_ratio`, `dormant_awaken_count`, `failure_ratio`, `mean_tx_interval_hours`, `std_tx_interval_hours`, `weekend_tx_ratio`, `txn_span_hours`, `night_tx_ratio`.

- **L2 – Risk Flags and Heuristics**: Boolean or categorical flags derived from AML intuition and identified fraud research patterns.
    - Key Features: `num_fraud_counterparties`, `circular_flow_flag`, `gas_anomaly_flag`, `rapid_bridging_flag`, `smart_contract_misuse_flag`, `mixer_flag`, `mixer_then_bridge_flag`, `mixer_exit_tx_count`, `same_recipient_ratio`.

- **L3 – MetaAI & Explainability**: Model-derived features from unsupervised anomaly detection and human-interpretable Explainable AI (XAI) tags.
    - Key Features: `anomaly_iso`, `anomaly_dbscan`, `combined_risk_tag`.
    - XAI Outputs: `xai_reason_code` (e.g., “burst_tx|fraud_link|model_anomaly”, “dormant_awakened”, “unusual_gas_fee”).

Each layer builds on the prior one, allowing for both human-interpretable and machine-driven insights into wallet behavior across Ethereum and Layer 2 ecosystems.

## Current Phase: Results Synthesis and Documentation

- Classification reports and confusion matrices collected across models
- GNN results visualized and evaluated (2D + 3D t-SNE, class clustering)
- Performance comparison: Random Forest vs. GNN vs. XGBoost
- Real-time fraud detection service (`realtime_inference_service.py`) implemented and validated
- Comprehensive technical workflow guide (`WORKFLOW_GUIDE.md`) created
- Interactive demonstration notebooks (`demo_workflow_walkthrough.ipynb`, `demo_live_fraud_capture.ipynb`) developed

## Dataset Overview

| Label     | Transactions | Wallets | Time Range     |
|-----------|--------------|---------|----------------|
| Fraud     | 187,604      | 33,628  | 2017–2025      |
| Normal    | 59,685       | 11,229  | 2015–2025      |
| Mixer     | 129,577      | 25,893  | 2019–2025      |
| Mixer Recipient | ~300+ traced txns | ~90+ wallets | 2015–2025 |

##  Graph Dataset

- Nodes: Wallets with feature vectors (L0–L3)
- Edges: Transactions between wallets (directed)
- Format: PyTorch Geometric `Data` object
- Stored at: `output/gnn/gnn_data.pt`

##  Repo Structure Highlights

```
noir-framework/
│
├── datasource/
│   ├── raw/              ← Raw fetched transaction data
│   └── processed/         ← Cleaned and engineered feature datasets
│
├── scripts/
│   ├── fetch/             ← Scripts for data acquisition
│   ├── process/           ← Scripts for data merging and preprocessing
│   ├── features/          ← Scripts for layered feature engineering (L0-L3)
│   ├── models/            ← Scripts for supervised model training and SHAP analysis
│   ├── gnn/               ← Scripts for GNN dataset preparation and training
│   └── live/              ← Scripts for real-time inference service
│
├── notebooks/             ← Exploratory analysis and demonstration notebooks
├── output/                ← Final predictions, visualizations, and reports
├── docs/                  ← Project documentation (README, workflow guide, reports)
├── requirements.txt
└── .gitignore
```

## Key Findings
 
- Fraudulent wallets exhibit distinct patterns in `burst_tx_ratio`, `dormant_awaken_count`, and `num_fraud_counterparties`.
- `combined_risk_tag` and `anomaly_flag` correlate strongly with the fraud label.
- SHAP analysis highlights `active_days`, `burst_tx_ratio`, and `failure_ratio` as influential features in supervised models.
- Graph analysis reveals dense clustering among high-risk wallets, suggesting strong relational patterns in fraud propagation.
- GNN accuracy is currently limited; future improvements are focused on larger training sets and temporal edge encoding.
- GNN model shows tendencies towards overfitting and class imbalance effects (e.g., fraud overrepresented in some runs).
- Accuracy across supervised models ranged from 58% to 71%; highest F1-score observed for suspicious wallets (class 2).
- Confusion matrices exhibit variability across runs due to random data splits and the inherent challenges of small, imbalanced samples.

## Future Work
 
- Expand GNN sample size and re-train with edge weights/temporal features.
- Conduct ablation study for most influential features to refine model inputs.
- Add interactive 3D visualization tools (Plotly/Altair) to explore GNN embeddings and feature spaces.
- Further refine GNN input sampling and edge construction methodologies.
- Cross-check class label integrity across the entire preprocessing pipeline.
- Explore DAO-integrated fraud reporting mechanisms (experimental).