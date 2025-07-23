# Noir Framework – Progress Snapshot

## Current Milestone: Model Evaluation and Graph Analysis Complete

- **L0 – Aggregate Statistics**: Basic wallet-level summaries such as total number of transactions (`total_tx`), average transaction value (`avg_tx_value`), and wallet age. These features form the foundation of behavioral profiling.

- **L1 – Behavioral Patterns**: Time-sensitive and pattern-based metrics. Examples include:
  - `burst_tx_ratio`: Measures the intensity of activity spikes (e.g., multiple transactions in a short time).
  - `dormant_awakening`: Flags wallets that were inactive for a long period before sudden activity.
  - `tx_timing_entropy`: Captures irregularity in transaction intervals.

- **L2 – Risk Flags and Heuristics**: Boolean or categorical flags derived from known risk behaviors. Includes:
  - `mixer_interaction_flag`: Wallets interacting with known mixers.
  - `fraud_counterparty_ratio`: Proportion of counterparties already labeled as fraud.
  - `anomaly_score_flag`: Thresholded score from Isolation Forest/DBSCAN output.

- **L3 – MetaAI & Explainability**: Model-derived and symbolic explainability features:
  - `unsupervised_anomaly_score`: Scaled anomaly score from unsupervised models.
  - `xai_reason_code`: Tagged explanation using SHAP or rule-based mapping (e.g., “sudden spike in failed txs”).
  - `meta_label_agreement`: Agreement level between different model types on a fraud prediction.

Each layer builds on the prior one, allowing for both human-interpretable and machine-driven insights into wallet behavior across Ethereum and Layer 2 ecosystems.

## Current Phase: Results Synthesis and Thesis Report Compilation

- Classification reports and confusion matrices collected across models
- GNN results visualized and evaluated (2D + 3D t-SNE, class clustering)
- Performance comparison: Random Forest vs. GNN vs. XGBoost
- Interactive dashboards and model interpretability outputs under preparation

##  Dataset Overview

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
notebooks/
├── data-collection.ipynb
├── eda_l3_insights.ipynb   <-- Coming up
scripts/features/
├── build_features_l0_stats.py
├── build_features_l1_behavior.py
├── build_features_l2_riskflags.py
├── build_features_l3_metaai.py
├── tag_xai_reasons_l3.py
scripts/models/
├── train_supervised_models.py
scripts/mixers/
├── fetch_mixer_recipients.py
├── trace_mixer_recipient_txns.py
scripts/gnn/
├── build_graph_dataset.py
├── prep_gnn_input.py
├── train_gnn_model.py
```

## Key Findings
 
- Fraud wallets exhibit distinct patterns in burst_tx_ratio, dormant awakenings, and high counterparty fraud counts.
- Combined risk tag and anomaly flags correlate strongly with fraud label.
- SHAP highlights `active_days`, `burst_tx_ratio`, and `failure_ratio` as influential.
- Graph analysis reveals dense clustering among high-risk wallets.
- GNN accuracy is currently limited; future improvements include larger training set and temporal edge encoding.
- GNN model shows overfitting tendencies and class imbalance effects (e.g., fraud overrepresented in some runs)
- Accuracy across models ranged from 58% to 71%; highest F1 for suspicious wallets (class 2)
- Confusion matrices vary across runs due to random splits and small sample sizes

## Next Steps
 
- Finalize explainability layer with rule-based tags (L3 XAI reason codes)
- Expand GNN sample size and re-train with edge weights/time
- Conduct ablation study for most influential features
- Prepare thesis demo visuals and integrated pipeline summary
- Refine GNN input sampling and edge construction (e.g., edge weights, temporal sorting)
- Add interactive 3D visualization tools (Plotly/Altair) to explore embeddings
- Finalize written evaluation report with interpretation of precision, recall, F1, and model tradeoffs
- Cross-check class label integrity across preprocessing pipeline