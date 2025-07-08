# Noir Framework – Progress Snapshot

## Current Milestone: Feature Engineering and Model Evaluation Complete

- L0: Raw stats from processed transactions (e.g., total tx, avg value)
- L1: Behavioral patterns (burst ratios, dormant awakenings, tx timing)
- L2: Heuristic risk flags (mixer links, fraud counterparties, anomaly flags)
- L3: MetaAI layers (unsupervised scores, reason tags, XAI explanation markers)
- SHAP Summary: Feature importance evaluated using Random Forest SHAP
- Mixer Tracebacks: Recipients and interactions extracted for enriched labels
- GNN Dataset: Wallet graph (nodes + edges) built for message-passing learning

## Current Phase: Full Model Evaluation and Interpretation

- Random Forest and XGBoost classifiers trained on final merged features
- Confusion matrices and classification reports generated for all classes
- SHAP analysis conducted (summary + beeswarm for top features)
- Feature distribution and correlation heatmap visualized by label
- GNN Model (GCN) trained on PyTorch Geometric graph
- GNN results reviewed with t-SNE and node visualizations
- Comparative limitations noted (e.g., GNN underfit due to small sample)

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

## Next Steps

- Finalize explainability layer with rule-based tags (L3 XAI reason codes)
- Expand GNN sample size and re-train with edge weights/time
- Conduct ablation study for most influential features
- Prepare thesis demo visuals and integrated pipeline summary