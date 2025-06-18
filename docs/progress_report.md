# Noir Framework – Progress Snapshot

##  Current Milestone: Feature Engineering Complete

All wallet-level features are now generated and saved. This includes:

-  L0: Raw stats from processed transactions
-  L1: Behavioral features (dormancy, burst activity)
-  L2: Risk flags from mixers, bridges, and label associations
-  L3: MetaAI tags with anomaly scores, counterparty risk profiling, and XAI reason codes
-  Mixer Recipient Phase: Newly integrated mixer recipient wallet tracebacks and transactions
-  GNN Graph Dataset: Wallet-to-wallet edge list and node features generated

##  Current Phase: Model Training & Evaluation

- Random Forest and XGBoost classifiers trained
- Evaluation metrics analyzed (precision, recall, f1-score)
- Mixer-linked wallet performance under review
- SHAP explanations and predictions exported
- GNN model (Graph Convolutional Network) trained on wallet graph
- Graph-based classification performance reviewed (early-stage)

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