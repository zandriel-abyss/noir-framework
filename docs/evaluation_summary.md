#  Evaluation Summary – Noir Framework (Supervised Models)

This document summarizes the performance and interpretability of our supervised ML models (Random Forest & XGBoost) trained on wallet-level features for blockchain fraud detection.

---

##  Model Setup

- **Input Features:** Combined L0–L3 wallet features (e.g., transaction volume, anomaly tags, counterparty risk)
- **Target Labels:** `fraud`, `normal`, `suspicious`
- **Models Used:**
  - Random Forest Classifier
  - XGBoost Classifier
  - Graph Neural Network (GCN) using PyTorch Geometric
- **Explainability Layer:** SHAP values (per-feature impact on predictions)
- **Dataset Split:** Stratified 80/20 train-test

---

##  Performance Overview

### Random Forest

| Metric        | Normal | Fraud | Suspicious |
|---------------|--------|-------|------------|
| Precision     | 0.75   | 0.88  | 0.75       |
| Recall        | 0.90   | 0.91  | 0.43       |
| F1-Score      | 0.82   | 0.89  | 0.55       |

- **Overall Accuracy:** 84%
- **Macro Avg F1:** 0.75
- **Key Strength:** Strong at classifying `fraud`, fair performance on `normal`, moderate struggles with `suspicious`

### XGBoost

| Metric        | Normal | Fraud | Suspicious |
|---------------|--------|-------|------------|
| Precision     | 0.69   | 0.88  | 0.75       |
| Recall        | 0.90   | 0.88  | 0.43       |
| F1-Score      | 0.78   | 0.88  | 0.55       |

- **Overall Accuracy:** 82%
- **Macro Avg F1:** 0.73

### GNN (Graph Convolutional Network)

| Metric        | Normal | Fraud | Suspicious |
|---------------|--------|-------|------------|
| Precision     | 0.00   | 0.67  | 0.00       |
| Recall        | 0.00   | 0.91  | 0.00       |
| F1-Score      | 0.00   | 0.77  | 0.00       |

- **Overall Accuracy:** 62%
- **Macro Avg F1:** 0.26

- **Observation:** The GCN model currently struggles to classify `normal` and `suspicious` labels due to class imbalance and limited graph depth. Fraud detection remains relatively strong, likely due to well-connected fraud clusters in the transaction graph.

---

##  SHAP Feature Importance (Random Forest)

Top impactful features:
1. `avg_tx_value`
2. `num_suspicious_counterparties`
3. `total_value`
4. `burst_tx_ratio`
5. `active_days`, `wallet_age_days`

Low-impact features:
- `failure_flag`, `anomaly_dbscan`, `num_normal_counterparties`

Interpretation:
- High transaction values and counterparties labeled as suspicious heavily influence fraud predictions.
- Wallets with burst activity and short lifespans tend to be flagged as anomalies.

---

##  Key Insights

- **Fraud detection is robust:** Both models show high precision/recall for the `fraud` class.
- **Suspicious class is underrepresented:** Limited samples result in weaker predictive performance.
- **Explainability works:** SHAP validates model reasoning aligns with domain heuristics.
- **Some mislabeled cases remain:** May benefit from active learning or better label curation.
- **GNN models show potential but need tuning:** The GCN's current underperformance highlights the need for better class balance and richer node features. Despite this, it captures fraud-linked hubs more consistently than random.

---

##  Recommendations

- Augment `suspicious` wallet samples for better learning.
- Consider fine-tuning thresholds for production scoring.
- Use GNN-based models next for temporal/graph behavior.
- SHAP + rules make compliance reporting easier.

---

## Outputs saved:
- `output/models/predictions.csv`
- `output/models/shap_rf_fraud_summary.png`
- `output/gnn/gnn_data.pt`
- `output/gnn/gnn_model_results.txt`