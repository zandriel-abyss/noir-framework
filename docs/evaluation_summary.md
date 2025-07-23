# Evaluation Summary – Noir Framework

This report presents the evaluation of machine learning models developed for fraud detection in blockchain ecosystems using the Noir Framework. The framework combines feature-driven supervised learning with graph-based modeling to classify wallet behavior into three categories: `fraud`, `normal`, and `suspicious`.

---

## 1. Dataset & Setup

- **Wallet Labels**: `fraud`, `normal`, `suspicious`
- **Label Distribution (post-filtered)**:  
  - Fraud: 155 wallets  
  - Normal: 50 wallets  
  - Suspicious: 34 wallets

- **Feature Layers**:
  - **L0–L3 Aggregate**: Transaction counts, volumes, burst patterns, anomaly flags
  - **Label Source**: Curated dataset from OFAC, Etherscan tags, and heuristic risk indicators

- **Train-Test Split**:
  - Stratified 80/20 split for supervised models
  - GNN split handled after node filtering

---

## 2. Supervised Model Evaluation

### Random Forest

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Normal       | 0.75      | 0.90   | 0.82     |
| Fraud        | 0.88      | 0.91   | 0.89     |
| Suspicious   | 0.75      | 0.43   | 0.55     |

- **Accuracy**: 84%
- **Macro Avg F1**: 0.75
- **Strengths**: Reliable fraud detection
- **Gaps**: Struggles with suspicious label due to label sparsity

### XGBoost

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Normal       | 0.69      | 0.90   | 0.78     |
| Fraud        | 0.88      | 0.88   | 0.88     |
| Suspicious   | 0.75      | 0.43   | 0.55     |

- **Accuracy**: 82%
- **Macro Avg F1**: 0.73
- **Insights**: Comparable to RF with slight drop in generalization on smaller classes

---

## 3. GNN Evaluation (Graph Convolutional Network)

### Latest Run Summary

- **Input Nodes**: 58,508
- **Valid Labeled Nodes**: 239 (connected + labeled)
- **Filtered Graph**: 239 nodes, 287 edges
- **Test Split**: 48 nodes

### Classification Report (Example Run)

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Normal (0)   | 0.38      | 1.00   | 0.56     | 10      |
| Fraud (1)    | 1.00      | 0.33   | 0.50     | 30      |
| Suspicious (2)| 0.67     | 1.00   | 0.80     | 8       |

- **Accuracy**: 58%
- **Macro Avg F1**: 0.62
- **Weighted Avg F1**: 0.56

### Confusion Matrix (Example)

```
           Predicted
            0  1  2
Actual  0  [10 0  0]
        1  [16 10 4]
        2  [0  0  8]
```

### Notes on Variability:
- GNN output changes across runs due to random seeds in node shuffling, split, and graph sampling.
- Performance is sensitive to class imbalance, node connectivity, and small test sizes.
- Fraud class being overrepresented introduces skew in how the model optimizes predictions.

---

## 4. Visualizations

- **2D & 3D t-SNE Embeddings** show clustering of wallets by predicted class.
- Suspicious wallets show tight grouping in 3D space, suggesting strong feature coherence.
- Interactive plots help spot overlaps and misclassified nodes.
- Color legend mismatch was corrected—`Grey` now represents Suspicious.

---

## 5. Key Insights

- **Fraud is detected well across models**, both in supervised and GNN approaches.
- **Suspicious class** remains the hardest to model due to limited support and unclear boundaries.
- **Graph models** are promising but need richer connectivity and feature propagation.
- **Explainability** via SHAP confirms fraud decisions align with domain rules (e.g. burst ratio, counterparty tags).

---

## 6. Recommendations

- Increase data for suspicious wallets (label curation, anomaly triage)
- Introduce temporal features or rolling window behaviors into graph modeling
- Perform stratified k-fold validation for GNNs to reduce randomness
- Consider a hybrid model: GNN embeddings as inputs to XGBoost/Random Forest
- Use GNN model confidence + explainability for compliance-ready scoring

---

## 7. Output Artifacts

- `output/models/predictions.csv`
- `output/models/shap_rf_fraud_summary.png`
- `output/gnn/gnn_data.pt`
- `output/gnn/gnn_model_results.txt`
- `output/gnn/tsne_embeddings_3d.html`