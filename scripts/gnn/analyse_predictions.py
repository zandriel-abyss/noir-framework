import pandas as pd

gnn = pd.read_csv("output/gnn/gnn_predictions.csv")
sup = pd.read_csv("output/models/predictions.csv")

merged = gnn.merge(sup, on="wallet_address", suffixes=('_gnn', '_sup'))
print(merged.columns)  # Check column names

# Use the correct true_label column
merged['true_label'] = merged['true_label_gnn']  # or 'true_label_sup'

# Agreement columns
merged['agree_rf'] = merged['gnn_pred'] == merged['rf_pred']
merged['agree_xgb'] = merged['gnn_pred'] == merged['xgb_pred']

print("GNN & RF agreement rate:", merged['agree_rf'].mean())
print("GNN & XGB agreement rate:", merged['agree_xgb'].mean())
print("All models agree on:", ((merged['gnn_pred'] == merged['rf_pred']) & (merged['rf_pred'] == merged['xgb_pred'])).mean())

# Error analysis
gnn_right_rf_wrong = merged[(merged['gnn_pred'] == merged['true_label']) & (merged['rf_pred'] != merged['true_label'])]
print("GNN correct, RF wrong:", len(gnn_right_rf_wrong))

rf_right_gnn_wrong = merged[(merged['rf_pred'] == merged['true_label']) & (merged['gnn_pred'] != merged['true_label'])]
print("RF correct, GNN wrong:", len(rf_right_gnn_wrong))

both_wrong = merged[(merged['gnn_pred'] != merged['true_label']) & (merged['rf_pred'] != merged['true_label'])]
print("Both wrong:", len(both_wrong))

from sklearn.metrics import classification_report

print("GNN Classification Report:")
print(classification_report(merged['true_label'], merged['gnn_pred']))

print("RF Classification Report:")
print(classification_report(merged['true_label'], merged['rf_pred']))

print("XGB Classification Report:")
print(classification_report(merged['true_label'], merged['xgb_pred']))

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(pd.crosstab(merged['gnn_pred'], merged['rf_pred']), annot=True, fmt='d')
plt.xlabel("RF Prediction")
plt.ylabel("GNN Prediction")
plt.title("GNN vs RF Prediction Agreement")
plt.show()

