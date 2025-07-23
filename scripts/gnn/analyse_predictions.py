import pandas as pd
import sys

gnn = pd.read_csv("output/gnn/gnn_predictions.csv")
sup = pd.read_csv("output/models/predictions.csv")

# Normalize wallet addresses before merging
gnn['wallet_address'] = gnn['wallet_address'].str.strip().str.lower()
sup['wallet_address'] = sup['wallet_address'].str.strip().str.lower()

merged = gnn.merge(sup, on="wallet_address", suffixes=('_gnn', '_sup'))
print(merged.columns)  # Check column names
print("Merged shape:", merged.shape)
print("Non-null true_label_gnn:", merged['true_label_gnn'].notnull().sum())
print("Non-null true_label_sup:", merged['true_label_sup'].notnull().sum())

# Exit if merge is empty
if merged.shape[0] == 0:
    print("⚠️ No overlapping wallet addresses found between GNN and supervised models.")
    sys.exit()

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

if merged['true_label'].notnull().sum() > 0:
    print("GNN Classification Report:")
    print(classification_report(merged['true_label'], merged['gnn_pred']))
    
    print("RF Classification Report:")
    print(classification_report(merged['true_label'], merged['rf_pred']))
    
    print("XGB Classification Report:")
    print(classification_report(merged['true_label'], merged['xgb_pred']))
else:
    print("⚠️ No valid labels for classification report.")

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(pd.crosstab(merged['gnn_pred'], merged['rf_pred']), annot=True, fmt='d')
plt.xlabel("RF Prediction")
plt.ylabel("GNN Prediction")
plt.title("GNN vs RF Prediction Agreement")
plt.show()

