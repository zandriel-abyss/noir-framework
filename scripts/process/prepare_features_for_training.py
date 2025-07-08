import pandas as pd
import numpy as np
from pathlib import Path

# Input path
INPUT_PATH = Path("datasource/processed/features_final_all_layers.csv")
OUTPUT_PATH = Path("datasource/processed/features_for_training.csv")

# Load merged feature dataset
df = pd.read_csv(INPUT_PATH)

# Drop non-numeric or non-feature columns
drop_cols = ['xai_reason_code', 'label']
X = df.drop(columns=drop_cols, errors='ignore')
y = df['label']

# Drop redundant or highly correlated features
redundant_features = ['anomaly_dbscan']  # keep 'anomaly_flag'
X = X.drop(columns=redundant_features, errors='ignore')

# Explicitly drop non-numeric cols for correlation matrix calculation
X_numeric = X.select_dtypes(include=[np.number])
# Optional: Drop highly correlated features (threshold > 0.95)
corr_matrix = X_numeric.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop_corr, errors='ignore')
print(f"Dropped {len(to_drop_corr)} highly correlated features:", to_drop_corr)

# Insert wallet_address at the front AFTER correlation step
if 'wallet_address' not in X.columns:
    X.insert(0, 'wallet_address', df['wallet_address'])

# Encode labels
label_mapping = {'normal': 0, 'fraud': 1, 'suspicious': 2}
y_encoded = y.map(label_mapping)

# Add label back
X['label'] = y_encoded

# Save processed data
X.to_csv(OUTPUT_PATH, index=False)
print(f"Saved cleaned feature dataset to: {OUTPUT_PATH}")