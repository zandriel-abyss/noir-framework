import pandas as pd
from pathlib import Path

# Input paths
INPUT_PATH = Path("datasource/processed/features_final_all_layers.csv")
NODE_FEATURES_PATH = Path("output/gnn/graph_node_features.csv")
OUTPUT_PATH = Path("datasource/processed/wallet_node_labels_merged.csv")

# Load the full dataset
df = pd.read_csv(INPUT_PATH)
node_df = pd.read_csv(NODE_FEATURES_PATH)

# Extract wallet_address and label
if 'wallet_address' not in df.columns or 'label' not in df.columns:
    raise ValueError("The input file must contain 'wallet_address' and 'label' columns.")

merged_df = pd.merge(node_df, df[['wallet_address', 'label']], on='wallet_address', how='inner')

# Save merged node features with labels
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved merged wallet label reference to: {OUTPUT_PATH}")