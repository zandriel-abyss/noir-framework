import pandas as pd
from pathlib import Path

# Input path
INPUT_PATH = Path("datasource/processed/features_final_all_layers.csv")
OUTPUT_PATH = Path("datasource/processed/wallet_labels_reference.csv")

# Load the full dataset
df = pd.read_csv(INPUT_PATH)

# Extract wallet_address and label
if 'wallet_address' not in df.columns or 'label' not in df.columns:
    raise ValueError("The input file must contain 'wallet_address' and 'label' columns.")

ref_df = df[['wallet_address', 'label']]

# Save reference mapping
ref_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved wallet label reference to: {OUTPUT_PATH}")