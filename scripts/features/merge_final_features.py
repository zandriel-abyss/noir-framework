# merge_final_features.py

import pandas as pd
from pathlib import Path

# Input paths
L0_PATH = Path("datasource/processed/features_l0_aggregate.csv")
L3_XAI_PATH = Path("datasource/processed/features_l3_metaai_xai.csv")
OUTPUT_PATH = Path("datasource/processed/features_final_all_layers.csv")

# Load datasets
df_l0 = pd.read_csv(L0_PATH)
df_l3 = pd.read_csv(L3_XAI_PATH)

# Merge on wallet_address
df_merged = pd.merge(df_l3, df_l0, how='left', on='wallet_address', suffixes=("", "_l0"))

# Drop duplicated columns if L1/L2 features already in L3
columns_to_drop = ['total_transactions_l0', 'wallet_age_days_l0', 'active_days_l0',
                   'dormant_awaken_count_l0', 'num_unique_from', 'num_unique_to',
                   'circular_tx_count']
df_merged = df_merged.drop(columns=[col for col in columns_to_drop if col in df_merged.columns])

# Save final combined feature set
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Final feature set saved to: {OUTPUT_PATH}")