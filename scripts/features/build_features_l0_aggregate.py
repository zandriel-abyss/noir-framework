"""
L0 - Raw Transaction Stats
  - Basic aggregate metrics: total txs, avg value, wallet age
  - Useful for establishing baseline activity patterns
  for each eallet->
    Counts total transactions.
    Sums and averages transaction values.
    Calculates wallet age (time between first and last transaction).
    Counts unique counterparties (from and to).
    Counts "circular" transactions (where from and to are the same wallet).
    Counts "dormant awakenings" (gaps >30 days between transactions).
    Assigns a label (if available).

"""

import pandas as pd
from pathlib import Path
from scripts.features.feature_utils import calculate_l0_features # Import the utility function

# Input and output paths
INPUT_FILE = Path("datasource/raw/all_transactions_labeled.csv")
OUTPUT_FILE = Path("datasource/processed/features_l0_aggregate.csv")

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Ensure timestamps are in datetime format
df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s', errors='coerce')

# Create a helper column for ETH value
try:
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
except Exception as e:
    print("Error parsing value column:", e)

# Compute features using the utility function
features = []
grouped = df.groupby('wallet_address')
for wallet, group in grouped:
    features.append(calculate_l0_features(group.copy(), wallet))

wallet_features = pd.DataFrame(features)

# Save
wallet_features.to_csv(OUTPUT_FILE, index=False)
print(f" Wallet features saved to: {OUTPUT_FILE}")