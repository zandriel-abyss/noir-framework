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

# Define function to compute wallet-level features
def compute_wallet_features(df):
    features = []
    
    grouped = df.groupby('wallet_address')
    for wallet, group in grouped:
        group = group.sort_values('timeStamp')
        tx_count = len(group)
        total_value = group['value'].sum()
        avg_value = group['value'].mean()
        wallet_age = (group['timeStamp'].max() - group['timeStamp'].min()).days
        active_days = group['timeStamp'].dt.date.nunique()
        
        unique_from = group['from'].nunique()
        unique_to = group['to'].nunique()
        circular_count = ((group['from'] == group['to']) & (group['from'] == wallet)).sum()

        # Detect dormant awakenings (if >30 days gap between tx)
        time_diffs = group['timeStamp'].diff().dt.days
        dormant_awakenings = (time_diffs > 30).sum()

        label = group['label'].mode()[0] if 'label' in group.columns else 'unknown'

        features.append({
            'wallet_address': wallet,
            'total_transactions': tx_count,
            'total_value': total_value,
            'avg_tx_value': avg_value,
            'wallet_age_days': wallet_age,
            'active_days': active_days,
            'num_unique_from': unique_from,
            'num_unique_to': unique_to,
            'circular_tx_count': circular_count,
            'dormant_awaken_count': dormant_awakenings,
            'label': label
        })
    return pd.DataFrame(features)

# Compute features
wallet_features = compute_wallet_features(df)

# Save
wallet_features.to_csv(OUTPUT_FILE, index=False)
print(f" Wallet features saved to: {OUTPUT_FILE}")