"""
L2 - Risk Flags (flags wallets based on their interactions with known risky counterparties)
  - Counterparty risk scores
  - Useful for identifying risky wallets
"""

import pandas as pd
from collections import defaultdict
from scripts.features.feature_utils import calculate_l2_features

# File paths
TRANSACTIONS_FILE = 'datasource/raw/all_transactions_labeled.csv'
FEATURES_L1_FILE = 'datasource/processed/features_l1_behavior.csv' # Renamed from L0 to L1 for clarity
OUTPUT_FILE = 'datasource/processed/features_l2_riskflags.csv'

# Load datasets
df_tx = pd.read_csv(TRANSACTIONS_FILE)

# Ensure timestamps are in datetime format for later use in real-time context if needed
df_tx['timeStamp'] = pd.to_datetime(df_tx['timeStamp'], unit='s', errors='coerce')

# Convert important numeric fields safely (these are from raw transactions, not aggregated yet for wallets)
if 'avg_gas_fee' in df_tx.columns:
    df_tx['avg_gas_fee'] = pd.to_numeric(df_tx['avg_gas_fee'], errors='coerce').fillna(0)
else:
    df_tx['avg_gas_fee'] = 0

if 'smart_contract_failures' in df_tx.columns:
    df_tx['smart_contract_failures'] = pd.to_numeric(df_tx['smart_contract_failures'], errors='coerce').fillna(0)
else:
    df_tx['smart_contract_failures'] = 0

if 'layer_hopping_count' in df_tx.columns:
    df_tx['layer_hopping_count'] = pd.to_numeric(df_tx['layer_hopping_count'], errors='coerce').fillna(0)
else:
    df_tx['layer_hopping_count'] = 0

df_l1_behavior = pd.read_csv(FEATURES_L1_FILE)

# Clean and prepare
df_tx = df_tx.dropna(subset=['from', 'to', 'wallet_address', 'label'])
df_tx['from'] = df_tx['from'].str.lower()
df_tx['to'] = df_tx['to'].str.lower()
df_tx['wallet_address'] = df_tx['wallet_address'].str.lower()

# Get risk scores for known counterparties
wallet_risk_labels = df_tx.groupby('wallet_address')['label'].first().to_dict()

# Calculate global 95th percentile for gas fee for consistent gas anomaly flag calculation
global_gas_95th_percentile = df_tx['gasPrice'].quantile(0.95) if 'gasPrice' in df_tx.columns and not df_tx['gasPrice'].empty else 0

# Compute counterparty features using the utility function
all_l2_features = []
# Group transactions by the 'from' wallet to calculate counterparty features for each sending wallet
for wallet, group in df_tx.groupby('from'):
    # Get the corresponding L0/L1 features for the current wallet
    wallet_l0_l1_features = df_l1_behavior[df_l1_behavior['wallet_address'] == wallet].iloc[0].to_dict() if wallet in df_l1_behavior['wallet_address'].values else {}
    all_l2_features.append(calculate_l2_features(group.copy(), wallet, wallet_risk_labels, wallet_l0_l1_features, global_gas_95th_percentile))

df_l2_computed_flags = pd.DataFrame(all_l2_features)

# Merge L1 features with the newly computed L2 flags
df_final = pd.merge(df_l1_behavior, df_l2_computed_flags, how='left', on='wallet_address')

# Fill any NaN values introduced by the merge for the new L2 features
df_final = df_final.fillna(0)

# Save output
df_final.to_csv(OUTPUT_FILE, index=False)
print(f" L2 counterparty and risk flags saved to: {OUTPUT_FILE}")