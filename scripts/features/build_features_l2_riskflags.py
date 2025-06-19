"""
L2 - Risk Flags (flags wallets based on their interactions with known risky counterparties)
  - Counterparty risk scores
  - Useful for identifying risky wallets
"""

import pandas as pd
from collections import defaultdict

# File paths
TRANSACTIONS_FILE = 'datasource/raw/all_transactions_labeled.csv'
FEATURES_L0_FILE = 'datasource/processed/features_l1_behavior.csv'
OUTPUT_FILE = 'datasource/processed/features_l2_riskflags.csv'

# Load datasets
df_tx = pd.read_csv(TRANSACTIONS_FILE)
df_l0 = pd.read_csv(FEATURES_L0_FILE)

# Clean and prepare
df_tx = df_tx.dropna(subset=['from', 'to', 'wallet_address', 'label'])
df_tx['from'] = df_tx['from'].str.lower()
df_tx['to'] = df_tx['to'].str.lower()
df_tx['wallet_address'] = df_tx['wallet_address'].str.lower()

# Get risk scores for known counterparties
wallet_risk = df_tx.groupby('wallet_address')['label'].first().to_dict()

# Initialize new L2 features
counterparty_scores = defaultdict(lambda: {'fraud': 0, 'suspicious': 0, 'normal': 0})

# Count how many times each wallet interacted with risky counterparties
for _, row in df_tx.iterrows():
    src = row['from']
    tgt = row['to']
    tgt_label = wallet_risk.get(tgt, None)
    if tgt_label in ['fraud', 'suspicious', 'normal']:
        counterparty_scores[src][tgt_label] += 1

# Convert to DataFrame
df_l2 = pd.DataFrame.from_dict(counterparty_scores, orient='index').reset_index()
df_l2 = df_l2.rename(columns={
    'index': 'wallet_address',
    'fraud': 'num_fraud_counterparties',
    'suspicious': 'num_suspicious_counterparties',
    'normal': 'num_normal_counterparties'
})

# Merge with L0
df_final = pd.merge(df_l0, df_l2, how='left', on='wallet_address')

# Fill missing values
df_final[['num_fraud_counterparties', 'num_suspicious_counterparties', 'num_normal_counterparties']] = \
    df_final[['num_fraud_counterparties', 'num_suspicious_counterparties', 'num_normal_counterparties']].fillna(0)

# Save output
df_final.to_csv(OUTPUT_FILE, index=False)
print(f" L2 counterparty features saved to: {OUTPUT_FILE}")