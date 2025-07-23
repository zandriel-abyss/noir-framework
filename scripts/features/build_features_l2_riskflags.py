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

# Convert important numeric fields safely
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

# Ensure required columns exist and convert to numeric
for col in ['avg_gas_fee', 'layer_hopping_count', 'smart_contract_failures']:
    if col not in df_final.columns:
        df_final[col] = 0
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

# Debug preview of non-zero values in raw transaction data
print("Example gas fees > 0:\n", df_tx[df_tx['avg_gas_fee'] > 0].head())
print("Example smart_contract_failures > 0:\n", df_tx[df_tx['smart_contract_failures'] > 0].head())
print("Example layer_hopping_count > 0:\n", df_tx[df_tx['layer_hopping_count'] > 0].head())

# Temporary placeholder: self-loop detection from raw transactions (can be replaced with graph logic)
circular_wallets = df_tx[df_tx['from'] == df_tx['to']]['from'].unique()
df_final['circular_flow_flag'] = df_final['wallet_address'].isin(circular_wallets).astype(int)

df_final['gas_anomaly_flag'] = (df_final['avg_gas_fee'] > df_final['avg_gas_fee'].quantile(0.95)).astype(int)

df_final['rapid_bridging_flag'] = (df_final['layer_hopping_count'] > df_final['layer_hopping_count'].quantile(0.90)).astype(int)

df_final['smart_contract_misuse_flag'] = (df_final['smart_contract_failures'] > 0).astype(int)

# --- New behavioral context features ---

# mixer_flag: 1 if wallet interacted with suspicious counterparty at least once
df_final['mixer_flag'] = (df_final['num_suspicious_counterparties'] > 0).astype(int)

# mixer_then_bridge_flag: True if mixer_flag == 1 and layer_hopping_count > 1
df_final['mixer_then_bridge_flag'] = ((df_final['mixer_flag'] == 1) & (df_final['layer_hopping_count'] > 1)).astype(int)

# mixer_exit_tx_count: count of outgoing transactions from wallet after any suspicious counterparty interaction

# First, identify wallets that have interacted with suspicious counterparties
suspicious_counterparties = set(df_l2[df_l2['num_suspicious_counterparties'] > 0]['wallet_address'])

# For each transaction, check if the sender has interacted with suspicious counterparties
df_tx['sender_has_mixer'] = df_tx['from'].isin(suspicious_counterparties).astype(int)

# Count outgoing transactions per wallet where sender_has_mixer == 1
mixer_exit_counts = df_tx[df_tx['sender_has_mixer'] == 1].groupby('from').size().to_dict()

df_final['mixer_exit_tx_count'] = df_final['wallet_address'].map(mixer_exit_counts).fillna(0).astype(int)

# same_recipient_ratio: ratio of transactions going to the most common recipient address out of all transactions sent by the wallet

# Compute counts of 'to' addresses per 'from' wallet
tx_counts = df_tx.groupby('from').size()
most_common_to_counts = df_tx.groupby(['from', 'to']).size().groupby(level=0).max()

same_recipient_ratio = (most_common_to_counts / tx_counts).fillna(0)

df_final['same_recipient_ratio'] = df_final['wallet_address'].map(same_recipient_ratio).fillna(0).astype(float)

# --- End new behavioral context features ---

# Debug summary before saving to help verify flag activation
print("Non-zero flag counts:")
print(df_final[['circular_flow_flag', 'gas_anomaly_flag', 'rapid_bridging_flag', 'smart_contract_misuse_flag']].sum())

print("Distribution checks:")
print("avg_gas_fee > 0:", (df_final['avg_gas_fee'] > 0).sum())
print("layer_hopping_count > 0:", (df_final['layer_hopping_count'] > 0).sum())
print("smart_contract_failures > 0:", (df_final['smart_contract_failures'] > 0).sum())

# Save output
df_final.to_csv(OUTPUT_FILE, index=False)
print(f" L2 counterparty and risk flags saved to: {OUTPUT_FILE}")