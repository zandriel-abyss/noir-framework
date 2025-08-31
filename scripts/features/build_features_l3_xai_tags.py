"""
L3 - MetaAI + XAI Tags
- Burst transaction flag
- Dormant/awake flag
- Counterparty fraud flag
- Anomaly flag
- Failure flag
"""

import pandas as pd
from scripts.features.feature_utils import calculate_l3_xai_tags

# Load the L3 MetaAI features
df = pd.read_csv("datasource/processed/features_l3_metaai.csv")

# Ensure necessary columns are present for XAI tag calculation
default_cols = {
    'smart_contract_failures': 0,
    'layer_hopping_count': 0,
    'circular_tx_ratio': 0.0,
    'avg_gas_fee': 0.0
}
for col, default_val in default_cols.items():
    if col not in df.columns:
        df[col] = default_val

# Calculate global thresholds for flags that depend on them (e.g., quantiles)
# For batch processing, it's efficient to calculate these once.
# In a real-time scenario, these thresholds would be pre-computed and passed.

# Calculate gas_anomaly_flag based on the full dataset's 95th percentile
if 'avg_gas_fee' in df.columns and not df['avg_gas_fee'].empty:
    gas_fee_95th_percentile = df['avg_gas_fee'].quantile(0.95)
    df['gas_anomaly_flag'] = (df['avg_gas_fee'] > gas_fee_95th_percentile).astype(int)
else:
    df['gas_anomaly_flag'] = 0

# The other flags' thresholds are absolute or depend on features already in the DF

# Apply the utility function row-wise (or vectorized if possible for specific flags)
# For simplicity and consistency with `calculate_l3_xai_tags`, we'll apply row-wise.

xai_features = df.apply(lambda row: calculate_l3_xai_tags(pd.DataFrame([row.to_dict()])), axis=1)

# Merge the new XAI tags back to the original DataFrame
# The utility function returns a dict, so we convert to DataFrame for easier merging
xai_df = pd.DataFrame(xai_features.tolist())
xai_df = xai_df.drop(columns=['wallet_address'], errors='ignore') # Drop redundant wallet_address

df_final = pd.concat([df, xai_df], axis=1)

# Drop duplicate columns that might arise from concat if they were in xai_df and df
# (e.g. if a flag was already calculated and then re-returned by the utility function)
# For better control, we should ensure the utility only returns the new XAI tags.

# For now, let's specifically add new XAI columns if not already there
for col in ['burst_tx_flag', 'dormant_awake_flag', 'counterparty_fraud_flag', 'anomaly_flag',
            'failure_flag', 'smart_contract_misuse_flag', 'rapid_bridging_flag',
            'circular_flow_flag', 'gas_anomaly_flag', 'xai_reason_code', 'xai_flag']:
    if col in xai_df.columns and col not in df.columns:
        df_final[col] = xai_df[col]
    elif col in xai_df.columns and col in df.columns: # If already in df, use the one from xai_df if preferred
        df_final[col] = xai_df[col]

# Clean up any duplicate columns after concatenation, keeping the latest (from xai_df)
# This is a robust way to handle potentially overlapping columns.
df_final = df_final.loc[:,~df_final.columns.duplicated()]

# Save final enriched L3 with XAI
df_final.to_csv("datasource/processed/features_l3_metaai_xai.csv", index=False)
print("L3 MetaAI + XAI tagging complete → features_l3_metaai_xai.csv")