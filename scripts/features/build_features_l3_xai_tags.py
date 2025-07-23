"""
L3 - MetaAI + XAI Tags
- Burst transaction flag
- Dormant/awake flag
- Counterparty fraud flag
- Anomaly flag
- Failure flag
"""

import pandas as pd

# Load the L3 MetaAI features
df = pd.read_csv("datasource/processed/features_l3_metaai.csv")

# XAI tags as boolean flags
df['burst_tx_flag'] = df['burst_tx_ratio'] > 0.95
df['dormant_awake_flag'] = df['dormant_awaken_count'] > 1
df['counterparty_fraud_flag'] = df['num_fraud_counterparties'] >= 1
df['anomaly_flag'] = (df['anomaly_iso'] == 1.0) | (df['anomaly_dbscan'] == 1)
df['failure_flag'] = df['failure_ratio'] > 0.5

# Additional symbolic XAI flags
df['smart_contract_misuse_flag'] = df['smart_contract_failures'] > 0
df['rapid_bridging_flag'] = df['layer_hopping_count'] > 2
df['circular_flow_flag'] = df['circular_tx_ratio'] > 0.6
df['gas_anomaly_flag'] = df['avg_gas_fee'] > df['avg_gas_fee'].quantile(0.95)

# Build human-readable reason codes
def generate_reason(row):
    reasons = []
    if row['burst_tx_flag']: reasons.append("burst_tx")
    if row['dormant_awake_flag']: reasons.append("dormant_awakened")
    if row['counterparty_fraud_flag']: reasons.append("fraud_link")
    if row['anomaly_flag']: reasons.append("model_anomaly")
    if row['failure_flag']: reasons.append("high_failure_rate")
    if row['smart_contract_misuse_flag']: reasons.append("smart_contract_misuse")
    if row['rapid_bridging_flag']: reasons.append("rapid_layer_hopping")
    if row['circular_flow_flag']: reasons.append("circular_fund_flow")
    if row['gas_anomaly_flag']: reasons.append("unusual_gas_fee")
    return "|".join(reasons) if reasons else "clean"

df['xai_reason_code'] = df.apply(generate_reason, axis=1)
df['xai_flag'] = df['xai_reason_code'] != "clean"

# Save final enriched L3 with XAI
df.to_csv("datasource/processed/features_l3_metaai_xai.csv", index=False)
print("L3 MetaAI + XAI tagging complete → features_l3_metaai_xai.csv")