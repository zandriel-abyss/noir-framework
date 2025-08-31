import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# --- Configuration Paths ---
MERGED_FEATURES_PATH = Path("datasource/processed/all_merged_features.csv")
RAW_TRANSACTIONS_PATH = Path("datasource/raw/all_transactions_labeled.csv")

# --- Global/In-Memory State (for demonstration) ---
# In a real system, this would be a persistent database (e.g., Redis, PostgreSQL)
wallet_features_cache = {}
wallet_transactions_history = defaultdict(list)

def load_initial_state():
    """
    Loads the existing merged features and raw transactions to simulate
    the system's initial knowledge.
    """
    global wallet_features_cache, wallet_transactions_history

    print(f"Loading initial merged features from {MERGED_FEATURES_PATH}...")
    if MERGED_FEATURES_PATH.exists():
        df_merged = pd.read_csv(MERGED_FEATURES_PATH)
        wallet_features_cache = df_merged.set_index('wallet_address').to_dict(orient='index')
        print(f"Loaded {len(wallet_features_cache)} wallets into feature cache.")
    else:
        print("Warning: Initial merged features not found. Starting with empty cache.")

    print(f"Loading initial transaction history from {RAW_TRANSACTIONS_PATH}...")
    if RAW_TRANSACTIONS_PATH.exists():
        df_tx = pd.read_csv(RAW_TRANSACTIONS_PATH)
        df_tx['timeStamp'] = pd.to_datetime(df_tx['timeStamp'], unit='s', errors='coerce')
        df_tx = df_tx.dropna(subset=['timeStamp', 'wallet_address'])
        # Store history per wallet, sorted by time
        for wallet, group in df_tx.groupby('wallet_address'):
            wallet_transactions_history[wallet] = group.sort_values('timeStamp').to_dict(orient='records')
        print(f"Loaded transaction history for {len(wallet_transactions_history)} wallets.")
    else:
        print("Warning: Raw transactions not found. Live feature updates will rely solely on new data.")


def _update_l0_features(current_features, new_tx, wallet_history):
    """
    Conceptually updates L0 features for a single wallet with a new transaction.
    In a real system, this would query a transaction DB for full history.
    """
    # Simulate retrieving full history (in reality, query DB or stream)
    temp_df = pd.DataFrame(wallet_history + [new_tx])
    temp_df['timeStamp'] = pd.to_datetime(temp_df['timeStamp'], unit='s', errors='coerce') # Ensure datetime

    # Recalculate L0 for the specific wallet using the updated history
    if temp_df.empty or temp_df['timeStamp'].isnull().all():
        return current_features # Cannot compute if no valid timestamps
    
    # Basic L0 recalculation (simplified)
    tx_count = len(temp_df)
    total_value = pd.to_numeric(temp_df['value'], errors='coerce').sum()
    avg_value = pd.to_numeric(temp_df['value'], errors='coerce').mean()
    wallet_age = (temp_df['timeStamp'].max() - temp_df['timeStamp'].min()).days + 1
    active_days = temp_df['timeStamp'].dt.date.nunique()
    
    # Dormant awakenings (re-calculate on full history)
    time_diffs_days = temp_df['timeStamp'].diff().dt.days.dropna()
    dormant_awakenings = (time_diffs_days > 30).sum()

    current_features.update({
        'total_transactions': tx_count,
        'total_value': total_value,
        'avg_tx_value': avg_value,
        'wallet_age_days': wallet_age,
        'active_days': active_days,
        'dormant_awaken_count': dormant_awakenings,
        # ... other L0 features would be updated here ...
    })
    return current_features

def _update_l1_features(current_features, new_tx, wallet_history):
    """
    Conceptually updates L1 behavioral features for a single wallet.
    This would be more complex, requiring sorted history for diffs.
    """
    temp_df = pd.DataFrame(wallet_history + [new_tx])
    temp_df['timeStamp'] = pd.to_datetime(temp_df['timeStamp'], unit='s', errors='coerce')
    temp_df = temp_df.sort_values('timeStamp').dropna(subset=['timeStamp'])

    if len(temp_df) < 2:
        # Cannot compute diff-based features with less than 2 transactions
        return current_features
    
    tx_times = temp_df["timeStamp"].values
    tx_diffs = np.diff(tx_times).astype('timedelta64[h]').astype(int)

    # Recalculate L1 features (simplified)
    burst_tx_ratio = (tx_diffs <= 1).sum() / len(tx_diffs) if len(tx_diffs) > 0 else 0
    failure_ratio = pd.to_numeric(temp_df["isError"], errors='coerce').sum() / len(temp_df)

    current_features.update({
        'burst_tx_ratio': burst_tx_ratio,
        'failure_ratio': failure_ratio,
        # ... other L1 features would be updated here ...
    })
    return current_features

def _update_l2_features(current_features, new_tx):
    """
    Conceptually updates L2 risk flags. This would involve querying known
    fraud/suspicious lists and potentially the network for counterparty labels.
    """
    # Example: if new_tx['to'] is a known fraud address, update num_fraud_counterparties
    # In a real system, 'wallet_risk' would be a lookup table/DB.
    if 'num_fraud_counterparties' not in current_features: current_features['num_fraud_counterparties'] = 0
    if 'num_suspicious_counterparties' not in current_features: current_features['num_suspicious_counterparties'] = 0
    # Simplified: assume we have a global lookup for known risky addresses
    # This part needs to be more robust, potentially loading specific lists.
    # For demo, let's assume the new_tx itself has a 'to_label' for simplicity.
    
    # This part is highly dependent on how 'wallet_risk' is maintained live.
    # For now, we will just pass through the existing values or assume no change for simplicity.

    # Placeholder for updating L2 flags
    # current_features['mixer_flag'] = (current_features.get('num_suspicious_counterparties', 0) > 0)
    # current_features['rapid_bridging_flag'] = (current_features.get('layer_hopping_count', 0) > 2) # Assume layer_hopping_count is updated elsewhere

    return current_features

def _update_l3_metaai_xai_tags(current_features):
    """
    Conceptually re-calculates L3 meta-AI and XAI tags based on updated features.
    This would involve re-running IsolationForest/DBSCAN or using pre-trained models.
    """
    # This is highly simplified. In reality, you'd feed the updated feature vector
    # into pre-trained IsolationForest/DBSCAN models and then re-evaluate XAI tags.
    
    # For demo, just re-evaluate xai tags based on thresholds
    current_features['burst_tx_flag'] = current_features.get('burst_tx_ratio', 0) > 0.95
    current_features['dormant_awake_flag'] = current_features.get('dormant_awaken_count', 0) > 1
    current_features['counterparty_fraud_flag'] = current_features.get('num_fraud_counterparties', 0) >= 1
    
    # Anomaly flags would come from actual model inference, not just simple thresholds
    # For demo, we assume they are updated based on some real-time anomaly score
    current_features['anomaly_iso'] = current_features.get('anomaly_iso', 0)
    current_features['anomaly_dbscan'] = current_features.get('anomaly_dbscan', 0)
    current_features['anomaly_flag'] = (current_features['anomaly_iso'] == 1) | (current_features['anomaly_dbscan'] == 1)

    current_features['failure_flag'] = current_features.get('failure_ratio', 0) > 0.5

    # --- Placeholder for other flags ---
    # These would need their base features updated first
    # For now, default to 0 if not present
    current_features['smart_contract_misuse_flag'] = current_features.get('smart_contract_failures', 0) > 0
    current_features['rapid_bridging_flag'] = current_features.get('layer_hopping_count', 0) > 2
    current_features['circular_flow_flag'] = current_features.get('circular_tx_ratio', 0) > 0.6 # Assuming ratio calculated
    current_features['gas_anomaly_flag'] = current_features.get('avg_gas_fee', 0) > current_features.get('avg_gas_fee_95_quantile', 999999999) # Requires quantile from full dataset

    # Re-generate xai_reason_code
    reasons = []
    if current_features.get('burst_tx_flag', False): reasons.append("burst_tx")
    if current_features.get('dormant_awake_flag', False): reasons.append("dormant_awakened")
    if current_features.get('counterparty_fraud_flag', False): reasons.append("fraud_link")
    if current_features.get('anomaly_flag', False): reasons.append("model_anomaly")
    if current_features.get('failure_flag', False): reasons.append("high_failure_rate")
    if current_features.get('smart_contract_misuse_flag', False): reasons.append("smart_contract_misuse")
    if current_features.get('rapid_bridging_flag', False): reasons.append("rapid_layer_hopping")
    if current_features.get('circular_flow_flag', False): reasons.append("circular_fund_flow")
    if current_features.get('gas_anomaly_flag', False): reasons.append("unusual_gas_fee")
    
    current_features['xai_reason_code'] = "|".join(reasons) if reasons else "clean"
    current_features['xai_flag'] = current_features['xai_reason_code'] != "clean"

    return current_features

def update_wallet_features_live(new_tx):
    """
    Orchestrates the live update of features for a wallet based on a new transaction.
    """
    wallet_address = new_tx.get('wallet_address')
    if not wallet_address: # Fallback if wallet_address is missing in tx for some reason
        if new_tx.get('from'): wallet_address = new_tx['from']
        elif new_tx.get('to'): wallet_address = new_tx['to']
        else: return None # Cannot process without a wallet address

    print(f"Processing new transaction for wallet: {wallet_address}")

    # Get current features for the wallet, or initialize if new
    current_features = wallet_features_cache.get(wallet_address, {'wallet_address': wallet_address})
    
    # Update transaction history for the wallet
    wallet_transactions_history[wallet_address].append(new_tx)
    # Keep history sorted (important for diffs)
    wallet_transactions_history[wallet_address] = sorted(wallet_transactions_history[wallet_address], key=lambda x: pd.to_datetime(x.get('timeStamp'), unit='s', errors='coerce'))

    # --- Update each feature layer conceptually ---
    current_features = _update_l0_features(current_features, new_tx, wallet_transactions_history[wallet_address])
    current_features = _update_l1_features(current_features, new_tx, wallet_transactions_history[wallet_address])
    current_features = _update_l2_features(current_features, new_tx) # L2 is harder to update incrementally without global context
    current_features = _update_l3_metaai_xai_tags(current_features) # Recalculate XAI tags based on updated features

    # Store updated features back in cache
    wallet_features_cache[wallet_address] = current_features

    return current_features

if __name__ == "__main__":
    print("Running realtime_feature_updater.py directly. Loading initial state...")
    load_initial_state()
    print("Feature updater ready. No direct execution of updates here; intended for import.") 