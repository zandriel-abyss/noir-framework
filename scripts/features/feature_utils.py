import pandas as pd
import numpy as np

def calculate_l0_features(wallet_transactions: pd.DataFrame, wallet_address: str) -> dict:
    """Calculates L0 (raw transaction stats) features for a single wallet's transactions."""
    if wallet_transactions.empty:
        return {
            'wallet_address': wallet_address,
            'total_transactions': 0,
            'total_value': 0,
            'avg_tx_value': 0,
            'wallet_age_days': 0,
            'active_days': 0,
            'num_unique_from': 0,
            'num_unique_to': 0,
            'circular_tx_count': 0,
            'dormant_awaken_count': 0, # L0 version, simpler
            'label': 'unknown'
        }

    wallet_transactions = wallet_transactions.sort_values('timeStamp')
    
    tx_count = len(wallet_transactions)
    total_value = wallet_transactions['value'].sum()
    avg_value = wallet_transactions['value'].mean()
    
    min_ts = wallet_transactions['timeStamp'].min()
    max_ts = wallet_transactions['timeStamp'].max()
    wallet_age = (max_ts - min_ts).days if pd.notna(min_ts) and pd.notna(max_ts) else 0

    active_days = wallet_transactions['timeStamp'].dt.date.nunique()
    
    unique_from = wallet_transactions['from'].nunique()
    unique_to = wallet_transactions['to'].nunique()
    circular_count = ((wallet_transactions['from'] == wallet_transactions['to']) & (wallet_transactions['from'] == wallet_address)).sum()

    time_diffs_days = wallet_transactions['timeStamp'].diff().dt.days
    dormant_awakenings_l0 = (time_diffs_days > 30).sum()
    
    label = wallet_transactions['label'].mode()[0] if 'label' in wallet_transactions.columns else 'unknown'

    return {
        'wallet_address': wallet_address,
        'total_transactions': tx_count,
        'total_value': total_value,
        'avg_tx_value': avg_value,
        'wallet_age_days': wallet_age,
        'active_days': active_days,
        'num_unique_from': unique_from,
        'num_unique_to': unique_to,
        'circular_tx_count': circular_count,
        'dormant_awaken_count': dormant_awakenings_l0,
        'label': label
    }

def calculate_l1_features(wallet_transactions: pd.DataFrame, wallet_address: str) -> dict:
    """Calculates L1 (behavioral) features for a single wallet's transactions."""
    if wallet_transactions.empty or len(wallet_transactions) < 2:
        return {
            "wallet_address": wallet_address,
            "total_transactions": len(wallet_transactions),
            "wallet_age_days": 0,
            "active_days": 0,
            "burst_tx_ratio": 0,
            "dormant_awaken_count": 0, # L1 version, in hours
            "failure_ratio": 0,
            "mean_tx_interval_hours": np.nan,
            "std_tx_interval_hours": np.nan,
            "weekend_tx_ratio": 0,
            "txn_span_hours": 0,
            "night_tx_ratio": 0,
        }

    wallet_transactions = wallet_transactions.sort_values("timeStamp")
    tx_times = wallet_transactions["timeStamp"].values
    tx_diffs_hours = np.diff(tx_times).astype('timedelta64[h]').astype(int)

    burst_tx_ratio = (tx_diffs_hours <= 1).sum() / len(tx_diffs_hours) if len(tx_diffs_hours) > 0 else 0
    dormant_awakenings_l1 = (tx_diffs_hours > 30*24).sum() if len(tx_diffs_hours) > 0 else 0
    
    # Handle 'isError' column which might not be present or might be mixed type
    if 'isError' in wallet_transactions.columns:
        failure_ratio = wallet_transactions["isError"].astype(int).sum() / len(wallet_transactions)
    else:
        failure_ratio = 0 # Assume no failures if column not present

    mean_tx_interval_hours = np.mean(tx_diffs_hours) if len(tx_diffs_hours) > 0 else np.nan
    std_tx_interval_hours = np.std(tx_diffs_hours) if len(tx_diffs_hours) > 0 else np.nan
    weekend_tx_ratio = (wallet_transactions["timeStamp"].dt.weekday >= 5).mean()
    txn_span_hours = (wallet_transactions["timeStamp"].max() - wallet_transactions["timeStamp"].min()).total_seconds() / 3600
    night_tx_ratio = ((wallet_transactions["timeStamp"].dt.hour < 6) | (wallet_transactions["timeStamp"].dt.hour >= 22)).mean()

    return {
        "wallet_address": wallet_address,
        "total_transactions": len(wallet_transactions),
        "wallet_age_days": (wallet_transactions["timeStamp"].max() - wallet_transactions["timeStamp"].min()).days + 1,
        "active_days": wallet_transactions["timeStamp"].dt.date.nunique(),
        "burst_tx_ratio": burst_tx_ratio,
        "dormant_awaken_count": dormant_awakenings_l1,
        "failure_ratio": failure_ratio,
        "mean_tx_interval_hours": mean_tx_interval_hours,
        "std_tx_interval_hours": std_tx_interval_hours,
        "weekend_tx_ratio": weekend_tx_ratio,
        "txn_span_hours": txn_span_hours,
        "night_tx_ratio": night_tx_ratio,
    }

def calculate_l2_features(wallet_transactions: pd.DataFrame, wallet_address: str, wallet_risk_labels: dict, 
                          wallet_l0_l1_features: dict, global_gas_95th_percentile: float) -> dict:
    """Calculates L2 (risk flags) features for a single wallet's transactions, using its aggregated L0/L1 features."""
    
    feature = {"wallet_address": wallet_address}

    # Initialize counts for counterparties
    num_fraud_counterparties = 0
    num_suspicious_counterparties = 0
    num_normal_counterparties = 0

    # Count interactions with risky counterparties (where current wallet is 'from')
    for _, row in wallet_transactions.iterrows():
        tgt = row['to'].lower()
        if tgt != wallet_address: # Only consider interactions with external parties
            tgt_label = wallet_risk_labels.get(tgt, None)
            if tgt_label == 'fraud':
                num_fraud_counterparties += 1
            elif tgt_label == 'suspicious':
                num_suspicious_counterparties += 1
            elif tgt_label == 'normal':
                num_normal_counterparties += 1
    
    feature['num_fraud_counterparties'] = num_fraud_counterparties
    feature['num_suspicious_counterparties'] = num_suspicious_counterparties
    feature['num_normal_counterparties'] = num_normal_counterparties

    # Now, calculate the other L2 flags that depend on L0/L1 aggregated features
    # These values come from wallet_l0_l1_features which is the aggregated state of the wallet
    
    # Ensure necessary L0/L1 features are available, provide defaults if not
    total_transactions = wallet_l0_l1_features.get('total_transactions', 0)
    circular_tx_count = wallet_l0_l1_features.get('circular_tx_count', 0)
    avg_gas_fee = wallet_l0_l1_features.get('avg_gas_fee', 0.0) # Assume avg_gas_fee is part of L0/L1 or derived
    layer_hopping_count = wallet_l0_l1_features.get('layer_hopping_count', 0) # Assume layer_hopping_count is part of L0/L1 or derived
    smart_contract_failures = wallet_l0_l1_features.get('smart_contract_failures', 0) # Assume smart_contract_failures is part of L0/L1 or derived
    
    # circular_flow_flag (using the definition from L2 original script, based on circular_tx_count/total_transactions)
    if total_transactions > 0:
        circular_tx_ratio = circular_tx_count / total_transactions
        feature['circular_flow_flag'] = 1 if circular_tx_ratio > 0 else 0 # Original L2 just checks if circular_tx_count > 0 from df_tx
    else:
        feature['circular_flow_flag'] = 0
    
    # gas_anomaly_flag (requires global 95th percentile, passed as argument)
    feature['gas_anomaly_flag'] = 1 if avg_gas_fee > global_gas_95th_percentile else 0
    
    # rapid_bridging_flag
    feature['rapid_bridging_flag'] = 1 if layer_hopping_count > 0 else 0 # Original L2 used quantile(0.90) from full dataset
    
    # smart_contract_misuse_flag
    feature['smart_contract_misuse_flag'] = 1 if smart_contract_failures > 0 else 0
    
    # mixer_flag
    feature['mixer_flag'] = 1 if num_suspicious_counterparties > 0 else 0
    
    # mixer_then_bridge_flag
    feature['mixer_then_bridge_flag'] = 1 if (feature['mixer_flag'] == 1 and layer_hopping_count > 1) else 0

    # mixer_exit_tx_count (requires sequence, simplified for now)
    # This is hard to calculate for a single new transaction without deep sequence analysis
    # For this utility, we'll simplify and make it depend on the mixer_flag and total_transactions
    # In a real system, this would need stateful processing or a specialized graph query
    feature['mixer_exit_tx_count'] = wallet_l0_l1_features.get('mixer_exit_tx_count', 0) # Propagate if already computed
    
    # same_recipient_ratio (requires aggregation of 'to' addresses over time)
    # This is also better from an aggregated feature set
    feature['same_recipient_ratio'] = wallet_l0_l1_features.get('same_recipient_ratio', 0.0) # Propagate if already computed

    return feature

def calculate_l3_metaai_features(wallet_features: pd.DataFrame, scaler, iso_model, db_model) -> dict:
    """Calculates L3 (meta-AI) features for a single wallet's combined features."""
    # wallet_features is expected to be a DataFrame (even for a single row) with all L0-L2 features
    
    # Drop non-numeric and identify wallet address
    # wallets = wallet_features['wallet_address'] # wallet_address will be in the dict from outside
    
    X = wallet_features.fillna(0)

    # Scale data (use the pre-fitted scaler)
    X_scaled = scaler.transform(X) # Use transform, not fit_transform

    # Isolation Forest
    anomaly_iso = iso_model.predict(X_scaled)[0]
    anomaly_iso = 0 if anomaly_iso == 1 else 1  # 1 = anomaly, 0 = normal
    
    # DBSCAN
    # For single sample prediction, DBSCAN needs context. This will likely classify single samples as noise (-1)
    db_clusters = db_model.fit_predict(X_scaled) 
    anomaly_dbscan = 1 if db_clusters[0] == -1 else 0 # 1 = anomaly, 0 = normal
    
    # Combined Behavioral Tag
    combined_risk_tag = int(
        (wallet_features['num_fraud_counterparties'].iloc[0] > 2) |
        (wallet_features['num_suspicious_counterparties'].iloc[0] > 2) |
        (anomaly_iso == 1) |
        (anomaly_dbscan == 1)
    )

    return {
        'anomaly_iso': anomaly_iso,
        'anomaly_dbscan': anomaly_dbscan,
        'combined_risk_tag': combined_risk_tag
    }

def calculate_l3_xai_tags(wallet_features: pd.DataFrame) -> dict:
    """Calculates L3 (XAI tags) for a single wallet's combined features."""
    
    # These flags require pre-calculated features from L0, L1, L2, L3_metaai
    
    # Create default values for features if not present to avoid KeyError during XAI tag calculation
    # In a complete system, these would always be present after previous layers.
    defaults = {
        'burst_tx_ratio': 0.0,
        'dormant_awaken_count': 0,
        'num_fraud_counterparties': 0,
        'anomaly_iso': 0,
        'anomaly_dbscan': 0,
        'failure_ratio': 0.0,
        'smart_contract_failures': 0,
        'layer_hopping_count': 0,
        'circular_tx_ratio': 0.0,
        'avg_gas_fee': 0.0,
        'gas_anomaly_flag': 0 # Ensure this is present for `gas_anomaly_flag` logic
    }
    for col, default_val in defaults.items():
        if col not in wallet_features.columns:
            wallet_features[col] = default_val

    burst_tx_flag = wallet_features['burst_tx_ratio'].iloc[0] > 0.95
    dormant_awake_flag = wallet_features['dormant_awaken_count'].iloc[0] > 1
    counterparty_fraud_flag = wallet_features['num_fraud_counterparties'].iloc[0] >= 1
    anomaly_flag = (wallet_features['anomaly_iso'].iloc[0] == 1.0) | (wallet_features['anomaly_dbscan'].iloc[0] == 1)
    failure_flag = wallet_features['failure_ratio'].iloc[0] > 0.5
    smart_contract_misuse_flag = wallet_features['smart_contract_failures'].iloc[0] > 0
    rapid_bridging_flag = wallet_features['layer_hopping_count'].iloc[0] > 2
    circular_flow_flag = wallet_features['circular_tx_ratio'].iloc[0] > 0.6
    
    # This part needs to be careful: the batch script calculates quantile on the full DF.
    # For single wallet, we should use the globally pre-computed threshold.
    # Assume `global_gas_95th_percentile` is available in `wallet_features` for the purpose of this utility.
    # If `gas_anomaly_flag` is already computed in L2, use that.
    # Otherwise, compare `avg_gas_fee` against the global threshold.
    if 'gas_anomaly_flag' in wallet_features.columns:
        gas_anomaly_flag = wallet_features['gas_anomaly_flag'].iloc[0] # Use if already computed in L2
    elif 'avg_gas_fee' in wallet_features.columns and 'gas_fee_95th_percentile_global' in wallet_features.columns:
        gas_anomaly_flag = wallet_features['avg_gas_fee'].iloc[0] > wallet_features['gas_fee_95th_percentile_global'].iloc[0]
    else:
        gas_anomaly_flag = False
    
    # Build human-readable reason codes
    reasons = []
    if burst_tx_flag: reasons.append("burst_tx")
    if dormant_awake_flag: reasons.append("dormant_awakened")
    if counterparty_fraud_flag: reasons.append("fraud_link")
    if anomaly_flag: reasons.append("model_anomaly")
    if failure_flag: reasons.append("high_failure_rate")
    if smart_contract_misuse_flag: reasons.append("smart_contract_misuse")
    if rapid_bridging_flag: reasons.append("rapid_layer_hopping")
    if circular_flow_flag: reasons.append("circular_fund_flow")
    if gas_anomaly_flag: reasons.append("unusual_gas_fee")
    
    xai_reason_code = "|".join(reasons) if reasons else "clean"
    xai_flag = xai_reason_code != "clean"

    return {
        'burst_tx_flag': int(burst_tx_flag),
        'dormant_awake_flag': int(dormant_awake_flag),
        'counterparty_fraud_flag': int(counterparty_fraud_flag),
        'anomaly_flag': int(anomaly_flag),
        'failure_flag': int(failure_flag),
        'smart_contract_misuse_flag': int(smart_contract_misuse_flag),
        'rapid_bridging_flag': int(rapid_bridging_flag),
        'circular_flow_flag': int(circular_flow_flag),
        'gas_anomaly_flag': int(gas_anomaly_flag),
        'xai_reason_code': xai_reason_code,
        'xai_flag': int(xai_flag)
    } 