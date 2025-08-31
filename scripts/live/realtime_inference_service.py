import asyncio
import websockets
import json
from web3 import Web3
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import time
import sys

# Adjust the path to import from the scripts/features directory
sys.path.append(str(Path(__file__).parent.parent))
from features.feature_utils import calculate_l0_features, calculate_l1_features, calculate_l2_features, calculate_l3_metaai_features, calculate_l3_xai_tags
from live.stream_transactions import stream_live_transactions # Import the live streaming function

# --- Configuration ---
# Your Alchemy WebSocket URL for Ethereum Mainnet (repeated here for clarity, though stream_transactions has it)
ALCHEMY_WSS_URL_ETH = "wss://eth-mainnet.g.alchemy.com/v2/CrTsyvhAZiKhQmqP7hz1E"

# New Configuration for Network Source
NETWORK_SOURCE = "Ethereum Mainnet" # Can be changed to "Arbitrum", "Optimism", etc.

MODELS_DIR = Path('output/models')
RAW_DATA_PATH = Path('datasource/raw/all_transactions_labeled.csv')

# --- Global variables for loaded models and data ---
rf_model = None
xgb_model = None
iso_model_l3 = None
dbscan_model_l3 = None
scaler_l3 = None

model_feature_columns = [] # Features expected by the final RF/XGB models
l3_metaai_feature_columns = [] # Features expected by L3 Meta-AI scaler/models
wallet_risk_labels_global = {}
gas_fee_95th_percentile_global = 0.0

# In-memory store for wallet transaction histories
# In a production system, this would be a persistent, scalable database (e.g., Redis, Cassandra)
wallet_histories = {}

async def load_all_models_and_data():
    global rf_model, xgb_model, iso_model_l3, dbscan_model_l3, scaler_l3
    global model_feature_columns, l3_metaai_feature_columns, wallet_risk_labels_global, gas_fee_95th_percentile_global
    global flagged_transactions_count # New global variable
    global flagged_transactions_log # New global variable for logging

    print("Loading models and data...")
    try:
        # Load supervised models
        rf_model = joblib.load(MODELS_DIR / 'random_forest_model.joblib')
        xgb_model = joblib.load(MODELS_DIR / 'xgboost_model.joblib')
        print("Supervised models loaded.")

        # Load L3 Meta-AI models and scaler
        iso_model_l3 = joblib.load(MODELS_DIR / 'iso_forest_l3_metaai.joblib')
        dbscan_model_l3 = joblib.load(MODELS_DIR / 'dbscan_l3_metaai.joblib')
        scaler_l3 = joblib.load(MODELS_DIR / 'scaler_l3_metaai.joblib')
        print("L3 Meta-AI models and scaler loaded.")

        # Load feature column lists
        model_feature_columns = joblib.load(MODELS_DIR / 'model_feature_columns.joblib')
        l3_metaai_feature_columns = joblib.load(MODELS_DIR / 'l3_metaai_feature_columns.joblib')
        print("Feature column lists loaded.")

        # Load wallet risk labels and global gas percentile
        wallet_risk_labels_global = joblib.load(MODELS_DIR / 'wallet_risk_labels.joblib')
        gas_fee_95th_percentile_global = joblib.load(MODELS_DIR / 'global_gas_95th_percentile.joblib')
        print("Wallet risk labels and global gas percentile loaded.")

        print("All models and data loaded successfully.")
        flagged_transactions_count = 0 # Initialize counter
        flagged_transactions_log = [] # Initialize log list
    except Exception as e:
        print(f"Error loading models or data: {e}")
        sys.exit(1) # Exit if essential components can't be loaded

# Define path for the flagged transactions log file
FLAGGED_TRANSACTIONS_FILE = Path('output/flagged_transactions.csv')

async def save_flagged_transactions_to_file():
    global flagged_transactions_log
    if not flagged_transactions_log:
        return

    # Convert list of dicts to DataFrame
    df_new_flags = pd.DataFrame(flagged_transactions_log)
    flagged_transactions_log = [] # Clear log after processing

    # Check if file exists to determine if header is needed
    file_exists = FLAGGED_TRANSACTIONS_FILE.exists()

    with open(FLAGGED_TRANSACTIONS_FILE, 'a') as f:
        df_new_flags.to_csv(f, header=not file_exists, index=False)
    print(f"Saved {len(df_new_flags)} flagged transactions to {FLAGGED_TRANSACTIONS_FILE}")


# --- Real-time Inference Function ---
def perform_realtime_inference(new_transaction: dict, wallet_history_df: pd.DataFrame) -> dict:
    """Simulates real-time inference for a new transaction for a specific wallet."""
    global rf_model, xgb_model, iso_model_l3, dbscan_model_l3, scaler_l3
    global model_feature_columns, l3_metaai_feature_columns, wallet_risk_labels_global, gas_fee_95th_percentile_global
    global flagged_transactions_count # New global variable
    global flagged_transactions_log # New global variable for logging
    print(f"\n--- Processing new transaction for wallet: {new_transaction['wallet_address']} ---")
    
    # Convert new transaction to DataFrame row and append to history
    new_tx_df = pd.DataFrame([new_transaction])
    updated_wallet_history = pd.concat([wallet_history_df, new_tx_df], ignore_index=True)
    updated_wallet_history['timeStamp'] = pd.to_datetime(updated_wallet_history['timeStamp'], unit='s', errors='coerce')
    updated_wallet_history['value'] = pd.to_numeric(updated_wallet_history['value'], errors='coerce').fillna(0)

    wallet_address = new_transaction['wallet_address']

    # 1. Calculate L0 Features
    l0_features = calculate_l0_features(updated_wallet_history.copy(), wallet_address)
    current_features_dict = {**l0_features}

    # 2. Calculate L1 Features
    l1_features = calculate_l1_features(updated_wallet_history.copy(), wallet_address)
    current_features_dict.update({k: v for k, v in l1_features.items() if k != 'wallet_address'})

    # Prepare L0/L1 features for L2 calculation (from the dictionary)
    wallet_l0_l1_features_dict = current_features_dict.copy() 

    # 3. Calculate L2 Features
    l2_features = calculate_l2_features(updated_wallet_history.copy(), wallet_address, 
                                        wallet_risk_labels_global, wallet_l0_l1_features_dict, 
                                        gas_fee_95th_percentile_global)
    current_features_dict.update({k: v for k, v in l2_features.items() if k != 'wallet_address'})

    # 4. Calculate L3 Meta-AI Features
    # Convert current_features_dict to DataFrame for scaling/model prediction. Ensure unique wallet_address for current_features before passing.
    current_features_df = pd.DataFrame([current_features_dict])

    # Add any missing columns to current_features_df with default values before filtering/reordering
    for col in l3_metaai_feature_columns:
        if col not in current_features_df.columns:
            current_features_df[col] = 0.0 # Default to 0.0 for numeric features
    
    current_features_df = current_features_df.fillna(0)

    # Filter and reorder current_features_df to only include columns the scaler was trained on, in the correct order
    X_for_metaai_scaling = current_features_df[l3_metaai_feature_columns]
    
    l3_metaai_features = calculate_l3_metaai_features(X_for_metaai_scaling.copy(), scaler_l3, iso_model_l3, dbscan_model_l3)
    current_features_dict.update({k: v for k, v in l3_metaai_features.items() if k != 'wallet_address'})

    # 5. Calculate L3 XAI Tags
    # Convert current_features_dict to DataFrame for XAI tag calculation.
    current_features_df = pd.DataFrame([current_features_dict])

    # Add any missing columns to current_features_df that L3 XAI expects
    xai_expected_cols = model_feature_columns # Use the comprehensive list of model features for XAI tags
    for col in xai_expected_cols:
        if col not in current_features_df.columns:
            current_features_df[col] = 0.0
    current_features_df = current_features_df.fillna(0) # Fill NaN after adding columns

    # Ensure 'avg_gas_fee' and 'gas_fee_95th_percentile_global' are correctly handled
    # They might not be in the feature list for XAI tag calculation if they were already part of L0/L1 and passed through
    temp_features_for_xai = current_features_df.copy()
    if 'avg_gas_fee' in temp_features_for_xai.columns and 'avg_gas_fee' in new_transaction:
        temp_features_for_xai['avg_gas_fee'].iloc[0] = new_transaction.get('avg_gas_fee', temp_features_for_xai['avg_gas_fee'].iloc[0])
    
    # Make sure global gas percentile is available for XAI if it needs it directly
    if 'gas_fee_95th_percentile_global' not in temp_features_for_xai.columns:
        temp_features_for_xai['gas_fee_95th_percentile_global'] = gas_fee_95th_percentile_global # Add it if missing
    

    l3_xai_tags = calculate_l3_xai_tags(temp_features_for_xai.copy())
    current_features_dict.update({k: v for k, v in l3_xai_tags.items() if k != 'wallet_address'})

    # Prepare features for final model prediction
    current_features_df = pd.DataFrame([current_features_dict])
    
    # The model_feature_columns list ensures the input to the model has the correct features in the correct order
    for col in model_feature_columns:
        if col not in current_features_df.columns:
            current_features_df[col] = 0.0 # Default to 0.0 for numeric features

    # Final X_predict: filter and strictly reorder
    X_predict = current_features_df[model_feature_columns]
    X_predict = X_predict.fillna(0)

    # Make prediction
    prediction = rf_model.predict(X_predict)[0] # Using Random Forest for final prediction
    prediction_proba = rf_model.predict_proba(X_predict)[0]

    # Get XAI Reason Code
    reason_code = l3_xai_tags.get('xai_reason_code', 'clean')
    xai_flag = l3_xai_tags.get('xai_flag', 0)

    return {
        'wallet_address': wallet_address,
        'transaction_hash': new_transaction.get('hash', 'N/A'),
        'predicted_label': int(prediction),
        'fraud_probability': prediction_proba[1],
        'xai_reason_code': reason_code,
        'xai_flag': int(xai_flag),
        'network_source': NETWORK_SOURCE, # New field
        'processing_timestamp': pd.Timestamp.now().isoformat() # New field for processing time
    }

async def main():
    await load_all_models_and_data()

    print("Starting real-time inference service for live Ethereum transactions...")
    try:
        async for live_transaction in stream_live_transactions():
            # More robust check: Ensure 'from' address is present, not None, and is a string
            if not isinstance(live_transaction.get('from'), str):
                print(f"Skipping transaction {live_transaction.get('hash', 'N/A')}: 'from' address is missing, None, or not a string (type: {type(live_transaction.get('from'))}).")
                continue

            wallet_address = live_transaction['from'].lower() # Or 'to' depending on focus
            
            # Update wallet history in memory
            if wallet_address not in wallet_histories:
                # For a new wallet, we initialize its history with just the current transaction
                wallet_histories[wallet_address] = pd.DataFrame([live_transaction])
            else:
                # Append new transaction to existing history
                new_tx_df = pd.DataFrame([live_transaction])
                wallet_histories[wallet_address] = pd.concat([wallet_histories[wallet_address], new_tx_df], ignore_index=True)
                wallet_histories[wallet_address] = wallet_histories[wallet_address].sort_values(by='timeStamp').reset_index(drop=True)
            
            # Perform inference for the current wallet with its updated history
            result = perform_realtime_inference(live_transaction, wallet_histories[wallet_address].copy())
            print(f"[LIVE INFERENCE] Wallet: {result['wallet_address']}, TxHash: {result['transaction_hash'][:10]}..., "
                  f"Predicted: {'Fraud' if result['predicted_label'] == 1 else 'Normal'} (Prob: {result['fraud_probability']:.2f}), "
                  f"Reason: {result['xai_reason_code']}")
            
            # If fraud is predicted, log it and increment counter
            if result['predicted_label'] == 1:
                global flagged_transactions_count
                global flagged_transactions_log
                flagged_transactions_count += 1
                flagged_transactions_log.append(result) # Add result to log
                print(f"[TOTAL FLAGGED: {flagged_transactions_count}] Current flagged transaction logged.")

                # Periodically save to file (e.g., every 10 flagged transactions, or every minute)
                # For now, let's save after every flagged transaction for immediate feedback, adjust as needed
                await save_flagged_transactions_to_file()

    except KeyboardInterrupt:
        print("Real-time inference service stopped by user.")
        # Save any remaining flagged transactions before exiting
        await save_flagged_transactions_to_file()
    except Exception as e:
        print(f"An error occurred during live inference: {e}")
        # Save any remaining flagged transactions before exiting on error
        await save_flagged_transactions_to_file()

if __name__ == "__main__":
    asyncio.run(main()) 