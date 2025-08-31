"""
L1 - Behavioral Features (from txn histories)
  - Transaction frequency patterns
  - Dormancy/awakening patterns
  - Error/failure patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime
from scripts.features.feature_utils import calculate_l1_features # Import the utility function

# Load merged transactions
TX_PATH = "datasource/raw/all_transactions_labeled.csv"
df = pd.read_csv(TX_PATH)

# --- Preprocessing ---
# Convert timestamp to datetime
df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit="s")

# Group by wallet and sort transactions chronologically
df = df.sort_values(by=["wallet_address", "timeStamp"])

# --- Feature Engineering ---
feature_rows = []

for wallet, group in df.groupby("wallet_address"):
    feature_rows.append(calculate_l1_features(group.copy(), wallet))

# --- Output ---
features_df = pd.DataFrame(feature_rows)
OUTPUT_PATH = "datasource/processed/features_l1_behavior.csv"
features_df.to_csv(OUTPUT_PATH, index=False)
print(f"L1 behavioral features saved to {OUTPUT_PATH}")