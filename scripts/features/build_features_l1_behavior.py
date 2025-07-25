"""
L1 - Behavioral Features (from txn histories)
  - Transaction frequency patterns
  - Dormancy/awakening patterns
  - Error/failure patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime

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
    group = group.sort_values("timeStamp")
    tx_times = group["timeStamp"].values
    tx_diffs = np.diff(tx_times).astype('timedelta64[h]').astype(int)

    feature = {
        "wallet_address": wallet,
        "total_transactions": len(group),
        "wallet_age_days": (group["timeStamp"].max() - group["timeStamp"].min()).days + 1,
        "active_days": group["timeStamp"].dt.date.nunique(),
        "burst_tx_ratio": (tx_diffs <= 1).sum() / len(tx_diffs) if len(tx_diffs) > 0 else 0, #fraction of transactions occurring within 1 hour of the previous
        "dormant_awaken_count": ((tx_diffs > 30*24).sum()) if len(tx_diffs) > 0 else 0,
        "failure_ratio": group["isError"].astype(int).sum() / len(group), #fraction of failed txns
        "mean_tx_interval_hours": np.mean(tx_diffs) if len(tx_diffs) > 0 else np.nan,
        "std_tx_interval_hours": np.std(tx_diffs) if len(tx_diffs) > 0 else np.nan,
        "weekend_tx_ratio": (group["timeStamp"].dt.weekday >= 5).mean(),
        "txn_span_hours": (group["timeStamp"].max() - group["timeStamp"].min()).total_seconds() / 3600,
        "night_tx_ratio": ((group["timeStamp"].dt.hour < 6) | (group["timeStamp"].dt.hour >= 22)).mean(),
    }

    feature_rows.append(feature)

# --- Output ---
features_df = pd.DataFrame(feature_rows)
OUTPUT_PATH = "datasource/processed/features_l1_behavior.csv"
features_df.to_csv(OUTPUT_PATH, index=False)
print(f"L1 behavioral features saved to {OUTPUT_PATH}")