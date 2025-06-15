# scripts/utils/merge_mixer_recipient_txns.py
import pandas as pd
from pathlib import Path

# Files
existing_file = Path("datasource/raw/all_transactions_labeled.csv")
new_file = Path("datasource/raw/mixer_recipients_transactions.csv")
output_file = Path("datasource/raw/all_transactions_labeled.csv")  # overwrite

# Load
df_existing = pd.read_csv(existing_file)
df_new = pd.read_csv(new_file)

# Normalize wallet address casing
df_existing['wallet_address'] = df_existing['wallet_address'].str.lower()
df_new['wallet_address'] = df_new['wallet_address'].str.lower()

# Merge (avoid duplicates)
df_merged = pd.concat([df_existing, df_new], ignore_index=True)
df_merged = df_merged.drop_duplicates(subset=["hash"])  # Drop duplicate txns

# Save back
df_merged.to_csv(output_file, index=False)
print(f"✅ Merged mixer-recipient txns into: {output_file}")