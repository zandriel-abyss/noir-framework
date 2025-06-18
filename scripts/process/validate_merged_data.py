# scripts/process/validate_merged_data.py

import pandas as pd
from pathlib import Path

# Path to merged CSV file
MERGED_CSV_PATH = Path("datasource/raw/all_transactions_labeled.csv")

# Load the data
df = pd.read_csv(MERGED_CSV_PATH)

# 1. Schema check
print("\n Columns Present:")
print(df.columns.tolist())

# 2. Class balance
print("\n Label Distribution:")
print(df['label'].value_counts())

print("\n Unique Wallets per Label:")
print(df.groupby('label')['wallet_address'].nunique())

# 3. Missing values
print("\n  Missing Values Summary:")
print(df[['wallet_address', 'from', 'to', 'value', 'label']].isnull().sum())

# 4. Timestamp analysis
try:
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')
    print("\n Timestamp Range:")
    print("Start:", df['timeStamp'].min())
    print("End  :", df['timeStamp'].max())
except Exception as e:
    print("\n Error parsing timestamps:", e)

# 5. Duplicate check
num_duplicates = df.duplicated(subset=['hash']).sum()
print(f"\n Duplicate Transactions (by tx hash): {num_duplicates}")

# 6. Basic stats
print("\n Value Stats (ETH wei):")
print(df['value'].describe())

print("\n Gas Used Stats:")
print(df['gasUsed'].describe())